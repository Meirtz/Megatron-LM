# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dynamic Sparse Attention.

The module is model-agnostic: callers pass architecture dimensions directly and
keep model config classes out of the primitive layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.primitive.modules.lora import LinearLoRA, LoraConfig, normalize_lora_config
from megatron.lite.primitive.parallel.cp import (
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
)
from megatron.lite.primitive.parallel.thd import (
    reconstruct_packed_from_cp_parts,
    split_packed_to_cp_local,
)

from megatron.lite.primitive.kernels import dsa_kernels as _dsa_kernels

if TYPE_CHECKING:
    from megatron.lite.primitive.modules.attention.mla import MultiLatentAttention


DSA_INDEX_SHARE_TOPK_HOLDER_ATTR = "_dsa_index_share_topk_holder"


def _fused_indexer_sparse_attn(*args, value_dim: int | None = None, **kwargs):
    try:
        return _dsa_kernels.fused_indexer_sparse_attn(*args, value_dim=value_dim, **kwargs)
    except TypeError as exc:
        if "value_dim" not in str(exc):
            raise
        return _dsa_kernels.fused_indexer_sparse_attn(*args, **kwargs)


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """Attach the DSA indexer loss to the output without changing forward values."""

    main_loss_backward_scale: torch.Tensor | None = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


RMSNorm = te.RMSNorm


def _hadamard_transform_torch(x: torch.Tensor, scale: float) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard rotation requires power-of-two dim, got {n}")
    original_shape = x.shape
    y = x.reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (h * 2), h * 2)
        left = y[..., :h]
        right = y[..., h:]
        y = torch.cat([left + right, left - right], dim=-1)
        h *= 2
    return y.reshape(original_shape) * scale


try:
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard_transform
except Exception:  # pragma: no cover - optional CUDA extension
    _fast_hadamard_transform = None


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
    scale = x.shape[-1] ** -0.5
    if _fast_hadamard_transform is not None and x.is_cuda:
        return _fast_hadamard_transform(x, scale=scale)
    return _hadamard_transform_torch(x, scale=scale)


def build_rope_cache(
    *, dim: int, max_position_embeddings: int, rope_theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    positions = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def build_rotary_embeddings(
    *, position_ids: torch.Tensor, dim: int, rope_theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    device = position_ids.device
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).to(torch.float32) / dim)
    )
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    interleaved: bool = True,
) -> torch.Tensor:
    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    input_dtype = x.dtype
    x = x.float()
    cos = cos.to(device=x.device)[position_ids].float().unsqueeze(2)
    sin = sin.to(device=x.device)[position_ids].float().unsqueeze(2)
    if interleaved:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out.to(input_dtype)

    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(input_dtype)


def _rotary_embeddings_from_cache(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3 and sin.dim() == 3:
        return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)

    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    cos = cos.to(device=device)[position_ids].float()
    sin = sin.to(device=device)[position_ids].float()
    if cos.shape[-1] * 2 == dim:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _all_gather_cp(tensor: torch.Tensor, *, cp_size: int, cp_group) -> list[torch.Tensor]:
    if cp_size <= 1:
        return [tensor]
    if cp_group is None:
        raise RuntimeError("CP>1 requires a context-parallel process group.")
    from torch.distributed.nn.functional import all_gather

    return list(all_gather(tensor.contiguous(), group=cp_group))


class DSAIndexer(nn.Module):
    """Compute per-token top-k key indices for Dynamic Sparse Attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rope_interleaved: bool = True,
        layer_norm_eps: float = 1e-5,
        rope_first: bool = False,
        use_hadamard: bool = True,
    ):
        super().__init__()
        if index_head_dim < qk_rope_head_dim:
            raise ValueError("index_head_dim must be >= qk_rope_head_dim")
        self.num_heads = index_n_heads
        self.head_dim = index_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = index_head_dim - qk_rope_head_dim
        self.index_topk = index_topk
        self.rope_interleaved = rope_interleaved
        self.rope_first = rope_first
        self.use_hadamard = use_hadamard

        self.wq_b = nn.Linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(index_head_dim, eps=layer_norm_eps)
        self.weights_proj = nn.Linear(hidden_size, index_n_heads, bias=False)
        self.softmax_scale = index_head_dim**-0.5

    def forward_before_topk(
        self,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project GLM5 indexer inputs for Megatron's fused DSA kernels."""
        if attention_mask is not None:
            raise NotImplementedError(
                "GLM5 fused DSA indexer only supports causal masking; custom "
                "attention_mask is not supported."
            )
        batch, seq_len, _ = x.shape
        cos, sin = _rotary_embeddings_from_cache(
            cos, sin, position_ids, device=x.device, dtype=x.dtype, dim=self.qk_rope_head_dim
        )

        q = self.wq_b(q_resid).view(batch, seq_len, self.num_heads, self.head_dim)

        k = self.k_norm(self.wk(x))
        if self.rope_first:
            q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.qk_nope_head_dim], dim=-1)
            k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.qk_nope_head_dim], dim=-1)
        else:
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_pe = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2)
        k_pe = k_pe.squeeze(2)

        if self.rope_first:
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)
        else:
            q = torch.cat([q_nope, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)
        if self.use_hadamard:
            q = rotate_activation(q)
            k = rotate_activation(k)

        weights = self.weights_proj(x) * (self.num_heads**-0.5)
        return (
            q.transpose(0, 1).contiguous(),
            k.transpose(0, 1).contiguous(),
            weights.transpose(0, 1).contiguous(),
        )

    def forward(
        self,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, weights = self.forward_before_topk(
            x, q_resid, cos, sin, position_ids, attention_mask=attention_mask
        )
        topk_indices, _ = _dsa_kernels.indexer_topk(
            q,
            k,
            weights,
            min(self.index_topk, k.shape[0]),
            1,
            indexer_softmax_scale=self.softmax_scale,
        )
        return topk_indices


class DynamicSparseAttention(nn.Module):
    """Correctness-first DSA attention path."""

    @staticmethod
    def dense_attention_cls() -> type[MultiLatentAttention]:
        from megatron.lite.primitive.modules.attention.mla import MultiLatentAttention

        return MultiLatentAttention

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rms_norm_eps: float,
        rope_interleaved: bool = True,
        latent_rms_norm_eps: float | None = None,
        indexer_layer_norm_eps: float = 1e-5,
        indexer_rope_interleaved: bool | None = None,
        indexer_rope_first: bool = False,
        indexer_use_hadamard: bool = True,
        indexer_loss_coeff: float = 0.0,
        indexer_use_sparse_loss: bool = False,
        calculate_per_token_loss: bool = False,
        layer_idx: int | None = None,
        dsa_index_share_enabled: bool = False,
        dsa_indexer_type: str = "full",
        dsa_source_layer_idx: int | None = None,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_group=None,
        lora_config: LoraConfig | Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if cp_size < 1:
            raise ValueError(f"cp_size must be >= 1, got {cp_size}")
        if not 0 <= cp_rank < cp_size:
            raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_interleaved = rope_interleaved
        self.softmax_scale = self.qk_head_dim**-0.5
        self.index_topk = index_topk
        self.indexer_softmax_scale = index_head_dim**-0.5
        self.indexer_loss_coeff = indexer_loss_coeff
        self.indexer_use_sparse_loss = indexer_use_sparse_loss
        self.calculate_per_token_loss = calculate_per_token_loss
        if dsa_indexer_type not in {"full", "shared"}:
            raise ValueError(f"dsa_indexer_type must be 'full' or 'shared', got {dsa_indexer_type!r}")
        self.layer_idx = layer_idx
        self.index_share = bool(dsa_index_share_enabled)
        self.skip_topk = dsa_indexer_type == "shared"
        if self.skip_topk and not self.index_share:
            raise ValueError("dsa_indexer_type='shared' requires dsa_index_share_enabled=True")
        if self.index_share and self.layer_idx is None:
            raise ValueError("DSA index-share requires a global layer_idx")
        self.dsa_source_layer_idx = dsa_source_layer_idx if dsa_source_layer_idx is not None else layer_idx
        if self.skip_topk and self.dsa_source_layer_idx is None:
            raise ValueError("DSA shared layer requires dsa_source_layer_idx")
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_group = cp_group
        latent_rms_norm_eps = rms_norm_eps if latent_rms_norm_eps is None else latent_rms_norm_eps
        indexer_rope_interleaved = (
            rope_interleaved if indexer_rope_interleaved is None else indexer_rope_interleaved
        )

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=latent_rms_norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=latent_rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)
        lora = normalize_lora_config(lora_config)
        input_projection_lora = lora.enabled and lora.targets_module("linear_qkv")
        self.q_a_lora: LinearLoRA | None = None
        self.q_b_lora: LinearLoRA | None = None
        self.kv_a_lora: LinearLoRA | None = None
        self.kv_b_lora: LinearLoRA | None = None
        self.o_lora: LinearLoRA | None = None
        if input_projection_lora or (lora.enabled and lora.targets_module("q_a_proj")):
            self.q_a_lora = LinearLoRA(
                hidden_size,
                q_lora_rank,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                use_rslora=lora.use_rslora,
            )
        if input_projection_lora or (lora.enabled and lora.targets_module("q_b_proj")):
            self.q_b_lora = LinearLoRA(
                q_lora_rank,
                num_attention_heads * self.qk_head_dim,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                use_rslora=lora.use_rslora,
            )
        if input_projection_lora or (lora.enabled and lora.targets_module("kv_a_proj_with_mqa")):
            self.kv_a_lora = LinearLoRA(
                hidden_size,
                kv_lora_rank + qk_rope_head_dim,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                use_rslora=lora.use_rslora,
            )
        if input_projection_lora or (lora.enabled and lora.targets_module("kv_b_proj")):
            self.kv_b_lora = LinearLoRA(
                kv_lora_rank,
                num_attention_heads * (qk_nope_head_dim + v_head_dim),
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                use_rslora=lora.use_rslora,
            )
        if lora.enabled and lora.targets_module("linear_proj"):
            self.o_lora = LinearLoRA(
                num_attention_heads * v_head_dim,
                hidden_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                use_rslora=lora.use_rslora,
            )
        self.indexer: DSAIndexer | None = None
        if not self.skip_topk:
            self.indexer = DSAIndexer(
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                rope_interleaved=indexer_rope_interleaved,
                layer_norm_eps=indexer_layer_norm_eps,
                rope_first=indexer_rope_first,
                use_hadamard=indexer_use_hadamard,
            )
        self.register_buffer(
            "attn_sink",
            torch.full((num_attention_heads,), -1.0e20, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_seq_params=None,
        dsa_index_share_topk_holder: dict[Any, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError(
                "GLM5 fused DSA only supports causal masking; custom attention_mask "
                "is not supported."
            )
        if packed_seq_params is not None:
            if self.cp_size > 1:
                x, position_ids = self._gather_packed_cp_inputs(x, position_ids, packed_seq_params)
                cos, sin = self._gather_packed_cp_rotary(cos, sin, packed_seq_params, x.device)
            out = self._forward_packed_full(
                x,
                cos,
                sin,
                position_ids,
                packed_seq_params,
                dsa_index_share_topk_holder=dsa_index_share_topk_holder,
            )
            if self.cp_size > 1:
                out = split_packed_to_cp_local(
                    out,
                    cu_seqlens_padded=self._packed_cu_seqlens(packed_seq_params, x.device),
                    cp_size=self.cp_size,
                    cp_rank=self.cp_rank,
                    dim=1,
                )
            return out

        cp_restore = self.cp_size > 1
        if cp_restore:
            x, position_ids, attention_mask = self._gather_cp_inputs(
                x, position_ids, attention_mask
            )

        out = self._forward_dense_full(
            x,
            cos,
            sin,
            position_ids,
            packed_seq_params=packed_seq_params,
            dsa_index_share_topk_holder=dsa_index_share_topk_holder,
        )
        if cp_restore:
            out = zigzag_slice_for_cp(out, self.cp_rank, self.cp_size, seq_dim=1)
        return out

    def _forward_packed_full(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        packed_seq_params,
        *,
        dsa_index_share_topk_holder: dict[Any, torch.Tensor] | None,
    ) -> torch.Tensor:
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params, x.device)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[-1] != x.shape[1]:
            raise ValueError(
                "GLM5 packed DynamicSparseAttention position_ids must cover the reconstructed packed tokens, "
                f"got {tuple(position_ids.shape)} for packed length {x.shape[1]}."
            )
        pieces = []
        for idx in range(int(cu_seqlens.numel()) - 1):
            start = int(cu_seqlens[idx].item())
            end = int(cu_seqlens[idx + 1].item())
            if end <= start:
                continue
            seg_cos, seg_sin = self._slice_rotary_cache(cos, sin, start, end)
            pieces.append(
                self._forward_dense_full(
                    x[:, start:end, :],
                    seg_cos,
                    seg_sin,
                    position_ids[:, start:end],
                    packed_seq_params=packed_seq_params,
                    dsa_index_share_topk_holder=dsa_index_share_topk_holder,
                    index_share_segment_idx=idx,
                )
            )
        if pieces:
            return torch.cat(pieces, dim=1)
        return x.new_empty(x.shape[0], 0, self.o_proj.out_features)

    def _forward_dense_full(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        packed_seq_params=None,
        dsa_index_share_topk_holder: dict[Any, torch.Tensor] | None = None,
        index_share_segment_idx: int | None = None,
    ) -> torch.Tensor:

        batch, seq_len, _ = x.shape
        q_a = self.q_a_proj(x)
        if self.q_a_lora is not None:
            q_a = q_a + self.q_a_lora(x)
        q_resid = self.q_a_layernorm(q_a)
        q = self.q_b_proj(q_resid)
        if self.q_b_lora is not None:
            q = q + self.q_b_lora(q_resid)
        q = q.view(batch, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        cos, sin = _rotary_embeddings_from_cache(
            cos, sin, position_ids, device=x.device, dtype=x.dtype, dim=self.qk_rope_head_dim
        )

        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        k_up_weight, v_up_weight = self._split_kv_b_weights()
        q_nope = torch.einsum("bshd,hdr->bshr", q_nope, k_up_weight)
        query_states = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1).contiguous()

        kv_a = self.kv_a_proj_with_mqa(x)
        if self.kv_a_lora is not None:
            kv_a = kv_a + self.kv_a_lora(x)
        kv_latent, k_pe = torch.split(kv_a, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        kv_full = torch.cat([kv_latent, k_pe], dim=-1).transpose(0, 1).contiguous()

        topk_indices = None
        q_indexer = k_indexer = weights_indexer = None
        if self.skip_topk:
            topk_indices = self._load_index_share_topk(
                packed_seq_params=packed_seq_params,
                dsa_index_share_topk_holder=dsa_index_share_topk_holder,
                index_share_segment_idx=index_share_segment_idx,
            )
        else:
            assert self.indexer is not None
            q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                x.detach(), q_resid.detach(), cos, sin, position_ids
            )
        effective_indexer_topk = min(self.index_topk, seq_len)

        use_indexer_loss = (
            self.training
            and torch.is_grad_enabled()
            and not self.index_share
            and self.indexer_loss_coeff > 0
        )
        if use_indexer_loss:
            assert q_indexer is not None and k_indexer is not None and weights_indexer is not None
            window_idxs = torch.empty(batch, seq_len, 0, device=x.device, dtype=torch.int32)
            out, indexer_loss = _fused_indexer_sparse_attn(
                query_states,
                kv_full,
                self.attn_sink.float(),
                window_idxs,
                q_indexer,
                k_indexer,
                weights_indexer,
                self.index_topk,
                1,
                self.softmax_scale,
                self.indexer_softmax_scale,
                self.indexer_loss_coeff,
                sparse_loss=self.indexer_use_sparse_loss,
                kv_offset=0,
                calculate_per_token_loss=self.calculate_per_token_loss,
                value_dim=self.kv_lora_rank,
            )
            if self.indexer_loss_coeff > 0:
                out = DSAIndexerLossAutoScaler.apply(out, indexer_loss)
        else:
            if topk_indices is None:
                if self.training and torch.is_grad_enabled() and self.indexer_loss_coeff > 0:
                    raise NotImplementedError(
                        "DSA IndexShare training with dsa_indexer_loss_coeff > 0 requires "
                        "the fused training kernel to return top-k indices. Disable index "
                        "sharing or set dsa_indexer_loss_coeff=0 until that kernel path is wired."
                    )
                assert q_indexer is not None and k_indexer is not None and weights_indexer is not None
                topk_indices, _ = _dsa_kernels.indexer_topk(
                    q_indexer,
                    k_indexer,
                    weights_indexer,
                    effective_indexer_topk,
                    1,
                    indexer_softmax_scale=self.indexer_softmax_scale,
                )
                self._store_index_share_topk(
                    topk_indices,
                    packed_seq_params=packed_seq_params,
                    dsa_index_share_topk_holder=dsa_index_share_topk_holder,
                    index_share_segment_idx=index_share_segment_idx,
                )
            flat_idxs, flat_tlen = _dsa_kernels.build_flat_topk_idxs(
                topk_indices, batch_size=batch, seqlen_kv=seq_len, compact=True
            )
            out = _dsa_kernels.dsa_sparse_attn(
                query_states,
                kv_full,
                self.attn_sink.float(),
                flat_idxs,
                self.softmax_scale,
                topk_length=flat_tlen,
                value_dim=self.kv_lora_rank,
            )

        out = out.view(seq_len, batch, self.num_heads, self.kv_lora_rank)
        out = out.permute(1, 0, 2, 3).contiguous()
        out = torch.einsum("bshr,hvr->bshv", out, v_up_weight)
        out = out.reshape(batch, seq_len, self.num_heads * self.v_head_dim)
        projected = self.o_proj(out)
        if self.o_lora is not None:
            projected = projected + self.o_lora(out)
        return projected

    def _get_index_share_topk_holder(
        self,
        *,
        packed_seq_params,
        dsa_index_share_topk_holder: dict[Any, torch.Tensor] | None,
    ) -> dict[Any, torch.Tensor]:
        if dsa_index_share_topk_holder is not None:
            return dsa_index_share_topk_holder
        if packed_seq_params is None:
            raise AssertionError(
                "DSA index-share requires a per-forward top-k holder. Pass "
                "dsa_index_share_topk_holder or use packed_seq_params carrying "
                f"{DSA_INDEX_SHARE_TOPK_HOLDER_ATTR}."
            )
        holder = getattr(packed_seq_params, DSA_INDEX_SHARE_TOPK_HOLDER_ATTR, None)
        if holder is None:
            holder = {}
            setattr(packed_seq_params, DSA_INDEX_SHARE_TOPK_HOLDER_ATTR, holder)
        return holder

    @staticmethod
    def _index_share_holder_key(layer_idx: int, segment_idx: int | None) -> Any:
        return layer_idx if segment_idx is None else (layer_idx, segment_idx)

    def _load_index_share_topk(
        self,
        *,
        packed_seq_params,
        dsa_index_share_topk_holder: dict[Any, torch.Tensor] | None,
        index_share_segment_idx: int | None,
    ) -> torch.Tensor:
        assert self.dsa_source_layer_idx is not None
        holder = self._get_index_share_topk_holder(
            packed_seq_params=packed_seq_params,
            dsa_index_share_topk_holder=dsa_index_share_topk_holder,
        )
        key = self._index_share_holder_key(self.dsa_source_layer_idx, index_share_segment_idx)
        if key not in holder:
            raise AssertionError(
                "DSA index-share skip layer "
                f"(layer_idx={self.layer_idx}) needs top-k indices from source "
                f"layer_idx={self.dsa_source_layer_idx}, but that layer did not run "
                "earlier in this forward context. Cross-PP top-k sharing is not supported. "
                f"Holder has keys {sorted(holder, key=repr)}."
            )
        return holder[key]

    def _store_index_share_topk(
        self,
        topk_indices: torch.Tensor,
        *,
        packed_seq_params,
        dsa_index_share_topk_holder: dict[Any, torch.Tensor] | None,
        index_share_segment_idx: int | None,
    ) -> None:
        if not self.index_share:
            return
        assert self.layer_idx is not None
        holder = self._get_index_share_topk_holder(
            packed_seq_params=packed_seq_params,
            dsa_index_share_topk_holder=dsa_index_share_topk_holder,
        )
        holder[self._index_share_holder_key(self.layer_idx, index_share_segment_idx)] = topk_indices

    def _split_kv_b_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.kv_b_proj.weight
        if self.kv_b_lora is not None:
            weight = weight + self.kv_b_lora.materialized_delta_weight().to(weight.dtype)
        kv_b = weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        return (kv_b[:, : self.qk_nope_head_dim, :], kv_b[:, self.qk_nope_head_dim :, :])

    def _gather_cp_inputs(
        self, x: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        local_batch, local_seq = x.shape[:2]
        x_parts = _all_gather_cp(x, cp_size=self.cp_size, cp_group=self.cp_group)
        full_x = zigzag_reconstruct_from_cp_parts(x_parts, seq_dim=1)
        full_seq = full_x.shape[1]

        full_position_ids = self._full_cp_position_ids(
            position_ids, batch=local_batch, local_seq=local_seq, full_seq=full_seq, device=x.device
        )
        if attention_mask is not None:
            expected = (full_seq, full_seq)
            if tuple(attention_mask.shape[-2:]) != expected:
                raise NotImplementedError(
                    "GLM5 DynamicSparseAttention CP attention_mask must already cover the reconstructed "
                    f"full sequence {expected}, got {tuple(attention_mask.shape)}."
                )
        return full_x, full_position_ids, attention_mask

    def _full_cp_position_ids(
        self,
        position_ids: torch.Tensor,
        *,
        batch: int,
        local_seq: int,
        full_seq: int,
        device: torch.device,
    ) -> torch.Tensor:
        if position_ids.dim() == 3:
            position_ids = position_ids[0]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        if position_ids.shape[-1] == full_seq:
            return position_ids.to(device=device, dtype=torch.long)
        if position_ids.shape[-1] != local_seq:
            raise ValueError(
                "GLM5 DynamicSparseAttention CP position_ids must be either local or full sequence length, "
                f"got {tuple(position_ids.shape)} for local_seq={local_seq}, full_seq={full_seq}."
            )

        pos_parts = _all_gather_cp(
            position_ids.to(device=device, dtype=torch.long),
            cp_size=self.cp_size,
            cp_group=self.cp_group,
        )
        return zigzag_reconstruct_from_cp_parts(pos_parts, seq_dim=1)

    def _gather_packed_cp_inputs(
        self, x: torch.Tensor, position_ids: torch.Tensor, packed_seq_params
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_seq = x.shape[1]
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params, x.device)
        full_seq = int(cu_seqlens[-1].item())
        x_parts = _all_gather_cp(x, cp_size=self.cp_size, cp_group=self.cp_group)
        full_x = reconstruct_packed_from_cp_parts(
            x_parts, cu_seqlens_padded=cu_seqlens, cp_size=self.cp_size, dim=1
        )

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.to(device=x.device, dtype=torch.long)
        if position_ids.shape[-1] == full_seq:
            return full_x, position_ids
        if position_ids.shape[-1] != local_seq:
            raise ValueError(
                "GLM5 packed DynamicSparseAttention CP position_ids must be either local or full packed length, "
                f"got {tuple(position_ids.shape)} for local_seq={local_seq}, full_seq={full_seq}."
            )
        pos_parts = _all_gather_cp(position_ids, cp_size=self.cp_size, cp_group=self.cp_group)
        full_position_ids = reconstruct_packed_from_cp_parts(
            pos_parts, cu_seqlens_padded=cu_seqlens, cp_size=self.cp_size, dim=1
        )
        return full_x, full_position_ids

    def _gather_packed_cp_rotary(
        self, cos: torch.Tensor, sin: torch.Tensor, packed_seq_params, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cos.dim() != 3 or sin.dim() != 3:
            return cos, sin
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params, device)
        full_seq = int(cu_seqlens[-1].item())
        if cos.shape[1] == full_seq and sin.shape[1] == full_seq:
            return cos, sin
        cos_parts = _all_gather_cp(cos, cp_size=self.cp_size, cp_group=self.cp_group)
        sin_parts = _all_gather_cp(sin, cp_size=self.cp_size, cp_group=self.cp_group)
        full_cos = reconstruct_packed_from_cp_parts(
            cos_parts, cu_seqlens_padded=cu_seqlens, cp_size=self.cp_size, dim=1
        )
        full_sin = reconstruct_packed_from_cp_parts(
            sin_parts, cu_seqlens_padded=cu_seqlens, cp_size=self.cp_size, dim=1
        )
        return full_cos, full_sin

    @staticmethod
    def _packed_cu_seqlens(packed_seq_params, device: torch.device) -> torch.Tensor:
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
        if cu_seqlens is None:
            cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
        if cu_seqlens is None:
            raise ValueError("GLM5 packed DynamicSparseAttention requires packed cu_seqlens.")
        return cu_seqlens.to(device=device, dtype=torch.int32)

    @staticmethod
    def _slice_rotary_cache(
        cos: torch.Tensor, sin: torch.Tensor, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cos.dim() == 3 and sin.dim() == 3:
            return cos[:, start:end, :], sin[:, start:end, :]
        return cos, sin


__all__ = [
    "DSA_INDEX_SHARE_TOPK_HOLDER_ATTR",
    "DSAIndexer",
    "DSAIndexerLossAutoScaler",
    "DynamicSparseAttention",
    "RMSNorm",
    "apply_rotary_emb",
    "apply_rotary_pos_emb",
    "build_rope_cache",
    "build_rotary_embeddings",
    "rotate_activation",
    "rotate_half",
]
