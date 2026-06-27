# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]

_FUSED_R2R_OUTPUT_ATOL = 2.0e-3
_FUSED_R2R_GRAD_ATOL = 5.0e-2
_FUSED_VS_REFERENCE_OUTPUT_ATOL = 2.0e-3
_FUSED_VS_REFERENCE_GRAD_ATOL = 5.0e-2
_FUSED_VS_REFERENCE_LOSS_ATOL = 1.0e-5
_INDEXER_TOPK = 512
_SEQUENCE_LENGTH = 1024
_INDEXER_LOSS_COEFF = 1.0e-2
_MIN_INDEXER_GRAD_MAX_ABS = 1.0e-8


def _make_dsa_pair(*, sparse_loss: bool):
    pytest.importorskip("cudnn", reason="GLM5 DSA accept-with-proof needs cudnn DSA.")
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention

    common = dict(
        hidden_size=128,
        num_attention_heads=64,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=_INDEXER_TOPK,
        rms_norm_eps=1e-5,
        rope_interleaved=True,
        indexer_rope_interleaved=True,
        index_topk_freq=2,
        index_skip_topk_offset=1,
        index_share_enabled=True,
        indexer_loss_coeff=_INDEXER_LOSS_COEFF,
        indexer_use_sparse_loss=sparse_loss,
    )
    source = DynamicSparseAttention(
        **common,
        layer_number=1,
        indexer_type="full",
        index_share_source_layer=1,
    )
    shared = DynamicSparseAttention(
        **common,
        layer_number=2,
        indexer_type="shared",
        index_share_source_layer=1,
    )
    assert source.indexer is not None
    assert shared.indexer is None
    return torch.nn.ModuleDict({"source": source, "shared": shared})


def _run_once(modules, x, cos, sin, position_ids, *, fused_training: bool):
    from megatron.lite.primitive.modules.attention import DSAIndexShareState

    class RecordingState(DSAIndexShareState):
        def __init__(self):
            super().__init__({1: 1})
            self.saved: list[torch.Tensor] = []
            self.consumed: list[torch.Tensor] = []

        def save_topk(self, layer_number, topk_indices, *, sequence_key=None):
            self.saved.append(topk_indices.detach().clone())
            return super().save_topk(
                layer_number, topk_indices, sequence_key=sequence_key
            )

        def get_topk(self, layer_number, source_layer, *, sequence_key=None):
            topk_indices = super().get_topk(
                layer_number, source_layer, sequence_key=sequence_key
            )
            self.consumed.append(topk_indices.detach().clone())
            return topk_indices

    modules.zero_grad(set_to_none=True)
    modules.train(fused_training)
    local_x = x.detach().clone().requires_grad_(True)
    index_share_state = RecordingState()
    source_out = modules["source"](
        local_x,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        index_share_state=index_share_state,
    )
    assert index_share_state.cached_tensor_count == 1
    hidden = local_x + source_out
    shared_out = modules["shared"](
        hidden,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        index_share_state=index_share_state,
    )
    out = hidden + shared_out
    assert len(index_share_state.saved) == 1
    assert len(index_share_state.consumed) == 1
    assert torch.equal(index_share_state.saved[0], index_share_state.consumed[0])
    assert index_share_state.cached_tensor_count == 0
    loss = out.float().square().mean()
    loss.backward()
    param_grads = {
        name: param.grad.detach().float().clone()
        for name, param in modules.named_parameters()
        if param.grad is not None
    }
    return {
        "loss": loss.detach().float().clone(),
        "out": out.detach().float().clone(),
        "x_grad": local_x.grad.detach().float().clone(),
        "param_grads": param_grads,
        "topk_indices": index_share_state.consumed[0],
    }


def _causal_mask(seq_q: int, seq_k: int, *, ratio: int, device: torch.device) -> torch.Tensor:
    q_idx = torch.arange(seq_q, device=device)
    k_idx = torch.arange(seq_k, device=device)
    valid_per_q = ((q_idx + 1) // ratio).clamp(max=seq_k)
    return k_idx.unsqueeze(0) < valid_per_q.unsqueeze(1)


def _torch_indexer_scores(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    *,
    ratio: int,
    indexer_softmax_scale: float,
) -> torch.Tensor:
    q_bshd = q_indexer.permute(1, 0, 2, 3).float()
    k_bsd = k_indexer.permute(1, 0, 2).float()
    w_bsh = weights.permute(1, 0, 2).float()
    scores = torch.einsum("bqhd,bkd->bqhk", q_bshd, k_bsd)
    scores = torch.relu(scores).mul(w_bsh.unsqueeze(-1)).sum(dim=2)
    scores = scores * float(indexer_softmax_scale)
    causal = _causal_mask(scores.shape[1], scores.shape[2], ratio=ratio, device=scores.device)
    return torch.where(causal.unsqueeze(0), scores, torch.full_like(scores, -torch.inf))


def _torch_topk_from_scores(scores: torch.Tensor, topk: int) -> torch.Tensor:
    effective_topk = min(topk, scores.shape[-1])
    values, indices = torch.topk(scores, k=effective_topk, dim=-1)
    indices = torch.where(torch.isfinite(values), indices, torch.full_like(indices, -1))
    if effective_topk < topk:
        pad = torch.full(
            (*indices.shape[:-1], topk - effective_topk),
            -1,
            device=indices.device,
            dtype=indices.dtype,
        )
        indices = torch.cat([indices, pad], dim=-1)
    return indices.int()


def _gather_sequence(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe_indices = indices.clamp(min=0).long()
    expanded = source.unsqueeze(1).expand(-1, indices.shape[1], -1, -1)
    gathered = torch.gather(
        expanded,
        dim=2,
        index=safe_indices.unsqueeze(-1).expand(-1, -1, -1, source.shape[-1]),
    )
    return torch.where(indices.unsqueeze(-1) >= 0, gathered, torch.zeros_like(gathered))


def _torch_sparse_attention(
    query_states: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float,
    value_dim: int,
) -> torch.Tensor:
    query_bshd = query_states.permute(1, 0, 2, 3).float()
    kv_bsd = kv_full.permute(1, 0, 2).float()
    selected_keys = _gather_sequence(kv_bsd, topk_indices)
    selected_values = _gather_sequence(kv_bsd[..., :value_dim], topk_indices)
    scores = torch.einsum("bqhd,bqtd->bqht", query_bshd, selected_keys)
    scores = scores * float(softmax_scale)
    scores = torch.where(
        topk_indices.unsqueeze(2) >= 0, scores, torch.full_like(scores, -torch.inf)
    )
    sink = attn_sink.float().view(1, 1, -1, 1).expand(scores.shape[0], scores.shape[1], -1, -1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    out = torch.einsum("bqht,bqtr->bqhr", probs, selected_values)
    return out.permute(1, 0, 2, 3).reshape(
        query_states.shape[0], query_states.shape[1], query_states.shape[2] * value_dim
    )


def _torch_sparse_indexer_loss(
    indexer_scores: torch.Tensor,
    query_states: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float,
    loss_coeff: float,
) -> torch.Tensor:
    """Independent sparse KL oracle for the fused indexer-loss backward."""

    valid = topk_indices >= 0
    query_bshd = query_states.detach().permute(1, 0, 2, 3).float()
    kv_bsd = kv_full.detach().permute(1, 0, 2).float()
    selected_keys = _gather_sequence(kv_bsd, topk_indices)
    attn_scores = torch.einsum("bqhd,bqtd->bqht", query_bshd, selected_keys)
    attn_scores = attn_scores * float(softmax_scale)
    attn_scores = torch.where(
        valid.unsqueeze(2), attn_scores, torch.full_like(attn_scores, -torch.inf)
    )
    sink = attn_sink.detach().float().view(1, 1, -1, 1)
    sink = sink.expand(attn_scores.shape[0], attn_scores.shape[1], -1, -1)
    attn_probs = torch.softmax(torch.cat([attn_scores, sink], dim=-1), dim=-1)[..., :-1]
    target_mass = attn_probs.sum(dim=2)
    target_mass = torch.where(valid, target_mass, torch.zeros_like(target_mass))

    row_valid = valid.any(dim=-1)
    target_denom = target_mass.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    target = target_mass / target_denom

    safe_indices = topk_indices.clamp(min=0).long()
    selected_indexer_scores = torch.gather(indexer_scores, dim=-1, index=safe_indices)
    selected_indexer_scores = torch.where(
        valid,
        selected_indexer_scores,
        torch.full_like(selected_indexer_scores, torch.finfo(torch.float32).min),
    )
    predict = torch.softmax(selected_indexer_scores, dim=-1)

    eps = torch.finfo(torch.float32).tiny
    target_clamped = target.clamp_min(eps)
    predict_clamped = predict.clamp_min(eps)
    kl_per_row = (
        target_clamped * (torch.log(target_clamped) - torch.log(predict_clamped))
    ).sum(dim=-1)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    return float(loss_coeff) * kl_per_row.mean()


def _torch_dense_indexer_loss(
    indexer_scores: torch.Tensor,
    query_states: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float,
    loss_coeff: float,
) -> torch.Tensor:
    """Independent full-KV KL oracle for GLM's default dense loss."""

    query_bshd = query_states.detach().permute(1, 0, 2, 3).float()
    kv_bsd = kv_full.detach().permute(1, 0, 2).float()
    attn_scores = torch.einsum("bqhd,bkd->bqhk", query_bshd, kv_bsd)
    attn_scores = attn_scores * float(softmax_scale)
    causal = _causal_mask(
        attn_scores.shape[1], attn_scores.shape[-1], ratio=1, device=attn_scores.device
    )
    attn_scores = torch.where(
        causal.view(1, causal.shape[0], 1, causal.shape[1]),
        attn_scores,
        torch.full_like(attn_scores, -torch.inf),
    )
    # FlashMLA reports the per-head LSE over the selected sparse keys plus
    # sink.  Dense DSA then recomputes all-key scores against that same LSE
    # before L1-normalising the head-summed target over the full KV axis.
    selected_keys = _gather_sequence(kv_bsd, topk_indices)
    selected_scores = torch.einsum("bqhd,bqtd->bqht", query_bshd, selected_keys)
    selected_scores = selected_scores * float(softmax_scale)
    selected_scores = torch.where(
        topk_indices.unsqueeze(2) >= 0,
        selected_scores,
        torch.full_like(selected_scores, -torch.inf),
    )
    sink = attn_sink.detach().float().view(1, 1, -1, 1)
    sink = sink.expand(selected_scores.shape[0], selected_scores.shape[1], -1, -1)
    selected_lse = torch.logsumexp(torch.cat([selected_scores, sink], dim=-1), dim=-1)
    target_mass = torch.exp(attn_scores - selected_lse.unsqueeze(-1)).sum(dim=2)
    target_mass = torch.where(causal.unsqueeze(0), target_mass, torch.zeros_like(target_mass))
    target_denom = target_mass.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    target = target_mass / target_denom

    row_valid = causal.any(dim=-1).unsqueeze(0)
    safe_indexer_scores = torch.where(
        row_valid.unsqueeze(-1), indexer_scores, torch.zeros_like(indexer_scores)
    )
    predict = torch.softmax(safe_indexer_scores, dim=-1)
    eps = torch.finfo(torch.float32).tiny
    terms = target.clamp_min(eps) * (
        torch.log(target.clamp_min(eps)) - torch.log(predict.clamp_min(eps))
    )
    terms = torch.where(causal.unsqueeze(0), terms, torch.zeros_like(terms))
    kl_per_row = terms.sum(dim=-1)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    return float(loss_coeff) * kl_per_row.mean()


def _torch_unfused_dsa_forward(
    module, x, cos, sin, position_ids, *, topk_indices: torch.Tensor | None = None
):
    from megatron.lite.primitive.modules.attention.dsa import (
        _rotary_embeddings_from_cache,
        apply_rotary_pos_emb,
    )

    batch, seq_len, _ = x.shape
    q_resid = module.q_a_layernorm(module.q_a_proj(x))
    q = module.q_b_proj(q_resid).view(batch, seq_len, module.num_heads, module.qk_head_dim)
    q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
    cos, sin = _rotary_embeddings_from_cache(
        cos, sin, position_ids, device=x.device, dtype=x.dtype, dim=module.qk_rope_head_dim
    )
    q_pe = apply_rotary_pos_emb(
        q_pe,
        cos,
        sin,
        unsqueeze_dim=2,
        mla_interleaved=module.rope_interleaved,
    )

    k_up_weight, v_up_weight = module._split_kv_b_weights()
    q_nope = torch.einsum("bshd,hdr->bshr", q_nope, k_up_weight)
    query_states = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1).contiguous()

    kv_latent, k_pe = torch.split(
        module.kv_a_proj_with_mqa(x), [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1
    )
    kv_latent = module.kv_a_layernorm(kv_latent)
    k_pe = apply_rotary_pos_emb(
        k_pe.unsqueeze(2),
        cos,
        sin,
        unsqueeze_dim=2,
        mla_interleaved=module.rope_interleaved,
    ).squeeze(2)
    kv_full = torch.cat([kv_latent, k_pe], dim=-1).transpose(0, 1).contiguous()

    indexer_loss = x.new_zeros((), dtype=torch.float32)
    if module.indexer is not None:
        assert topk_indices is None
        q_indexer, k_indexer, weights_indexer = module.indexer.forward_before_topk(
            x.detach(), q_resid.detach(), cos, sin, position_ids
        )
        indexer_scores = _torch_indexer_scores(
            q_indexer,
            k_indexer,
            weights_indexer,
            ratio=1,
            indexer_softmax_scale=module.indexer_softmax_scale,
        )
        topk_indices = _torch_topk_from_scores(
            indexer_scores, min(module.index_topk, indexer_scores.shape[-1])
        )
        if module.indexer_use_sparse_loss:
            indexer_loss = _torch_sparse_indexer_loss(
                indexer_scores,
                query_states,
                kv_full,
                module.attn_sink,
                topk_indices,
                softmax_scale=module.softmax_scale,
                loss_coeff=module.indexer_loss_coeff,
            )
        else:
            indexer_loss = _torch_dense_indexer_loss(
                indexer_scores,
                query_states,
                kv_full,
                module.attn_sink,
                topk_indices,
                softmax_scale=module.softmax_scale,
                loss_coeff=module.indexer_loss_coeff,
            )
    else:
        assert topk_indices is not None
    out = _torch_sparse_attention(
        query_states,
        kv_full,
        module.attn_sink,
        topk_indices,
        softmax_scale=module.softmax_scale,
        value_dim=module.kv_lora_rank,
    )
    out = out.to(x.dtype).view(seq_len, batch, module.num_heads, module.kv_lora_rank)
    out = out.permute(1, 0, 2, 3).contiguous()
    out = torch.einsum("bshr,hvr->bshv", out, v_up_weight)
    out = out.reshape(batch, seq_len, module.num_heads * module.v_head_dim)
    out = module.o_proj(out)
    return out, topk_indices, indexer_loss


def _run_once_torch_unfused(modules, x, cos, sin, position_ids):
    modules.zero_grad(set_to_none=True)
    modules.train(True)
    local_x = x.detach().clone().requires_grad_(True)
    source_out, topk_indices, indexer_loss = _torch_unfused_dsa_forward(
        modules["source"], local_x, cos, sin, position_ids
    )
    hidden = local_x + source_out
    shared_out, reused_topk_indices, shared_indexer_loss = _torch_unfused_dsa_forward(
        modules["shared"],
        hidden,
        cos,
        sin,
        position_ids,
        topk_indices=topk_indices,
    )
    assert torch.equal(reused_topk_indices, topk_indices)
    assert float(shared_indexer_loss.item()) == 0.0
    out = hidden + shared_out
    loss = out.float().square().mean()
    (loss + indexer_loss).backward()
    param_grads = {
        name: param.grad.detach().float().clone()
        for name, param in modules.named_parameters()
        if param.grad is not None
    }
    return {
        "loss": loss.detach().float().clone(),
        "indexer_loss": indexer_loss.detach().float().clone(),
        "out": out.detach().float().clone(),
        "x_grad": local_x.grad.detach().float().clone(),
        "param_grads": param_grads,
        "topk_indices": topk_indices.detach().clone(),
    }


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _max_param_grad_abs(a: dict, b: dict) -> float:
    a_keys = set(a["param_grads"])
    b_keys = set(b["param_grads"])
    assert a_keys == b_keys, (
        f"gradient key mismatch: only_a={sorted(a_keys - b_keys)}, "
        f"only_b={sorted(b_keys - a_keys)}"
    )
    if not a_keys:
        return 0.0
    return max(_max_abs(a["param_grads"][name], b["param_grads"][name]) for name in a_keys)


def _assert_meaningful_indexer_grads(result: dict) -> float:
    expected = {
        "source.indexer.wq_b.weight",
        "source.indexer.wk.weight",
        "source.indexer.k_norm.weight",
        "source.indexer.k_norm.bias",
        "source.indexer.weights_proj.weight",
    }
    grads = {
        name: grad
        for name, grad in result["param_grads"].items()
        if name.startswith("source.indexer.")
    }
    assert set(grads) == expected
    assert all(torch.isfinite(grad).all() for grad in grads.values())
    assert all(torch.count_nonzero(grad).item() > 0 for grad in grads.values())
    grad_max_abs = {
        name: float(grad.abs().max().item()) for name, grad in grads.items()
    }
    assert all(value > _MIN_INDEXER_GRAD_MAX_ABS for value in grad_max_abs.values()), (
        grad_max_abs
    )
    return max(grad_max_abs.values())


@pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse-loss", "dense-loss"])
def test_glm5_dsa_run_to_run_accept_with_proof(sparse_loss: bool):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GLM5 DSA accept-with-proof smoke.")

    from megatron.lite.primitive.modules.attention import build_rope_cache

    device = torch.device("cuda", int(torch.cuda.current_device()))
    torch.manual_seed(20260626)
    fused = _make_dsa_pair(sparse_loss=sparse_loss).to(device=device, dtype=torch.bfloat16)
    unfused = copy.deepcopy(fused).to(device=device, dtype=torch.bfloat16)

    batch, seq, hidden = 1, _SEQUENCE_LENGTH, 128
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.bfloat16)
    cos, sin = build_rope_cache(
        dim=64,
        max_position_embeddings=seq,
        rope_theta=1_000_000.0,
        device=device,
    )
    position_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0)

    fused_a = _run_once(fused, x, cos, sin, position_ids, fused_training=True)
    fused_b = _run_once(fused, x, cos, sin, position_ids, fused_training=True)
    unfused_a = _run_once_torch_unfused(unfused, x, cos, sin, position_ids)
    unfused_b = _run_once_torch_unfused(unfused, x, cos, sin, position_ids)

    fused_r2r_out = _max_abs(fused_a["out"], fused_b["out"])
    fused_r2r_loss = abs(float(fused_a["loss"].item()) - float(fused_b["loss"].item()))
    fused_r2r_x_grad = _max_abs(fused_a["x_grad"], fused_b["x_grad"])
    fused_r2r_param_grad = _max_param_grad_abs(fused_a, fused_b)
    unfused_r2r_out = _max_abs(unfused_a["out"], unfused_b["out"])
    unfused_r2r_loss = abs(
        float(unfused_a["loss"].item()) - float(unfused_b["loss"].item())
    )
    unfused_r2r_x_grad = _max_abs(unfused_a["x_grad"], unfused_b["x_grad"])
    unfused_r2r_param_grad = _max_param_grad_abs(unfused_a, unfused_b)
    fused_vs_unfused_out = _max_abs(fused_a["out"], unfused_a["out"])
    fused_vs_unfused_x_grad = _max_abs(fused_a["x_grad"], unfused_a["x_grad"])
    fused_vs_unfused_param_grad = _max_param_grad_abs(fused_a, unfused_a)
    loss_diff = abs(float(fused_a["loss"].item()) - float(unfused_a["loss"].item()))
    unfused_indexer_loss_diff = abs(
        float(unfused_a["indexer_loss"].item())
        - float(unfused_b["indexer_loss"].item())
    )
    fused_indexer_grad_max_abs = _assert_meaningful_indexer_grads(fused_a)
    _assert_meaningful_indexer_grads(fused_b)
    unfused_indexer_grad_max_abs = _assert_meaningful_indexer_grads(unfused_a)
    _assert_meaningful_indexer_grads(unfused_b)

    assert torch.isfinite(fused_a["loss"])
    assert torch.isfinite(unfused_a["loss"])
    assert torch.isfinite(unfused_a["indexer_loss"])
    assert float(unfused_a["indexer_loss"].item()) > 0.0
    assert torch.equal(fused_a["topk_indices"], fused_b["topk_indices"])
    assert torch.equal(unfused_a["topk_indices"], unfused_b["topk_indices"])
    assert torch.equal(fused_a["topk_indices"], unfused_a["topk_indices"])
    valid_topk = (fused_a["topk_indices"] >= 0).sum(dim=-1)
    assert _INDEXER_TOPK < seq
    assert fused_a["topk_indices"].shape[-1] == _INDEXER_TOPK
    assert int(valid_topk[0, -1].item()) == _INDEXER_TOPK
    assert int(valid_topk.max().item()) < seq
    assert fused_r2r_loss <= _FUSED_VS_REFERENCE_LOSS_ATOL
    assert fused_r2r_out <= _FUSED_R2R_OUTPUT_ATOL
    assert fused_r2r_x_grad <= _FUSED_R2R_GRAD_ATOL
    assert fused_r2r_param_grad <= _FUSED_R2R_GRAD_ATOL
    assert unfused_r2r_loss == 0.0
    assert unfused_r2r_out == 0.0
    assert unfused_r2r_x_grad == 0.0
    assert unfused_r2r_param_grad == 0.0
    assert unfused_indexer_loss_diff == 0.0
    assert loss_diff <= _FUSED_VS_REFERENCE_LOSS_ATOL
    assert fused_vs_unfused_out <= _FUSED_VS_REFERENCE_OUTPUT_ATOL
    assert fused_vs_unfused_x_grad <= _FUSED_VS_REFERENCE_GRAD_ATOL
    assert fused_vs_unfused_param_grad <= _FUSED_VS_REFERENCE_GRAD_ATOL

    print(
        "NON_SKIP_GLM5_DSA_RUN_TO_RUN_ACCEPT_WITH_PROOF "
        f"fused_loss={float(fused_a['loss'].item()):.6e} "
        f"unfused_loss={float(unfused_a['loss'].item()):.6e} "
        f"unfused_indexer_loss={float(unfused_a['indexer_loss'].item()):.6e} "
        f"index_heads=32 sparse_loss={sparse_loss} "
        f"indexer_topk={_INDEXER_TOPK} seq={seq} "
        f"loss_diff={loss_diff:.6e} "
        f"fused_r2r_loss_diff={fused_r2r_loss:.6e} "
        f"fused_r2r_out_max_abs={fused_r2r_out:.6e} "
        f"fused_r2r_x_grad_max_abs={fused_r2r_x_grad:.6e} "
        f"fused_r2r_param_grad_max_abs={fused_r2r_param_grad:.6e} "
        f"unfused_r2r_out_max_abs={unfused_r2r_out:.6e} "
        f"unfused_r2r_loss_diff={unfused_r2r_loss:.6e} "
        f"unfused_r2r_x_grad_max_abs={unfused_r2r_x_grad:.6e} "
        f"unfused_r2r_param_grad_max_abs={unfused_r2r_param_grad:.6e} "
        f"fused_vs_unfused_out_max_abs={fused_vs_unfused_out:.6e} "
        f"fused_vs_unfused_x_grad_max_abs={fused_vs_unfused_x_grad:.6e} "
        f"fused_vs_unfused_param_grad_max_abs={fused_vs_unfused_param_grad:.6e} "
        f"fused_indexer_grad_max_abs={fused_indexer_grad_max_abs:.6e} "
        f"unfused_indexer_grad_max_abs={unfused_indexer_grad_max_abs:.6e}"
    )


def test_dsv4_fused_dsa_legacy_two_output_api_real_gpu():
    """Run the legacy DSv4-facing two-output autograd contract on real kernels."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the fused DSA API regression smoke.")
    pytest.importorskip("cudnn", reason="Fused DSA API smoke needs the cuDNN DSA stack.")
    from megatron.lite.primitive.kernels import dsa_kernels

    device = torch.device("cuda", int(torch.cuda.current_device()))
    seq, batch, heads = 512, 1, 64
    query_dim, value_dim = 576, 512
    index_heads, index_dim, topk = 32, 128, 512

    def make_args():
        torch.manual_seed(20260627)

        def leaf(*shape, dtype=torch.bfloat16):
            return torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)

        return (
            leaf(seq, batch, heads, query_dim),
            leaf(seq, batch, query_dim),
            leaf(heads, dtype=torch.float32),
            torch.empty(batch, seq, 0, device=device, dtype=torch.int32),
            leaf(seq, batch, index_heads, index_dim),
            leaf(seq, batch, index_dim),
            leaf(seq, batch, index_heads),
            topk,
            1,
            query_dim**-0.5,
            index_dim**-0.5,
            0.0,
            False,
            0,
            False,
            value_dim,
        )

    results = []
    for name, call in (
        ("direct", dsa_kernels.FusedIndexerSparseAttnFunc.apply),
        ("public", dsa_kernels.fused_indexer_sparse_attn),
    ):
        args = make_args()
        result = call(*args)
        assert isinstance(result, tuple) and len(result) == 2
        output, indexer_loss = result
        assert output.shape == (seq, batch, heads * value_dim)
        assert indexer_loss.ndim == 0
        objective = output.float().square().mean() + indexer_loss
        objective.backward()
        differentiable_inputs = (args[0], args[1], args[2], args[4], args[5], args[6])
        assert all(tensor.grad is not None for tensor in differentiable_inputs)
        assert all(torch.isfinite(tensor.grad).all() for tensor in differentiable_inputs)
        results.append(output.detach().float())
        print(
            "DSV4_FUSED_DSA_LEGACY_API "
            f"path={name} loss={float(objective.detach().item()):.8e}"
        )

    torch.testing.assert_close(results[0], results[1], rtol=0.0, atol=0.0)


def test_glm52_model_preserves_multisegment_positions_through_indexshare_and_mtp():
    """A real packed model must not replace reset positions with flat arange."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GLM5 packed-position smoke.")
    pytest.importorskip("cudnn", reason="GLM5 packed-position smoke needs cudnn DSA.")

    from types import SimpleNamespace

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5Model
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    device = torch.device("cuda", int(torch.cuda.current_device()))
    cfg = Glm5Config(
        num_hidden_layers=2,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=1024,
        initializer_range=0.002,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=512,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=3,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_nextn_predict_layers=1,
        index_topk_freq=2,
        index_skip_topk_offset=1,
        indexer_types=["full", "shared"],
        rope_interleave=True,
        indexer_rope_interleave=True,
    )
    ps = ParallelState()
    train_cfg = SimpleNamespace(
        tp=1,
        ep=1,
        etp=1,
        pp=1,
        cp=1,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        deterministic=True,
    )

    torch.manual_seed(20260627)
    model = Glm5Model(
        cfg,
        train_cfg,
        ps,
        mtp_enable=True,
        mtp_enable_train=False,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()
    assert model.mtp is not None

    segment_length = 512
    total_tokens = segment_length * 2
    input_ids = torch.randint(
        0, cfg.vocab_size, (1, total_tokens), device=device, dtype=torch.long
    )
    segment_positions = torch.arange(segment_length, device=device, dtype=torch.long)
    reset_positions = torch.cat((segment_positions, segment_positions)).unsqueeze(0)
    cu_seqlens = torch.tensor(
        [0, segment_length, total_tokens], device=device, dtype=torch.int32
    )
    packed_seq_params = PackedSeqParams.from_cu_seqlens(
        cu_seqlens, max_seqlen=segment_length
    )

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def capture_positions(module, _args, kwargs):
        captured[module.layer_number] = kwargs["position_ids"].detach().clone()

    attention_modules = [
        layer.self_attention.self_attention for layer in model.layers
    ] + [
        model.mtp.layers[0].transformer_layer.self_attention.self_attention
    ]
    for attention in attention_modules:
        handles.append(
            attention.register_forward_pre_hook(capture_positions, with_kwargs=True)
        )

    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                position_ids=reset_positions,
                packed_seq_params=packed_seq_params,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert output["logits"].shape == (1, total_tokens, cfg.vocab_size)
    assert torch.isfinite(output["logits"].float()).all()
    assert "mtp_logits" in output and len(output["mtp_logits"]) == 1
    assert torch.equal(captured[1], reset_positions)
    assert torch.equal(captured[2], reset_positions)

    rolled_segment_positions = torch.cat(
        (segment_positions[1:], torch.zeros(1, device=device, dtype=torch.long))
    )
    expected_mtp_positions = torch.cat(
        (rolled_segment_positions, rolled_segment_positions)
    ).unsqueeze(0)
    assert torch.equal(captured[3], expected_mtp_positions)
    print(
        "NON_SKIP_GLM52_PACKED_POSITION_IDS_PASSED "
        f"segments=2 segment_length={segment_length} "
        f"trunk_reset={torch.equal(captured[2], reset_positions)} "
        f"mtp_segment_roll={torch.equal(captured[3], expected_mtp_positions)}"
    )
