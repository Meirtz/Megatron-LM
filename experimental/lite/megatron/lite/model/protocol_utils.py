# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Helpers shared by model protocol forward steps.

The verl/runtime layers hand each protocol a raw, model-agnostic ``PackedBatch``
(true per-sequence lengths, no padding, no ``PackedSeqParams``). Each model owns
its pack/unpack pair: ``pack_thd_forward_kwargs`` pads + CP-splits the batch into
model forward kwargs, and ``unpack_thd_forward_output`` reverses a model output
back to jagged true-length form. THD models share the zigzag-CP pair below;
models with a different CP layout (e.g. DeepSeek-V4 contiguous DSA) provide their
own pair.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import torch
from megatron.lite.primitive.modules.router_replay import (
    RouterReplayAction,
    get_routed_experts_dtype,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    parallel_state_from_model,
    prepare_packed_thd_kwargs_for_context_parallel,
    split_packed_to_cp_local,
    thd_pack_meta,
    unpack_thd_to_nested,
)
from megatron.lite.primitive.utils.packed_seq import PackedSeqParams
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.loss import get_loss_context


def _parallel_state(model) -> ParallelState:
    return parallel_state_from_model(model) or ParallelState()


def _unwrap_router_replay_model(model):
    current = model
    seen = set()
    for _ in range(8):
        if hasattr(current, "router_replay_instances"):
            return current
        ident = id(current)
        if ident in seen:
            break
        seen.add(ident)
        next_model = None
        for attr in ("module", "_module", "model", "wrapped_module", "_fsdp_wrapped_module"):
            if hasattr(current, attr):
                candidate = getattr(current, attr)
                if candidate is not None and candidate is not current:
                    next_model = candidate
                    break
        if next_model is None:
            break
        current = next_model
    return model


def _router_replay_action_from_batch(batch: PackedBatch) -> RouterReplayAction | None:
    extras = getattr(batch, "extras", None) or {}
    action_value = extras.get("router_replay_action")
    if action_value is None:
        if getattr(batch, "routed_experts", None) is not None:
            return RouterReplayAction.REPLAY_FORWARD
        if extras.get("record_routed_experts") or extras.get("router_replay_record"):
            return RouterReplayAction.RECORD
        return None
    if isinstance(action_value, RouterReplayAction):
        return action_value
    if isinstance(action_value, str):
        try:
            return RouterReplayAction(action_value)
        except ValueError as exc:
            valid = ", ".join(action.value for action in RouterReplayAction)
            raise ValueError(
                f"Unsupported router_replay_action={action_value!r}; expected one of: {valid}."
            ) from exc
    raise TypeError(
        "router_replay_action must be a RouterReplayAction or string value, "
        f"got {type(action_value)!r}."
    )


def padded_routed_experts_to_list(
    routed_experts: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_routers: int | None = None,
    compact: bool = True,
) -> list[torch.Tensor]:
    """Convert padded R3 traces to per-router true-token tensors.

    The padded input follows the rollout-side shape used by ROLL/SGLang:
    ``[batch, seq, routers, topk]``. ``seq_lens`` selects each sample's valid
    right-padded prefix. The return value is ordered by router/layer, with each
    tensor shaped ``[sum(seq_lens), topk]``.
    """

    if not isinstance(routed_experts, torch.Tensor):
        raise TypeError(
            "padded routed_experts must be a tensor, "
            f"got {type(routed_experts)!r}."
        )
    if routed_experts.ndim != 4:
        raise ValueError(
            "padded routed_experts must have shape [batch, seq, routers, topk], "
            f"got {tuple(routed_experts.shape)}."
        )
    if (
        routed_experts.dtype == torch.bool
        or torch.is_floating_point(routed_experts)
        or torch.is_complex(routed_experts)
    ):
        raise TypeError(
            "padded routed_experts must use an integer tensor dtype, "
            f"got {routed_experts.dtype}."
        )
    if routed_experts.numel():
        min_expert_idx = int(routed_experts.min().item())
        if min_expert_idx < 0:
            raise ValueError(
                "padded routed_experts must contain non-negative expert ids, "
                f"got min={min_expert_idx}."
            )
    if not isinstance(seq_lens, torch.Tensor):
        raise TypeError(f"seq_lens must be a tensor, got {type(seq_lens)!r}.")
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must have shape [batch], got {tuple(seq_lens.shape)}.")
    if seq_lens.numel() != routed_experts.size(0):
        raise ValueError(
            "padded routed_experts batch size must match seq_lens: "
            f"got batch={routed_experts.size(0)}, seq_lens={seq_lens.numel()}."
        )
    if num_routers is not None and routed_experts.size(2) != num_routers:
        raise ValueError(
            "padded routed_experts router dimension mismatch: "
            f"got {routed_experts.size(2)}, expected {num_routers}."
        )

    lengths = [int(length.item()) for length in seq_lens.to(device="cpu")]
    max_seq = int(routed_experts.size(1))
    if any(length < 0 for length in lengths):
        raise ValueError(f"seq_lens must be non-negative, got {lengths}.")
    if any(length > max_seq for length in lengths):
        raise ValueError(
            "seq_lens cannot exceed padded routed_experts sequence dimension: "
            f"seq_lens={lengths}, seq_dim={max_seq}."
        )

    pieces = [
        routed_experts[sample_idx, :length]
        for sample_idx, length in enumerate(lengths)
        if length > 0
    ]
    if pieces:
        true_tokens = torch.cat(pieces, dim=0)
    else:
        true_tokens = routed_experts.new_empty(
            (0, routed_experts.size(2), routed_experts.size(3))
        )
    if compact and true_tokens.numel():
        max_expert_idx = int(true_tokens.max().item())
        true_tokens = true_tokens.to(get_routed_experts_dtype(max_expert_idx))
    return [tensor.contiguous() for tensor in true_tokens.unbind(dim=1)]


def _validate_routed_experts_tensor(tensor: torch.Tensor, *, context: str) -> None:
    if (
        tensor.dtype == torch.bool
        or torch.is_floating_point(tensor)
        or torch.is_complex(tensor)
    ):
        raise TypeError(f"{context} must use an integer tensor dtype, got {tensor.dtype}.")
    if tensor.numel():
        min_expert_idx = int(tensor.min().item())
        if min_expert_idx < 0:
            raise ValueError(
                f"{context} must contain non-negative expert ids, got min={min_expert_idx}."
            )


def _validate_per_router_token_rows(replay_data: list[torch.Tensor], *, context: str) -> None:
    if not replay_data:
        return
    expected_tokens = int(replay_data[0].size(0))
    for idx, tensor in enumerate(replay_data[1:], start=1):
        tokens = int(tensor.size(0))
        if tokens != expected_tokens:
            raise ValueError(
                f"{context} must use the same token count for every router; "
                f"router 0 has tokens={expected_tokens}, router {idx} has tokens={tokens}."
            )


def _compact_routed_experts_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.numel():
        return tensor
    min_expert_idx = int(tensor.min().item())
    if min_expert_idx < 0:
        raise ValueError(
            f"Router Replay tensor must contain non-negative expert ids, got min={min_expert_idx}."
        )
    max_expert_idx = int(tensor.max().item())
    return tensor.to(get_routed_experts_dtype(max_expert_idx))


def _routed_experts_segment_to_list(
    routed_experts,
    *,
    num_routers: int | None,
    seq_lens: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    if routed_experts is None:
        raise ValueError("Router Replay segment cannot be None.")
    if isinstance(routed_experts, torch.Tensor):
        _validate_routed_experts_tensor(routed_experts, context="Router Replay segment")
        if routed_experts.dim() == 2:
            if num_routers is not None and num_routers != 1:
                raise ValueError(
                    "Router Replay 2D segment implies one router, "
                    f"but num_routers={num_routers}."
                )
            return [routed_experts]
        if routed_experts.dim() == 3:
            if num_routers is None:
                raise ValueError(
                    "Router Replay 3D segment requires num_routers to disambiguate "
                    "[routers, tokens, topk] from [tokens, routers, topk]."
                )
            routers_first = routed_experts.size(0) == num_routers
            tokens_first = routed_experts.size(1) == num_routers
            if routers_first and tokens_first:
                raise ValueError(
                    "Router Replay segment is ambiguous because both the first and "
                    "second dimensions equal num_routers. Pass per-router tensors."
                )
            if routers_first:
                return [tensor for tensor in routed_experts.unbind(0)]
            if tokens_first:
                return [tensor for tensor in routed_experts.unbind(1)]
            raise ValueError(
                "Router Replay 3D segment must have shape [routers, tokens, topk] "
                "or [tokens, routers, topk]."
            )
        if routed_experts.dim() == 4:
            if seq_lens is None:
                raise ValueError(
                    "Router Replay padded segment requires seq_lens for "
                    "[batch, seq, routers, topk] conversion."
                )
            return padded_routed_experts_to_list(
                routed_experts,
                seq_lens,
                num_routers=num_routers,
                compact=False,
            )
        raise ValueError(
            "Router Replay segment tensor must have shape [tokens, topk], "
            "[routers, tokens, topk], [tokens, routers, topk], or "
            "[batch, seq, routers, topk]."
        )
    if not isinstance(routed_experts, (list, tuple)):
        raise TypeError(
            "Router Replay segment must be a tensor, list of tensors, or tuple of tensors; "
            f"got {type(routed_experts)!r}."
        )
    replay_data = list(routed_experts)
    if num_routers is not None and len(replay_data) != num_routers:
        raise ValueError(
            f"Router Replay segment expected {num_routers} per-router tensors, "
            f"got {len(replay_data)}."
        )
    if any(item is None for item in replay_data):
        raise ValueError("Router Replay segment cannot contain None entries.")
    for idx, item in enumerate(replay_data):
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                "Router Replay segment entries must be tensors, "
                f"got {type(item)!r} at index {idx}."
            )
        if item.dim() != 2:
            raise ValueError(
                "Router Replay segment entries must have shape [tokens, topk], "
                f"got {tuple(item.shape)} at index {idx}."
            )
        _validate_routed_experts_tensor(item, context="Router Replay segment entries")
    _validate_per_router_token_rows(replay_data, context="Router Replay segment entries")
    return replay_data


def concat_routed_experts_segments(
    segments: Sequence[Any],
    seq_lens: Sequence[torch.Tensor | None] | None = None,
    *,
    num_routers: int | None = None,
    compact: bool = True,
) -> list[torch.Tensor]:
    """Concatenate multi-turn/agentic R3 traces along the token dimension.

    ROLL's user-defined rollout loop carries previous ``routed_experts`` and
    concatenates later turns along tokens. This helper gives MLite the same data
    contract while accepting the layouts already supported by the protocol:
    per-router lists, ``[routers, tokens, topk]``, ``[tokens, routers, topk]``,
    and padded ROLL/SGLang ``[batch, seq, routers, topk]`` segments.
    """

    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError("Router Replay segments must be a non-empty list or tuple.")
    if seq_lens is None:
        seq_lens = [None] * len(segments)
    elif not isinstance(seq_lens, (list, tuple)) or len(seq_lens) != len(segments):
        raise ValueError("Router Replay seq_lens must match the number of segments.")

    pieces_by_router: list[list[torch.Tensor]] | None = None
    inferred_num_routers = num_routers
    topk_by_router: list[int] | None = None
    for segment_idx, (segment, segment_seq_lens) in enumerate(zip(segments, seq_lens, strict=True)):
        per_router = _routed_experts_segment_to_list(
            segment,
            num_routers=inferred_num_routers,
            seq_lens=segment_seq_lens,
        )
        if inferred_num_routers is None:
            inferred_num_routers = len(per_router)
        if len(per_router) != inferred_num_routers:
            raise ValueError(
                "Router Replay segment router count mismatch: "
                f"segment {segment_idx} has {len(per_router)}, expected {inferred_num_routers}."
            )
        if pieces_by_router is None:
            pieces_by_router = [[] for _ in range(inferred_num_routers)]
            topk_by_router = [int(tensor.size(1)) for tensor in per_router]
        assert pieces_by_router is not None
        assert topk_by_router is not None
        for router_idx, tensor in enumerate(per_router):
            if tensor.dim() != 2:
                raise ValueError(
                    "Router Replay per-router segment must have shape [tokens, topk], "
                    f"got {tuple(tensor.shape)} for router {router_idx}."
                )
            expected_topk = topk_by_router[router_idx]
            if int(tensor.size(1)) != expected_topk:
                raise ValueError(
                    "Router Replay segment topk mismatch: "
                    f"segment {segment_idx} router {router_idx} has topk={tensor.size(1)}, "
                    f"expected {expected_topk}."
                )
            pieces_by_router[router_idx].append(tensor)

    assert pieces_by_router is not None
    concatenated = [torch.cat(pieces, dim=0).contiguous() for pieces in pieces_by_router]
    if compact:
        concatenated = [_compact_routed_experts_tensor(tensor) for tensor in concatenated]
    return concatenated


_ROUTED_EXPERTS_DIGEST_VERSION = b"mlite-router-replay-routed-experts-v1"


def _hash_int64_tensor_values(
    hasher: "hashlib._Hash",
    tensor: torch.Tensor,
    *,
    context: str,
) -> None:
    _validate_routed_experts_tensor(tensor, context=context)
    if tensor.numel():
        max_expert_idx = int(tensor.max().item())
        if max_expert_idx > torch.iinfo(torch.int64).max:
            raise ValueError(
                f"{context} expert ids must fit in int64 for digesting, "
                f"got max={max_expert_idx}."
            )
    canonical = tensor.detach().to(device="cpu", dtype=torch.int64).contiguous()
    for value in canonical.reshape(-1).tolist():
        hasher.update(int(value).to_bytes(8, byteorder="little", signed=True))


def _routed_experts_digest_from_list(replay_data: list[torch.Tensor]) -> str:
    if not replay_data:
        raise ValueError("Router Replay routed_experts digest requires at least one router.")
    for router_idx, tensor in enumerate(replay_data):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Router Replay digest data entries must be tensors, "
                f"got {type(tensor)!r} at index {router_idx}."
            )
        if tensor.dim() != 2:
            raise ValueError(
                "Router Replay digest data entries must have shape [tokens, topk], "
                f"got {tuple(tensor.shape)} at index {router_idx}."
            )
        _validate_routed_experts_tensor(
            tensor,
            context=f"Router Replay digest data entry {router_idx}",
        )
    _validate_per_router_token_rows(replay_data, context="Router Replay digest data")

    hasher = hashlib.sha256()
    hasher.update(_ROUTED_EXPERTS_DIGEST_VERSION)
    hasher.update(b"\0")
    hasher.update(f"routers={len(replay_data)}\n".encode("ascii"))
    for router_idx, tensor in enumerate(replay_data):
        rows, cols = (int(dim) for dim in tensor.shape)
        hasher.update(f"router={router_idx};shape={rows},{cols};dtype=int64-le\n".encode("ascii"))
        _hash_int64_tensor_values(
            hasher,
            tensor,
            context=f"Router Replay digest data entry {router_idx}",
        )
        hasher.update(b"\n")
    return f"sha256:{hasher.hexdigest()}"


def routed_experts_digest(
    routed_experts,
    *,
    num_routers: int | None = None,
    seq_lens: torch.Tensor | None = None,
) -> str:
    """Return a deterministic content digest for R3 routed expert ids.

    The digest canonicalizes all supported Router Replay layouts to ordered
    per-router true-token tensors before hashing values as int64 little-endian
    bytes. Equivalent replay ids therefore produce the same ``sha256:...``
    digest across compact dtypes and across list, routers-first, tokens-first,
    or padded ROLL/SGLang layouts.
    """

    replay_data = _routed_experts_segment_to_list(
        routed_experts,
        num_routers=num_routers,
        seq_lens=seq_lens,
    )
    return _routed_experts_digest_from_list(replay_data)


def routed_experts_segments_digest(
    segments: Sequence[Any],
    seq_lens: Sequence[torch.Tensor | None] | None = None,
    *,
    num_routers: int | None = None,
) -> str:
    """Return a deterministic digest for concatenated multi-turn R3 segments."""

    replay_data = concat_routed_experts_segments(
        segments,
        seq_lens,
        num_routers=num_routers,
        compact=False,
    )
    return _routed_experts_digest_from_list(replay_data)


def _routed_experts_to_list(
    routed_experts, num_routers: int, batch: PackedBatch
) -> list[torch.Tensor] | None:
    if routed_experts is None:
        return None
    if isinstance(routed_experts, torch.Tensor):
        _validate_routed_experts_tensor(routed_experts, context="Router Replay tensor data")
        if routed_experts.dim() == 2 and num_routers == 1:
            return [routed_experts]
        if routed_experts.dim() == 3:
            routers_first = routed_experts.size(0) == num_routers
            tokens_first = routed_experts.size(1) == num_routers
            if routers_first and tokens_first:
                raise ValueError(
                    "Router Replay tensor data is ambiguous because both the first and "
                    "second dimensions equal the number of routers. Pass a list of "
                    "per-router tensors, or use a non-ambiguous "
                    "[routers, tokens, topk] / [tokens, routers, topk] tensor."
                )
            if routers_first:
                return [tensor for tensor in routed_experts.unbind(0)]
            if tokens_first:
                return [tensor for tensor in routed_experts.unbind(1)]
        if routed_experts.dim() == 4:
            return padded_routed_experts_to_list(
                routed_experts,
                batch.seq_lens,
                num_routers=num_routers,
            )
        raise ValueError(
            "Router Replay tensor data must have shape [tokens, topk] for one router, "
            "[routers, tokens, topk], [tokens, routers, topk], or "
            "[batch, seq, routers, topk]."
        )
    if not isinstance(routed_experts, (list, tuple)):
        raise TypeError(
            "Router Replay data must be a tensor, list of tensors, tuple of tensors, or None; "
            f"got {type(routed_experts)!r}."
        )
    replay_data = list(routed_experts)
    if len(replay_data) != num_routers:
        raise ValueError(
            f"Router Replay expected {num_routers} tensors for this model chunk, "
            f"got {len(replay_data)}."
        )
    if any(item is None for item in replay_data):
        raise ValueError("Router Replay replay data cannot contain None entries.")
    for idx, item in enumerate(replay_data):
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                "Router Replay replay data entries must be tensors, "
                f"got {type(item)!r} at index {idx}."
            )
        if item.dim() != 2:
            raise ValueError(
                "Router Replay replay data entries must have shape [tokens, topk], "
                f"got {tuple(item.shape)} at index {idx}."
            )
        _validate_routed_experts_tensor(item, context="Router Replay replay data entries")
    _validate_per_router_token_rows(replay_data, context="Router Replay replay data entries")
    return replay_data


_ROUTER_REPLAY_LAYOUT_ALIASES = {
    "true": "true_tokens",
    "true_token": "true_tokens",
    "true_tokens": "true_tokens",
    "full": "full_padded",
    "full_padded": "full_padded",
    "cp_local": "cp_local",
    "local": "cp_local",
}


def _router_replay_layout_from_batch(batch: PackedBatch) -> str | None:
    extras = getattr(batch, "extras", None) or {}
    layout = extras.get("router_replay_layout")
    if layout is None:
        return None
    if not isinstance(layout, str):
        raise TypeError(
            "router_replay_layout must be a string when provided; "
            f"got {type(layout)!r}."
        )
    normalized = _ROUTER_REPLAY_LAYOUT_ALIASES.get(layout)
    if normalized is None:
        valid = ", ".join(sorted(set(_ROUTER_REPLAY_LAYOUT_ALIASES)))
        raise ValueError(
            f"Unsupported router_replay_layout={layout!r}; expected one of: {valid}."
        )
    return normalized


def _pack_replay_tensor_for_thd(
    topk_indices: torch.Tensor,
    batch: PackedBatch,
    ps: ParallelState,
    *,
    layout: str | None = None,
) -> torch.Tensor:
    if topk_indices.dim() != 2:
        raise ValueError(
            "Router Replay replay data must have shape [tokens, topk], "
            f"got {tuple(topk_indices.shape)}."
        )

    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=int(getattr(ps, "tp_size", 1) or 1),
        cp_size=int(getattr(ps, "cp_size", 1) or 1),
        cp_group=getattr(ps, "cp_group", None),
    )
    true_tokens = int(batch.seq_lens.sum().item())
    full_padded_tokens = int(meta.cu_seqlens_padded[-1].item())
    cp_size = int(getattr(ps, "cp_size", 1) or 1)
    cp_rank = int(getattr(ps, "cp_rank", 0) or 0)
    local_padded_tokens = full_padded_tokens // cp_size
    rows = int(topk_indices.size(0))

    if layout is not None:
        if layout == "true_tokens":
            expected_rows = true_tokens
        elif layout == "full_padded":
            expected_rows = full_padded_tokens
        elif layout == "cp_local":
            if cp_size <= 1:
                raise ValueError(
                    "Router Replay cp_local layout is only valid when cp_size > 1; "
                    "use router_replay_layout='true_tokens' or 'full_padded' instead."
                )
            expected_rows = local_padded_tokens
        else:
            raise AssertionError(f"unexpected normalized router_replay_layout={layout!r}")
        if rows != expected_rows:
            raise ValueError(
                "Router Replay replay data token count does not match "
                f"router_replay_layout={layout!r}: got {rows}, expected {expected_rows}."
            )

    if layout in {None, "true_tokens"} and rows == true_tokens:
        full = topk_indices.new_zeros((full_padded_tokens, topk_indices.size(1)))
        replay_offset = 0
        for idx, length_t in enumerate(batch.seq_lens):
            length = int(length_t.item())
            full_start = int(meta.cu_seqlens_padded[idx].item())
            full[full_start : full_start + length] = topk_indices[
                replay_offset : replay_offset + length
            ]
            replay_offset += length
    elif layout in {None, "full_padded"} and rows == full_padded_tokens:
        full = topk_indices
    elif layout in {None, "cp_local"} and rows == local_padded_tokens:
        return topk_indices
    else:
        raise ValueError(
            "Router Replay replay data token count does not match the PackedBatch THD layout: "
            f"got {rows}, expected true={true_tokens}, full_padded={full_padded_tokens}, "
            f"or cp_local={local_padded_tokens}."
        )

    if cp_size <= 1:
        return full
    return split_packed_to_cp_local(
        full,
        cu_seqlens_padded=meta.cu_seqlens_padded,
        cp_size=cp_size,
        cp_rank=cp_rank,
        dim=0,
    )


def _pack_router_replay_data_for_thd(
    replay_data: list[torch.Tensor] | None,
    *,
    model,
    batch: PackedBatch,
) -> list[torch.Tensor] | None:
    if replay_data is None:
        return None
    ps = _parallel_state(model)
    layout = _router_replay_layout_from_batch(batch)
    return [
        _pack_replay_tensor_for_thd(topk_indices, batch, ps, layout=layout)
        for topk_indices in replay_data
    ]


@contextmanager
def router_replay_context(model, batch: PackedBatch):
    action = _router_replay_action_from_batch(batch)
    replay_model = _unwrap_router_replay_model(model)
    instances_fn = getattr(replay_model, "router_replay_instances", None)
    routers = instances_fn() if callable(instances_fn) else []
    if not routers:
        if action is not None:
            raise ValueError(
                "Router Replay action/data was provided, but this model chunk has no "
                "router replay instances. Enable router_replay in the model ImplConfig."
            )
        yield
        return

    if action is None:
        yield
        return

    if (
        action == RouterReplayAction.RECORD
        and getattr(batch, "routed_experts", None) is not None
    ):
        raise ValueError(
            "Router Replay RECORD does not accept batch.routed_experts; "
            "record mode computes fresh routing decisions instead of replaying provided ids."
        )

    replay_data = _routed_experts_to_list(
        getattr(batch, "routed_experts", None),
        len(routers),
        batch,
    )
    replay_data = _pack_router_replay_data_for_thd(replay_data, model=replay_model, batch=batch)
    if action == RouterReplayAction.REPLAY_FORWARD and replay_data is None:
        raise ValueError("Router Replay forward replay requires batch.routed_experts.")
    if replay_data is not None and action in {
        RouterReplayAction.REPLAY_FORWARD,
        RouterReplayAction.REPLAY_BACKWARD,
    }:
        for router, topk_indices in zip(routers, replay_data, strict=True):
            router.set_target_indices(
                topk_indices,
                save_for_backward=(
                    action == RouterReplayAction.REPLAY_BACKWARD
                    or bool(getattr(router, "replay_backward_enabled", False))
                ),
            )

    for router in routers:
        router.set_router_replay_action(action)
    try:
        yield
    finally:
        for router in routers:
            router.clear_router_replay_action()
        if action == RouterReplayAction.RECORD:
            for router in routers:
                router.clear_indices()


def nested_from_packed(tensor: torch.Tensor | None, seq_lens: torch.Tensor):
    """Split a 1-D packed (true, unpadded) tensor back into a jagged nested tensor."""
    if tensor is None:
        return None
    if tensor.dim() == 2 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 1:
        raise ValueError(f"PackedBatch tensor must be 1-D, got {tuple(tensor.shape)}.")
    pieces = []
    offset = 0
    for length_t in seq_lens:
        length = int(length_t.item())
        pieces.append(tensor.narrow(0, offset, length))
        offset += length
    if offset != tensor.numel():
        raise ValueError(f"PackedBatch sizes sum to {offset}, tensor has {tensor.numel()} tokens.")
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def pack_thd_forward_kwargs(model, batch: PackedBatch) -> dict[str, Any]:
    """Pad + zigzag-CP-split a raw THD batch into model forward kwargs.

    Pads each sequence to the TE/zigzag alignment, then CP-splits tokens,
    labels, masks and position ids through the shared THD primitive — the same
    layout the model was validated against, now produced inside the protocol
    rather than the connector.
    """
    ps = _parallel_state(model)
    seq_lens = batch.seq_lens
    packed = pack_nested_thd(
        nested_from_packed(batch.input_ids, seq_lens),
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
        split_cp=False,
        labels=nested_from_packed(batch.labels, seq_lens),
        roll_labels=batch.labels is not None,
        loss_mask=nested_from_packed(batch.loss_mask, seq_lens),
        roll_loss_mask=batch.loss_mask is not None,
    )
    max_seqlen = int(packed.padded_lengths.max().item()) if packed.padded_lengths.numel() else 0
    # pack_nested_thd already returns [1, T] token rows; do not unsqueeze again.
    kwargs: dict[str, Any] = {
        "input_ids": packed.input_ids,
        "labels": packed.labels,
        "loss_mask": packed.loss_mask,
        "position_ids": packed.position_ids,
        "packed_seq_params": PackedSeqParams.from_cu_seqlens(
            packed.cu_seqlens_padded, max_seqlen=max_seqlen
        ),
    }
    prepare_packed_thd_kwargs_for_context_parallel(model, kwargs)
    return kwargs


def unpack_thd_forward_output(model, batch: PackedBatch, output: torch.Tensor) -> torch.Tensor:
    """Reverse a zigzag-CP THD model output back to jagged true-length form."""
    ps = _parallel_state(model)
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
    )
    return unpack_thd_to_nested(output, meta, contiguous=False)


def add_loss_context_kwargs(kwargs: dict[str, Any], *, include_return_log_probs: bool = False) -> None:
    loss_context = get_loss_context()
    if loss_context is None:
        return
    kwargs["temperature"] = loss_context.temperature
    kwargs["calculate_entropy"] = loss_context.calculate_entropy
    if include_return_log_probs:
        kwargs["return_log_probs"] = loss_context.return_log_probs


def add_cross_entropy_fusion(kwargs: dict[str, Any], model) -> None:
    kwargs["use_fused_kernels"] = bool(getattr(model, "cross_entropy_fusion", False))


def set_cross_entropy_fusion(chunks: list, enabled: bool) -> None:
    for chunk in chunks:
        chunk.cross_entropy_fusion = bool(enabled)


__all__ = [
    "add_cross_entropy_fusion",
    "add_loss_context_kwargs",
    "concat_routed_experts_segments",
    "nested_from_packed",
    "padded_routed_experts_to_list",
    "pack_thd_forward_kwargs",
    "routed_experts_digest",
    "routed_experts_segments_digest",
    "router_replay_context",
    "set_cross_entropy_fusion",
    "unpack_thd_forward_output",
]
