# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE router replay utilities.

Router Replay records or reuses the discrete top-k expert indices selected by
MoE routers. It is intentionally separate from DSA IndexShare: replayed values
are expert ids, not attention key positions.
"""

from __future__ import annotations

import re
import weakref
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Literal

import torch

RouterReplayTraceSource = Literal["routed_experts", "routed_experts_segments"]
RouterReplayTraceCollectionPath = Literal[
    "generate_request",
    "router_generate_request",
    "user_defined_rollout_loop",
    "mlite_engine_bridge",
]
RouterReplayLayout = Literal["true_tokens", "full_padded", "cp_local"]

ROUTER_REPLAY_TRACE_COLLECTION_PATHS: tuple[RouterReplayTraceCollectionPath, ...] = (
    "generate_request",
    "router_generate_request",
    "user_defined_rollout_loop",
    "mlite_engine_bridge",
)
ROUTER_REPLAY_TRACE_SOURCES: tuple[RouterReplayTraceSource, ...] = (
    "routed_experts",
    "routed_experts_segments",
)
ROUTER_REPLAY_LAYOUTS: tuple[RouterReplayLayout, ...] = (
    "true_tokens",
    "full_padded",
    "cp_local",
)
ROUTER_REPLAY_TRACE_BOOL_EXTRA_KEYS: tuple[str, ...] = (
    "router_replay_trace_preserved_through_scheduler",
    "router_replay_postprocess_wrote_trace_to_batch",
)
ROUTER_REPLAY_TRACE_CONFLICTS_KEY = "router_replay_trace_metadata_conflicts"

_SHA256_DIGEST_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")


def _type_name(value: Any) -> str:
    return type(value).__name__


def _normalize_string_choice(value: Any, *, name: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {_type_name(value)}.")
    normalized = value.strip()
    if normalized not in choices:
        valid = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {valid}; got {value!r}.")
    return normalized


def normalize_routed_experts_digest(value: Any) -> str:
    """Normalize an R3 routed-experts content digest to ``sha256:<hex>``."""

    if not isinstance(value, str):
        raise TypeError(
            "routed_experts_digest must be a string, "
            f"got {_type_name(value)}."
        )
    match = _SHA256_DIGEST_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            "routed_experts_digest must be a sha256 digest formatted as "
            "sha256:<64 hex chars> or <64 hex chars>."
        )
    return f"sha256:{match.group(1).lower()}"


def normalize_router_replay_trace_collection_path(
    value: Any,
) -> RouterReplayTraceCollectionPath:
    """Validate the rollout path that preserved an R3 route trace."""

    return _normalize_string_choice(
        value,
        name="router_replay_trace_collection_path",
        choices=ROUTER_REPLAY_TRACE_COLLECTION_PATHS,
    )  # type: ignore[return-value]


def normalize_router_replay_trace_source(value: Any) -> RouterReplayTraceSource:
    """Validate the carrier used for an R3 route trace."""

    return _normalize_string_choice(
        value,
        name="router_replay_trace_source",
        choices=ROUTER_REPLAY_TRACE_SOURCES,
    )  # type: ignore[return-value]


def normalize_router_replay_layout(value: Any) -> RouterReplayLayout:
    """Validate the token layout for an R3 replay payload."""

    return _normalize_string_choice(
        value,
        name="router_replay_layout",
        choices=ROUTER_REPLAY_LAYOUTS,
    )  # type: ignore[return-value]


def _normalize_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise TypeError(f"{name} must be a boolean, got {_type_name(value)}.")


def _normalize_conflicts(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.strip()
        return (value,) if value else ()
    if not isinstance(value, Sequence):
        raise TypeError(
            "router_replay_trace_metadata_conflicts must be a sequence of strings, "
            f"got {_type_name(value)}."
        )
    conflicts = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(
                "router_replay_trace_metadata_conflicts entries must be strings, "
                f"got {_type_name(item)} at index {idx}."
            )
        item = item.strip()
        if item:
            conflicts.append(item)
    return tuple(conflicts)


@dataclass(frozen=True)
class RouterReplayTraceContract:
    """No-launch contract for an R3 rollout trace handed to MLite training.

    This records the source, token layout, and content digest that must move
    with ``routed_experts`` until ``RouterReplay`` consumes the payload. It is a
    local evidence contract only: it must not claim that SGLang, Ray, GPUs, or a
    paper-scale RL smoke actually ran.
    """

    collection_path: RouterReplayTraceCollectionPath
    trace_source: RouterReplayTraceSource
    routed_experts_digest: str
    router_replay_layout: RouterReplayLayout = "true_tokens"
    trace_preserved_through_scheduler: bool = True
    postprocess_wrote_trace_to_batch: bool = True
    metadata_conflicts: Sequence[str] = field(default_factory=tuple)
    starts_sglang: bool = False
    launches_training: bool = False
    claims_real_smoke: bool = False
    claims_paper_results: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "collection_path",
            normalize_router_replay_trace_collection_path(self.collection_path),
        )
        object.__setattr__(
            self,
            "trace_source",
            normalize_router_replay_trace_source(self.trace_source),
        )
        object.__setattr__(
            self,
            "routed_experts_digest",
            normalize_routed_experts_digest(self.routed_experts_digest),
        )
        object.__setattr__(
            self,
            "router_replay_layout",
            normalize_router_replay_layout(self.router_replay_layout),
        )
        object.__setattr__(
            self,
            "trace_preserved_through_scheduler",
            _normalize_bool(
                self.trace_preserved_through_scheduler,
                name="router_replay_trace_preserved_through_scheduler",
            ),
        )
        object.__setattr__(
            self,
            "postprocess_wrote_trace_to_batch",
            _normalize_bool(
                self.postprocess_wrote_trace_to_batch,
                name="router_replay_postprocess_wrote_trace_to_batch",
            ),
        )
        object.__setattr__(
            self,
            "metadata_conflicts",
            _normalize_conflicts(self.metadata_conflicts),
        )
        for name in (
            "starts_sglang",
            "launches_training",
            "claims_real_smoke",
            "claims_paper_results",
        ):
            object.__setattr__(self, name, _normalize_bool(getattr(self, name), name=name))
        if self.trace_source == "routed_experts_segments" and self.router_replay_layout != "true_tokens":
            raise ValueError(
                "routed_experts_segments traces must use router_replay_layout='true_tokens'."
            )
        if not self.trace_preserved_through_scheduler:
            raise ValueError("RouterReplayTraceContract requires trace preservation through scheduler.")
        if not self.postprocess_wrote_trace_to_batch:
            raise ValueError("RouterReplayTraceContract requires postprocess to write trace to batch.")
        if self.metadata_conflicts:
            raise ValueError(
                "RouterReplayTraceContract requires empty trace metadata conflicts, "
                f"got {list(self.metadata_conflicts)!r}."
            )
        if self.starts_sglang or self.launches_training or self.claims_real_smoke:
            raise ValueError(
                "RouterReplayTraceContract is a local no-launch contract and must not "
                "claim real SGLang/GPU smoke execution."
            )
        if self.claims_paper_results:
            raise ValueError("RouterReplayTraceContract must not claim paper results.")

    def to_packed_batch_extras(self) -> dict[str, Any]:
        """Return the MLite/verl side-channel fields that accompany replay data."""

        return {
            "router_replay_layout": self.router_replay_layout,
            "router_replay_trace_collection_path": self.collection_path,
            "router_replay_trace_source": self.trace_source,
            "router_replay_trace_preserved_through_scheduler": (
                self.trace_preserved_through_scheduler
            ),
            "router_replay_postprocess_wrote_trace_to_batch": (
                self.postprocess_wrote_trace_to_batch
            ),
            "routed_experts_digest": self.routed_experts_digest,
            ROUTER_REPLAY_TRACE_CONFLICTS_KEY: list(self.metadata_conflicts),
        }

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an audit-friendly no-result contract payload."""

        segment_layout_ok = (
            self.trace_source != "routed_experts_segments"
            or self.router_replay_layout == "true_tokens"
        )
        return {
            "status": "pass_router_replay_trace_contract",
            "paper_exact": False,
            "claims_paper_results": self.claims_paper_results,
            "claims_real_smoke": self.claims_real_smoke,
            "starts_sglang": self.starts_sglang,
            "launches_training": self.launches_training,
            "extras": self.to_packed_batch_extras(),
            "invariants": {
                "digest_is_sha256": self.routed_experts_digest.startswith("sha256:")
                and len(self.routed_experts_digest) == len("sha256:") + 64,
                "trace_source_is_supported": self.trace_source in ROUTER_REPLAY_TRACE_SOURCES,
                "collection_path_is_trace_preserving": (
                    self.collection_path in ROUTER_REPLAY_TRACE_COLLECTION_PATHS
                ),
                "layout_is_supported": self.router_replay_layout in ROUTER_REPLAY_LAYOUTS,
                "segment_traces_use_true_tokens_layout": segment_layout_ok,
                "trace_preserved_through_scheduler": self.trace_preserved_through_scheduler,
                "postprocess_wrote_trace_to_batch": self.postprocess_wrote_trace_to_batch,
                "metadata_conflicts_empty": not self.metadata_conflicts,
                "local_contract_does_not_launch": not self.starts_sglang
                and not self.launches_training,
                "local_contract_does_not_claim_results": not self.claims_real_smoke
                and not self.claims_paper_results,
            },
        }


def build_router_replay_trace_contract(
    *,
    collection_path: Any,
    trace_source: Any,
    routed_experts_digest: Any,
    router_replay_layout: Any = "true_tokens",
    trace_preserved_through_scheduler: Any = True,
    postprocess_wrote_trace_to_batch: Any = True,
    metadata_conflicts: Any = (),
    starts_sglang: Any = False,
    launches_training: Any = False,
    claims_real_smoke: Any = False,
    claims_paper_results: Any = False,
) -> RouterReplayTraceContract:
    """Build the local R3 trace contract used before real-smoke evidence exists."""

    return RouterReplayTraceContract(
        collection_path=collection_path,
        trace_source=trace_source,
        routed_experts_digest=routed_experts_digest,
        router_replay_layout=router_replay_layout,
        trace_preserved_through_scheduler=trace_preserved_through_scheduler,
        postprocess_wrote_trace_to_batch=postprocess_wrote_trace_to_batch,
        metadata_conflicts=metadata_conflicts,
        starts_sglang=starts_sglang,
        launches_training=launches_training,
        claims_real_smoke=claims_real_smoke,
        claims_paper_results=claims_paper_results,
    )


def get_routed_experts_dtype(max_expert_idx: int) -> torch.dtype:
    """Return the compact integer dtype for routed expert ids."""

    if not isinstance(max_expert_idx, int):
        raise TypeError(
            "max_expert_idx must be an int when selecting routed_experts dtype, "
            f"got {type(max_expert_idx)!r}."
        )
    if max_expert_idx < 0:
        raise ValueError(
            f"max_expert_idx must be non-negative, got {max_expert_idx}."
        )
    if max_expert_idx <= 255:
        return torch.uint8
    if max_expert_idx <= 32767:
        return torch.int16
    raise ValueError(
        f"Expert index {max_expert_idx} exceeds int16 range (0-32767); "
        "RouterReplay routed_experts compact storage needs a larger dtype."
    )


class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplay:
    """Per-router state for recording and replaying top-k expert indices."""

    global_router_replay_instances: ClassVar[list[weakref.ReferenceType["RouterReplay"]]] = []

    def __init__(self, *, layer_idx: int | None = None):
        self.layer_idx = layer_idx
        self.target_topk_idx: torch.Tensor | None = None
        self.recorded_topk_idx: torch.Tensor | None = None
        self.router_replay_action: RouterReplayAction | None = None
        self.replay_backward_enabled = False
        self.replay_backward_list: list[torch.Tensor] = []
        RouterReplay.global_router_replay_instances.append(weakref.ref(self))

    @classmethod
    def _live_global_instances(cls) -> list["RouterReplay"]:
        live: list[RouterReplay] = []
        live_refs: list[weakref.ReferenceType[RouterReplay]] = []
        for ref in cls.global_router_replay_instances:
            router = ref()
            if router is not None:
                live.append(router)
                live_refs.append(ref)
        cls.global_router_replay_instances = live_refs
        return live

    @property
    def active(self) -> bool:
        return self.router_replay_action is not None

    @staticmethod
    def set_replay_data(all_layers_topk_indices: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> None:
        if not isinstance(all_layers_topk_indices, (list, tuple)):
            raise TypeError(
                "RouterReplay.set_replay_data expects a list or tuple of per-router "
                f"tensors, got {type(all_layers_topk_indices)!r}."
            )
        routers = RouterReplay._live_global_instances()
        if len(all_layers_topk_indices) != len(routers):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) does not "
                f"match router replay instances ({len(routers)})."
            )
        for router, topk_indices in zip(routers, all_layers_topk_indices, strict=True):
            router.set_target_indices(topk_indices)

    @staticmethod
    def get_recorded_data() -> list[torch.Tensor | None]:
        return [router.get_recorded_indices() for router in RouterReplay._live_global_instances()]

    @staticmethod
    def clear_global_indices() -> None:
        for router in RouterReplay._live_global_instances():
            router.clear_indices()

    @staticmethod
    def set_global_router_replay_action(action: RouterReplayAction | str) -> None:
        for router in RouterReplay._live_global_instances():
            router.set_router_replay_action(action)

    @staticmethod
    def clear_global_router_replay_action() -> None:
        for router in RouterReplay._live_global_instances():
            router.clear_router_replay_action()

    @staticmethod
    def clear_global_router_replay_instances() -> None:
        RouterReplay.global_router_replay_instances.clear()

    def enable_replay_backward_queue(self) -> None:
        self.replay_backward_enabled = True

    def set_target_indices(
        self, topk_indices: torch.Tensor, *, save_for_backward: bool | None = None
    ) -> None:
        topk_indices = self._snapshot_index_tensor(
            self._validate_index_tensor(topk_indices, description="target indices")
        )
        self.target_topk_idx = topk_indices
        if save_for_backward is None:
            save_for_backward = self.replay_backward_enabled
        if save_for_backward:
            self.replay_backward_list.append(self._snapshot_index_tensor(topk_indices))

    @staticmethod
    def _validate_index_tensor(topk_indices: torch.Tensor, *, description: str) -> torch.Tensor:
        if not isinstance(topk_indices, torch.Tensor):
            raise TypeError(
                f"RouterReplay {description} must be a tensor, got {type(topk_indices)!r}."
            )
        if topk_indices.ndim != 2:
            raise ValueError(
                f"RouterReplay {description} must have shape [tokens, topk], "
                f"got {tuple(topk_indices.shape)}."
            )
        if (
            topk_indices.dtype == torch.bool
            or torch.is_floating_point(topk_indices)
            or torch.is_complex(topk_indices)
        ):
            raise TypeError(
                f"RouterReplay {description} must use an integer tensor dtype, "
                f"got {topk_indices.dtype}."
            )
        return topk_indices

    @staticmethod
    def _snapshot_index_tensor(topk_indices: torch.Tensor) -> torch.Tensor:
        return topk_indices.detach().clone()

    def get_recorded_indices(self) -> torch.Tensor | None:
        if self.recorded_topk_idx is None:
            return None
        return self._snapshot_index_tensor(self.recorded_topk_idx)

    def clear_indices(self) -> None:
        self.target_topk_idx = None
        self.recorded_topk_idx = None
        self.replay_backward_list = []

    @staticmethod
    def _coerce_action(action: RouterReplayAction | str) -> RouterReplayAction:
        if isinstance(action, RouterReplayAction):
            return action
        if isinstance(action, str):
            try:
                return RouterReplayAction(action)
            except ValueError as exc:
                valid = ", ".join(item.value for item in RouterReplayAction)
                raise ValueError(
                    f"Unsupported RouterReplay action={action!r}; expected one of: {valid}."
                ) from exc
        raise TypeError(
            "RouterReplay action must be a RouterReplayAction or string value, "
            f"got {type(action)!r}."
        )

    def set_router_replay_action(self, action: RouterReplayAction | str) -> None:
        self.router_replay_action = self._coerce_action(action)

    def clear_router_replay_action(self) -> None:
        self.router_replay_action = None

    def record_indices(self, topk_indices: torch.Tensor) -> None:
        self.recorded_topk_idx = self._snapshot_index_tensor(
            self._validate_index_tensor(topk_indices, description="recorded indices")
        )

    @contextmanager
    def action_context(self, action: RouterReplayAction | str):
        previous = self.router_replay_action
        self.set_router_replay_action(action)
        try:
            yield
        finally:
            self.router_replay_action = previous

    def _validated_replay_indices(
        self,
        scores: torch.Tensor,
        topk: int,
        indices: torch.Tensor,
        *,
        description: str = "replay indices",
    ) -> torch.Tensor:
        self._validate_index_tensor(indices, description=description)
        indices = indices.to(device=scores.device, dtype=torch.long)
        expected = (scores.shape[0], topk)
        if tuple(indices.shape) != expected:
            raise ValueError(
                f"RouterReplay {description} shape mismatch: "
                f"got {tuple(indices.shape)}, expected {expected} for scores {tuple(scores.shape)}."
            )
        if indices.numel():
            min_idx = int(indices.min().item())
            max_idx = int(indices.max().item())
            if min_idx < 0 or max_idx >= scores.shape[1]:
                raise ValueError(
                    f"RouterReplay {description} are out of range for router scores: "
                    f"min={min_idx}, max={max_idx}, num_experts={scores.shape[1]}."
                )
        return indices

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: int | None,
        group_topk: int | None,
        default_compute_topk: Callable[
            [torch.Tensor, int, int | None, int | None], tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.router_replay_action == RouterReplayAction.RECORD:
            values, indices = default_compute_topk(scores, topk, num_groups, group_topk)
            indices = self._validated_replay_indices(
                scores, topk, indices, description="recorded indices"
            )
            self.record_indices(indices)
            return values, indices
        if self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            if self.target_topk_idx is None:
                raise RuntimeError("RouterReplay REPLAY_FORWARD requires target_topk_idx.")
            indices = self._validated_replay_indices(scores, topk, self.target_topk_idx)
            return scores.gather(1, indices), indices
        if self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            if not self.replay_backward_list:
                raise RuntimeError("RouterReplay REPLAY_BACKWARD has no saved forward indices.")
            indices = self._validated_replay_indices(scores, topk, self.replay_backward_list[0])
            self.replay_backward_list.pop(0)
            return scores.gather(1, indices), indices
        return default_compute_topk(scores, topk, num_groups, group_topk)


__all__ = [
    "ROUTER_REPLAY_LAYOUTS",
    "ROUTER_REPLAY_TRACE_BOOL_EXTRA_KEYS",
    "ROUTER_REPLAY_TRACE_COLLECTION_PATHS",
    "ROUTER_REPLAY_TRACE_CONFLICTS_KEY",
    "ROUTER_REPLAY_TRACE_SOURCES",
    "RouterReplay",
    "RouterReplayAction",
    "RouterReplayLayout",
    "RouterReplayTraceCollectionPath",
    "RouterReplayTraceContract",
    "RouterReplayTraceSource",
    "build_router_replay_trace_contract",
    "get_routed_experts_dtype",
    "normalize_routed_experts_digest",
    "normalize_router_replay_layout",
    "normalize_router_replay_trace_collection_path",
    "normalize_router_replay_trace_source",
]
