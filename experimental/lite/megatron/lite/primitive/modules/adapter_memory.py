# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA memory-capacity planning primitives for Megatron Lite.

The Scaling PEFT paper defines capacity efficiency as memory tokens per
trainable LoRA parameter and studies target-module ablations. This module keeps
that accounting local and deterministic: it estimates LoRA parameter counts,
classifies capacity-ratio regimes, and builds target-module ablation plans. It
does not load datasets, train adapters, evaluate accuracy, or claim the paper's
empirical module ranking as a local result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

CAPACITY_HIGH_ACCURACY_MAX = 1e-3
CAPACITY_COLLAPSE_MIN = 1e-2

CapacityRegime = Literal[
    "below_transition_expected_high_accuracy",
    "transition_band_expected_degradation",
    "above_collapse_expected_failure",
]
TargetGroup = Literal["mlp", "attention", "all", "unembed"]

TARGET_GROUPS: tuple[TargetGroup, ...] = ("mlp", "attention", "all", "unembed")
PAPER_REFERENCE_MODULE_RANKING: tuple[TargetGroup, ...] = ("mlp", "attention", "all", "unembed")


def _type_name(value: Any) -> str:
    return type(value).__name__


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {_type_name(value)}.")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {_type_name(value)}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return int(value)


def _positive_int(value: Any, *, name: str) -> int:
    value = _non_negative_int(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _finite_non_negative_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {_type_name(value)}.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}.")
    return value


def _normalize_target_group(value: Any, *, name: str = "target_group") -> TargetGroup:
    value = _non_empty_string(value, name=name).lower()
    if value not in TARGET_GROUPS:
        raise ValueError("target_group must be one of mlp, attention, all, or unembed.")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class LinearModuleSpec:
    """Shape and target group for one linear module that could receive LoRA."""

    name: str
    in_features: int
    out_features: int
    target_group: TargetGroup
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, name="name"))
        object.__setattr__(self, "in_features", _positive_int(self.in_features, name="in_features"))
        object.__setattr__(self, "out_features", _positive_int(self.out_features, name="out_features"))
        object.__setattr__(self, "target_group", _normalize_target_group(self.target_group))
        if not isinstance(self.metadata, Mapping):
            raise TypeError(f"metadata must be a mapping, got {_type_name(self.metadata)}.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def lora_trainable_params(self, rank: int) -> int:
        return lora_trainable_params(self.in_features, self.out_features, rank=rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "target_group": self.target_group,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LoraMemoryPlan:
    """Capacity-efficiency plan for one target-module group."""

    target_group: TargetGroup
    memory_tokens: int
    rank: int
    modules: tuple[LinearModuleSpec, ...]
    trainable_params: int
    tokens_per_trainable_param: float
    capacity_regime: CapacityRegime
    paper_accuracy_claimed: bool = False
    paper_module_ranking_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_group", _normalize_target_group(self.target_group))
        object.__setattr__(self, "memory_tokens", _non_negative_int(self.memory_tokens, name="memory_tokens"))
        object.__setattr__(self, "rank", _positive_int(self.rank, name="rank"))
        if not isinstance(self.modules, tuple) or not all(isinstance(item, LinearModuleSpec) for item in self.modules):
            raise TypeError("modules must be a tuple of LinearModuleSpec.")
        object.__setattr__(self, "trainable_params", _positive_int(self.trainable_params, name="trainable_params"))
        object.__setattr__(
            self,
            "tokens_per_trainable_param",
            _finite_non_negative_number(
                self.tokens_per_trainable_param,
                name="tokens_per_trainable_param",
            ),
        )
        object.__setattr__(
            self,
            "capacity_regime",
            classify_capacity_efficiency(self.tokens_per_trainable_param),
        )
        if not isinstance(self.paper_accuracy_claimed, bool):
            raise TypeError("paper_accuracy_claimed must be a boolean.")
        if not isinstance(self.paper_module_ranking_claimed, bool):
            raise TypeError("paper_module_ranking_claimed must be a boolean.")

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(module.name for module in self.modules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_group": self.target_group,
            "memory_tokens": self.memory_tokens,
            "rank": self.rank,
            "module_names": list(self.module_names),
            "trainable_params": self.trainable_params,
            "tokens_per_trainable_param": self.tokens_per_trainable_param,
            "capacity_regime": self.capacity_regime,
            "paper_accuracy_claimed": self.paper_accuracy_claimed,
            "paper_module_ranking_claimed": self.paper_module_ranking_claimed,
        }


def lora_trainable_params(in_features: int, out_features: int, *, rank: int) -> int:
    """Return LoRA A/B trainable parameters for one dense linear projection."""

    in_features = _positive_int(in_features, name="in_features")
    out_features = _positive_int(out_features, name="out_features")
    rank = _positive_int(rank, name="rank")
    return rank * (in_features + out_features)


def capacity_efficiency(*, memory_tokens: int, trainable_params: int) -> float:
    """Return memory tokens per trainable LoRA parameter."""

    memory_tokens = _non_negative_int(memory_tokens, name="memory_tokens")
    trainable_params = _positive_int(trainable_params, name="trainable_params")
    return float(memory_tokens) / float(trainable_params)


def classify_capacity_efficiency(ratio: float) -> CapacityRegime:
    """Classify the paper's capacity-efficiency regimes without claiming accuracy."""

    ratio = _finite_non_negative_number(ratio, name="capacity efficiency ratio")
    if ratio < CAPACITY_HIGH_ACCURACY_MAX:
        return "below_transition_expected_high_accuracy"
    if ratio <= CAPACITY_COLLAPSE_MIN:
        return "transition_band_expected_degradation"
    return "above_collapse_expected_failure"


def trainable_params_for_modules(modules: Sequence[LinearModuleSpec], *, rank: int) -> int:
    rank = _positive_int(rank, name="rank")
    modules = tuple(modules)
    if not modules:
        raise ValueError("modules must be non-empty.")
    if not all(isinstance(module, LinearModuleSpec) for module in modules):
        raise TypeError("modules must contain LinearModuleSpec entries.")
    return sum(module.lora_trainable_params(rank) for module in modules)


def select_target_group_modules(
    modules: Sequence[LinearModuleSpec],
    target_group: TargetGroup,
) -> tuple[LinearModuleSpec, ...]:
    target_group = _normalize_target_group(target_group)
    modules = tuple(modules)
    if not modules:
        raise ValueError("modules must be non-empty.")
    if not all(isinstance(module, LinearModuleSpec) for module in modules):
        raise TypeError("modules must contain LinearModuleSpec entries.")
    if target_group == "all":
        selected = tuple(module for module in modules if module.target_group in {"attention", "mlp"})
    else:
        selected = tuple(module for module in modules if module.target_group == target_group)
    if not selected:
        raise ValueError(f"target group {target_group!r} selected no modules.")
    return selected


def build_lora_memory_plan(
    modules: Sequence[LinearModuleSpec],
    *,
    target_group: TargetGroup,
    rank: int,
    memory_tokens: int,
) -> LoraMemoryPlan:
    """Build one local capacity-efficiency plan for a target-module group."""

    selected = select_target_group_modules(modules, target_group)
    trainable = trainable_params_for_modules(selected, rank=rank)
    ratio = capacity_efficiency(memory_tokens=memory_tokens, trainable_params=trainable)
    return LoraMemoryPlan(
        target_group=_normalize_target_group(target_group),
        memory_tokens=memory_tokens,
        rank=rank,
        modules=selected,
        trainable_params=trainable,
        tokens_per_trainable_param=ratio,
        capacity_regime=classify_capacity_efficiency(ratio),
    )


def build_target_module_ablation_plans(
    modules: Sequence[LinearModuleSpec],
    *,
    rank: int,
    memory_tokens: int,
) -> tuple[LoraMemoryPlan, ...]:
    """Return MLP, attention, all, and unembed LoRA capacity plans."""

    return tuple(
        build_lora_memory_plan(
            modules,
            target_group=target_group,
            rank=rank,
            memory_tokens=memory_tokens,
        )
        for target_group in TARGET_GROUPS
    )


def build_rank_shift_plans(
    modules: Sequence[LinearModuleSpec],
    *,
    target_group: TargetGroup,
    ranks: Sequence[int],
    memory_tokens: int,
) -> tuple[LoraMemoryPlan, ...]:
    """Show how increasing rank moves a fixed-memory task across regimes."""

    ranks = tuple(_positive_int(rank, name="rank") for rank in ranks)
    if not ranks:
        raise ValueError("ranks must be non-empty.")
    return tuple(
        build_lora_memory_plan(
            modules,
            target_group=target_group,
            rank=rank,
            memory_tokens=memory_tokens,
        )
        for rank in ranks
    )


def paper_reference_target_order() -> tuple[TargetGroup, ...]:
    """Return the paper's target-module ranking as reference context only."""

    return PAPER_REFERENCE_MODULE_RANKING


__all__ = [
    "CAPACITY_COLLAPSE_MIN",
    "CAPACITY_HIGH_ACCURACY_MAX",
    "CapacityRegime",
    "LinearModuleSpec",
    "LoraMemoryPlan",
    "PAPER_REFERENCE_MODULE_RANKING",
    "TARGET_GROUPS",
    "TargetGroup",
    "build_lora_memory_plan",
    "build_rank_shift_plans",
    "build_target_module_ablation_plans",
    "capacity_efficiency",
    "classify_capacity_efficiency",
    "lora_trainable_params",
    "paper_reference_target_order",
    "select_target_group_modules",
    "trainable_params_for_modules",
]
