# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA rank-regime and rsLoRA transfer planning primitives for Megatron Lite.

The Scaling PEFT paper treats LoRA rank as an experimental axis, not just a
shape knob. This module makes the paper's no-result contract explicit: it
constructs the Qwen3-8B 216-arm rank-regime sweep, records the three rsLoRA
alpha-scaling conventions, checks the Eq.2 early-step perturbation term, tracks
the Fig.15 rank-1 OLoRA-tail stability grid, and analyzes externally supplied
result tables against the paper's qualitative patterns. It does not launch training, load weights, evaluate models, or claim the paper's empirical rank-regime, LR-transfer, or rank-1 stability results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

DEFAULT_RANKS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_BATCH_SIZES: tuple[int, ...] = (16, 32, 64, 128)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
DEFAULT_STEPS = 500
DEFAULT_REFERENCE_RANK = 16
DEFAULT_REFERENCE_ALPHA = 32.0
PAPER_RANK_REGIME_RUN_COUNT = 216
PAPER_RANK1_OLORA_TAIL_RUN_COUNT = 48

AlphaScalingRule = Literal["const_alpha", "fixed_alpha_over_rank", "sqrt_rank_rslora"]
RuntimeScalingConvention = Literal["alpha_over_rank", "alpha_over_sqrt_rank"]
Rank1Init = Literal["standard", "olora_tail"]

ALPHA_SCALING_RULES: tuple[AlphaScalingRule, ...] = (
    "const_alpha",
    "fixed_alpha_over_rank",
    "sqrt_rank_rslora",
)
RANK1_INIT_VALUES: tuple[Rank1Init, ...] = ("standard", "olora_tail")


def _type_name(value: Any) -> str:
    return type(value).__name__


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {_type_name(value)}.")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {_type_name(value)}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return int(value)


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {_type_name(value)}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return int(value)


def _finite_positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {_type_name(value)}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}.")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {_type_name(value)}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return value


def _required_true(value: Any, *, name: str) -> bool:
    if value is not True:
        raise ValueError(f"{name} must be true.")
    return True


def _invariant_value_passes(value: bool | int) -> bool:
    return value is True or (isinstance(value, int) and not isinstance(value, bool))


def _sorted_unique_positive_ints(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    values = tuple(_positive_int(value, name=name) for value in values)
    if not values:
        raise ValueError(f"{name} must be non-empty.")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be sorted ascending.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")
    return values


def _sorted_unique_non_negative_ints(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    values = tuple(_non_negative_int(value, name=name) for value in values)
    if not values:
        raise ValueError(f"{name} must be non-empty.")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be sorted ascending.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")
    return values


def _sorted_unique_positive_numbers(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of finite positive numbers.")
    values = tuple(_finite_positive_number(value, name=name) for value in values)
    if not values:
        raise ValueError(f"{name} must be non-empty.")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be sorted ascending.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")
    return values


def _normalize_alpha_rule(value: Any) -> AlphaScalingRule:
    value = _non_empty_string(value, name="alpha_rule")
    if value not in ALPHA_SCALING_RULES:
        raise ValueError("alpha_rule must be const_alpha, fixed_alpha_over_rank, or sqrt_rank_rslora.")
    return value  # type: ignore[return-value]


def _normalize_rank1_init(value: Any) -> Rank1Init:
    value = _non_empty_string(value, name="init_lora_weights")
    if value in {"standard", "lora", "standard_lora"}:
        return "standard"
    if value in {"olora_tail", "olora-tail"}:
        return "olora_tail"
    raise ValueError("init_lora_weights must be standard or olora_tail.")


def lora_runtime_scale(rank: int, alpha: float, *, use_rslora: bool) -> float:
    """Return the runtime LoRA scale alpha/r or alpha/sqrt(r)."""

    rank = _positive_int(rank, name="rank")
    alpha = _finite_positive_number(alpha, name="alpha")
    if not isinstance(use_rslora, bool):
        raise TypeError("use_rslora must be a boolean.")
    denominator = math.sqrt(float(rank)) if use_rslora else float(rank)
    return alpha / denominator


def eq2_alpha_squared_over_rank(rank: int, alpha: float) -> float:
    """Return the paper Eq.2 alpha_r squared over rank factor."""

    rank = _positive_int(rank, name="rank")
    alpha = _finite_positive_number(alpha, name="alpha")
    return alpha * alpha / float(rank)


def alpha_for_scaling_rule(
    rule: AlphaScalingRule,
    rank: int,
    *,
    reference_rank: int = DEFAULT_REFERENCE_RANK,
    reference_alpha: float = DEFAULT_REFERENCE_ALPHA,
) -> tuple[float, bool, RuntimeScalingConvention]:
    """Return alpha, use_rslora, and runtime convention for one paper rule."""

    rule = _normalize_alpha_rule(rule)
    rank = _positive_int(rank, name="rank")
    reference_rank = _positive_int(reference_rank, name="reference_rank")
    reference_alpha = _finite_positive_number(reference_alpha, name="reference_alpha")
    if rule == "const_alpha":
        return reference_alpha, False, "alpha_over_rank"
    if rule == "fixed_alpha_over_rank":
        return float(rank), False, "alpha_over_rank"
    base_scale = reference_alpha / math.sqrt(float(reference_rank))
    return base_scale * math.sqrt(float(rank)), True, "alpha_over_sqrt_rank"


def mlite_lora_overrides(
    *,
    rank: int,
    alpha: float,
    use_rslora: bool,
    lora_init: str = "standard",
) -> tuple[str, ...]:
    """Return Hydra-style overrides used by MLite/verl adapters."""

    rank = _positive_int(rank, name="rank")
    alpha = _finite_positive_number(alpha, name="alpha")
    if not isinstance(use_rslora, bool):
        raise TypeError("use_rslora must be a boolean.")
    lora_init = _non_empty_string(lora_init, name="lora_init")
    return (
        f"+actor_rollout_ref.actor.engine.impl_cfg.lora.rank={rank}",
        f"+actor_rollout_ref.actor.engine.impl_cfg.lora.alpha={alpha:.12g}",
        "+actor_rollout_ref.actor.engine.impl_cfg.lora.dropout=0.0",
        "+actor_rollout_ref.actor.engine.impl_cfg.lora.use_rslora="
        + ("true" if use_rslora else "false"),
        f"+actor_rollout_ref.actor.engine.impl_cfg.lora_init={lora_init}",
    )


@dataclass(frozen=True)
class RsLoraTransferArm:
    """One no-result rsLoRA LR-transfer planning arm."""

    alpha_rule: AlphaScalingRule
    rank: int
    alpha: float
    use_rslora: bool
    scaling_convention: RuntimeScalingConvention
    runtime_scale: float
    eq2_alpha_squared_over_rank: float
    result_status: str = "pending_gpu_training_and_lr_grid"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    claims_paper_results: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha_rule", _normalize_alpha_rule(self.alpha_rule))
        object.__setattr__(self, "rank", _positive_int(self.rank, name="rank"))
        object.__setattr__(self, "alpha", _finite_positive_number(self.alpha, name="alpha"))
        if not isinstance(self.use_rslora, bool):
            raise TypeError("use_rslora must be a boolean.")
        if self.scaling_convention not in {"alpha_over_rank", "alpha_over_sqrt_rank"}:
            raise ValueError("scaling_convention must be alpha_over_rank or alpha_over_sqrt_rank.")
        object.__setattr__(
            self,
            "runtime_scale",
            _finite_positive_number(self.runtime_scale, name="runtime_scale"),
        )
        object.__setattr__(
            self,
            "eq2_alpha_squared_over_rank",
            _finite_positive_number(
                self.eq2_alpha_squared_over_rank,
                name="eq2_alpha_squared_over_rank",
            ),
        )
        object.__setattr__(self, "result_status", _non_empty_string(self.result_status, name="result_status"))
        if not isinstance(self.metadata, Mapping):
            raise TypeError(f"metadata must be a mapping, got {_type_name(self.metadata)}.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not isinstance(self.claims_paper_results, bool):
            raise TypeError("claims_paper_results must be a boolean.")
        if self.claims_paper_results:
            raise ValueError("RsLoraTransferArm must not claim paper metrics.")

    @property
    def arm_id(self) -> str:
        return f"rslora_lr_transfer/{self.alpha_rule}/r{self.rank}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "alpha_rule": self.alpha_rule,
            "rank": self.rank,
            "alpha": self.alpha,
            "use_rslora": self.use_rslora,
            "scaling_convention": self.scaling_convention,
            "runtime_scale": self.runtime_scale,
            "eq2_alpha_squared_over_rank": self.eq2_alpha_squared_over_rank,
            "mlite_overrides": list(
                mlite_lora_overrides(
                    rank=self.rank,
                    alpha=self.alpha,
                    use_rslora=self.use_rslora,
                )
            ),
            "requires_lr_grid": True,
            "result_status": self.result_status,
            "metadata": dict(self.metadata),
            "claims_paper_results": self.claims_paper_results,
        }


@dataclass(frozen=True)
class RankRegimeArm:
    """One no-result Qwen3-8B rank-regime PPO planning arm."""

    rank: int
    effective_batch_size: int
    seed: int
    steps: int = DEFAULT_STEPS
    model_family: str = "Qwen3-8B"
    algorithm: str = "PPO"
    domain: str = "math"
    result_status: str = "pending_gpu_training"
    claims_paper_results: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank", _positive_int(self.rank, name="rank"))
        object.__setattr__(
            self,
            "effective_batch_size",
            _positive_int(self.effective_batch_size, name="effective_batch_size"),
        )
        object.__setattr__(self, "seed", _non_negative_int(self.seed, name="seed"))
        object.__setattr__(self, "steps", _positive_int(self.steps, name="steps"))
        object.__setattr__(self, "model_family", _non_empty_string(self.model_family, name="model_family"))
        object.__setattr__(self, "algorithm", _non_empty_string(self.algorithm, name="algorithm"))
        object.__setattr__(self, "domain", _non_empty_string(self.domain, name="domain"))
        object.__setattr__(self, "result_status", _non_empty_string(self.result_status, name="result_status"))
        if not isinstance(self.claims_paper_results, bool):
            raise TypeError("claims_paper_results must be a boolean.")
        if self.claims_paper_results:
            raise ValueError("RankRegimeArm must not claim paper metrics.")

    @property
    def arm_id(self) -> str:
        return (
            f"rank_regime/qwen3_8b_ppo/r{self.rank}/"
            f"bs{self.effective_batch_size}/seed{self.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "model_family": self.model_family,
            "algorithm": self.algorithm,
            "domain": self.domain,
            "rank": self.rank,
            "effective_batch_size": self.effective_batch_size,
            "seed": self.seed,
            "steps": self.steps,
            "mlite_overrides": [
                f"+actor_rollout_ref.actor.engine.impl_cfg.lora.rank={self.rank}",
                f"data.train_batch_size={self.effective_batch_size}",
                f"trainer.seed={self.seed}",
                f"actor_rollout_ref.actor.engine.seed={self.seed}",
                f"+trainer.total_training_steps={self.steps}",
            ],
            "result_status": self.result_status,
            "claims_paper_results": self.claims_paper_results,
        }


@dataclass(frozen=True)
class Rank1OloraTailArm:
    """One no-result Fig.15 rank-1 OLoRA-tail stability planning arm."""

    init_lora_weights: Rank1Init
    effective_batch_size: int
    seed: int
    rank: int = 1
    alpha: float = 2.0
    steps: int = DEFAULT_STEPS
    model_family: str = "Qwen3-8B"
    algorithm: str = "PPO"
    domain: str = "math"
    result_status: str = "pending_gpu_training"
    claims_paper_results: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "init_lora_weights", _normalize_rank1_init(self.init_lora_weights))
        object.__setattr__(self, "effective_batch_size", _positive_int(self.effective_batch_size, name="effective_batch_size"))
        object.__setattr__(self, "seed", _non_negative_int(self.seed, name="seed"))
        object.__setattr__(self, "rank", _positive_int(self.rank, name="rank"))
        if self.rank != 1:
            raise ValueError("Rank1OloraTailArm rank must be 1.")
        object.__setattr__(self, "alpha", _finite_positive_number(self.alpha, name="alpha"))
        object.__setattr__(self, "steps", _positive_int(self.steps, name="steps"))
        object.__setattr__(self, "model_family", _non_empty_string(self.model_family, name="model_family"))
        object.__setattr__(self, "algorithm", _non_empty_string(self.algorithm, name="algorithm"))
        object.__setattr__(self, "domain", _non_empty_string(self.domain, name="domain"))
        object.__setattr__(self, "result_status", _non_empty_string(self.result_status, name="result_status"))
        if not isinstance(self.claims_paper_results, bool):
            raise TypeError("claims_paper_results must be a boolean.")
        if self.claims_paper_results:
            raise ValueError("Rank1OloraTailArm must not claim paper metrics.")

    @property
    def arm_id(self) -> str:
        return (
            f"rank1_olora_tail/{self.init_lora_weights}/"
            f"bs{self.effective_batch_size}/seed{self.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "paper_anchor": "Fig.15 rank-1 OLoRA-tail stability comparison",
            "model_family": self.model_family,
            "algorithm": self.algorithm,
            "domain": self.domain,
            "rank": self.rank,
            "alpha": self.alpha,
            "init_lora_weights": self.init_lora_weights,
            "effective_batch_size": self.effective_batch_size,
            "seed": self.seed,
            "steps": self.steps,
            "mlite_overrides": [
                *mlite_lora_overrides(
                    rank=self.rank,
                    alpha=self.alpha,
                    use_rslora=False,
                    lora_init=self.init_lora_weights,
                ),
                f"data.train_batch_size={self.effective_batch_size}",
                f"trainer.seed={self.seed}",
                f"actor_rollout_ref.actor.engine.seed={self.seed}",
                f"+trainer.total_training_steps={self.steps}",
            ],
            "result_status": self.result_status,
            "claims_paper_results": self.claims_paper_results,
        }


@dataclass(frozen=True)
class RankSweepContract:
    """No-launch, no-result local contract for the paper's rank axes."""

    ranks: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    rslora_lr_transfer_arms: tuple[RsLoraTransferArm, ...]
    rank_regime_arms: tuple[RankRegimeArm, ...]
    rank1_olora_tail_arms: tuple[Rank1OloraTailArm, ...] = field(default_factory=tuple)
    paper_exact: bool = False
    claims_paper_results: bool = False
    launches_training: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", _sorted_unique_positive_ints(self.ranks, name="ranks"))
        object.__setattr__(
            self,
            "batch_sizes",
            _sorted_unique_positive_ints(self.batch_sizes, name="batch_sizes"),
        )
        object.__setattr__(self, "seeds", _sorted_unique_non_negative_ints(self.seeds, name="seeds"))
        if not isinstance(self.rslora_lr_transfer_arms, tuple) or not all(
            isinstance(arm, RsLoraTransferArm) for arm in self.rslora_lr_transfer_arms
        ):
            raise TypeError("rslora_lr_transfer_arms must be a tuple of RsLoraTransferArm.")
        if not isinstance(self.rank_regime_arms, tuple) or not all(
            isinstance(arm, RankRegimeArm) for arm in self.rank_regime_arms
        ):
            raise TypeError("rank_regime_arms must be a tuple of RankRegimeArm.")
        if not isinstance(self.rank1_olora_tail_arms, tuple) or not all(
            isinstance(arm, Rank1OloraTailArm) for arm in self.rank1_olora_tail_arms
        ):
            raise TypeError("rank1_olora_tail_arms must be a tuple of Rank1OloraTailArm.")
        if not isinstance(self.paper_exact, bool):
            raise TypeError("paper_exact must be a boolean.")
        if not isinstance(self.claims_paper_results, bool):
            raise TypeError("claims_paper_results must be a boolean.")
        if not isinstance(self.launches_training, bool):
            raise TypeError("launches_training must be a boolean.")
        if self.paper_exact or self.claims_paper_results or self.launches_training:
            raise ValueError("RankSweepContract is a local contract and must stay no-launch/no-claim.")

    def invariants(self) -> dict[str, bool | int]:
        return rank_sweep_invariants(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass_rank_sweep_contract"
            if all(_invariant_value_passes(value) for value in self.invariants().values())
            and self.invariants()["rank_regime_matches_paper_216_shape"] is True
            else "fail_rank_sweep_contract",
            "paper_exact": self.paper_exact,
            "claims_paper_results": self.claims_paper_results,
            "launches_training": self.launches_training,
            "ranks": list(self.ranks),
            "batch_sizes": list(self.batch_sizes),
            "seeds": list(self.seeds),
            "rslora_lr_transfer_arms": [arm.to_dict() for arm in self.rslora_lr_transfer_arms],
            "rank_regime_arms": [arm.to_dict() for arm in self.rank_regime_arms],
            "rank1_olora_tail_arms": [arm.to_dict() for arm in self.rank1_olora_tail_arms],
            "invariants": self.invariants(),
            "missing_for_paper_exact": [
                "exact LR grid and per-rank metric table",
                "216 completed Qwen3-8B PPO optimizer-step/eval artifacts",
                "48 completed rank-1 OLoRA-tail stability artifacts",
                "best-band analysis proving cross-rank LR transfer",
            ],
        }


def build_rslora_lr_transfer_arms(
    *,
    ranks: Sequence[int] = DEFAULT_RANKS,
    reference_rank: int = DEFAULT_REFERENCE_RANK,
    reference_alpha: float = DEFAULT_REFERENCE_ALPHA,
) -> tuple[RsLoraTransferArm, ...]:
    """Build the three-rule rsLoRA transfer matrix without running it."""

    ranks = _sorted_unique_positive_ints(ranks, name="ranks")
    arms: list[RsLoraTransferArm] = []
    for rule in ALPHA_SCALING_RULES:
        for rank in ranks:
            alpha, use_rslora, convention = alpha_for_scaling_rule(
                rule,
                rank,
                reference_rank=reference_rank,
                reference_alpha=reference_alpha,
            )
            arms.append(
                RsLoraTransferArm(
                    alpha_rule=rule,
                    rank=rank,
                    alpha=alpha,
                    use_rslora=use_rslora,
                    scaling_convention=convention,
                    runtime_scale=lora_runtime_scale(rank, alpha, use_rslora=use_rslora),
                    eq2_alpha_squared_over_rank=eq2_alpha_squared_over_rank(rank, alpha),
                )
            )
    return tuple(arms)


def build_rank_regime_arms(
    *,
    ranks: Sequence[int] = DEFAULT_RANKS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    steps: int = DEFAULT_STEPS,
) -> tuple[RankRegimeArm, ...]:
    """Build the paper-shaped 9 x 4 x 6 PPO rank-regime matrix."""

    ranks = _sorted_unique_positive_ints(ranks, name="ranks")
    batch_sizes = _sorted_unique_positive_ints(batch_sizes, name="batch_sizes")
    seeds = _sorted_unique_non_negative_ints(seeds, name="seeds")
    steps = _positive_int(steps, name="steps")
    return tuple(
        RankRegimeArm(rank=rank, effective_batch_size=batch_size, seed=seed, steps=steps)
        for rank in ranks
        for batch_size in batch_sizes
        for seed in seeds
    )


def build_rank1_olora_tail_arms(
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    steps: int = DEFAULT_STEPS,
    alpha: float = 2.0,
) -> tuple[Rank1OloraTailArm, ...]:
    """Build the paper-shaped Fig.15 rank-1 standard-vs-OLoRA-tail grid."""

    batch_sizes = _sorted_unique_positive_ints(batch_sizes, name="batch_sizes")
    seeds = _sorted_unique_non_negative_ints(seeds, name="seeds")
    steps = _positive_int(steps, name="steps")
    alpha = _finite_positive_number(alpha, name="alpha")
    return tuple(
        Rank1OloraTailArm(
            init_lora_weights=init_lora_weights,
            effective_batch_size=batch_size,
            seed=seed,
            alpha=alpha,
            steps=steps,
        )
        for init_lora_weights in RANK1_INIT_VALUES
        for batch_size in batch_sizes
        for seed in seeds
    )


def build_rank_sweep_contract(
    *,
    ranks: Sequence[int] = DEFAULT_RANKS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    reference_rank: int = DEFAULT_REFERENCE_RANK,
    reference_alpha: float = DEFAULT_REFERENCE_ALPHA,
) -> RankSweepContract:
    """Return a local no-result contract for rank regimes and rsLoRA transfer."""

    ranks = _sorted_unique_positive_ints(ranks, name="ranks")
    batch_sizes = _sorted_unique_positive_ints(batch_sizes, name="batch_sizes")
    seeds = _sorted_unique_non_negative_ints(seeds, name="seeds")
    if _positive_int(reference_rank, name="reference_rank") not in ranks:
        raise ValueError("reference_rank must be included in ranks.")
    return RankSweepContract(
        ranks=ranks,
        batch_sizes=batch_sizes,
        seeds=seeds,
        rslora_lr_transfer_arms=build_rslora_lr_transfer_arms(
            ranks=ranks,
            reference_rank=reference_rank,
            reference_alpha=reference_alpha,
        ),
        rank_regime_arms=build_rank_regime_arms(
            ranks=ranks,
            batch_sizes=batch_sizes,
            seeds=seeds,
        ),
        rank1_olora_tail_arms=build_rank1_olora_tail_arms(
            batch_sizes=batch_sizes,
            seeds=seeds,
        ),
    )


def rank_sweep_invariants(contract: RankSweepContract) -> dict[str, bool | int]:
    """Return local invariants; these do not prove paper metrics."""

    if not isinstance(contract, RankSweepContract):
        raise TypeError("contract must be a RankSweepContract.")
    sqrt_arms = [
        arm for arm in contract.rslora_lr_transfer_arms if arm.alpha_rule == "sqrt_rank_rslora"
    ]
    fixed_arms = [
        arm for arm in contract.rslora_lr_transfer_arms if arm.alpha_rule == "fixed_alpha_over_rank"
    ]
    return {
        "sqrt_rank_rslora_eq2_factor_constant": len(
            {round(arm.eq2_alpha_squared_over_rank, 12) for arm in sqrt_arms}
        )
        == 1,
        "sqrt_rank_rslora_runtime_scale_constant": len(
            {round(arm.runtime_scale, 12) for arm in sqrt_arms}
        )
        == 1,
        "sqrt_rank_arms_use_rslora": all(arm.use_rslora for arm in sqrt_arms),
        "fixed_alpha_over_rank_runtime_scale_is_one": all(
            math.isclose(arm.runtime_scale, 1.0) for arm in fixed_arms
        ),
        "rank_regime_run_count": len(contract.rank_regime_arms),
        "rank_regime_expected_run_count": len(contract.ranks)
        * len(contract.batch_sizes)
        * len(contract.seeds),
        "rank_regime_matches_paper_216_shape": len(contract.rank_regime_arms)
        == PAPER_RANK_REGIME_RUN_COUNT,
        "rank1_olora_tail_run_count": len(contract.rank1_olora_tail_arms),
        "rank1_olora_tail_expected_run_count": len(RANK1_INIT_VALUES)
        * len(contract.batch_sizes)
        * len(contract.seeds),
        "rank1_olora_tail_matches_paper_48_shape": len(contract.rank1_olora_tail_arms)
        == PAPER_RANK1_OLORA_TAIL_RUN_COUNT,
        "all_rank_regime_arms_pending_results": all(
            arm.result_status == "pending_gpu_training" for arm in contract.rank_regime_arms
        ),
        "all_rank1_olora_tail_arms_pending_results": all(
            arm.result_status == "pending_gpu_training"
            for arm in contract.rank1_olora_tail_arms
        ),
        "all_rslora_arms_pending_lr_grid": all(
            arm.result_status == "pending_gpu_training_and_lr_grid"
            for arm in contract.rslora_lr_transfer_arms
        ),
        "paper_results_not_claimed": not contract.paper_exact
        and not contract.claims_paper_results
        and not contract.launches_training,
        "standard_library_only": True,
    }


def validate_rank_sweep_contract(contract: RankSweepContract) -> None:
    invariants = rank_sweep_invariants(contract)
    if invariants["rank_regime_matches_paper_216_shape"] is not True:
        raise ValueError("rank sweep contract does not match the paper 216-arm shape.")
    failed = [name for name, value in invariants.items() if not _invariant_value_passes(value)]
    if failed:
        raise ValueError(f"rank sweep contract failed invariants: {', '.join(failed)}")


def _rank_regime_result_metric(result: Mapping[str, Any], metric_name: str) -> float:
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping) and metric_name in metrics:
        return _finite_number(metrics[metric_name], name=f"metrics.{metric_name}")
    if metric_name in result:
        return _finite_number(result[metric_name], name=metric_name)
    raise ValueError(f"rank-regime result is missing metric {metric_name!r}.")


def _rank_regime_result_key(result: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        _positive_int(result.get("rank"), name="rank"),
        _positive_int(result.get("effective_batch_size"), name="effective_batch_size"),
        _non_negative_int(result.get("seed"), name="seed"),
    )


def _rank_regime_result_failures(
    result: Mapping[str, Any],
    *,
    expected: RankRegimeArm,
    metric_name: str,
) -> list[str]:
    failures: list[str] = []
    if result.get("status") != "pass":
        failures.append(f"{expected.arm_id}.status must be pass")
    if result.get("evidence_type") not in {"paper_scale_rl_run", "rank_regime_ppo_run"}:
        failures.append(f"{expected.arm_id}.evidence_type must be paper_scale_rl_run or rank_regime_ppo_run")
    for key in ("explicit_user_confirmation", "gpu_or_training_launched", "optimizer_step_evidence"):
        try:
            _required_true(result.get(key), name=key)
        except ValueError as exc:
            failures.append(f"{expected.arm_id}.{exc}")
    steps = result.get("steps")
    if steps != expected.steps:
        failures.append(f"{expected.arm_id}.steps must be {expected.steps}")
    optimizer_steps = result.get("optimizer_steps")
    if isinstance(optimizer_steps, bool) or not isinstance(optimizer_steps, int):
        failures.append(f"{expected.arm_id}.optimizer_steps must be an integer")
    elif optimizer_steps < expected.steps:
        failures.append(f"{expected.arm_id}.optimizer_steps must be >= {expected.steps}")
    try:
        _rank_regime_result_metric(result, metric_name)
    except (TypeError, ValueError) as exc:
            failures.append(f"{expected.arm_id}.{exc}")
    return failures


def _rslora_result_metric(result: Mapping[str, Any], metric_name: str) -> float:
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping) and metric_name in metrics:
        return _finite_number(metrics[metric_name], name=f"metrics.{metric_name}")
    if metric_name in result:
        return _finite_number(result[metric_name], name=metric_name)
    raise ValueError(f"rsLoRA result is missing metric {metric_name!r}.")


def _rslora_result_learning_rate(result: Mapping[str, Any]) -> float:
    if "learning_rate" in result:
        return _finite_positive_number(result["learning_rate"], name="learning_rate")
    if "lr" in result:
        return _finite_positive_number(result["lr"], name="learning_rate")
    raise ValueError("rsLoRA result is missing learning_rate.")


def _rslora_result_key(result: Mapping[str, Any]) -> tuple[AlphaScalingRule, int, float, int]:
    return (
        _normalize_alpha_rule(result.get("alpha_rule")),
        _positive_int(result.get("rank"), name="rank"),
        _rslora_result_learning_rate(result),
        _non_negative_int(result.get("seed"), name="seed"),
    )


def _rslora_result_failures(
    result: Mapping[str, Any],
    *,
    expected: RsLoraTransferArm,
    learning_rate: float,
    seed: int,
    metric_name: str,
    min_optimizer_steps: int,
) -> list[str]:
    failures: list[str] = []
    prefix = f"{expected.arm_id}/lr{learning_rate:.12g}/seed{seed}"
    if result.get("status") != "pass":
        failures.append(f"{prefix}.status must be pass")
    if result.get("evidence_type") not in {"paper_scale_lr_transfer_run", "rslora_lr_transfer_run"}:
        failures.append(f"{prefix}.evidence_type must be paper_scale_lr_transfer_run or rslora_lr_transfer_run")
    for key in ("explicit_user_confirmation", "gpu_or_training_launched", "optimizer_step_evidence"):
        try:
            _required_true(result.get(key), name=key)
        except ValueError as exc:
            failures.append(f"{prefix}.{exc}")
    if result.get("paper_exact") is True or result.get("claims_paper_results") is True:
        failures.append(f"{prefix}.must not claim paper-exact results")

    optimizer_steps = result.get("optimizer_steps")
    if isinstance(optimizer_steps, bool) or not isinstance(optimizer_steps, int):
        failures.append(f"{prefix}.optimizer_steps must be an integer")
    elif optimizer_steps < min_optimizer_steps:
        failures.append(f"{prefix}.optimizer_steps must be >= {min_optimizer_steps}")

    try:
        alpha = _finite_positive_number(result.get("alpha"), name="alpha")
        if not math.isclose(alpha, expected.alpha, rel_tol=1e-9, abs_tol=1e-12):
            failures.append(f"{prefix}.alpha must match {expected.alpha:.12g}")
    except (TypeError, ValueError) as exc:
        failures.append(f"{prefix}.{exc}")

    if result.get("use_rslora") is not expected.use_rslora:
        failures.append(f"{prefix}.use_rslora must be {expected.use_rslora}")
    if result.get("scaling_convention") != expected.scaling_convention:
        failures.append(f"{prefix}.scaling_convention must be {expected.scaling_convention}")

    try:
        _rslora_result_metric(result, metric_name)
    except (TypeError, ValueError) as exc:
        failures.append(f"{prefix}.{exc}")
    return failures


def _rank1_result_metric(result: Mapping[str, Any], metric_name: str) -> float:
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping) and metric_name in metrics:
        return _finite_number(metrics[metric_name], name=f"metrics.{metric_name}")
    if metric_name in result:
        return _finite_number(result[metric_name], name=metric_name)
    raise ValueError(f"rank-1 OLoRA-tail result is missing metric {metric_name!r}.")


def _rank1_result_key(result: Mapping[str, Any]) -> tuple[Rank1Init, int, int]:
    return (
        _normalize_rank1_init(result.get("init_lora_weights", result.get("lora_init"))),
        _positive_int(result.get("effective_batch_size"), name="effective_batch_size"),
        _non_negative_int(result.get("seed"), name="seed"),
    )


def _rank1_result_failures(
    result: Mapping[str, Any],
    *,
    expected: Rank1OloraTailArm,
    metric_name: str,
    min_optimizer_steps: int,
) -> list[str]:
    failures: list[str] = []
    if result.get("status") != "pass":
        failures.append(f"{expected.arm_id}.status must be pass")
    if result.get("evidence_type") not in {
        "paper_scale_rank1_stability_run",
        "rank1_olora_tail_run",
    }:
        failures.append(
            f"{expected.arm_id}.evidence_type must be paper_scale_rank1_stability_run "
            "or rank1_olora_tail_run"
        )
    for key in ("explicit_user_confirmation", "gpu_or_training_launched", "optimizer_step_evidence"):
        try:
            _required_true(result.get(key), name=key)
        except ValueError as exc:
            failures.append(f"{expected.arm_id}.{exc}")
    if result.get("paper_exact") is True or result.get("claims_paper_results") is True:
        failures.append(f"{expected.arm_id}.must not claim paper-exact results")

    try:
        rank = _positive_int(result.get("rank"), name="rank")
        if rank != expected.rank:
            failures.append(f"{expected.arm_id}.rank must be {expected.rank}")
    except (TypeError, ValueError) as exc:
        failures.append(f"{expected.arm_id}.{exc}")
    try:
        alpha = _finite_positive_number(result.get("alpha"), name="alpha")
        if not math.isclose(alpha, expected.alpha, rel_tol=1e-9, abs_tol=1e-12):
            failures.append(f"{expected.arm_id}.alpha must match {expected.alpha:.12g}")
    except (TypeError, ValueError) as exc:
        failures.append(f"{expected.arm_id}.{exc}")

    steps = result.get("steps")
    if steps is not None and steps != expected.steps:
        failures.append(f"{expected.arm_id}.steps must be {expected.steps}")
    optimizer_steps = result.get("optimizer_steps")
    if isinstance(optimizer_steps, bool) or not isinstance(optimizer_steps, int):
        failures.append(f"{expected.arm_id}.optimizer_steps must be an integer")
    elif optimizer_steps < max(min_optimizer_steps, expected.steps):
        failures.append(
            f"{expected.arm_id}.optimizer_steps must be >= {max(min_optimizer_steps, expected.steps)}"
        )

    try:
        _rank1_result_metric(result, metric_name)
    except (TypeError, ValueError) as exc:
        failures.append(f"{expected.arm_id}.{exc}")
    return failures


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty.")
    return sum(values) / float(len(values))


def _population_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / float(len(values)))


def _nonincreasing_with_drop(values: Sequence[float]) -> bool:
    if len(values) <= 1:
        return False
    return all(values[index] >= values[index + 1] for index in range(len(values) - 1)) and values[0] > values[-1]


def _ratio(values: Sequence[float]) -> float:
    if not values:
        return math.inf
    low = min(values)
    if low <= 0.0:
        return math.inf
    return max(values) / low


def analyze_rank_regime_results(
    results: Sequence[Mapping[str, Any]],
    *,
    contract: RankSweepContract | None = None,
    metric_name: str = "math_accuracy",
    pattern_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Analyze completed rank-regime results without launching or claiming exact paper reproduction.

    The returned ``paper_pattern_supported`` is a local evidence judgment over
    supplied results. It requires exact coverage of the 216-arm paper-shaped
    matrix, real-run provenance fields, finite metrics, and the paper's
    qualitative rank-regime pattern: r16/32 best mean, r1-4 best-run ceiling
    close to deployment, wider frontier spread, and flat high-rank ceiling.
    """

    if contract is None:
        contract = build_rank_sweep_contract()
    if not isinstance(contract, RankSweepContract):
        raise TypeError("contract must be a RankSweepContract.")
    metric_name = _non_empty_string(metric_name, name="metric_name")
    pattern_tolerance = _finite_number(pattern_tolerance, name="pattern_tolerance")
    if pattern_tolerance < 0.0:
        raise ValueError("pattern_tolerance must be non-negative.")

    expected_by_key = {
        (arm.rank, arm.effective_batch_size, arm.seed): arm for arm in contract.rank_regime_arms
    }
    observed_by_key: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    failures: list[str] = []

    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError("results must be a sequence of mappings.")
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            failures.append(f"results[{index}] must be a mapping")
            continue
        try:
            key = _rank_regime_result_key(result)
        except (TypeError, ValueError) as exc:
            failures.append(f"results[{index}].{exc}")
            continue
        expected = expected_by_key.get(key)
        if expected is None:
            failures.append(
                "unexpected rank-regime result "
                f"rank={key[0]} effective_batch_size={key[1]} seed={key[2]}"
            )
            continue
        if key in observed_by_key:
            failures.append(
                "duplicate rank-regime result "
                f"rank={key[0]} effective_batch_size={key[1]} seed={key[2]}"
            )
            continue
        observed_by_key[key] = result
        failures.extend(_rank_regime_result_failures(result, expected=expected, metric_name=metric_name))

    missing_keys = sorted(set(expected_by_key) - set(observed_by_key))
    if missing_keys:
        preview = [
            {"rank": rank, "effective_batch_size": batch_size, "seed": seed}
            for rank, batch_size, seed in missing_keys[:5]
        ]
        failures.append(f"missing {len(missing_keys)} rank-regime results; first_missing={preview}")

    unexpected_count = sum(1 for failure in failures if failure.startswith("unexpected rank-regime result"))
    duplicate_count = sum(1 for failure in failures if failure.startswith("duplicate rank-regime result"))
    values_by_rank: dict[int, list[float]] = {rank: [] for rank in contract.ranks}
    if not failures:
        for key, result in observed_by_key.items():
            values_by_rank[key[0]].append(_rank_regime_result_metric(result, metric_name))

    per_rank: dict[str, dict[str, float | int]] = {}
    for rank in contract.ranks:
        values = values_by_rank.get(rank, [])
        if not values:
            continue
        per_rank[str(rank)] = {
            "count": len(values),
            "mean": _mean(values),
            "best": max(values),
            "worst": min(values),
            "std": _population_std(values),
            "spread": max(values) - min(values),
        }

    def _regime_stats(ranks: Sequence[int]) -> dict[str, float | int | list[int]]:
        values = [value for rank in ranks for value in values_by_rank.get(rank, [])]
        rank_stats = [per_rank[str(rank)] for rank in ranks if str(rank) in per_rank]
        if not values or not rank_stats:
            return {"ranks": list(ranks), "count": 0}
        return {
            "ranks": list(ranks),
            "count": len(values),
            "mean": _mean(values),
            "best": max(values),
            "worst": min(values),
            "mean_rank_std": _mean([float(item["std"]) for item in rank_stats]),
            "mean_rank_spread": _mean([float(item["spread"]) for item in rank_stats]),
        }

    frontier = _regime_stats(tuple(rank for rank in (1, 2, 4) if rank in contract.ranks))
    deployment = _regime_stats(tuple(rank for rank in (16, 32) if rank in contract.ranks))
    cost_warning = _regime_stats(tuple(rank for rank in (64, 128, 256) if rank in contract.ranks))

    pattern_checks = {
        "exact_216_run_coverage": len(observed_by_key) == PAPER_RANK_REGIME_RUN_COUNT
        and len(missing_keys) == 0
        and unexpected_count == 0
        and duplicate_count == 0,
        "all_results_have_real_run_evidence": not failures,
        "deployment_default_has_highest_mean": False,
        "frontier_best_matches_deployment_ceiling": False,
        "frontier_reliability_is_lower": False,
        "cost_warning_ceiling_is_flat": False,
    }
    if not failures and per_rank and int(deployment.get("count", 0)) > 0:
        deployment_mean = float(deployment["mean"])
        deployment_best = float(deployment["best"])
        all_rank_means = [float(item["mean"]) for item in per_rank.values()]
        pattern_checks["deployment_default_has_highest_mean"] = (
            deployment_mean >= max(all_rank_means) - pattern_tolerance
        )
        if int(frontier.get("count", 0)) > 0:
            pattern_checks["frontier_best_matches_deployment_ceiling"] = (
                float(frontier["best"]) >= deployment_best - pattern_tolerance
            )
            pattern_checks["frontier_reliability_is_lower"] = (
                float(frontier["mean"]) < deployment_mean
                and float(frontier["mean_rank_spread"]) > float(deployment["mean_rank_spread"])
            )
        if int(cost_warning.get("count", 0)) > 0:
            pattern_checks["cost_warning_ceiling_is_flat"] = (
                float(cost_warning["best"]) <= deployment_best + pattern_tolerance
            )

    paper_pattern_supported = all(pattern_checks.values())
    status = "pass_rank_regime_result_analysis" if not failures else "fail_rank_regime_result_analysis"
    return {
        "status": status,
        "paper_exact": False,
        "claims_exact_paper_results": False,
        "metric_name": metric_name,
        "pattern_tolerance": pattern_tolerance,
        "expected_run_count": PAPER_RANK_REGIME_RUN_COUNT,
        "observed_run_count": len(observed_by_key),
        "missing_run_count": len(missing_keys),
        "unexpected_run_count": unexpected_count,
        "duplicate_run_count": duplicate_count,
        "failures": failures,
        "per_rank": per_rank,
        "regimes": {
            "frontier_ranks_1_4": frontier,
            "deployment_default_ranks_16_32": deployment,
            "cost_warning_ranks_64_plus": cost_warning,
        },
        "pattern_checks": pattern_checks,
        "paper_pattern_supported": paper_pattern_supported,
        "paper_pattern_status": (
            "pass_paper_rank_regime_pattern"
            if paper_pattern_supported
            else "blocked_or_failed_paper_rank_regime_pattern"
        ),
    }


def analyze_rslora_lr_transfer_results(
    results: Sequence[Mapping[str, Any]],
    *,
    contract: RankSweepContract | None = None,
    expected_learning_rates: Sequence[float] | None = None,
    expected_seeds: Sequence[int] | None = None,
    metric_name: str = "eval_accuracy",
    reusable_band_tolerance: float = 0.0,
    metric_tolerance: float = 0.0,
    lr_same_order_ratio: float = 10.0,
    min_optimizer_steps: int = 1,
) -> dict[str, Any]:
    """Analyze completed rsLoRA LR-transfer results without launching training.

    The Scaling PEFT LR-transfer claim is not just that ``alpha/sqrt(rank)``
    exists. A result table must show that the sqrt-rank convention has a
    reusable LR band across ranks, while the fixed ``alpha/r`` convention's
    best LR shifts downward with rank. This analyzer checks that qualitative
    pattern over externally supplied, real-run result rows.
    """

    if contract is None:
        contract = build_rank_sweep_contract()
    if not isinstance(contract, RankSweepContract):
        raise TypeError("contract must be a RankSweepContract.")
    metric_name = _non_empty_string(metric_name, name="metric_name")
    reusable_band_tolerance = _finite_number(reusable_band_tolerance, name="reusable_band_tolerance")
    metric_tolerance = _finite_number(metric_tolerance, name="metric_tolerance")
    lr_same_order_ratio = _finite_positive_number(lr_same_order_ratio, name="lr_same_order_ratio")
    min_optimizer_steps = _positive_int(min_optimizer_steps, name="min_optimizer_steps")
    if reusable_band_tolerance < 0.0:
        raise ValueError("reusable_band_tolerance must be non-negative.")
    if metric_tolerance < 0.0:
        raise ValueError("metric_tolerance must be non-negative.")
    if lr_same_order_ratio < 1.0:
        raise ValueError("lr_same_order_ratio must be >= 1.")

    expected_arm_by_rule_rank = {
        (arm.alpha_rule, arm.rank): arm for arm in contract.rslora_lr_transfer_arms
    }
    observed_by_key: dict[tuple[AlphaScalingRule, int, float, int], Mapping[str, Any]] = {}
    observed_learning_rates: set[float] = set()
    observed_seeds: set[int] = set()
    failures: list[str] = []

    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError("results must be a sequence of mappings.")
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            failures.append(f"results[{index}] must be a mapping")
            continue
        try:
            key = _rslora_result_key(result)
        except (TypeError, ValueError) as exc:
            failures.append(f"results[{index}].{exc}")
            continue
        alpha_rule, rank, learning_rate, seed = key
        observed_learning_rates.add(learning_rate)
        observed_seeds.add(seed)
        expected = expected_arm_by_rule_rank.get((alpha_rule, rank))
        if expected is None:
            failures.append(f"unexpected rsLoRA result alpha_rule={alpha_rule} rank={rank}")
            continue
        if key in observed_by_key:
            failures.append(
                "duplicate rsLoRA result "
                f"alpha_rule={alpha_rule} rank={rank} learning_rate={learning_rate:.12g} seed={seed}"
            )
            continue
        observed_by_key[key] = result
        failures.extend(
            _rslora_result_failures(
                result,
                expected=expected,
                learning_rate=learning_rate,
                seed=seed,
                metric_name=metric_name,
                min_optimizer_steps=min_optimizer_steps,
            )
        )

    if expected_learning_rates is None:
        if observed_learning_rates:
            learning_rates = tuple(sorted(observed_learning_rates))
        else:
            learning_rates = ()
            failures.append("no rsLoRA learning-rate results were supplied")
    else:
        learning_rates = _sorted_unique_positive_numbers(expected_learning_rates, name="expected_learning_rates")
    if expected_seeds is None:
        if observed_seeds:
            seeds = tuple(sorted(observed_seeds))
        else:
            seeds = ()
            failures.append("no rsLoRA seeds were supplied")
    else:
        seeds = _sorted_unique_non_negative_ints(expected_seeds, name="expected_seeds")

    expected_keys = {
        (arm.alpha_rule, arm.rank, learning_rate, seed)
        for arm in contract.rslora_lr_transfer_arms
        for learning_rate in learning_rates
        for seed in seeds
    }
    missing_keys = sorted(expected_keys - set(observed_by_key))
    unexpected_keys = sorted(set(observed_by_key) - expected_keys)
    if missing_keys:
        preview = [
            {
                "alpha_rule": rule,
                "rank": rank,
                "learning_rate": learning_rate,
                "seed": seed,
            }
            for rule, rank, learning_rate, seed in missing_keys[:5]
        ]
        failures.append(f"missing {len(missing_keys)} rsLoRA LR-transfer results; first_missing={preview}")
    if unexpected_keys:
        preview = [
            {
                "alpha_rule": rule,
                "rank": rank,
                "learning_rate": learning_rate,
                "seed": seed,
            }
            for rule, rank, learning_rate, seed in unexpected_keys[:5]
        ]
        failures.append(f"unexpected {len(unexpected_keys)} rsLoRA LR-transfer results; first_unexpected={preview}")

    duplicate_count = sum(1 for failure in failures if failure.startswith("duplicate rsLoRA result"))
    metric_values_by_rule_rank_lr: dict[tuple[AlphaScalingRule, int, float], list[float]] = {}
    if not failures:
        for (alpha_rule, rank, learning_rate, _seed), result in observed_by_key.items():
            metric_values_by_rule_rank_lr.setdefault((alpha_rule, rank, learning_rate), []).append(
                _rslora_result_metric(result, metric_name)
            )

    rule_summaries: dict[str, dict[str, Any]] = {}
    for rule in ALPHA_SCALING_RULES:
        per_rank: dict[str, dict[str, Any]] = {}
        best_lrs: list[float] = []
        best_scores: list[float] = []
        for rank in contract.ranks:
            lr_scores = {
                learning_rate: _mean(metric_values_by_rule_rank_lr[(rule, rank, learning_rate)])
                for learning_rate in learning_rates
                if (rule, rank, learning_rate) in metric_values_by_rule_rank_lr
            }
            if not lr_scores:
                continue
            best_lr = max(lr_scores, key=lambda lr: (lr_scores[lr], -lr))
            best_score = lr_scores[best_lr]
            best_lrs.append(best_lr)
            best_scores.append(best_score)
            per_rank[str(rank)] = {
                "best_lr": best_lr,
                "best_score": best_score,
                "lr_scores": {f"{learning_rate:.12g}": score for learning_rate, score in lr_scores.items()},
            }

        shared_good_lrs: list[float] = []
        if len(per_rank) == len(contract.ranks):
            for learning_rate in learning_rates:
                good_for_all_ranks = True
                for rank in contract.ranks:
                    rank_summary = per_rank[str(rank)]
                    score = float(rank_summary["lr_scores"].get(f"{learning_rate:.12g}", -math.inf))
                    if score < float(rank_summary["best_score"]) - reusable_band_tolerance:
                        good_for_all_ranks = False
                        break
                if good_for_all_ranks:
                    shared_good_lrs.append(learning_rate)

        rule_summaries[rule] = {
            "rank_count": len(per_rank),
            "per_rank": per_rank,
            "best_lrs_by_rank": best_lrs,
            "best_scores_by_rank": best_scores,
            "mean_best_score": _mean(best_scores) if best_scores else None,
            "best_lr_spread_ratio": _ratio(best_lrs),
            "reusable_lr_band": [f"{learning_rate:.12g}" for learning_rate in shared_good_lrs],
            "reusable_lr_band_count": len(shared_good_lrs),
        }

    pattern_checks = {
        "exact_rule_rank_lr_seed_coverage": len(observed_by_key) == len(expected_keys)
        and len(missing_keys) == 0
        and len(unexpected_keys) == 0
        and duplicate_count == 0,
        "all_results_have_real_run_evidence": not failures,
        "sqrt_rank_has_largest_reusable_lr_band": False,
        "sqrt_rank_best_lr_same_order_across_ranks": False,
        "sqrt_rank_best_score_competitive": False,
        "fixed_alpha_over_rank_best_lr_moves_down_with_rank": False,
        "const_alpha_best_point_flat": False,
    }
    if not failures:
        sqrt_summary = rule_summaries["sqrt_rank_rslora"]
        fixed_summary = rule_summaries["fixed_alpha_over_rank"]
        const_summary = rule_summaries["const_alpha"]
        other_band_counts = [
            int(const_summary["reusable_lr_band_count"]),
            int(fixed_summary["reusable_lr_band_count"]),
        ]
        mean_best_scores = [
            float(summary["mean_best_score"])
            for summary in rule_summaries.values()
            if summary["mean_best_score"] is not None
        ]
        sqrt_mean_best = sqrt_summary["mean_best_score"]
        pattern_checks["sqrt_rank_has_largest_reusable_lr_band"] = (
            int(sqrt_summary["reusable_lr_band_count"]) > 0
            and int(sqrt_summary["reusable_lr_band_count"]) >= max(other_band_counts)
        )
        pattern_checks["sqrt_rank_best_lr_same_order_across_ranks"] = (
            float(sqrt_summary["best_lr_spread_ratio"]) <= lr_same_order_ratio
        )
        pattern_checks["sqrt_rank_best_score_competitive"] = (
            sqrt_mean_best is not None
            and float(sqrt_mean_best) >= max(mean_best_scores) - metric_tolerance
        )
        pattern_checks["fixed_alpha_over_rank_best_lr_moves_down_with_rank"] = _nonincreasing_with_drop(
            [float(value) for value in fixed_summary["best_lrs_by_rank"]]
        )
        pattern_checks["const_alpha_best_point_flat"] = (
            float(const_summary["best_lr_spread_ratio"]) <= lr_same_order_ratio
        )

    rslora_transfer_pattern_supported = all(pattern_checks.values())
    status = "pass_rslora_lr_transfer_result_analysis" if not failures else "fail_rslora_lr_transfer_result_analysis"
    return {
        "status": status,
        "paper_exact": False,
        "claims_exact_paper_results": False,
        "metric_name": metric_name,
        "learning_rates": [f"{learning_rate:.12g}" for learning_rate in learning_rates],
        "seeds": list(seeds),
        "reusable_band_tolerance": reusable_band_tolerance,
        "metric_tolerance": metric_tolerance,
        "lr_same_order_ratio": lr_same_order_ratio,
        "expected_result_count": len(expected_keys),
        "observed_result_count": len(observed_by_key),
        "missing_result_count": len(missing_keys),
        "unexpected_result_count": len(unexpected_keys),
        "duplicate_result_count": duplicate_count,
        "failures": failures,
        "rule_summaries": rule_summaries,
        "pattern_checks": pattern_checks,
        "rslora_transfer_pattern_supported": rslora_transfer_pattern_supported,
        "paper_pattern_supported": rslora_transfer_pattern_supported,
        "paper_pattern_status": (
            "pass_paper_rslora_lr_transfer_pattern"
            if rslora_transfer_pattern_supported
            else "blocked_or_failed_paper_rslora_lr_transfer_pattern"
        ),
    }


def analyze_rank1_olora_tail_results(
    results: Sequence[Mapping[str, Any]],
    *,
    contract: RankSweepContract | None = None,
    metric_name: str = "gain_percent",
    min_olora_tail_mean_gain: float = 15.0,
    min_olora_tail_worst_gain: float = 0.0,
    max_olora_tail_batch_spread: float = 10.0,
    standard_large_batch_max_gain: float = 0.0,
    min_standard_degradation_drop: float = 20.0,
    superiority_margin: float = 0.0,
    min_optimizer_steps: int = DEFAULT_STEPS,
) -> dict[str, Any]:
    """Analyze Fig.15 rank-1 OLoRA-tail stability results without launching training.

    The paper's rank-1 claim is a stability comparison, not just an init flag:
    OLoRA-tail should keep positive, near-constant gains across batch sizes,
    while standard LoRA degrades as batch size increases. This checker consumes
    externally supplied result rows and reports whether that qualitative pattern
    is supported.
    """

    if contract is None:
        contract = build_rank_sweep_contract()
    if not isinstance(contract, RankSweepContract):
        raise TypeError("contract must be a RankSweepContract.")
    metric_name = _non_empty_string(metric_name, name="metric_name")
    min_olora_tail_mean_gain = _finite_number(
        min_olora_tail_mean_gain,
        name="min_olora_tail_mean_gain",
    )
    min_olora_tail_worst_gain = _finite_number(
        min_olora_tail_worst_gain,
        name="min_olora_tail_worst_gain",
    )
    max_olora_tail_batch_spread = _finite_positive_number(
        max_olora_tail_batch_spread,
        name="max_olora_tail_batch_spread",
    )
    standard_large_batch_max_gain = _finite_number(
        standard_large_batch_max_gain,
        name="standard_large_batch_max_gain",
    )
    min_standard_degradation_drop = _finite_positive_number(
        min_standard_degradation_drop,
        name="min_standard_degradation_drop",
    )
    superiority_margin = _finite_number(superiority_margin, name="superiority_margin")
    if superiority_margin < 0.0:
        raise ValueError("superiority_margin must be non-negative.")
    min_optimizer_steps = _positive_int(min_optimizer_steps, name="min_optimizer_steps")

    expected_by_key = {
        (arm.init_lora_weights, arm.effective_batch_size, arm.seed): arm
        for arm in contract.rank1_olora_tail_arms
    }
    observed_by_key: dict[tuple[Rank1Init, int, int], Mapping[str, Any]] = {}
    failures: list[str] = []

    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError("results must be a sequence of mappings.")
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            failures.append(f"results[{index}] must be a mapping")
            continue
        try:
            key = _rank1_result_key(result)
        except (TypeError, ValueError) as exc:
            failures.append(f"results[{index}].{exc}")
            continue
        expected = expected_by_key.get(key)
        if expected is None:
            failures.append(
                "unexpected rank-1 OLoRA-tail result "
                f"init_lora_weights={key[0]} effective_batch_size={key[1]} seed={key[2]}"
            )
            continue
        if key in observed_by_key:
            failures.append(
                "duplicate rank-1 OLoRA-tail result "
                f"init_lora_weights={key[0]} effective_batch_size={key[1]} seed={key[2]}"
            )
            continue
        observed_by_key[key] = result
        failures.extend(
            _rank1_result_failures(
                result,
                expected=expected,
                metric_name=metric_name,
                min_optimizer_steps=min_optimizer_steps,
            )
        )

    missing_keys = sorted(set(expected_by_key) - set(observed_by_key))
    if missing_keys:
        preview = [
            {
                "init_lora_weights": init_lora_weights,
                "effective_batch_size": batch_size,
                "seed": seed,
            }
            for init_lora_weights, batch_size, seed in missing_keys[:5]
        ]
        failures.append(f"missing {len(missing_keys)} rank-1 OLoRA-tail results; first_missing={preview}")

    unexpected_count = sum(
        1 for failure in failures if failure.startswith("unexpected rank-1 OLoRA-tail result")
    )
    duplicate_count = sum(
        1 for failure in failures if failure.startswith("duplicate rank-1 OLoRA-tail result")
    )

    values_by_init_batch: dict[tuple[Rank1Init, int], list[float]] = {}
    if not failures:
        for (init_lora_weights, batch_size, _seed), result in observed_by_key.items():
            values_by_init_batch.setdefault((init_lora_weights, batch_size), []).append(
                _rank1_result_metric(result, metric_name)
            )

    batch_sizes = contract.batch_sizes
    per_init: dict[str, dict[str, Any]] = {}
    for init_lora_weights in RANK1_INIT_VALUES:
        per_batch: dict[str, dict[str, float | int]] = {}
        batch_means: list[float] = []
        batch_worsts: list[float] = []
        for batch_size in batch_sizes:
            values = values_by_init_batch.get((init_lora_weights, batch_size), [])
            if not values:
                continue
            mean = _mean(values)
            batch_means.append(mean)
            batch_worsts.append(min(values))
            per_batch[str(batch_size)] = {
                "count": len(values),
                "mean": mean,
                "best": max(values),
                "worst": min(values),
                "std": _population_std(values),
                "spread": max(values) - min(values),
            }
        per_init[init_lora_weights] = {
            "batch_count": len(per_batch),
            "per_batch": per_batch,
            "batch_means": batch_means,
            "batch_worsts": batch_worsts,
            "mean_gain": _mean(batch_means) if batch_means else None,
            "batch_mean_spread": max(batch_means) - min(batch_means) if batch_means else None,
        }

    pattern_checks = {
        "exact_rank1_batch_seed_coverage": len(observed_by_key)
        == PAPER_RANK1_OLORA_TAIL_RUN_COUNT
        and len(missing_keys) == 0
        and unexpected_count == 0
        and duplicate_count == 0,
        "all_results_have_real_run_evidence": not failures,
        "olora_tail_positive_across_batches": False,
        "olora_tail_mean_meets_reference_floor": False,
        "olora_tail_batch_stability": False,
        "standard_lora_degrades_with_batch": False,
        "standard_lora_large_batch_below_floor": False,
        "olora_tail_beats_standard_every_batch": False,
    }
    if not failures:
        olora = per_init["olora_tail"]
        standard = per_init["standard"]
        olora_means = [float(value) for value in olora["batch_means"]]
        olora_worsts = [float(value) for value in olora["batch_worsts"]]
        standard_means = [float(value) for value in standard["batch_means"]]
        pattern_checks["olora_tail_positive_across_batches"] = (
            len(olora_worsts) == len(batch_sizes)
            and min(olora_worsts) >= min_olora_tail_worst_gain
        )
        pattern_checks["olora_tail_mean_meets_reference_floor"] = (
            len(olora_means) == len(batch_sizes)
            and min(olora_means) >= min_olora_tail_mean_gain
        )
        pattern_checks["olora_tail_batch_stability"] = (
            len(olora_means) == len(batch_sizes)
            and max(olora_means) - min(olora_means) <= max_olora_tail_batch_spread
        )
        pattern_checks["standard_lora_degrades_with_batch"] = (
            len(standard_means) == len(batch_sizes)
            and _nonincreasing_with_drop(standard_means)
            and standard_means[0] - standard_means[-1] >= min_standard_degradation_drop
        )
        pattern_checks["standard_lora_large_batch_below_floor"] = (
            len(standard_means) == len(batch_sizes)
            and standard_means[-1] <= standard_large_batch_max_gain
        )
        pattern_checks["olora_tail_beats_standard_every_batch"] = (
            len(olora_means) == len(batch_sizes)
            and len(standard_means) == len(batch_sizes)
            and all(
                olora_mean >= standard_mean + superiority_margin
                for olora_mean, standard_mean in zip(olora_means, standard_means, strict=True)
            )
        )

    rank1_stability_pattern_supported = all(pattern_checks.values())
    status = "pass_rank1_olora_tail_result_analysis" if not failures else "fail_rank1_olora_tail_result_analysis"
    return {
        "status": status,
        "paper_exact": False,
        "claims_exact_paper_results": False,
        "metric_name": metric_name,
        "expected_run_count": PAPER_RANK1_OLORA_TAIL_RUN_COUNT,
        "observed_run_count": len(observed_by_key),
        "missing_run_count": len(missing_keys),
        "unexpected_run_count": unexpected_count,
        "duplicate_run_count": duplicate_count,
        "thresholds": {
            "min_olora_tail_mean_gain": min_olora_tail_mean_gain,
            "min_olora_tail_worst_gain": min_olora_tail_worst_gain,
            "max_olora_tail_batch_spread": max_olora_tail_batch_spread,
            "standard_large_batch_max_gain": standard_large_batch_max_gain,
            "min_standard_degradation_drop": min_standard_degradation_drop,
            "superiority_margin": superiority_margin,
        },
        "failures": failures,
        "per_init": per_init,
        "pattern_checks": pattern_checks,
        "rank1_stability_pattern_supported": rank1_stability_pattern_supported,
        "paper_pattern_supported": rank1_stability_pattern_supported,
        "paper_pattern_status": (
            "pass_paper_rank1_olora_tail_stability_pattern"
            if rank1_stability_pattern_supported
            else "blocked_or_failed_paper_rank1_olora_tail_stability_pattern"
        ),
    }


def paper_reference_rank_regime_contract() -> dict[str, Any]:
    """Return reference-only paper anchors, not local results."""

    return {
        "rank_regimes": "Qwen3-8B PPO sweep, 9 ranks x 4 batch sizes x 6 seeds, 500 steps",
        "rslora": "alpha proportional to sqrt(rank) keeps Eq.2 alpha squared over rank constant",
        "rank1_olora_tail": "Fig.15 rank-1 standard-vs-OLoRA-tail grid, 2 inits x 4 batch sizes x 6 seeds",
        "paper_reference_not_local_result": True,
        "claims_paper_results": False,
    }


__all__ = [
    "ALPHA_SCALING_RULES",
    "AlphaScalingRule",
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_RANKS",
    "DEFAULT_REFERENCE_ALPHA",
    "DEFAULT_REFERENCE_RANK",
    "DEFAULT_SEEDS",
    "DEFAULT_STEPS",
    "PAPER_RANK1_OLORA_TAIL_RUN_COUNT",
    "PAPER_RANK_REGIME_RUN_COUNT",
    "RANK1_INIT_VALUES",
    "Rank1Init",
    "Rank1OloraTailArm",
    "RankRegimeArm",
    "RankSweepContract",
    "RsLoraTransferArm",
    "RuntimeScalingConvention",
    "alpha_for_scaling_rule",
    "analyze_rank1_olora_tail_results",
    "analyze_rank_regime_results",
    "analyze_rslora_lr_transfer_results",
    "build_rank1_olora_tail_arms",
    "build_rank_regime_arms",
    "build_rank_sweep_contract",
    "build_rslora_lr_transfer_arms",
    "eq2_alpha_squared_over_rank",
    "lora_runtime_scale",
    "mlite_lora_overrides",
    "paper_reference_rank_regime_contract",
    "rank_sweep_invariants",
    "validate_rank_sweep_contract",
]
