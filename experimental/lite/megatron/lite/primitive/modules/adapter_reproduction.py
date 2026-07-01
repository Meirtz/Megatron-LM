# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paper-reproduction result contracts for Megatron Lite adapters.

These helpers encode strict local analyzers for externally supplied paper-scale
result tables. They do not launch training, load weights, evaluate models, or
claim paper reproduction without matching evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

FIG14_METRICS: tuple[str, ...] = ("GSM8K", "MATH500", "AIME22", "AIME23", "AIME24", "AIME25")
FIG14_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
FIG14_PAPER_AVERAGES = {"lora_baseline": 56.3, "olora_tail": 58.3}
FIG14_PAPER_AVERAGE_ATOL = 0.5
FIG14_EXPECTED_ABSOLUTE_DELTA = 2.0
FIG14_MODEL_FAMILY = "DeepSeek-R1-Distill-Qwen-1.5B"
FIG14_DATASET = "DAPO-Math-17k"
FIG14_ALGORITHM = "DAPO"

Fig14ArmName = Literal["lora_baseline", "olora_tail"]


def _type_name(value: Any) -> str:
    return type(value).__name__


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {_type_name(value)}.")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {_type_name(value)}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {_type_name(value)}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return int(value)


def _finite_positive_number(value: Any, *, name: str) -> float:
    value = _finite_number(value, name=name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _required_true(value: Any, *, name: str) -> None:
    if value is not True:
        raise ValueError(f"{name} must be true.")


def _normalize_fig14_arm_name(value: Any) -> Fig14ArmName:
    value = _non_empty_string(value, name="arm_name")
    if value not in {"lora_baseline", "olora_tail"}:
        raise ValueError("arm_name must be lora_baseline or olora_tail.")
    return value  # type: ignore[return-value]


def _normalize_init(value: Any) -> str:
    value = _non_empty_string(value, name="init_lora_weights")
    if value in {"standard", "lora", "standard_lora"}:
        return "standard"
    if value in {"olora_tail", "olora-tail"}:
        return "olora_tail"
    raise ValueError("init_lora_weights must be standard or olora_tail.")


def _metric_dict(arm: Mapping[str, Any], metric_names: Sequence[str]) -> dict[str, float]:
    metrics = arm.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be an object.")
    values: dict[str, float] = {}
    for metric_name in metric_names:
        values[metric_name] = _finite_number(metrics.get(metric_name), name=f"metrics.{metric_name}")
    return values


@dataclass(frozen=True)
class Fig14ArmContract:
    """One no-launch Fig.14 OLoRA-tail-vs-LoRA result arm."""

    name: Fig14ArmName
    init_lora_weights: str
    olora_tail_applied: bool
    rank: int = 16
    alpha: float = 32.0
    steps: int = 500
    effective_batch_size: int = 32
    learning_rate: float = 1e-5
    model_family: str = FIG14_MODEL_FAMILY
    dataset: str = FIG14_DATASET
    algorithm: str = FIG14_ALGORITHM
    claims_paper_results: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_fig14_arm_name(self.name))
        object.__setattr__(self, "init_lora_weights", _normalize_init(self.init_lora_weights))
        if not isinstance(self.olora_tail_applied, bool):
            raise TypeError("olora_tail_applied must be a boolean.")
        if (self.name == "olora_tail") != self.olora_tail_applied:
            raise ValueError("olora_tail arm must set olora_tail_applied=true and baseline must set false.")
        if self.name == "olora_tail" and self.init_lora_weights != "olora_tail":
            raise ValueError("olora_tail arm must use init_lora_weights=olora_tail.")
        if self.name == "lora_baseline" and self.init_lora_weights != "standard":
            raise ValueError("lora_baseline arm must use init_lora_weights=standard.")
        object.__setattr__(self, "rank", _positive_int(self.rank, name="rank"))
        object.__setattr__(self, "alpha", _finite_positive_number(self.alpha, name="alpha"))
        object.__setattr__(self, "steps", _positive_int(self.steps, name="steps"))
        object.__setattr__(self, "effective_batch_size", _positive_int(self.effective_batch_size, name="effective_batch_size"))
        object.__setattr__(self, "learning_rate", _finite_positive_number(self.learning_rate, name="learning_rate"))
        object.__setattr__(self, "model_family", _non_empty_string(self.model_family, name="model_family"))
        object.__setattr__(self, "dataset", _non_empty_string(self.dataset, name="dataset"))
        object.__setattr__(self, "algorithm", _non_empty_string(self.algorithm, name="algorithm"))
        if not isinstance(self.claims_paper_results, bool):
            raise TypeError("claims_paper_results must be a boolean.")
        if self.claims_paper_results:
            raise ValueError("Fig14ArmContract must not claim paper metrics.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_family": self.model_family,
            "dataset": self.dataset,
            "algorithm": self.algorithm,
            "rank": self.rank,
            "alpha": self.alpha,
            "steps": self.steps,
            "effective_batch_size": self.effective_batch_size,
            "learning_rate": self.learning_rate,
            "target_modules": list(FIG14_TARGET_MODULES),
            "metrics": list(FIG14_METRICS),
            "init_lora_weights": self.init_lora_weights,
            "olora_tail_applied": self.olora_tail_applied,
            "result_status": "pending_gpu_training_and_eval",
            "claims_paper_results": self.claims_paper_results,
        }


@dataclass(frozen=True)
class Fig14OloraTailContract:
    """No-launch local contract for the Fig.14 paper reproduction table."""

    arms: tuple[Fig14ArmContract, ...]
    paper_exact: bool = False
    claims_paper_results: bool = False
    launches_training: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.arms, tuple) or not all(isinstance(arm, Fig14ArmContract) for arm in self.arms):
            raise TypeError("arms must be a tuple of Fig14ArmContract.")
        if {arm.name for arm in self.arms} != {"lora_baseline", "olora_tail"}:
            raise ValueError("Fig14 contract must contain lora_baseline and olora_tail arms.")
        if not isinstance(self.paper_exact, bool):
            raise TypeError("paper_exact must be a boolean.")
        if not isinstance(self.claims_paper_results, bool):
            raise TypeError("claims_paper_results must be a boolean.")
        if not isinstance(self.launches_training, bool):
            raise TypeError("launches_training must be a boolean.")
        if self.paper_exact or self.claims_paper_results or self.launches_training:
            raise ValueError("Fig14OloraTailContract is a local contract and must stay no-launch/no-claim.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass_fig14_olora_tail_contract",
            "paper_exact": self.paper_exact,
            "claims_paper_results": self.claims_paper_results,
            "launches_training": self.launches_training,
            "paper_reported_averages": dict(FIG14_PAPER_AVERAGES),
            "paper_average_atol": FIG14_PAPER_AVERAGE_ATOL,
            "arms": [arm.to_dict() for arm in self.arms],
            "missing_for_paper_exact": [
                "user-confirmed LoRA baseline 500-step DAPO run",
                "user-confirmed OLoRA-tail 500-step DAPO run",
                "GSM8K/MATH500/AIME22/AIME23/AIME24/AIME25 evaluation outputs",
                "adapter artifacts and optimizer-step evidence for both arms",
            ],
        }


def build_fig14_olora_tail_contract() -> Fig14OloraTailContract:
    """Return the no-launch contract for Fig.14 OLoRA-tail-vs-LoRA reproduction."""

    return Fig14OloraTailContract(
        arms=(
            Fig14ArmContract(
                name="lora_baseline",
                init_lora_weights="standard",
                olora_tail_applied=False,
            ),
            Fig14ArmContract(
                name="olora_tail",
                init_lora_weights="olora_tail",
                olora_tail_applied=True,
            ),
        )
    )


def _arms_by_name(results: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if isinstance(results, Mapping):
        raw_arms = results.get("arms", results)
        if isinstance(raw_arms, Mapping):
            return {
                _normalize_fig14_arm_name(name): arm
                for name, arm in raw_arms.items()
                if isinstance(arm, Mapping)
            }
        results = raw_arms  # type: ignore[assignment]
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError("results must be a sequence of arm mappings or an object with arms.")
    arms: dict[str, Mapping[str, Any]] = {}
    for index, arm in enumerate(results):
        if not isinstance(arm, Mapping):
            raise TypeError(f"results[{index}] must be a mapping.")
        name = _normalize_fig14_arm_name(arm.get("name"))
        if name in arms:
            raise ValueError(f"duplicate Fig.14 arm {name}.")
        arms[name] = arm
    return arms


def _arm_failures(arm: Mapping[str, Any], expected: Fig14ArmContract) -> list[str]:
    failures: list[str] = []
    prefix = expected.name
    if arm.get("status") != "pass":
        failures.append(f"{prefix}.status must be pass")
    if arm.get("evidence_type") not in {"paper_scale_rl_run", "fig14_olora_tail_run"}:
        failures.append(f"{prefix}.evidence_type must be paper_scale_rl_run or fig14_olora_tail_run")
    for key in ("explicit_user_confirmation", "gpu_or_training_launched", "optimizer_step_evidence"):
        try:
            _required_true(arm.get(key), name=key)
        except ValueError as exc:
            failures.append(f"{prefix}.{exc}")
    if arm.get("paper_exact") is True or arm.get("claims_paper_results") is True:
        failures.append(f"{prefix}.must not claim paper-exact results")
    checks = {
        "rank": expected.rank,
        "steps": expected.steps,
        "optimizer_steps": expected.steps,
        "effective_batch_size": expected.effective_batch_size,
    }
    for key, expected_value in checks.items():
        try:
            value = _positive_int(arm.get(key), name=key)
        except (TypeError, ValueError) as exc:
            failures.append(f"{prefix}.{exc}")
            continue
        if key == "optimizer_steps":
            if value < expected_value:
                failures.append(f"{prefix}.{key} must be >= {expected_value}")
        elif value != expected_value:
            failures.append(f"{prefix}.{key} must be == {expected_value}")
    for key in ("model_family", "dataset", "algorithm"):
        expected_value = getattr(expected, key)
        try:
            value = _non_empty_string(arm.get(key), name=key)
        except (TypeError, ValueError) as exc:
            failures.append(f"{prefix}.{exc}")
            continue
        if value != expected_value:
            failures.append(f"{prefix}.{key} must match {expected_value}")
    try:
        alpha = _finite_positive_number(arm.get("lora_alpha", arm.get("alpha")), name="alpha")
        if not math.isclose(alpha, expected.alpha, rel_tol=1e-9, abs_tol=1e-12):
            failures.append(f"{prefix}.alpha must match {expected.alpha:.12g}")
    except (TypeError, ValueError) as exc:
        failures.append(f"{prefix}.{exc}")
    try:
        learning_rate = _finite_positive_number(arm.get("learning_rate"), name="learning_rate")
        if not math.isclose(learning_rate, expected.learning_rate, rel_tol=1e-9, abs_tol=1e-12):
            failures.append(f"{prefix}.learning_rate must match {expected.learning_rate:.12g}")
    except (TypeError, ValueError) as exc:
        failures.append(f"{prefix}.{exc}")
    try:
        init = _normalize_init(arm.get("init_lora_weights", arm.get("lora_init")))
        if init != expected.init_lora_weights:
            failures.append(f"{prefix}.init_lora_weights must be {expected.init_lora_weights}")
    except (TypeError, ValueError) as exc:
        failures.append(f"{prefix}.{exc}")
    if arm.get("olora_tail_applied") is not expected.olora_tail_applied:
        failures.append(f"{prefix}.olora_tail_applied must be {expected.olora_tail_applied}")
    targets = arm.get("target_modules")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        failures.append(f"{prefix}.target_modules must be a sequence")
    elif tuple(targets) != FIG14_TARGET_MODULES:
        failures.append(f"{prefix}.target_modules must match the Fig.14 projection set")
    try:
        _metric_dict(arm, FIG14_METRICS)
    except (TypeError, ValueError) as exc:
        failures.append(f"{prefix}.{exc}")
    return failures


def analyze_fig14_olora_tail_results(
    results: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    contract: Fig14OloraTailContract | None = None,
    paper_average_atol: float = FIG14_PAPER_AVERAGE_ATOL,
) -> dict[str, Any]:
    """Analyze externally supplied Fig.14 results without launching or claiming them."""

    if contract is None:
        contract = build_fig14_olora_tail_contract()
    if not isinstance(contract, Fig14OloraTailContract):
        raise TypeError("contract must be a Fig14OloraTailContract.")
    paper_average_atol = _finite_number(paper_average_atol, name="paper_average_atol")
    if paper_average_atol < 0.0:
        raise ValueError("paper_average_atol must be non-negative.")

    failures: list[str] = []
    try:
        observed = _arms_by_name(results)
    except (TypeError, ValueError) as exc:
        observed = {}
        failures.append(str(exc))

    expected = {arm.name: arm for arm in contract.arms}
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    for name in missing:
        failures.append(f"missing Fig.14 arm {name}")
    for name in unexpected:
        failures.append(f"unexpected Fig.14 arm {name}")
    for name, expected_arm in expected.items():
        arm = observed.get(name)
        if arm is not None:
            failures.extend(_arm_failures(arm, expected_arm))

    averages: dict[str, float] = {}
    if not failures:
        for name, arm in observed.items():
            metrics = _metric_dict(arm, FIG14_METRICS)
            averages[name] = sum(metrics.values()) / float(len(FIG14_METRICS))

    pattern_checks = {
        "two_arm_coverage": set(observed) == {"lora_baseline", "olora_tail"} and not missing and not unexpected,
        "all_results_have_real_run_evidence": not failures,
        "olora_tail_beats_lora_average": False,
        "averages_match_paper_reported_values": False,
        "absolute_delta_matches_paper": False,
    }
    if not failures:
        delta = averages["olora_tail"] - averages["lora_baseline"]
        expected_delta = FIG14_PAPER_AVERAGES["olora_tail"] - FIG14_PAPER_AVERAGES["lora_baseline"]
        pattern_checks["olora_tail_beats_lora_average"] = delta > 0.0
        pattern_checks["averages_match_paper_reported_values"] = all(
            abs(averages[name] - target) <= paper_average_atol
            for name, target in FIG14_PAPER_AVERAGES.items()
        )
        pattern_checks["absolute_delta_matches_paper"] = abs(delta - expected_delta) <= (2.0 * paper_average_atol)

    paper_pattern_supported = all(pattern_checks.values())
    status = "pass_fig14_olora_tail_result_analysis" if not failures else "fail_fig14_olora_tail_result_analysis"
    return {
        "status": status,
        "paper_exact": False,
        "claims_exact_paper_results": False,
        "metrics": list(FIG14_METRICS),
        "paper_reported_averages": dict(FIG14_PAPER_AVERAGES),
        "paper_average_atol": paper_average_atol,
        "observed_averages": averages,
        "observed_absolute_delta": (
            averages["olora_tail"] - averages["lora_baseline"] if len(averages) == 2 else None
        ),
        "failures": failures,
        "pattern_checks": pattern_checks,
        "paper_pattern_supported": paper_pattern_supported,
        "paper_pattern_status": (
            "pass_paper_fig14_olora_tail_pattern"
            if paper_pattern_supported
            else "blocked_or_failed_paper_fig14_olora_tail_pattern"
        ),
    }


def paper_reference_fig14_contract() -> dict[str, Any]:
    """Return reference-only Fig.14 anchors, not local results."""

    return {
        "fig14": "DeepSeek-R1-Distill-Qwen-1.5B DAPO, 500 steps, bs32, LR1e-5, r16 alpha32",
        "metrics": list(FIG14_METRICS),
        "paper_reported_averages": dict(FIG14_PAPER_AVERAGES),
        "paper_reference_not_local_result": True,
        "claims_paper_results": False,
    }


__all__ = [
    "FIG14_ALGORITHM",
    "FIG14_DATASET",
    "FIG14_EXPECTED_ABSOLUTE_DELTA",
    "FIG14_METRICS",
    "FIG14_MODEL_FAMILY",
    "FIG14_PAPER_AVERAGES",
    "FIG14_PAPER_AVERAGE_ATOL",
    "FIG14_TARGET_MODULES",
    "Fig14ArmContract",
    "Fig14ArmName",
    "Fig14OloraTailContract",
    "analyze_fig14_olora_tail_results",
    "build_fig14_olora_tail_contract",
    "paper_reference_fig14_contract",
]
