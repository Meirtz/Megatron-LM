# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-user adapter-population primitives for Megatron Lite.

The Scaling PEFT paper reports OASIS user-simulator results where per-user LoRA policies preserve heterogeneity and richer action ecology relative to a
shared-base baseline. This module represents the local metadata contract needed
before that result can be reproduced: base deployment identity, per-user LoRA
policy records, shared-base evaluation arms, heterogeneity/action-ecology metric
schemas, and structural scaling axes. It does not create OASIS data, run user
simulators, train adapters, load data or weights, or claim paper metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

PER_USER_LORA = "per_user_lora"
SHARED_BASE = "shared_base"
LORA_ADAPTER = "lora"
NO_ADAPTER = "none"
COOL_STORED = "cool_stored"

ArmType = Literal["per_user_lora", "shared_base"]

DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

HETEROGENEITY_METRICS: tuple[str, ...] = (
    "between_user_action_distribution_jsd",
    "between_user_state_transition_jsd",
    "per_user_preference_retention",
)

ACTION_ECOLOGY_METRICS: tuple[str, ...] = (
    "unique_action_count",
    "action_entropy",
    "rare_action_rate",
    "role_specific_action_share",
)


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


def _pending_status(value: Any, *, name: str) -> str:
    value = _non_empty_string(value, name=name)
    if not value.startswith("pending_real_"):
        raise ValueError(f"{name} must start with 'pending_real_', got {value!r}.")
    return value


def _normalize_arm_type(value: Any, *, name: str = "arm_type") -> ArmType:
    value = _non_empty_string(value, name=name)
    if value not in {PER_USER_LORA, SHARED_BASE}:
        raise ValueError("arm_type must be per_user_lora or shared_base.")
    return value  # type: ignore[return-value]


def _target_modules(value: Sequence[str]) -> tuple[str, ...]:
    value = tuple(_non_empty_string(item, name="target_modules entry") for item in value)
    if not value:
        raise ValueError("target_modules must be non-empty.")
    if len(set(value)) != len(value):
        raise ValueError("target_modules must not contain duplicates.")
    return value


@dataclass(frozen=True)
class BaseDeploymentSpec:
    """Frozen shared base deployment for an adapter population."""

    base_deployment_id: str
    model_family: str = "Qwen3"
    weights_status: str = "not_materialized"
    shared_across_users: bool = True
    paper_metric_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_deployment_id",
            _non_empty_string(self.base_deployment_id, name="base_deployment_id"),
        )
        object.__setattr__(self, "model_family", _non_empty_string(self.model_family, name="model_family"))
        object.__setattr__(
            self,
            "weights_status",
            _non_empty_string(self.weights_status, name="weights_status"),
        )
        if self.shared_across_users is not True:
            raise ValueError("base deployment must be shared_across_users=True.")
        if self.paper_metric_claimed is not False:
            raise ValueError("BaseDeploymentSpec must not claim paper metrics.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_deployment_id": self.base_deployment_id,
            "model_family": self.model_family,
            "weights_status": self.weights_status,
            "shared_across_users": True,
            "paper_metric_claimed": False,
        }


@dataclass(frozen=True)
class UserPolicyRecord:
    """Per-user LoRA policy identity; no adapter weights are loaded here."""

    user_id: str
    base_deployment_id: str
    rank: int
    policy_id: str | None = None
    adapter_revision_id: str | None = None
    adapter_type: str = LORA_ADAPTER
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    residency: str = COOL_STORED
    result_status: str = "pending_real_user_simulator_training"
    paper_metric_claimed: bool = False

    def __post_init__(self) -> None:
        user_id = _non_empty_string(self.user_id, name="user_id")
        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(
            self,
            "base_deployment_id",
            _non_empty_string(self.base_deployment_id, name="base_deployment_id"),
        )
        object.__setattr__(self, "rank", _positive_int(self.rank, name="rank"))
        object.__setattr__(self, "policy_id", self.policy_id or f"policy/{user_id}")
        object.__setattr__(
            self,
            "adapter_revision_id",
            self.adapter_revision_id or f"adapter/{user_id}/rev0",
        )
        if self.adapter_type != LORA_ADAPTER:
            raise ValueError("per-user policy adapter_type must be lora.")
        object.__setattr__(self, "target_modules", _target_modules(self.target_modules))
        object.__setattr__(self, "residency", _non_empty_string(self.residency, name="residency"))
        object.__setattr__(
            self,
            "result_status",
            _pending_status(self.result_status, name="result_status"),
        )
        if self.paper_metric_claimed is not False:
            raise ValueError("UserPolicyRecord must not claim paper metrics.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "user_id": self.user_id,
            "base_deployment_id": self.base_deployment_id,
            "adapter_revision_id": self.adapter_revision_id,
            "adapter_type": self.adapter_type,
            "rank": self.rank,
            "target_modules": list(self.target_modules),
            "residency": self.residency,
            "result_status": self.result_status,
            "paper_metric_claimed": False,
        }


@dataclass(frozen=True)
class SharedBaselinePolicy:
    """Shared-base baseline arm with no adapter attached."""

    base_deployment_id: str
    policy_id: str = "policy/shared_base"
    adapter_revision_id: None = None
    adapter_type: str = NO_ADAPTER
    result_status: str = "pending_real_shared_baseline_evaluation"
    paper_metric_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_deployment_id",
            _non_empty_string(self.base_deployment_id, name="base_deployment_id"),
        )
        object.__setattr__(self, "policy_id", _non_empty_string(self.policy_id, name="policy_id"))
        if self.adapter_revision_id is not None:
            raise ValueError("shared baseline must not have an adapter_revision_id.")
        if self.adapter_type != NO_ADAPTER:
            raise ValueError("shared baseline adapter_type must be none.")
        object.__setattr__(
            self,
            "result_status",
            _pending_status(self.result_status, name="result_status"),
        )
        if self.paper_metric_claimed is not False:
            raise ValueError("SharedBaselinePolicy must not claim paper metrics.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "base_deployment_id": self.base_deployment_id,
            "adapter_revision_id": None,
            "adapter_type": self.adapter_type,
            "result_status": self.result_status,
            "paper_metric_claimed": False,
        }


@dataclass(frozen=True)
class PopulationEvaluationArm:
    """Pending OASIS evaluation arm for one user and one policy type."""

    user_id: str
    policy_id: str
    arm_type: ArmType
    metrics_status: str = "pending_real_oasis_evaluation"
    claim_ready: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _non_empty_string(self.user_id, name="user_id"))
        object.__setattr__(self, "policy_id", _non_empty_string(self.policy_id, name="policy_id"))
        object.__setattr__(self, "arm_type", _normalize_arm_type(self.arm_type))
        object.__setattr__(
            self,
            "metrics_status",
            _pending_status(self.metrics_status, name="metrics_status"),
        )
        if self.claim_ready is not False:
            raise ValueError("PopulationEvaluationArm must keep claim_ready=False.")

    @property
    def arm_id(self) -> str:
        return f"oasis_user_sim/{self.user_id}/{self.arm_type}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "user_id": self.user_id,
            "policy_id": self.policy_id,
            "arm_type": self.arm_type,
            "metrics_status": self.metrics_status,
            "claim_ready": False,
        }


@dataclass(frozen=True)
class AdapterPopulationMetricSchema:
    """Metric names required before OASIS user-simulator claims are made."""

    heterogeneity: tuple[str, ...] = HETEROGENEITY_METRICS
    action_ecology: tuple[str, ...] = ACTION_ECOLOGY_METRICS
    scaling: tuple[str, ...] = ("user_count", "adapter_count", "mean_metric", "confidence_interval")

    def __post_init__(self) -> None:
        object.__setattr__(self, "heterogeneity", _target_modules(self.heterogeneity))
        object.__setattr__(self, "action_ecology", _target_modules(self.action_ecology))
        object.__setattr__(self, "scaling", _target_modules(self.scaling))

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "heterogeneity": list(self.heterogeneity),
            "action_ecology": list(self.action_ecology),
            "scaling": list(self.scaling),
        }


@dataclass(frozen=True)
class StructuralScalingArm:
    """Declared population-size axis; metrics remain pending."""

    population_size: int
    per_user_adapter_count: int | None = None
    shared_base_count: int = 1
    result_status: str = "pending_real_structural_scaling"

    def __post_init__(self) -> None:
        population_size = _positive_int(self.population_size, name="population_size")
        object.__setattr__(self, "population_size", population_size)
        adapter_count = population_size if self.per_user_adapter_count is None else self.per_user_adapter_count
        object.__setattr__(
            self,
            "per_user_adapter_count",
            _positive_int(adapter_count, name="per_user_adapter_count"),
        )
        object.__setattr__(self, "shared_base_count", _positive_int(self.shared_base_count, name="shared_base_count"))
        object.__setattr__(
            self,
            "result_status",
            _pending_status(self.result_status, name="result_status"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "per_user_adapter_count": self.per_user_adapter_count,
            "shared_base_count": self.shared_base_count,
            "result_status": self.result_status,
        }


@dataclass(frozen=True)
class AdapterPopulationPlan:
    """Local adapter-population contract with all paper-result claims disabled."""

    base_deployment: BaseDeploymentSpec
    policy_records: tuple[UserPolicyRecord, ...]
    shared_baseline_policy: SharedBaselinePolicy
    evaluation_arms: tuple[PopulationEvaluationArm, ...]
    metric_schema: AdapterPopulationMetricSchema = field(default_factory=AdapterPopulationMetricSchema)
    scaling_matrix: tuple[StructuralScalingArm, ...] = (
        StructuralScalingArm(1),
        StructuralScalingArm(3),
        StructuralScalingArm(16),
    )
    runs_user_simulators: bool = False
    paper_results_claimed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_deployment, BaseDeploymentSpec):
            raise TypeError("base_deployment must be a BaseDeploymentSpec.")
        if not isinstance(self.shared_baseline_policy, SharedBaselinePolicy):
            raise TypeError("shared_baseline_policy must be a SharedBaselinePolicy.")
        if not isinstance(self.metric_schema, AdapterPopulationMetricSchema):
            raise TypeError("metric_schema must be an AdapterPopulationMetricSchema.")
        object.__setattr__(self, "policy_records", tuple(self.policy_records))
        object.__setattr__(self, "evaluation_arms", tuple(self.evaluation_arms))
        object.__setattr__(self, "scaling_matrix", tuple(self.scaling_matrix))
        if not self.policy_records:
            raise ValueError("policy_records must be non-empty.")
        if not all(isinstance(record, UserPolicyRecord) for record in self.policy_records):
            raise TypeError("policy_records must contain UserPolicyRecord entries.")
        if not all(isinstance(arm, PopulationEvaluationArm) for arm in self.evaluation_arms):
            raise TypeError("evaluation_arms must contain PopulationEvaluationArm entries.")
        if not all(isinstance(item, StructuralScalingArm) for item in self.scaling_matrix):
            raise TypeError("scaling_matrix must contain StructuralScalingArm entries.")
        if self.runs_user_simulators is not False:
            raise ValueError("AdapterPopulationPlan does not run user simulators.")
        if self.paper_results_claimed is not False:
            raise ValueError("AdapterPopulationPlan must not claim paper results.")
        validate_adapter_population_plan(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_deployment": self.base_deployment.to_dict(),
            "policy_records": [record.to_dict() for record in self.policy_records],
            "shared_baseline_policy": self.shared_baseline_policy.to_dict(),
            "evaluation_arms": [arm.to_dict() for arm in self.evaluation_arms],
            "metric_schema": self.metric_schema.to_dict(),
            "scaling_matrix": [item.to_dict() for item in self.scaling_matrix],
            "runs_user_simulators": False,
            "paper_results_claimed": False,
            "invariants": adapter_population_invariants(self),
        }


def build_population_evaluation_arms(
    policy_records: Sequence[UserPolicyRecord],
    *,
    shared_policy_id: str = "policy/shared_base",
) -> tuple[PopulationEvaluationArm, ...]:
    policy_records = tuple(policy_records)
    if not policy_records:
        raise ValueError("policy_records must be non-empty.")
    arms: list[PopulationEvaluationArm] = []
    for record in policy_records:
        arms.append(PopulationEvaluationArm(record.user_id, record.policy_id or "", PER_USER_LORA))
        arms.append(PopulationEvaluationArm(record.user_id, shared_policy_id, SHARED_BASE))
    return tuple(arms)


def build_adapter_population_plan(
    *,
    user_ids: Sequence[str] = ("user_careful_planner", "user_fast_actor", "user_social_explorer"),
    base_deployment_id: str = "base/qwen3-oasis-shared",
    rank: int = 16,
) -> AdapterPopulationPlan:
    user_ids = tuple(_non_empty_string(user_id, name="user_id") for user_id in user_ids)
    if not user_ids:
        raise ValueError("user_ids must be non-empty.")
    if len(set(user_ids)) != len(user_ids):
        raise ValueError("user_ids must be unique.")
    base = BaseDeploymentSpec(base_deployment_id=base_deployment_id)
    policies = tuple(
        UserPolicyRecord(user_id=user_id, base_deployment_id=base.base_deployment_id, rank=rank)
        for user_id in user_ids
    )
    shared = SharedBaselinePolicy(base_deployment_id=base.base_deployment_id)
    return AdapterPopulationPlan(
        base_deployment=base,
        policy_records=policies,
        shared_baseline_policy=shared,
        evaluation_arms=build_population_evaluation_arms(policies, shared_policy_id=shared.policy_id),
    )


def adapter_population_invariants(plan: AdapterPopulationPlan) -> dict[str, bool]:
    user_ids = [record.user_id for record in plan.policy_records]
    adapter_ids = [record.adapter_revision_id for record in plan.policy_records]
    per_user_arms = [arm for arm in plan.evaluation_arms if arm.arm_type == PER_USER_LORA]
    shared_arms = [arm for arm in plan.evaluation_arms if arm.arm_type == SHARED_BASE]
    metric_schema = plan.metric_schema
    return {
        "policy_identity_registry_has_unique_users": len(set(user_ids)) == len(user_ids),
        "every_user_has_dedicated_lora_policy_record": all(
            record.adapter_type == LORA_ADAPTER and record.user_id for record in plan.policy_records
        ),
        "all_user_adapters_share_same_base_deployment": {
            record.base_deployment_id for record in plan.policy_records
        }
        == {plan.base_deployment.base_deployment_id},
        "adapter_revision_ids_are_unique_per_user": len(set(adapter_ids)) == len(adapter_ids),
        "shared_baseline_has_no_adapter": plan.shared_baseline_policy.adapter_type == NO_ADAPTER
        and plan.shared_baseline_policy.adapter_revision_id is None,
        "per_user_and_shared_arms_present_for_every_user": {arm.user_id for arm in per_user_arms} == set(user_ids)
        and {arm.user_id for arm in shared_arms} == set(user_ids),
        "evaluation_arms_do_not_claim_metrics": all(
            arm.metrics_status == "pending_real_oasis_evaluation" and arm.claim_ready is False
            for arm in plan.evaluation_arms
        ),
        "heterogeneity_metric_schema_present": set(metric_schema.heterogeneity)
        == set(HETEROGENEITY_METRICS),
        "action_ecology_metric_schema_present": set(metric_schema.action_ecology)
        == set(ACTION_ECOLOGY_METRICS),
        "structural_scaling_axis_declared_not_measured": [item.population_size for item in plan.scaling_matrix]
        == [1, 3, 16]
        and all(item.result_status == "pending_real_structural_scaling" for item in plan.scaling_matrix),
        "does_not_run_user_simulators": plan.runs_user_simulators is False,
        "paper_results_not_claimed": plan.paper_results_claimed is False,
    }


def validate_adapter_population_plan(plan: AdapterPopulationPlan) -> None:
    invariants = adapter_population_invariants(plan)
    failures = [name for name, passed in invariants.items() if not passed]
    if failures:
        raise ValueError(f"adapter population invariants failed: {', '.join(failures)}")


def paper_reference_population_contract() -> dict[str, Any]:
    return {
        "benchmark": "OASIS",
        "reported_claim": "per-user LoRA preserves heterogeneity and richer action ecology versus shared-base",
        "reported_evidence": "Tables 5-7 structural scaling",
        "metric_status": "paper_reference_not_local_result",
    }


__all__ = [
    "ACTION_ECOLOGY_METRICS",
    "AdapterPopulationMetricSchema",
    "AdapterPopulationPlan",
    "ArmType",
    "BaseDeploymentSpec",
    "COOL_STORED",
    "DEFAULT_TARGET_MODULES",
    "HETEROGENEITY_METRICS",
    "LORA_ADAPTER",
    "NO_ADAPTER",
    "PER_USER_LORA",
    "PopulationEvaluationArm",
    "SHARED_BASE",
    "SharedBaselinePolicy",
    "StructuralScalingArm",
    "UserPolicyRecord",
    "adapter_population_invariants",
    "build_adapter_population_plan",
    "build_population_evaluation_arms",
    "paper_reference_population_contract",
    "validate_adapter_population_plan",
]
