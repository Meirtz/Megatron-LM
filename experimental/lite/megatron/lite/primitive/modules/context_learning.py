# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Context Learning write-policy primitives for Megatron Lite.

The Scaling PEFT paper frames Context Learning as repeated Context Distillation:
roll out a query-only student policy, score the candidate with a
teacher that sees query plus context, and write the reward back into the
query-only LoRA policy. This module only represents and validates that local
dataflow. It does not call a teacher model, load datasets or weights, launch
training, evaluate ALFWorld, or claim the paper's empirical result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

QUERY_ONLY_ROLLOUT = "query_only_rollout"
TEACHER_CONTEXT_SCORING = "teacher_context_scoring"
QUERY_ONLY_POLICY_UPDATE = "query_only_policy_update"
QUERY_ONLY_INFERENCE_CONTRACT = "query_only_inference_contract"
LORA_ADAPTER_TARGET = "lora_adapter"
RL_STYLE_POLICY_UPDATE = "rl_style_policy_update"

ContextLearningStage = Literal[
    "query_only_rollout",
    "teacher_context_scoring",
    "query_only_policy_update",
    "query_only_inference_contract",
]

EXPECTED_STAGE_ORDER: tuple[ContextLearningStage, ...] = (
    QUERY_ONLY_ROLLOUT,
    TEACHER_CONTEXT_SCORING,
    QUERY_ONLY_POLICY_UPDATE,
    QUERY_ONLY_INFERENCE_CONTRACT,
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


def _pending_status(value: Any, *, name: str) -> str:
    value = _non_empty_string(value, name=name)
    if not value.startswith("pending_real_"):
        raise ValueError(f"{name} must start with 'pending_real_', got {value!r}.")
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryOnlyPayload:
    """Student-policy input where context is intentionally absent."""

    query: str
    context: None = None
    context_visible: bool = False
    inference_context_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _non_empty_string(self.query, name="query"))
        if self.context is not None:
            raise ValueError("query-only payload must not carry context.")
        if self.context_visible is not False:
            raise ValueError("query-only payload must keep context_visible=False.")
        if self.inference_context_required is not False:
            raise ValueError("query-only payload must not require inference context.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "context": None,
            "context_visible": False,
            "inference_context_required": False,
        }


@dataclass(frozen=True)
class TeacherContextScoringPayload:
    """Teacher input that may see context during distillation scoring only."""

    query: str
    context: str
    candidate: str
    context_visible: bool = True
    score_status: str = "pending_real_teacher"

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _non_empty_string(self.query, name="query"))
        object.__setattr__(self, "context", _non_empty_string(self.context, name="context"))
        object.__setattr__(self, "candidate", _non_empty_string(self.candidate, name="candidate"))
        if self.context_visible is not True:
            raise ValueError("teacher scoring payload must keep context_visible=True.")
        object.__setattr__(self, "score_status", _pending_status(self.score_status, name="score_status"))

    @property
    def context_sha256(self) -> str:
        return _sha256(self.context)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "context": self.context,
            "candidate": self.candidate,
            "context_visible": True,
            "score_status": self.score_status,
        }


@dataclass(frozen=True)
class QueryOnlyRolloutRecord:
    """Pending query-only rollout from the current student LoRA policy."""

    policy: str
    input: QueryOnlyPayload
    candidate: str
    result_status: str = "pending_real_rollout"
    stage: ContextLearningStage = QUERY_ONLY_ROLLOUT

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _non_empty_string(self.policy, name="policy"))
        if not isinstance(self.input, QueryOnlyPayload):
            raise TypeError("input must be a QueryOnlyPayload.")
        object.__setattr__(self, "candidate", _non_empty_string(self.candidate, name="candidate"))
        object.__setattr__(self, "result_status", _pending_status(self.result_status, name="result_status"))
        if self.stage != QUERY_ONLY_ROLLOUT:
            raise ValueError(f"stage must be {QUERY_ONLY_ROLLOUT!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "policy": self.policy,
            "input": self.input.to_dict(),
            "candidate": self.candidate,
            "result_status": self.result_status,
        }


@dataclass(frozen=True)
class TeacherContextScoringRequest:
    """Pending teacher scoring request; the primitive never calls the teacher."""

    teacher: str
    input: TeacherContextScoringPayload
    result_status: str = "pending_real_teacher_score"
    stage: ContextLearningStage = TEACHER_CONTEXT_SCORING

    def __post_init__(self) -> None:
        object.__setattr__(self, "teacher", _non_empty_string(self.teacher, name="teacher"))
        if not isinstance(self.input, TeacherContextScoringPayload):
            raise TypeError("input must be a TeacherContextScoringPayload.")
        object.__setattr__(self, "result_status", _pending_status(self.result_status, name="result_status"))
        if self.stage != TEACHER_CONTEXT_SCORING:
            raise ValueError(f"stage must be {TEACHER_CONTEXT_SCORING!r}.")

    @property
    def context_sha256(self) -> str:
        return self.input.context_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "teacher": self.teacher,
            "input": self.input.to_dict(),
            "context_sha256": self.context_sha256,
            "result_status": self.result_status,
        }


@dataclass(frozen=True)
class QueryOnlyAdapterUpdatePlan:
    """Pending LoRA write that consumes a teacher reward without copying context."""

    policy_input: QueryOnlyPayload
    candidate: str
    reward_source: ContextLearningStage = TEACHER_CONTEXT_SCORING
    update_target: str = LORA_ADAPTER_TARGET
    update_style: str = RL_STYLE_POLICY_UPDATE
    context_visible_to_policy_update: bool = False
    result_status: str = "pending_real_rl_update"
    stage: ContextLearningStage = QUERY_ONLY_POLICY_UPDATE

    def __post_init__(self) -> None:
        if not isinstance(self.policy_input, QueryOnlyPayload):
            raise TypeError("policy_input must be a QueryOnlyPayload.")
        object.__setattr__(self, "candidate", _non_empty_string(self.candidate, name="candidate"))
        if self.reward_source != TEACHER_CONTEXT_SCORING:
            raise ValueError("reward_source must be teacher_context_scoring.")
        if self.update_target != LORA_ADAPTER_TARGET:
            raise ValueError("update_target must be lora_adapter.")
        if self.update_style != RL_STYLE_POLICY_UPDATE:
            raise ValueError("update_style must be rl_style_policy_update.")
        if self.context_visible_to_policy_update is not False:
            raise ValueError("policy update must keep context_visible_to_policy_update=False.")
        object.__setattr__(self, "result_status", _pending_status(self.result_status, name="result_status"))
        if self.stage != QUERY_ONLY_POLICY_UPDATE:
            raise ValueError(f"stage must be {QUERY_ONLY_POLICY_UPDATE!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "input": {
                "policy_input": self.policy_input.to_dict(),
                "candidate": self.candidate,
                "reward_source": self.reward_source,
                "update_target": self.update_target,
                "update_style": self.update_style,
                "context_visible_to_policy_update": False,
            },
            "result_status": self.result_status,
        }


@dataclass(frozen=True)
class QueryOnlyInferenceContract:
    """Inference-side contract: context must not be required after the write."""

    input: QueryOnlyPayload
    context_visible: bool = False
    inference_context_required: bool = False
    result_status: str = "pending_real_evaluation"
    stage: ContextLearningStage = QUERY_ONLY_INFERENCE_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.input, QueryOnlyPayload):
            raise TypeError("input must be a QueryOnlyPayload.")
        if self.context_visible is not False:
            raise ValueError("inference contract must keep context_visible=False.")
        if self.inference_context_required is not False:
            raise ValueError("inference contract must not require context.")
        object.__setattr__(self, "result_status", _pending_status(self.result_status, name="result_status"))
        if self.stage != QUERY_ONLY_INFERENCE_CONTRACT:
            raise ValueError(f"stage must be {QUERY_ONLY_INFERENCE_CONTRACT!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "input": self.input.to_dict(),
            "context_visible": False,
            "inference_context_required": False,
            "result_status": self.result_status,
        }


@dataclass(frozen=True)
class ContextLearningPlan:
    """Local Context Learning dataflow plan with paper-result claims disabled."""

    rollout: QueryOnlyRolloutRecord
    scoring: TeacherContextScoringRequest
    update: QueryOnlyAdapterUpdatePlan
    inference: QueryOnlyInferenceContract
    calls_teacher_model: bool = False
    paper_results_claimed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.rollout, QueryOnlyRolloutRecord):
            raise TypeError("rollout must be a QueryOnlyRolloutRecord.")
        if not isinstance(self.scoring, TeacherContextScoringRequest):
            raise TypeError("scoring must be a TeacherContextScoringRequest.")
        if not isinstance(self.update, QueryOnlyAdapterUpdatePlan):
            raise TypeError("update must be a QueryOnlyAdapterUpdatePlan.")
        if not isinstance(self.inference, QueryOnlyInferenceContract):
            raise TypeError("inference must be a QueryOnlyInferenceContract.")
        if self.calls_teacher_model is not False:
            raise ValueError("ContextLearningPlan does not call a teacher model.")
        if self.paper_results_claimed is not False:
            raise ValueError("ContextLearningPlan must not claim paper results.")
        assert_context_learning_plan(self)

    @property
    def workflow(self) -> tuple[
        QueryOnlyRolloutRecord,
        TeacherContextScoringRequest,
        QueryOnlyAdapterUpdatePlan,
        QueryOnlyInferenceContract,
    ]:
        return (self.rollout, self.scoring, self.update, self.inference)

    @property
    def stages(self) -> tuple[ContextLearningStage, ...]:
        return tuple(item.stage for item in self.workflow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": [item.to_dict() for item in self.workflow],
            "calls_teacher_model": False,
            "paper_results_claimed": False,
            "invariants": context_learning_invariants(self),
        }


def context_learning_invariants(plan: ContextLearningPlan) -> dict[str, bool]:
    """Return the Context Distillation invariants without launching work."""

    workflow_dicts = [item.to_dict() for item in plan.workflow]
    student_side_json = json.dumps(
        {
            "rollout": workflow_dicts[0],
            "update": workflow_dicts[2],
            "inference": workflow_dicts[3],
        },
        sort_keys=True,
    )
    context = plan.scoring.input.context
    return {
        "stage_order_matches_context_distillation": plan.stages == EXPECTED_STAGE_ORDER,
        "student_rollout_is_query_only": plan.rollout.input.context is None
        and plan.rollout.input.context_visible is False,
        "teacher_scoring_sees_query_and_context": plan.scoring.input.context_visible is True
        and bool(plan.scoring.input.query)
        and bool(plan.scoring.input.context),
        "policy_update_is_query_only": plan.update.policy_input.context is None
        and plan.update.context_visible_to_policy_update is False,
        "inference_does_not_require_context": plan.inference.input.context is None
        and plan.inference.inference_context_required is False,
        "adapter_write_target_is_lora": plan.update.update_target == LORA_ADAPTER_TARGET,
        "teacher_score_is_reward_source_for_update": plan.update.reward_source == plan.scoring.stage,
        "context_not_copied_into_rollout_or_update_payload": context not in student_side_json,
        "all_real_results_pending": all(
            item.to_dict()["result_status"].startswith("pending_real_")
            for item in plan.workflow
        ),
        "does_not_call_teacher_model": plan.calls_teacher_model is False,
        "paper_results_not_claimed": plan.paper_results_claimed is False,
    }


def assert_context_learning_plan(plan: ContextLearningPlan) -> None:
    invariants = context_learning_invariants(plan)
    failures = [name for name, passed in invariants.items() if not passed]
    if failures:
        raise ValueError(f"Context Learning invariants failed: {', '.join(failures)}")


def build_context_learning_plan(
    *,
    query: str,
    context: str,
    candidate: str,
    policy: str = "student_lora_policy",
    teacher: str = "query_plus_context_teacher",
) -> ContextLearningPlan:
    """Build the paper-shaped local Context Learning write-policy plan."""

    query_payload = QueryOnlyPayload(query=query)
    teacher_payload = TeacherContextScoringPayload(
        query=query,
        context=context,
        candidate=candidate,
    )
    return ContextLearningPlan(
        rollout=QueryOnlyRolloutRecord(
            policy=policy,
            input=query_payload,
            candidate=candidate,
        ),
        scoring=TeacherContextScoringRequest(
            teacher=teacher,
            input=teacher_payload,
        ),
        update=QueryOnlyAdapterUpdatePlan(
            policy_input=query_payload,
            candidate=candidate,
        ),
        inference=QueryOnlyInferenceContract(input=query_payload),
    )


__all__ = [
    "ContextLearningPlan",
    "ContextLearningStage",
    "EXPECTED_STAGE_ORDER",
    "LORA_ADAPTER_TARGET",
    "QUERY_ONLY_INFERENCE_CONTRACT",
    "QUERY_ONLY_POLICY_UPDATE",
    "QUERY_ONLY_ROLLOUT",
    "QueryOnlyAdapterUpdatePlan",
    "QueryOnlyInferenceContract",
    "QueryOnlyPayload",
    "QueryOnlyRolloutRecord",
    "RL_STYLE_POLICY_UPDATE",
    "TEACHER_CONTEXT_SCORING",
    "TeacherContextScoringPayload",
    "TeacherContextScoringRequest",
    "assert_context_learning_plan",
    "build_context_learning_plan",
    "context_learning_invariants",
]
