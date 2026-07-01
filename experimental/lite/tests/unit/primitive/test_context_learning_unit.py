# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron"
    / "lite"
    / "primitive"
    / "modules"
    / "context_learning.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("context_learning_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_context_learning_plan_keeps_context_only_in_teacher_scoring():
    context_learning = _load_module()
    context = "private environment state visible only during distillation"

    plan = context_learning.build_context_learning_plan(
        query="choose the next action",
        context=context,
        candidate="open the drawer",
    )
    payload = plan.to_dict()
    workflow = payload["workflow"]

    assert plan.stages == (
        "query_only_rollout",
        "teacher_context_scoring",
        "query_only_policy_update",
        "query_only_inference_contract",
    )
    assert workflow[0]["input"]["context"] is None
    assert workflow[0]["input"]["context_visible"] is False
    assert workflow[1]["input"]["context"] == context
    assert workflow[1]["input"]["context_visible"] is True
    assert workflow[2]["input"]["policy_input"]["context"] is None
    assert workflow[2]["input"]["context_visible_to_policy_update"] is False
    assert workflow[2]["input"]["update_target"] == "lora_adapter"
    assert workflow[3]["inference_context_required"] is False
    assert context not in json.dumps(
        {"rollout": workflow[0], "update": workflow[2], "inference": workflow[3]},
        sort_keys=True,
    )
    assert all(payload["invariants"].values())
    assert plan.calls_teacher_model is False
    assert plan.paper_results_claimed is False


def test_context_learning_plan_rejects_context_leaks_and_wrong_reward_source():
    context_learning = _load_module()

    with pytest.raises(ValueError, match="query-only payload must not carry context"):
        context_learning.QueryOnlyPayload(query="q", context="leaked")
    with pytest.raises(ValueError, match="reward_source must be teacher_context_scoring"):
        context_learning.QueryOnlyAdapterUpdatePlan(
            policy_input=context_learning.QueryOnlyPayload(query="q"),
            candidate="a",
            reward_source="query_only_rollout",
        )
    with pytest.raises(ValueError, match="update_target must be lora_adapter"):
        context_learning.QueryOnlyAdapterUpdatePlan(
            policy_input=context_learning.QueryOnlyPayload(query="q"),
            candidate="a",
            update_target="full_model",
        )
    with pytest.raises(ValueError, match="context must be non-empty"):
        context_learning.build_context_learning_plan(
            query="q",
            context="",
            candidate="a",
        )


def test_context_learning_plan_keeps_paper_results_unclaimed():
    context_learning = _load_module()
    plan = context_learning.build_context_learning_plan(
        query="q",
        context="teacher-only context",
        candidate="a",
    )

    with pytest.raises(ValueError, match="does not call a teacher model"):
        context_learning.ContextLearningPlan(
            rollout=plan.rollout,
            scoring=plan.scoring,
            update=plan.update,
            inference=plan.inference,
            calls_teacher_model=True,
        )
    with pytest.raises(ValueError, match="must not claim paper results"):
        context_learning.ContextLearningPlan(
            rollout=plan.rollout,
            scoring=plan.scoring,
            update=plan.update,
            inference=plan.inference,
            paper_results_claimed=True,
        )
    assert context_learning.context_learning_invariants(plan)["paper_results_not_claimed"] is True
