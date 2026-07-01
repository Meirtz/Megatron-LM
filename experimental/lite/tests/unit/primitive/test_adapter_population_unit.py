# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron"
    / "lite"
    / "primitive"
    / "modules"
    / "adapter_population.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("adapter_population_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_population_plan_builds_per_user_and_shared_baseline_arms():
    population = _load_module()

    plan = population.build_adapter_population_plan()
    payload = plan.to_dict()
    by_arm = {(arm["user_id"], arm["arm_type"]): arm for arm in payload["evaluation_arms"]}

    assert payload["base_deployment"]["base_deployment_id"] == "base/qwen3-oasis-shared"
    assert payload["base_deployment"]["shared_across_users"] is True
    assert len(payload["policy_records"]) == 3
    assert {record["adapter_type"] for record in payload["policy_records"]} == {"lora"}
    assert len({record["adapter_revision_id"] for record in payload["policy_records"]}) == 3
    assert payload["shared_baseline_policy"]["adapter_type"] == "none"
    assert payload["shared_baseline_policy"]["adapter_revision_id"] is None
    assert len(payload["evaluation_arms"]) == 6
    for record in payload["policy_records"]:
        user_id = record["user_id"]
        assert by_arm[(user_id, "per_user_lora")]["policy_id"] == record["policy_id"]
        assert by_arm[(user_id, "shared_base")]["policy_id"] == "policy/shared_base"
    assert payload["metric_schema"]["heterogeneity"] == [
        "between_user_action_distribution_jsd",
        "between_user_state_transition_jsd",
        "per_user_preference_retention",
    ]
    assert payload["metric_schema"]["action_ecology"] == [
        "unique_action_count",
        "action_entropy",
        "rare_action_rate",
        "role_specific_action_share",
    ]
    assert [item["population_size"] for item in payload["scaling_matrix"]] == [1, 3, 16]
    assert all(payload["invariants"].values())
    assert plan.runs_user_simulators is False
    assert plan.paper_results_claimed is False


def test_adapter_population_plan_rejects_duplicate_users_and_metric_claims():
    population = _load_module()

    with pytest.raises(ValueError, match="user_ids must be unique"):
        population.build_adapter_population_plan(user_ids=("user_a", "user_a"))
    with pytest.raises(ValueError, match="must not claim paper metrics"):
        population.UserPolicyRecord(
            user_id="user_a",
            base_deployment_id="base",
            rank=16,
            paper_metric_claimed=True,
        )
    with pytest.raises(ValueError, match="must not claim paper results"):
        plan = population.build_adapter_population_plan()
        population.AdapterPopulationPlan(
            base_deployment=plan.base_deployment,
            policy_records=plan.policy_records,
            shared_baseline_policy=plan.shared_baseline_policy,
            evaluation_arms=plan.evaluation_arms,
            paper_results_claimed=True,
        )
    with pytest.raises(ValueError, match="does not run user simulators"):
        plan = population.build_adapter_population_plan()
        population.AdapterPopulationPlan(
            base_deployment=plan.base_deployment,
            policy_records=plan.policy_records,
            shared_baseline_policy=plan.shared_baseline_policy,
            evaluation_arms=plan.evaluation_arms,
            runs_user_simulators=True,
        )


def test_adapter_population_shared_baseline_and_arm_guards():
    population = _load_module()

    with pytest.raises(ValueError, match="shared baseline must not have an adapter_revision_id"):
        population.SharedBaselinePolicy(
            base_deployment_id="base",
            adapter_revision_id="adapter/user/rev0",
        )
    with pytest.raises(ValueError, match="arm_type must be per_user_lora or shared_base"):
        population.PopulationEvaluationArm(
            user_id="user_a",
            policy_id="policy/user_a",
            arm_type="bad",
        )
    with pytest.raises(ValueError, match="claim_ready=False"):
        population.PopulationEvaluationArm(
            user_id="user_a",
            policy_id="policy/user_a",
            arm_type="per_user_lora",
            claim_ready=True,
        )
    with pytest.raises(ValueError, match="target_modules must not contain duplicates"):
        population.UserPolicyRecord(
            user_id="user_a",
            base_deployment_id="base",
            rank=16,
            target_modules=("q_proj", "q_proj"),
        )


def test_adapter_population_reference_contract_is_not_local_metric():
    population = _load_module()

    reference = population.paper_reference_population_contract()

    assert reference["benchmark"] == "OASIS"
    assert "per-user LoRA preserves heterogeneity" in reference["reported_claim"]
    assert reference["metric_status"] == "paper_reference_not_local_result"
    assert population.adapter_population_invariants(
        population.build_adapter_population_plan()
    )["paper_results_not_claimed"] is True
