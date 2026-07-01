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
    / "adapter_lifecycle.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("adapter_lifecycle_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_policy_registry_tracks_policy_revision_and_session_lifecycle():
    lifecycle = _load_module()
    registry = lifecycle.AdapterPolicyRegistry()
    registry.register_base_deployment(
        lifecycle.BaseDeployment(
            base_deployment_id="base/qwen3-30b",
            model_name="Qwen3-30B",
            model_version="revA",
        )
    )
    policy = registry.create_policy(
        policy_id="policy/user-001",
        base_deployment_id="base/qwen3-30b",
        rank=16,
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )
    assert policy.active_revision_id is None

    rev0 = registry.create_revision(
        policy_id=policy.policy_id,
        revision_id="adapter/user-001/rev0",
        adapter_path="/adapters/user-001/rev0",
        metadata={"format": "peft_lora"},
    )
    assert registry.active_revision(policy.policy_id) == rev0
    assert registry.get_policy(policy.policy_id).adapter_revision_ids == (rev0.revision_id,)

    rev1 = registry.create_revision(
        policy_id=policy.policy_id,
        revision_id="adapter/user-001/rev1",
        adapter_path="/adapters/user-001/rev1",
        parent_revision_id=rev0.revision_id,
    )
    registry.set_active_revision(policy.policy_id, rev0.revision_id)
    assert registry.active_revision(policy.policy_id) == rev0
    assert registry.get_revision(rev1.revision_id).parent_revision_id == rev0.revision_id

    session = registry.create_policy_session(
        session_id="session/user-001/train0",
        policy_id=policy.policy_id,
        trainer_state_path="/trainer/session0",
    )
    assert session.adapter_revision_id == rev0.revision_id
    assert registry.close_policy_session(session.session_id).closed is True

    manifest = registry.to_manifest()
    assert manifest["base_deployments"][0]["base_deployment_id"] == "base/qwen3-30b"
    assert len(manifest["adapter_revisions"]) == 2
    assert manifest["policy_sessions"][0]["closed"] is True

    with pytest.raises(ValueError, match="policy already exists"):
        registry.create_policy(
            policy_id=policy.policy_id,
            base_deployment_id="base/qwen3-30b",
            rank=16,
            target_modules="linear_fc1",
        )
    with pytest.raises(KeyError, match="unknown base deployment"):
        registry.create_policy(
            policy_id="policy/missing-base",
            base_deployment_id="base/missing",
            rank=16,
            target_modules="linear_fc1",
        )


def test_adapter_residency_manager_separates_identity_from_compute_residency():
    lifecycle = _load_module()
    registry = lifecycle.AdapterPolicyRegistry()
    registry.register_base_deployment(
        lifecycle.BaseDeployment(
            base_deployment_id="base/qwen3-30b",
            model_name="Qwen3-30B",
        )
    )
    registry.create_policy(
        policy_id="policy/user-001",
        base_deployment_id="base/qwen3-30b",
        rank=8,
        target_modules="linear_fc1",
    )
    rev0 = registry.create_revision(
        policy_id="policy/user-001",
        revision_id="adapter/user-001/rev0",
        adapter_path="/adapters/user-001/rev0",
    )
    rev1 = registry.create_revision(
        policy_id="policy/user-001",
        revision_id="adapter/user-001/rev1",
        adapter_path="/adapters/user-001/rev1",
        parent_revision_id=rev0.revision_id,
    )

    manager = lifecycle.AdapterResidencyManager(warm_capacity=1, hot_capacity=1)
    manager.register_revision(rev0)
    manager.register_revision(rev1)

    manager.promote_to_warm(rev0.revision_id)
    manager.promote_to_warm(rev1.revision_id)
    assert manager.get(rev0.revision_id).tier == lifecycle.COOL_STORED
    assert manager.get(rev1.revision_id).tier == lifecycle.WARM_CPU

    manager.promote_to_hot(rev0.revision_id, device="cuda:0", batch_key="batch/a")
    manager.promote_to_hot(rev1.revision_id, device="cuda:1", batch_key="batch/b")
    assert manager.get(rev0.revision_id).tier == lifecycle.COOL_STORED
    assert manager.get(rev1.revision_id).tier == lifecycle.HOT_GPU

    plan = manager.plan_batch([rev0.revision_id, rev1.revision_id])
    assert plan[lifecycle.COOL_STORED] == [rev0.revision_id]
    assert plan[lifecycle.HOT_GPU] == [rev1.revision_id]
    assert registry.active_revision("policy/user-001") == rev1

    summary = manager.summary()
    assert summary["counts"][lifecycle.COOL_STORED] == 1
    assert summary["counts"][lifecycle.HOT_GPU] == 1

    with pytest.raises(ValueError, match="one of cool, warm, or hot"):
        manager.set_residency(rev0.revision_id, tier="tepid")
