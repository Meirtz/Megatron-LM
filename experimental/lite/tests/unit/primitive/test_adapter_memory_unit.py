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
    / "adapter_memory.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("adapter_memory_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_modules(memory):
    return (
        memory.LinearModuleSpec("self_attn.q_proj", 64, 64, "attention"),
        memory.LinearModuleSpec("self_attn.o_proj", 64, 64, "attention"),
        memory.LinearModuleSpec("mlp.gate_proj", 64, 128, "mlp"),
        memory.LinearModuleSpec("mlp.down_proj", 128, 64, "mlp"),
        memory.LinearModuleSpec("lm_head", 64, 256, "unembed"),
    )


def test_lora_memory_capacity_classifies_paper_ratio_regimes():
    memory = _load_module()

    assert memory.lora_trainable_params(64, 128, rank=4) == 4 * (64 + 128)
    assert memory.capacity_efficiency(memory_tokens=4, trainable_params=8000) == 0.0005
    assert (
        memory.classify_capacity_efficiency(0.0005)
        == "below_transition_expected_high_accuracy"
    )
    assert (
        memory.classify_capacity_efficiency(0.005)
        == "transition_band_expected_degradation"
    )
    assert (
        memory.classify_capacity_efficiency(0.02)
        == "above_collapse_expected_failure"
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        memory.classify_capacity_efficiency(float("nan"))
    with pytest.raises(ValueError, match="positive"):
        memory.lora_trainable_params(64, 128, rank=0)
    with pytest.raises(ValueError, match="positive"):
        memory.capacity_efficiency(memory_tokens=1, trainable_params=0)


def test_lora_memory_target_module_ablation_plan_keeps_paper_ranking_unclaimed():
    memory = _load_module()
    modules = _toy_modules(memory)

    plans = memory.build_target_module_ablation_plans(
        modules,
        rank=4,
        memory_tokens=8,
    )
    by_group = {plan.target_group: plan for plan in plans}

    assert tuple(by_group) == ("mlp", "attention", "all", "unembed")
    assert by_group["attention"].module_names == ("self_attn.q_proj", "self_attn.o_proj")
    assert by_group["mlp"].module_names == ("mlp.gate_proj", "mlp.down_proj")
    assert by_group["all"].module_names == (
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    )
    assert by_group["unembed"].module_names == ("lm_head",)
    assert by_group["mlp"].trainable_params == 4 * ((64 + 128) + (128 + 64))
    assert by_group["all"].trainable_params == (
        by_group["attention"].trainable_params + by_group["mlp"].trainable_params
    )
    assert all(plan.paper_accuracy_claimed is False for plan in plans)
    assert all(plan.paper_module_ranking_claimed is False for plan in plans)
    assert memory.paper_reference_target_order() == ("mlp", "attention", "all", "unembed")
    assert by_group["mlp"].to_dict()["paper_module_ranking_claimed"] is False


def test_lora_memory_rank_shift_moves_fixed_memory_across_regimes():
    memory = _load_module()
    modules = (
        memory.LinearModuleSpec("mlp.fc1", 64, 64, "mlp"),
    )

    plans = memory.build_rank_shift_plans(
        modules,
        target_group="mlp",
        ranks=(1, 8, 64),
        memory_tokens=2,
    )

    assert [plan.rank for plan in plans] == [1, 8, 64]
    assert [plan.trainable_params for plan in plans] == [128, 1024, 8192]
    assert [plan.tokens_per_trainable_param for plan in plans] == [2 / 128, 2 / 1024, 2 / 8192]
    assert [plan.capacity_regime for plan in plans] == [
        "above_collapse_expected_failure",
        "transition_band_expected_degradation",
        "below_transition_expected_high_accuracy",
    ]

    with pytest.raises(ValueError, match="selected no modules"):
        memory.build_lora_memory_plan(
            modules,
            target_group="attention",
            rank=4,
            memory_tokens=8,
        )
    with pytest.raises(ValueError, match="must be one of"):
        memory.LinearModuleSpec("bad", 1, 1, "router")
