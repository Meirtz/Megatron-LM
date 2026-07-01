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
    / "model"
    / "glm5"
    / "lite"
    / "adapter_semantics.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("glm5_adapter_semantics_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_glm5_target_aliases_expand_to_mla_dsa_projection_surfaces():
    semantics = _load_module()

    assert semantics.dedupe_target_modules("all-linear") == semantics.ADAPTER_TARGET_MODULES
    assert semantics.dedupe_target_modules("qkv") == (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
    )
    assert semantics.dedupe_target_modules(["q_a", "q_a_proj", "gate_proj"]) == (
        "q_a_proj",
        "gate_proj",
        "up_proj",
    )
    assert "indexer" not in semantics.ADAPTER_TARGET_MODULES

    with pytest.raises(ValueError, match="Unsupported GLM5 LoRA target module"):
        semantics.dedupe_target_modules("indexer")
    with pytest.raises(TypeError, match="sequence of strings"):
        semantics.dedupe_target_modules({"q_a_proj": True})


def test_glm5_adapter_key_parser_accepts_only_adapter_surfaces():
    semantics = _load_module()

    attn_key = "base_model.model.model.layers.2.self_attn.kv_a_proj_with_mqa.lora_A.weight"
    spec = semantics.parse_adapter_tensor_key(attn_key)
    assert spec.layer_idx == 2
    assert spec.path_kind == "self_attn"
    assert spec.module == "kv_a_proj_with_mqa"
    assert spec.suffix == "lora_A.weight"
    assert spec.to_key() == attn_key

    expert_key = "base_model.model.model.layers.3.mlp.experts.7.gate_proj.lora_B.weight"
    expert = semantics.parse_adapter_tensor_key(expert_key)
    assert expert.path_kind == "expert_mlp"
    assert expert.expert_idx == 7
    assert expert.to_dict()["key"] == expert_key

    shared_key = "base_model.model.model.layers.4.mlp.shared_experts.down_proj.lora_A.weight"
    shared = semantics.parse_adapter_tensor_key(shared_key)
    assert shared.path_kind == "shared_experts"
    assert shared.to_key() == shared_key

    assert semantics.is_supported_adapter_tensor_key(attn_key) is True
    assert semantics.is_supported_adapter_tensor_key(
        "base_model.model.model.layers.2.self_attn.indexer.lora_A.weight"
    ) is False
    assert semantics.is_supported_adapter_tensor_key(
        "base_model.model.model.layers.2.self_attn.q_a_proj.weight"
    ) is False


def test_glm5_mtp_layer_keys_use_offset_after_main_layers():
    semantics = _load_module()

    assert semantics.mtp_layer_index(2, 0) == 2
    assert semantics.mtp_layer_index(2, 1) == 3
    keys = semantics.adapter_tensor_keys_for_main_and_mtp_layers(
        main_layer_indices=(0,),
        num_hidden_layers=2,
        num_mtp_layers=1,
        target_modules="all-linear",
    )

    assert len(keys) == 32
    assert "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight" in keys
    assert "base_model.model.model.layers.2.self_attn.q_a_proj.lora_A.weight" in keys
    assert "base_model.model.model.layers.2.mlp.gate_proj.lora_B.weight" in keys
    assert "base_model.model.model.layers.2.mlp.up_proj.lora_B.weight" in keys
    assert not any(".indexer." in key for key in keys)


def test_glm5_lora_semantics_contract_is_reference_not_result():
    semantics = _load_module()

    payload = semantics.glm5_lora_semantics_contract()
    assert payload["status"] == "pass_glm5_lora_semantics_contract"
    assert payload["paper_exact"] is False
    assert payload["claims_paper_results"] is False
    assert payload["claims_glm5_training_smoke"] is False
    assert payload["claims_weight_materialization"] is False
    assert payload["invariants"]["mla_dsa_attention_targets_present"] is True
    assert payload["invariants"]["dsa_indexer_targets_excluded"] is True
    assert payload["invariants"]["model_metadata_tracks_mtp"] is True
    assert payload["invariants"]["lora_metadata_tracks_rslora"] is True
    assert "real load/training smoke with MTP enabled" in payload["missing_for_paper_exact"]
    semantics.validate_glm5_lora_semantics_contract()
