# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static CPU tests for native DeepSeek-V4 lite config and registry."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlite


def test_deepseek_v4_registry_resolves_lite():
    from megatron.lite.model.registry import (
        TRAIN_RUNTIME_MODULES,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    runtime_name = resolve_runtime_model_name("deepseek_v4", "lite")
    assert runtime_name == "deepseek_v4"
    assert resolve_model_type_from_hf({"model_type": "deepseek_v4"}) == "deepseek_v4"
    assert TRAIN_RUNTIME_MODULES[runtime_name] == "megatron.lite.model.deepseek_v4.lite.protocol"


def test_deepseek_v4_config_reads_native_layer_types_and_hash_layers():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    cfg = DeepseekV4Config._from_hf_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 129280,
            "hidden_size": 4096,
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 1,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
            "compress_rates": {
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
            "mlp_layer_types": ["hash_moe", "hash_moe", "hash_moe", "moe"],
            "tie_word_embeddings": False,
        }
    )

    assert cfg.compress_ratios == [0, 0, 4, 128, 0]
    assert cfg.num_hash_layers == 3
    assert cfg.tie_word_embeddings is False


def test_deepseek_v4_config_reads_nested_rope_parameters():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    cfg = DeepseekV4Config._from_hf_dict(
        {
            "model_type": "deepseek_v4",
            "rope_theta": 999.0,
            "compress_rope_theta": 999.0,
            "rope_scaling": {
                "factor": 40.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
            "rope_parameters": {
                "main": {"rope_theta": 10000},
                "compress": {
                    "factor": 16,
                    "original_max_position_embeddings": 65536,
                    "rope_theta": 160000,
                    "beta_fast": 48,
                    "beta_slow": 2,
                    "type": "yarn",
                },
                "rope_theta": 10000,
            },
        }
    )

    assert cfg.rope_theta == 10000.0
    assert cfg.compress_rope_theta == 160000.0
    assert cfg.rotary_scaling_factor == 16.0
    assert cfg.original_max_position_embeddings == 65536
    assert cfg.beta_fast == 48.0
    assert cfg.beta_slow == 2.0


def test_deepseek_v4_config_rejects_non_prefix_hash_layers():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    with pytest.raises(ValueError, match="contiguous prefix"):
        DeepseekV4Config._from_hf_dict(
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 3,
                "mlp_layer_types": ["hash_moe", "moe", "hash_moe"],
            }
        )


def test_deepseek_v4_config_rejects_tied_embeddings():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    with pytest.raises(ValueError, match="untied embeddings"):
        DeepseekV4Config._from_hf_dict(
            {
                "model_type": "deepseek_v4",
                "tie_word_embeddings": True,
            }
        )


def test_deepseek_v4_mhc_matches_hf_reference_formula_markers():
    """Keep this static: local developer Macs may not have torch installed."""
    from pathlib import Path

    lite_root = Path(__file__).resolve().parents[3]
    hca_text = (lite_root / "megatron/lite/primitive/modules/attention/hca.py").read_text()
    mhc_text = (lite_root / "megatron/lite/primitive/modules/attention/mhc.py").read_text()
    model_text = (
        lite_root / "megatron/lite/model/deepseek_v4/lite/model.py"
    ).read_text()

    for marker in (
        "torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.rms_norm_eps)",
        "torch.softmax(comb_logits, dim=-1) + eps",
        "for _ in range(iters - 1):",
        "comb.to(dtype).transpose(-1, -2)",
    ):
        assert marker in hca_text
    for marker in (
        "torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.rms_norm_eps)",
        "torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps",
    ):
        assert marker in mhc_text
    assert "config.hc_eps, config.rms_norm_eps" in model_text


def test_deepseek_v4_hash_moe_duplicate_routes_are_coalesced():
    """Keep this static: local developer Macs may not have torch installed."""
    from pathlib import Path

    dispatcher_py = (
        Path(__file__).resolve().parents[3]
        / "megatron/lite/primitive/modules/dispatcher.py"
    )
    text = dispatcher_py.read_text()

    for marker in (
        "_coalesced_routing_map_and_probs",
        "scatter_add_(1, topk_indices, topk_scores)",
        "DeepSeek-V4 hash routing may map multiple top-k slots",
        "routing_map = probs_2d != 0",
    ):
        assert marker in text


def test_deepseek_v4_ep_alltoall_uses_global_expert_count():
    """Keep this static: local developer Macs may not have torch installed."""
    from pathlib import Path

    dispatcher_py = (
        Path(__file__).resolve().parents[3]
        / "megatron/lite/primitive/modules/dispatcher.py"
    )
    text = dispatcher_py.read_text()

    assert "num_global_experts = tokens_per_expert.numel()" in text
    assert "self.ep_size * e" not in text
    assert "view(self.ep_size, e)" not in text


def test_deepseek_v4_mtp_contract_stays_inside_forward_call():
    """Keep MTP contract under Module.__call__ so FSDP2 hooks materialize params."""
    from pathlib import Path

    model_py = (
        Path(__file__).resolve().parents[3]
        / "megatron/lite/model/deepseek_v4/lite/model.py"
    )
    text = model_py.read_text()

    assert "return_contract: bool = False" in text
    assert "return source, self.contract(source)" in text
    assert "return_contract=True" in text
    assert "mtp_layer.contract(source)" not in text


def test_deepseek_v4_checkpoint_loader_has_legacy_alias_markers():
    """Keep this static: local developer Macs may not have torch installed."""
    from pathlib import Path

    checkpoint_py = (
        Path(__file__).resolve().parents[3]
        / "megatron/lite/model/deepseek_v4/lite/checkpoint.py"
    )
    text = checkpoint_py.read_text()

    for marker in (
        "_hf_name_groups_for_state_key",
        "_legacy_hf_names_for_state_key",
        "model.embed_tokens.weight",
        "model.layers.{index}",
        "self_attn.{_legacy_attention_sub",
        "q_a_proj",
        "kv_proj",
        "compressor.indexer.",
        "mlp.gate.",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.experts..{int(expert_id)}",
        "attn_hc",
    ):
        assert marker in text
