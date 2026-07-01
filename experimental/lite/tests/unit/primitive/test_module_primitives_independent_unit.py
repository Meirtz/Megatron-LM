# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import math
from types import MappingProxyType

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.modules.delta_mem import (
    DeltaMemAttentionCorrection,
    DeltaMemConfig,
    DeltaMemState,
    DeltaMemStateManager,
    DeltaMemStatePolicy,
    DeltaMemWriteProjection,
    delta_mem_read,
    delta_mem_write,
    normalize_delta_mem_config,
    normalize_delta_mem_state_policy,
)
from megatron.lite.primitive.modules.lora import (
    GroupedLinearLoRA,
    LinearLoRA,
    LoraConfig,
    SharedGroupedLinearLoRA,
    freeze_non_lora_params,
    lora_scale,
    normalize_lora_config,
    trainable_param_stats,
)

pytestmark = pytest.mark.mlite


def test_delta_mem_write_matches_paper_formula_and_readback():
    state = torch.tensor(
        [
            [0.20, -0.10, 0.05],
            [0.00, 0.30, -0.20],
            [0.10, 0.25, 0.40],
        ],
        dtype=torch.float64,
    )
    key = torch.tensor([0.50, -0.25, 0.75], dtype=torch.float64)
    value = torch.tensor([0.30, -0.40, 0.20], dtype=torch.float64)
    retention = torch.tensor([0.90, 0.50, 0.10], dtype=torch.float64)
    write_strength = torch.tensor([0.20, 0.60, 1.00], dtype=torch.float64)

    projected = state @ key
    expected = retention.unsqueeze(-1) * state + (
        write_strength * (value - projected)
    ).unsqueeze(-1) * key.unsqueeze(0)

    next_state = delta_mem_write(state, key, value, retention, write_strength)
    torch.testing.assert_close(next_state, expected)
    torch.testing.assert_close(delta_mem_read(next_state, key), next_state @ key)


def test_delta_mem_module_handles_batch_head_state_and_one_hot_write():
    module = DeltaMemState(rank=3)
    state = module.initial_state(2, 1, dtype=torch.float32)
    key = torch.zeros(2, 1, 3)
    key[..., 1] = 1.0
    value = torch.tensor([[[1.25, -0.50, 0.75]], [[0.25, 0.50, -1.25]]])
    retention = torch.zeros_like(value)
    write_strength = torch.ones_like(value)

    readout, next_state = module(state, key, value, retention, write_strength)

    assert next_state.shape == (2, 1, 3, 3)
    expected = torch.zeros_like(next_state)
    expected[..., :, 1] = value
    torch.testing.assert_close(next_state, expected)
    torch.testing.assert_close(readout, value)
    torch.testing.assert_close(module.read(next_state, key), value)


def test_delta_mem_rejects_mismatched_shapes_dtype_and_bad_rank():
    with pytest.raises(ValueError, match="rank must be positive"):
        DeltaMemState(rank=0)

    module = DeltaMemState(rank=2)
    state = module.initial_state(1)
    key = torch.ones(1, 2)
    value = torch.ones(1, 2)
    retention = torch.ones(1, 2)
    write_strength = torch.ones(1, 2)

    with pytest.raises(ValueError, match="must have shape"):
        module.write(state, torch.ones(2), value, retention, write_strength)
    with pytest.raises(TypeError, match="dtype"):
        module.write(state, key.double(), value, retention, write_strength)
    with pytest.raises(ValueError, match="trailing shape"):
        module.write(torch.zeros(1, 2, 3), key, value, retention, write_strength)


def test_delta_mem_write_projection_is_trainable_and_updates_state():
    projector = DeltaMemWriteProjection(hidden_size=5, rank=2)
    hidden = torch.randn(3, 5)
    state = torch.zeros(3, 2, 2)

    inputs = projector(hidden)
    assert inputs.key.shape == (3, 2)
    assert inputs.value.shape == (3, 2)
    assert inputs.retention.shape == (3, 2)
    assert inputs.write_strength.shape == (3, 2)
    assert torch.all(inputs.retention >= 0.0)
    assert torch.all(inputs.retention <= 1.0)
    assert torch.all(inputs.write_strength >= 0.0)
    assert torch.all(inputs.write_strength <= 1.0)
    assert sum(param.numel() for param in projector.parameters()) == 5 * 8 + 8

    readout, next_state = projector.write(state, hidden)
    expected_state = delta_mem_write(
        state,
        inputs.key,
        inputs.value,
        inputs.retention,
        inputs.write_strength,
        rank=2,
    )
    torch.testing.assert_close(next_state, expected_state)
    torch.testing.assert_close(readout, delta_mem_read(expected_state, inputs.key, rank=2))


def test_delta_mem_write_projection_rejects_bad_inputs():
    with pytest.raises(TypeError, match="hidden_size must be an integer"):
        DeltaMemWriteProjection(hidden_size=True, rank=2)
    with pytest.raises(ValueError, match="rank must be positive"):
        DeltaMemWriteProjection(hidden_size=4, rank=0)
    with pytest.raises(TypeError, match="bias must be a boolean"):
        DeltaMemWriteProjection(hidden_size=4, rank=2, bias="yes")

    projector = DeltaMemWriteProjection(hidden_size=4, rank=2)
    with pytest.raises(ValueError, match="trailing dimension"):
        projector(torch.ones(3, 5))
    with pytest.raises(TypeError, match="floating-point"):
        projector(torch.ones(3, 4, dtype=torch.long))


def test_delta_mem_config_normalizes_mapping_and_aliases():
    disabled = normalize_delta_mem_config(None)
    assert disabled == DeltaMemConfig()
    assert not disabled.enabled
    assert disabled.rank == 0

    cfg = normalize_delta_mem_config({"r": 4, "write_bias": False, "correction_bias": True})
    assert cfg.enabled
    assert cfg.rank == 4
    assert cfg.write_bias is False
    assert cfg.correction_bias is True
    assert cfg.state_policy == DeltaMemStatePolicy()
    assert normalize_delta_mem_config(cfg) is cfg

    disabled_ranked = normalize_delta_mem_config({"enabled": False, "rank": 2})
    assert not disabled_ranked.enabled
    assert disabled_ranked.rank == 2

    sequence_cfg = normalize_delta_mem_config(
        {"rank": 2, "state_policy": {"granularity": "sequence-state", "detach": True}}
    )
    assert sequence_cfg.state_policy.granularity == "sequence"
    assert sequence_cfg.state_policy.detach_after_write is True

    multi_cfg = normalize_delta_mem_config({"rank": 2, "mode": "MSW", "states": 4})
    assert multi_cfg.state_policy.granularity == "multi"
    assert multi_cfg.state_policy.num_states == 4

    with pytest.raises(TypeError, match="DeltaMem config must be"):
        normalize_delta_mem_config([("rank", 2)])
    with pytest.raises(ValueError, match="Unsupported DeltaMem config keys"):
        normalize_delta_mem_config({"rank": 2, "bad": True})
    with pytest.raises(ValueError, match="enabled=True requires a positive rank"):
        normalize_delta_mem_config({"enabled": True})
    with pytest.raises(TypeError, match="rank must be an integer"):
        normalize_delta_mem_config({"rank": True})
    with pytest.raises(ValueError, match="rank must be non-negative"):
        normalize_delta_mem_config({"enabled": False, "rank": -1})
    with pytest.raises(TypeError, match="write_bias must be a boolean"):
        normalize_delta_mem_config({"rank": 2, "write_bias": 1})
    with pytest.raises(ValueError, match="state_policy cannot be combined"):
        normalize_delta_mem_config({"rank": 2, "state_policy": "token", "granularity": "sequence"})


def test_delta_mem_state_policy_normalizes_aliases_and_rejects_bad_values():
    assert normalize_delta_mem_state_policy(None) == DeltaMemStatePolicy()
    assert normalize_delta_mem_state_policy("sequence-state").granularity == "sequence"
    multi = normalize_delta_mem_state_policy({"mode": "MSW", "states": 3, "detach": True})
    assert multi.granularity == "multi"
    assert multi.num_states == 3
    assert multi.detach_after_write is True

    with pytest.raises(TypeError, match="state policy must be"):
        normalize_delta_mem_state_policy(3)
    with pytest.raises(ValueError, match="Unsupported DeltaMem state policy keys"):
        normalize_delta_mem_state_policy({"granularity": "token", "bad": True})
    with pytest.raises(ValueError, match="one of token, sequence, or multi"):
        normalize_delta_mem_state_policy("bad")
    with pytest.raises(ValueError, match="num_states must be positive"):
        normalize_delta_mem_state_policy({"granularity": "multi", "num_states": 0})
    with pytest.raises(ValueError, match="require num_states=1"):
        normalize_delta_mem_state_policy({"granularity": "token", "num_states": 2})
    with pytest.raises(TypeError, match="detach_after_write must be a boolean"):
        normalize_delta_mem_state_policy({"granularity": "sequence", "detach_after_write": 1})


def test_delta_mem_token_state_manager_matches_vectorized_write_and_detach():
    manager = DeltaMemStateManager(rank=2, policy={"granularity": "token-state", "detach_after_write": True})
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    value = torch.tensor([[0.25, 0.75], [-0.50, 1.25]])
    retention = torch.zeros_like(value)
    write_strength = torch.ones_like(value)
    state = manager.initial_state_for_write(key)
    state.requires_grad_()

    out = manager(state, key, value, retention, write_strength)

    expected = delta_mem_write(state, key, value, retention, write_strength, rank=2)
    torch.testing.assert_close(out.next_state, expected)
    torch.testing.assert_close(out.readout, delta_mem_read(expected, key, rank=2))
    assert out.next_state.requires_grad is False
    assert out.state_history is None

    with pytest.raises(ValueError, match="does not accept state_indices"):
        manager(state.detach(), key, value, retention, write_strength, state_indices=torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="does not produce recurrent state history"):
        manager(state.detach(), key, value, retention, write_strength, return_state_history=True)


def test_delta_mem_sequence_state_manager_recurrently_updates_one_state():
    manager = DeltaMemStateManager(rank=2, policy="sequence")
    key = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    value = torch.tensor([[0.25, 0.75], [-0.50, 1.25], [1.50, -1.00]])
    retention = torch.ones_like(value)
    write_strength = torch.ones_like(value)
    state = manager.initial_state_for_write(key)

    out = manager(state, key, value, retention, write_strength, return_state_history=True)

    expected_final = torch.zeros(2, 2)
    expected_final[:, 0] = value[-1]
    torch.testing.assert_close(out.next_state, expected_final)
    torch.testing.assert_close(out.readout, value)
    assert out.state_history is not None
    assert out.state_history.shape == (3, 2, 2)
    torch.testing.assert_close(out.state_history[-1], expected_final)

    with pytest.raises(ValueError, match="does not accept state_indices"):
        manager(state, key, value, retention, write_strength, state_indices=torch.tensor([0, 0, 0]))


def test_delta_mem_multi_state_manager_routes_writes_to_state_banks():
    manager = DeltaMemStateManager(rank=2, policy={"granularity": "multi-state", "num_states": 2})
    key = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    value = torch.tensor([[0.25, 0.75], [-0.50, 1.25], [1.50, -1.00], [0.80, 0.40]])
    retention = torch.ones_like(value)
    write_strength = torch.ones_like(value)
    state = manager.initial_state_for_write(key)
    indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    out = manager(
        state,
        key,
        value,
        retention,
        write_strength,
        state_indices=indices,
        return_state_history=True,
    )

    expected = torch.zeros(2, 2, 2)
    expected[0, :, 1] = value[2]
    expected[1, :, 1] = value[3]
    torch.testing.assert_close(out.next_state, expected)
    torch.testing.assert_close(out.readout, value)
    assert out.state_history is not None
    assert out.state_history.shape == (4, 2, 2, 2)
    torch.testing.assert_close(out.state_indices, indices)

    default_out = manager(state, key, value, retention, write_strength)
    torch.testing.assert_close(default_out.next_state, expected)
    with pytest.raises(ValueError, match="out of range"):
        manager(state, key, value, retention, write_strength, state_indices=torch.tensor([0, 1, 2, 0]))


def test_delta_mem_attention_correction_emits_query_and_output_deltas():
    correction = DeltaMemAttentionCorrection(
        hidden_size=5,
        query_size=6,
        output_size=4,
        rank=2,
        correction_bias=False,
    )
    hidden = torch.randn(3, 5)
    state = torch.zeros(3, 2, 2)

    result = correction(hidden, state)
    expected_state = delta_mem_write(
        state,
        result.write_inputs.key,
        result.write_inputs.value,
        result.write_inputs.retention,
        result.write_inputs.write_strength,
        rank=2,
    )
    expected_readout = delta_mem_read(expected_state, result.write_inputs.key, rank=2)

    assert result.next_state.shape == (3, 2, 2)
    assert result.readout.shape == (3, 2)
    assert result.query_delta.shape == (3, 6)
    assert result.output_delta.shape == (3, 4)
    torch.testing.assert_close(result.next_state, expected_state)
    torch.testing.assert_close(result.readout, expected_readout)
    torch.testing.assert_close(result.query_delta, correction.query_delta_proj(expected_readout))
    torch.testing.assert_close(result.output_delta, correction.output_delta_proj(expected_readout))
    assert sum(param.numel() for param in correction.parameters()) == 5 * 8 + 8 + 2 * 6 + 2 * 4
    assert all(param.requires_grad for param in correction.parameters())


def test_delta_mem_attention_correction_supports_batch_heads_and_read_key_override():
    correction = DeltaMemAttentionCorrection(
        hidden_size=4,
        query_size=4,
        output_size=3,
        rank=2,
        write_bias=False,
        correction_bias=True,
    )
    hidden = torch.randn(2, 1, 4)
    state = torch.zeros(2, 1, 2, 2)
    read_key = torch.randn(2, 1, 2)

    result = correction(hidden, state, read_key=read_key)

    assert result.query_delta.shape == (2, 1, 4)
    assert result.output_delta.shape == (2, 1, 3)
    torch.testing.assert_close(
        result.readout,
        delta_mem_read(result.next_state, read_key, rank=2),
    )

    with pytest.raises(TypeError, match="correction_bias must be a boolean"):
        DeltaMemAttentionCorrection(hidden_size=4, query_size=4, output_size=3, rank=2, correction_bias="yes")
    with pytest.raises(ValueError, match="query_size must be positive"):
        DeltaMemAttentionCorrection(hidden_size=4, query_size=0, output_size=3, rank=2)
    with pytest.raises(ValueError, match="must have shape"):
        correction(hidden, torch.zeros(2, 2, 2), read_key=read_key)


def test_delta_mem_attention_correction_uses_sequence_state_policy():
    correction = DeltaMemAttentionCorrection(
        hidden_size=4,
        query_size=4,
        output_size=3,
        rank=2,
        state_policy="sequence-state",
    )
    hidden = torch.randn(3, 2, 4)

    state = correction.initial_state(hidden)
    result = correction(hidden, state, return_state_history=True)

    assert state.shape == (2, 2, 2)
    assert result.readout.shape == (3, 2, 2)
    assert result.query_delta.shape == (3, 2, 4)
    assert result.output_delta.shape == (3, 2, 3)
    assert result.next_state.shape == (2, 2, 2)
    assert result.state_history is not None
    assert result.state_history.shape == (3, 2, 2, 2)
    with pytest.raises(ValueError, match="does not accept state_indices"):
        correction(hidden, state, state_indices=torch.tensor([0, 0, 0]))


def test_delta_mem_attention_correction_uses_multi_state_policy_indices():
    correction = DeltaMemAttentionCorrection(
        hidden_size=4,
        query_size=4,
        output_size=3,
        rank=2,
        state_policy={"granularity": "multi-state", "num_states": 2},
    )
    hidden = torch.randn(4, 1, 4)
    indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    state = correction.initial_state(hidden)
    result = correction(hidden, state, state_indices=indices, return_state_history=True)

    assert state.shape == (2, 1, 2, 2)
    assert result.readout.shape == (4, 1, 2)
    assert result.query_delta.shape == (4, 1, 4)
    assert result.output_delta.shape == (4, 1, 3)
    assert result.next_state.shape == (2, 1, 2, 2)
    assert result.state_history is not None
    assert result.state_history.shape == (4, 2, 1, 2, 2)
    torch.testing.assert_close(result.state_indices, indices)


def test_lora_config_aliases_and_trainable_param_accounting():
    cfg = normalize_lora_config({"enabled": True, "rank": 2, "alpha": 6, "targets": ["qkv", "fc2"]})

    assert cfg.enabled
    assert cfg.scale == 3.0
    assert cfg.targets() == {"linear_qkv", "linear_fc2"}
    assert cfg.targets_module("qkv")
    assert cfg.targets_module("linear_fc2")

    peft_cfg = normalize_lora_config(
        {
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.25,
            "target_modules": "gate_proj",
            "use_rslora": True,
            "peft_type": "LORA",
            "task_type": "causal_lm",
            "base_model_name_or_path": "local/tiny",
            "inference_mode": True,
            "bias": "none",
            "fan_in_fan_out": False,
            "init_lora_weights": "olora_tail",
            "modules_to_save": [],
        }
    )
    assert peft_cfg.rank == 4
    assert peft_cfg.alpha == 8
    assert peft_cfg.dropout == 0.25
    assert peft_cfg.use_rslora is True
    assert peft_cfg.scale == 4.0
    assert peft_cfg.targets() == {"linear_fc1"}
    assert peft_cfg.targets_module("up_proj")

    mapping_cfg = normalize_lora_config(
        MappingProxyType({"enabled": True, "rank": 2, "target_modules": "linear_qkv"})
    )
    assert mapping_cfg.enabled
    assert mapping_cfg.rank == 2
    assert mapping_cfg.target_modules == ("linear_qkv",)
    with pytest.raises(TypeError, match="LoRA config must be LoraConfig, mapping, or None"):
        normalize_lora_config([("rank", 2)])
    with pytest.raises(TypeError, match="target_modules must be a string or sequence"):
        normalize_lora_config(
            {
                "rank": 2,
                "target_modules": MappingProxyType({"linear_qkv": True}),
            }
        )

    assert not normalize_lora_config({"enabled": False, "rank": 8}).enabled
    assert not normalize_lora_config({"enabled": False}).enabled
    with pytest.raises(TypeError, match="LoRA config enabled must be a boolean"):
        normalize_lora_config({"enabled": "false", "rank": 2})
    with pytest.raises(TypeError, match="LoRA config enabled must be a boolean"):
        normalize_lora_config({"enabled": 0, "rank": 2})
    with pytest.raises(TypeError, match="LoRA config rank must be an integer"):
        normalize_lora_config({"enabled": False, "rank": "bad"})
    with pytest.raises(ValueError, match="enabled=True requires a positive rank"):
        normalize_lora_config({"enabled": True})
    with pytest.raises(ValueError, match="LoRA config rank must be positive"):
        normalize_lora_config({"enabled": True, "rank": 0})
    with pytest.raises(TypeError, match="base_model_name_or_path must be a string"):
        normalize_lora_config({"rank": 2, "base_model_name_or_path": 123})
    with pytest.raises(ValueError, match="peft_type='BAD'"):
        normalize_lora_config({"rank": 2, "peft_type": "BAD"})
    with pytest.raises(TypeError, match="peft_type must be a string"):
        normalize_lora_config({"rank": 2, "peft_type": 123})
    with pytest.raises(TypeError, match="task_type must be a string"):
        normalize_lora_config({"rank": 2, "task_type": ["CAUSAL_LM"]})
    with pytest.raises(ValueError, match="task_type='SEQ_CLS'"):
        normalize_lora_config({"rank": 2, "task_type": "SEQ_CLS"})
    with pytest.raises(TypeError, match="inference_mode must be a boolean"):
        normalize_lora_config({"rank": 2, "inference_mode": "true"})
    with pytest.raises(ValueError, match="bias='all'"):
        normalize_lora_config({"rank": 2, "bias": "all"})
    with pytest.raises(TypeError, match="bias must be a string"):
        normalize_lora_config({"rank": 2, "bias": 123})
    with pytest.raises(TypeError, match="fan_in_fan_out must be a boolean"):
        normalize_lora_config({"rank": 2, "fan_in_fan_out": "false"})
    with pytest.raises(ValueError, match="fan_in_fan_out=True"):
        normalize_lora_config({"rank": 2, "fan_in_fan_out": True})
    with pytest.raises(ValueError, match="modules_to_save"):
        normalize_lora_config({"rank": 2, "modules_to_save": ["lm_head"]})
    with pytest.raises(ValueError, match="modules_to_save"):
        normalize_lora_config({"rank": 2, "modules_to_save": ("lm_head",)})
    normalize_lora_config({"rank": 2, "modules_to_save": ()})
    with pytest.raises(TypeError, match="init_lora_weights"):
        normalize_lora_config({"rank": 2, "init_lora_weights": {"mode": "olora_tail"}})
    with pytest.raises(TypeError, match="LoRA config"):
        normalize_lora_config(object())
    with pytest.raises(TypeError, match="LoRA config rank must be an integer"):
        normalize_lora_config({"r": "4"})
    with pytest.raises(TypeError, match="LoRA config rank must be an integer"):
        LoraConfig(rank=True)
    with pytest.raises(TypeError, match="LoRA config alpha must be a finite number"):
        normalize_lora_config({"rank": 2, "lora_alpha": "8"})
    with pytest.raises(TypeError, match="LoRA config dropout must be a finite number"):
        normalize_lora_config({"rank": 2, "lora_dropout": [0.0]})
    with pytest.raises(ValueError, match="LoRA config dropout must be between 0 and 1"):
        normalize_lora_config({"rank": 2, "lora_dropout": -0.1})
    with pytest.raises(ValueError, match="LoRA config dropout must be between 0 and 1"):
        normalize_lora_config({"rank": 2, "lora_dropout": 1.5})
    with pytest.raises(TypeError, match="LoRA config use_rslora must be a boolean"):
        normalize_lora_config({"rank": 2, "use_rslora": 1})
    with pytest.raises(ValueError, match="target_modules must be non-empty"):
        normalize_lora_config({"rank": 2, "target_modules": []})
    with pytest.raises(ValueError, match="target_modules entries must be non-empty"):
        normalize_lora_config({"rank": 2, "target_modules": ["linear_qkv", " "]})
    disabled_empty_targets = normalize_lora_config(
        {"enabled": False, "rank": 2, "target_modules": []}
    )
    assert not disabled_empty_targets.enabled
    assert disabled_empty_targets.target_modules == ()
    with pytest.raises(TypeError, match="LoRA alpha must be a finite number"):
        lora_scale(2, alpha="4")
    with pytest.raises(ValueError, match="LoRA alpha must be finite"):
        lora_scale(2, alpha=float("inf"))
    with pytest.raises(TypeError, match="LoRA use_rslora must be a boolean"):
        lora_scale(2, alpha=4, use_rslora="false")
    assert lora_scale(0, alpha=4) == 0.0
    with pytest.raises(TypeError, match="LoRA alpha must be a finite number"):
        lora_scale(0, alpha="4")
    with pytest.raises(ValueError, match="LoRA alpha must be finite"):
        lora_scale(0, alpha=float("inf"))
    with pytest.raises(TypeError, match="LoRA use_rslora must be a boolean"):
        lora_scale(0, use_rslora="false")
    with pytest.raises(TypeError, match="LoRA rank must be an integer"):
        lora_scale(True)
    with pytest.raises(TypeError, match="LoRA rank must be an integer"):
        lora_scale("2")
    with pytest.raises(ValueError, match="LoRA rank must be non-negative"):
        lora_scale(-1)
    with pytest.raises(TypeError, match="LinearLoRA rank must be an integer"):
        LinearLoRA(2, 2, rank=True)
    with pytest.raises(TypeError, match="LinearLoRA rank must be an integer"):
        LinearLoRA(2, 2, rank=1.0)
    with pytest.raises(ValueError, match="LinearLoRA rank must be positive"):
        LinearLoRA(2, 2, rank=0)
    with pytest.raises(TypeError, match="LinearLoRA in_features must be an integer"):
        LinearLoRA(True, 2, rank=1)
    with pytest.raises(TypeError, match="LinearLoRA out_features must be an integer"):
        LinearLoRA(2, True, rank=1)
    with pytest.raises(ValueError, match="LinearLoRA in_features must be positive"):
        LinearLoRA(0, 2, rank=1)
    with pytest.raises(TypeError, match="LoRA rank partition size must be an integer"):
        LinearLoRA(2, 2, rank=1, rank_partitioned_a=True, rank_partition_size=True)
    with pytest.raises(TypeError, match="LoRA output partition size must be an integer"):
        LinearLoRA(2, 2, rank=1, output_partitioned_b=True, output_partition_size=True)
    with pytest.raises(TypeError, match="LinearLoRA tp_rank must be an integer"):
        LinearLoRA(2, 2, rank=1, tp_rank=True)
    with pytest.raises(ValueError, match="LinearLoRA tp_rank must be non-negative"):
        LinearLoRA(2, 2, rank=1, tp_rank=-1)
    with pytest.raises(TypeError, match="LinearLoRA alpha must be a finite number"):
        LinearLoRA(2, 2, rank=1, alpha="4")
    with pytest.raises(ValueError, match="LinearLoRA dropout must be between 0 and 1"):
        LinearLoRA(2, 2, rank=1, dropout=1.5)
    with pytest.raises(TypeError, match="LinearLoRA use_rslora must be a boolean"):
        LinearLoRA(2, 2, rank=1, use_rslora="false")
    for flag in (
        "sequence_parallel_input",
        "row_parallel_output",
        "sequence_parallel_scatter_output",
        "rank_partitioned_a",
        "input_parallel_reduce",
        "output_partitioned_b",
        "a_tensor_model_parallel",
        "b_tensor_model_parallel",
    ):
        with pytest.raises(TypeError, match=f"LinearLoRA {flag} must be a boolean"):
            LinearLoRA(2, 2, rank=1, **{flag: "false"})
    with pytest.raises(TypeError, match="GroupedLinearLoRA rank must be an integer"):
        GroupedLinearLoRA(1, 2, 2, rank=True)
    with pytest.raises(ValueError, match="GroupedLinearLoRA rank must be positive"):
        GroupedLinearLoRA(1, 2, 2, rank=0)
    with pytest.raises(TypeError, match="GroupedLinearLoRA num_local_experts must be an integer"):
        GroupedLinearLoRA(True, 2, 2, rank=1)
    with pytest.raises(ValueError, match="GroupedLinearLoRA num_local_experts must be positive"):
        GroupedLinearLoRA(0, 2, 2, rank=1)
    with pytest.raises(TypeError, match="GroupedLinearLoRA in_features must be an integer"):
        GroupedLinearLoRA(1, True, 2, rank=1)
    with pytest.raises(ValueError, match="GroupedLinearLoRA out_features must be positive"):
        GroupedLinearLoRA(1, 2, 0, rank=1)
    with pytest.raises(ValueError, match="GroupedLinearLoRA alpha must be finite"):
        GroupedLinearLoRA(1, 2, 2, rank=1, alpha=float("nan"))
    with pytest.raises(TypeError, match="GroupedLinearLoRA dropout must be a finite number"):
        GroupedLinearLoRA(1, 2, 2, rank=1, dropout="0.1")
    with pytest.raises(TypeError, match="GroupedLinearLoRA use_rslora must be a boolean"):
        GroupedLinearLoRA(1, 2, 2, rank=1, use_rslora="false")
    with pytest.raises(TypeError, match="SharedGroupedLinearLoRA rank must be an integer"):
        SharedGroupedLinearLoRA(1, 2, 2, rank=True)
    with pytest.raises(ValueError, match="SharedGroupedLinearLoRA rank must be positive"):
        SharedGroupedLinearLoRA(1, 2, 2, rank=0)
    with pytest.raises(TypeError, match="SharedGroupedLinearLoRA num_local_experts must be an integer"):
        SharedGroupedLinearLoRA(True, 2, 2, rank=1)
    with pytest.raises(ValueError, match="SharedGroupedLinearLoRA num_local_experts must be positive"):
        SharedGroupedLinearLoRA(0, 2, 2, rank=1)
    with pytest.raises(TypeError, match="SharedGroupedLinearLoRA in_features must be an integer"):
        SharedGroupedLinearLoRA(1, 1.0, 2, rank=1)
    with pytest.raises(TypeError, match="SharedGroupedLinearLoRA out_features must be an integer"):
        SharedGroupedLinearLoRA(1, 2, True, rank=1)
    with pytest.raises(TypeError, match="SharedGroupedLinearLoRA alpha must be a finite number"):
        SharedGroupedLinearLoRA(1, 2, 2, rank=1, alpha="4")
    with pytest.raises(ValueError, match="SharedGroupedLinearLoRA dropout must be between 0 and 1"):
        SharedGroupedLinearLoRA(1, 2, 2, rank=1, dropout=-0.1)
    with pytest.raises(TypeError, match="SharedGroupedLinearLoRA use_rslora must be a boolean"):
        SharedGroupedLinearLoRA(1, 2, 2, rank=1, use_rslora="false")

    class TinyAdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(3, 2)
            self.lora_adapter = nn.Linear(3, 2)

    model = TinyAdapterModel()
    stats = freeze_non_lora_params(model)

    assert stats["lora_tensors"] == 2
    assert stats["frozen_tensors"] == 2
    assert not model.base.weight.requires_grad
    assert model.lora_adapter.weight.requires_grad
    assert trainable_param_stats(model) == {
        "trainable_tensors": 2,
        "trainable_numel": model.lora_adapter.weight.numel() + model.lora_adapter.bias.numel(),
    }


def test_rslora_sqrt_rank_alpha_keeps_scale_constant_across_ranks():
    expected_scale = 2.0

    for rank in (1, 4, 16, 64):
        alpha = expected_scale * math.sqrt(rank)
        assert math.isclose(
            lora_scale(rank, alpha=alpha, use_rslora=True),
            expected_scale,
            rel_tol=0.0,
            abs_tol=1e-6,
        )

    assert not math.isclose(
        lora_scale(16, alpha=expected_scale * math.sqrt(16), use_rslora=False),
        expected_scale,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_linear_lora_forward_backward_matches_low_rank_delta():
    layer = LinearLoRA(3, 2, rank=2, alpha=4, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        layer.lora_b.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    output = layer(x)

    torch.testing.assert_close(output, torch.tensor([[10.0, 22.0]]))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[8.0, 12.0, 0.0]]))


def test_grouped_lora_respects_per_expert_splits():
    layer = GroupedLinearLoRA(2, 2, 2, rank=1, alpha=1, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
        layer.lora_b.copy_(torch.tensor([[[2.0], [3.0]], [[5.0], [7.0]]]))

    x = torch.tensor([[2.0, 9.0], [4.0, 1.0], [6.0, 3.0]])
    output = layer(x, [1, 2])

    torch.testing.assert_close(output, torch.tensor([[4.0, 6.0], [5.0, 7.0], [15.0, 21.0]]))
    with pytest.raises(ValueError, match="expected 2 splits"):
        layer(x, [3])
    with pytest.raises(ValueError, match="split sizes sum to 2"):
        layer(x, [2, 0])
    with pytest.raises(ValueError, match=r"splits\[1\] must be non-negative"):
        layer(x, [2, -1])
    with pytest.raises(TypeError, match=r"splits\[0\] must be an integer"):
        layer(x, [True, 2])


def test_shared_grouped_lora_uses_one_adapter_for_all_experts():
    layer = SharedGroupedLinearLoRA(2, 2, 2, rank=1, alpha=2, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[1.0, -1.0]]))
        layer.lora_b.copy_(torch.tensor([[2.0], [3.0]]))

    x = torch.tensor([[3.0, 1.0], [4.0, 7.0]])
    output = layer(x, [1, 1])

    torch.testing.assert_close(output, torch.tensor([[8.0, 12.0], [-12.0, -18.0]]))
    with pytest.raises(ValueError, match="split sizes sum to 1"):
        layer(x, [1, 0])
    with pytest.raises(ValueError, match=r"splits\[1\] must be non-negative"):
        layer(x, [3, -1])
    with pytest.raises(TypeError, match=r"splits\[0\] must be an integer"):
        layer(x, [True, 1])


def test_mrope_interleaves_text_height_and_width_sections():
    from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding

    base = torch.arange(3 * 2 * 6, dtype=torch.float32).reshape(3, 2, 6)

    interleaved = MultimodalRotaryEmbedding._apply_interleaved_mrope(base, mrope_section=[1, 1, 1])

    expected = base[0].clone()
    expected[..., 1] = base[1, ..., 1]
    expected[..., 2] = base[2, ..., 2]
    torch.testing.assert_close(interleaved, expected)


def test_mtp_aux_loss_scaler_threads_independent_gradient(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    MTPLossAutoScaler.set_loss_scale(torch.tensor(0.125))
    output = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    mtp_loss = torch.tensor(4.0, requires_grad=True)

    MTPLossAutoScaler.apply(output * 3.0, mtp_loss).sum().backward()

    torch.testing.assert_close(output.grad, torch.full_like(output, 3.0))
    torch.testing.assert_close(mtp_loss.grad, torch.tensor(0.125))
    MTPLossAutoScaler.main_loss_backward_scale = 1.0


def test_gated_delta_static_helpers_are_finite_and_shape_stable(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet

    alpha = torch.tensor([[[0.0, 1.0], [-1.0, 2.0]]])
    beta = torch.tensor([[[0.0, 2.0], [-2.0, 4.0]]])

    g, beta_sigmoid = GatedDeltaNet._compute_g_and_beta(torch.zeros(2), torch.ones(2), alpha, beta)

    assert g.shape == alpha.shape
    assert beta_sigmoid.shape == beta.shape
    assert torch.isfinite(g).all()
    assert torch.isfinite(beta_sigmoid).all()
    assert torch.all(g < 0)
