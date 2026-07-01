"""Runtime gates for GLM5 LoRA adapter import/export and OLoRA-tail init."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


class _FakeDTensorForExport:
    def __init__(self, full_tensor, local_tensor):
        self._full_tensor = full_tensor
        self._local_tensor = local_tensor
        self.full_tensor_calls = 0
        self.to_local_calls = 0

    @property
    def shape(self):
        return self._full_tensor.shape

    @property
    def ndim(self):
        return self._full_tensor.ndim

    def detach(self):
        return self

    def full_tensor(self):
        self.full_tensor_calls += 1
        return self._full_tensor

    def to_local(self):
        self.to_local_calls += 1
        return self._local_tensor


def _init_single_rank_cpu_mesh(torch):
    pytest.importorskip("torch.distributed.tensor")
    from torch.distributed import DeviceMesh

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    init_path = temp_file.name
    temp_file.close()
    created = False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_path}",
            rank=0,
            world_size=1,
        )
        created = True
    return DeviceMesh("cpu", [0]), created, init_path


def _cleanup_single_rank_cpu_mesh(torch, created, init_path):
    try:
        if created and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    finally:
        Path(init_path).unlink(missing_ok=True)


def _dtensor_param(torch, tensor, mesh, *, shard_dim=0):
    from torch.distributed.tensor import Shard, distribute_tensor

    return torch.nn.Parameter(
        distribute_tensor(tensor.contiguous(), mesh, placements=[Shard(shard_dim)])
    )


def test_lora_adapter_helpers_import_without_transformer_engine():
    pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite import lora_adapter as glm5_lora_adapter
    from megatron.lite.model.qwen3_moe.lite import lora_adapter as qwen_lora_adapter

    assert hasattr(glm5_lora_adapter, "_validate_adapter_state_is_lora_only")
    assert hasattr(qwen_lora_adapter, "_validate_adapter_state_is_lora_only")


def test_lora_adapter_state_guard_rejects_base_weight_keys():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        _validate_adapter_state_is_lora_only as validate_glm5,
    )
    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _validate_adapter_state_is_lora_only as validate_qwen,
    )

    glm5_lora_only = {
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.empty(1, 1),
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.empty(1, 1),
    }
    qwen_lora_only = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.empty(1, 1),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.empty(1, 1),
    }
    validate_glm5(glm5_lora_only)
    validate_qwen(qwen_lora_only)

    with pytest.raises(ValueError, match="adapter state is empty"):
        validate_glm5({})
    with pytest.raises(ValueError, match="adapter state is empty"):
        validate_qwen({})
    with pytest.raises(ValueError, match="non-adapter tensor keys"):
        validate_glm5({"base_model.model.model.layers.0.self_attn.q_a_proj.weight": torch.empty(1)})
    with pytest.raises(ValueError, match="non-adapter tensor keys"):
        validate_qwen({"base_model.model.model.layers.0.self_attn.q_proj.weight": torch.empty(1)})
    with pytest.raises(ValueError, match="unsupported adapter tensor keys"):
        validate_glm5(
            {
                "base_model.model.model.layers.0.self_attn.fake_proj.lora_A.weight": torch.empty(
                    1, 1
                )
            }
        )
    with pytest.raises(ValueError, match="unsupported adapter tensor keys"):
        validate_glm5(
            {
                "base_model.model.model.layers.0.self_attn.indexer.lora_A.weight": torch.empty(
                    1, 1
                )
            }
        )
    with pytest.raises(ValueError, match="unsupported adapter tensor keys"):
        validate_qwen(
            {
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.empty(
                    1, 1
                )
            }
        )
    with pytest.raises(ValueError, match="unsupported adapter tensor keys"):
        validate_qwen(
            {
                "base_model.model.model.layers.0.mlp.shared_experts.gate_proj.lora_A.weight": torch.empty(
                    1, 1
                )
            }
        )
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        validate_glm5(
            {"base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": [[1.0]]}
        )
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        validate_qwen(
            {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": [[1.0]]}
        )
    with pytest.raises(TypeError, match="floating-point dtype"):
        validate_glm5(
            {
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.zeros(
                    1, 1, dtype=torch.int64
                )
            }
        )
    with pytest.raises(TypeError, match="floating-point dtype"):
        validate_qwen(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(
                    1, 1, dtype=torch.bool
                )
            }
        )
    with pytest.raises(TypeError, match="floating-point dtype"):
        validate_qwen(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(
                    1, 1, dtype=torch.complex64
                )
            }
        )
    with pytest.raises(ValueError, match="must be 2-D"):
        validate_glm5(
            {"base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.empty(1)}
        )
    with pytest.raises(ValueError, match="must be 2-D"):
        validate_qwen(
            {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.empty(1)}
        )
    with pytest.raises(ValueError, match="positive dimensions"):
        validate_glm5(
            {
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.empty(
                    0, 1
                )
            }
        )
    with pytest.raises(ValueError, match="positive dimensions"):
        validate_qwen(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.empty(
                    1, 0
                )
            }
        )
    with pytest.raises(ValueError, match="finite values"):
        validate_glm5(
            {
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.tensor(
                    [[float("nan")]]
                )
            }
        )
    with pytest.raises(ValueError, match="finite values"):
        validate_qwen(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor(
                    [[float("inf")]]
                )
            }
        )
    with pytest.raises(ValueError, match="missing paired LoRA tensor keys"):
        validate_glm5(
            {
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.zeros(
                    1, 1
                )
            }
        )
    with pytest.raises(ValueError, match="missing paired LoRA tensor keys"):
        validate_qwen(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(
                    1, 1
                )
            }
        )
    def expect_single_pair_rank_mismatch(validate, state):
        try:
            validate(state)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected inconsistent paired LoRA ranks")
        assert "inconsistent paired LoRA ranks" in message
        assert message.count("lora_A rows=2, lora_B columns=3") == 1

    expect_single_pair_rank_mismatch(
        validate_glm5,
        {
            "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.zeros(
                2, 4
            ),
            "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.zeros(
                4, 3
            ),
        },
    )
    expect_single_pair_rank_mismatch(
        validate_qwen,
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4, 3),
        },
    )
    with pytest.raises(ValueError, match="inconsistent LoRA ranks"):
        validate_glm5(
            {
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.zeros(
                    2, 4
                ),
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.zeros(
                    4, 2
                ),
                "base_model.model.model.layers.0.self_attn.q_b_proj.lora_A.weight": torch.zeros(
                    3, 4
                ),
                "base_model.model.model.layers.0.self_attn.q_b_proj.lora_B.weight": torch.zeros(
                    4, 3
                ),
            }
        )
    with pytest.raises(ValueError, match="inconsistent LoRA ranks"):
        validate_qwen(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(
                    2, 4
                ),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(
                    4, 2
                ),
                "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.zeros(
                    3, 4
                ),
                "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": torch.zeros(
                    4, 3
                ),
            }
        )


def test_glm5_lora_adapter_export_materializes_full_tensor_before_local_shard():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import export_lora_adapter_state
    from megatron.lite.primitive.parallel import ParallelState

    class FakeDTensor:
        def __init__(self, full_tensor, local_tensor):
            self._full_tensor = full_tensor
            self._local_tensor = local_tensor
            self.full_tensor_calls = 0
            self.to_local_calls = 0

        def detach(self):
            return self

        def full_tensor(self):
            self.full_tensor_calls += 1
            return self._full_tensor

        def to_local(self):
            self.to_local_calls += 1
            return self._local_tensor

    full_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    full_b = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    local_a = full_a[:1].clone()
    local_b = full_b[:2].clone()
    fake_a = FakeDTensor(full_a, local_a)
    fake_b = FakeDTensor(full_b, local_b)
    dsa = SimpleNamespace(
        q_a_lora=SimpleNamespace(lora_a=fake_a, lora_b=fake_b),
        q_b_lora=None,
        kv_a_lora=None,
        kv_b_lora=None,
        o_lora=None,
    )
    layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=dsa),
        mlp=None,
        moe=None,
    )
    chunk = SimpleNamespace(layers=[layer], mtp=None)
    model_cfg = SimpleNamespace(num_hidden_layers=1)

    state = export_lora_adapter_state([chunk], model_cfg, ParallelState())

    key_a = "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"
    key_b = "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight"
    torch.testing.assert_close(state[key_a], full_a)
    torch.testing.assert_close(state[key_b], full_b)
    assert fake_a.full_tensor_calls == 1
    assert fake_b.full_tensor_calls == 1
    assert fake_a.to_local_calls == 0
    assert fake_b.to_local_calls == 0


def test_qwen3_lora_adapter_export_materializes_full_tensor_before_local_shard():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _attn_key,
        export_lora_adapter_state,
    )
    from megatron.lite.primitive.parallel import ParallelState

    class FakeDTensor:
        def __init__(self, full_tensor, local_tensor):
            self._full_tensor = full_tensor
            self._local_tensor = local_tensor
            self.full_tensor_calls = 0
            self.to_local_calls = 0

        @property
        def shape(self):
            return self._full_tensor.shape

        def detach(self):
            return self

        def full_tensor(self):
            self.full_tensor_calls += 1
            return self._full_tensor

        def to_local(self):
            self.to_local_calls += 1
            return self._local_tensor

    full_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    full_b = torch.tensor(
        [
            [10.0, 11.0],
            [12.0, 13.0],
            [20.0, 21.0],
            [22.0, 23.0],
            [30.0, 31.0],
            [32.0, 33.0],
        ]
    )
    local_a = full_a[:1].clone()
    local_b = full_b[:2].clone()
    fake_a = FakeDTensor(full_a, local_a)
    fake_b = FakeDTensor(full_b, local_b)
    qkv_lora = SimpleNamespace(
        lora_a=fake_a,
        lora_b=fake_b,
        rank=2,
        rank_partitioned_a=False,
        output_partitioned_b=False,
    )
    chunk = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_idx=0,
                attn=SimpleNamespace(qkv_lora=qkv_lora, proj_lora=None),
                moe=SimpleNamespace(
                    experts=SimpleNamespace(
                        fc1_lora=None,
                        fc2_lora=None,
                        num_local_experts=0,
                    )
                ),
            )
        ]
    )
    model_cfg = SimpleNamespace(
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=0,
    )

    state = export_lora_adapter_state([chunk], model_cfg, ParallelState())

    torch.testing.assert_close(state[_attn_key(0, "q_proj", "lora_A")], full_a)
    torch.testing.assert_close(state[_attn_key(0, "k_proj", "lora_A")], full_a)
    torch.testing.assert_close(state[_attn_key(0, "v_proj", "lora_A")], full_a)
    torch.testing.assert_close(state[_attn_key(0, "q_proj", "lora_B")], full_b[:2])
    torch.testing.assert_close(state[_attn_key(0, "k_proj", "lora_B")], full_b[2:4])
    torch.testing.assert_close(state[_attn_key(0, "v_proj", "lora_B")], full_b[4:])
    assert fake_a.full_tensor_calls == 1
    assert fake_b.full_tensor_calls == 1
    assert fake_a.to_local_calls == 0
    assert fake_b.to_local_calls == 0


def test_glm5_expert_lora_adapter_export_materializes_full_tensor_before_local_shard():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        _expert_key,
        export_lora_adapter_state,
    )
    from megatron.lite.primitive.parallel import ParallelState

    full_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    full_b = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 10
    fake_a = _FakeDTensorForExport(full_a, full_a[:1].clone())
    fake_b = _FakeDTensorForExport(full_b, full_b[:2].clone())
    fc1_lora = SimpleNamespace(lora_a=fake_a, lora_b=fake_b, shared_across_experts=True)
    experts = SimpleNamespace(num_local_experts=2, fc1_lora=fc1_lora, fc2_lora=None)
    layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=_fake_dsa_without_lora()),
        mlp=None,
        moe=SimpleNamespace(
            experts=experts,
            shared_expert=SimpleNamespace(gate_up_lora=None, down_lora=None),
        ),
    )
    chunk = SimpleNamespace(layers=[layer], mtp=None)
    model_cfg = SimpleNamespace(num_hidden_layers=1, num_experts=2)

    state = export_lora_adapter_state([chunk], model_cfg, ParallelState())

    torch.testing.assert_close(state[_expert_key(0, 0, "gate_proj", "lora_A")], full_a)
    torch.testing.assert_close(state[_expert_key(0, 1, "up_proj", "lora_A")], full_a)
    torch.testing.assert_close(state[_expert_key(0, 0, "gate_proj", "lora_B")], full_b[:2])
    torch.testing.assert_close(state[_expert_key(0, 1, "up_proj", "lora_B")], full_b[2:])
    assert fake_a.full_tensor_calls == 1
    assert fake_b.full_tensor_calls == 1
    assert fake_a.to_local_calls == 0
    assert fake_b.to_local_calls == 0


def test_qwen3_expert_lora_adapter_export_materializes_full_tensor_before_local_shard():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _expert_key,
        export_lora_adapter_state,
    )
    from megatron.lite.primitive.parallel import ParallelState

    full_a = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    full_b = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2) + 100
    fake_a = _FakeDTensorForExport(full_a, full_a[:1].clone())
    fake_b = _FakeDTensorForExport(full_b, full_b[:1].clone())
    fc1_lora = SimpleNamespace(lora_a=fake_a, lora_b=fake_b, shared_across_experts=False)
    chunk = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_idx=0,
                attn=SimpleNamespace(qkv_lora=None, proj_lora=None),
                moe=SimpleNamespace(
                    experts=SimpleNamespace(
                        fc1_lora=fc1_lora,
                        fc2_lora=None,
                        num_local_experts=2,
                    )
                ),
            )
        ]
    )
    model_cfg = SimpleNamespace(
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=2,
    )

    state = export_lora_adapter_state([chunk], model_cfg, ParallelState())

    torch.testing.assert_close(state[_expert_key(0, 0, "gate_proj", "lora_A")], full_a[0])
    torch.testing.assert_close(state[_expert_key(0, 1, "up_proj", "lora_A")], full_a[1])
    torch.testing.assert_close(state[_expert_key(0, 0, "gate_proj", "lora_B")], full_b[0, :3])
    torch.testing.assert_close(state[_expert_key(0, 1, "up_proj", "lora_B")], full_b[1, 3:])
    assert fake_a.full_tensor_calls == 1
    assert fake_b.full_tensor_calls == 1
    assert fake_a.to_local_calls == 0
    assert fake_b.to_local_calls == 0


def test_glm5_load_lora_adapter_state_copies_into_dtensor_lora_params_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        _attn_key,
        load_lora_adapter_state,
    )
    from megatron.lite.primitive.parallel import ParallelState

    mesh, created, temp_file = _init_single_rank_cpu_mesh(torch)
    try:
        key_a = _attn_key(0, "q_a_proj", "lora_A")
        key_b = _attn_key(0, "q_a_proj", "lora_B")
        full_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        full_b = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
        lora = SimpleNamespace(
            lora_a=_dtensor_param(torch, torch.zeros_like(full_a), mesh),
            lora_b=_dtensor_param(torch, torch.zeros_like(full_b), mesh),
        )
        dsa = SimpleNamespace(
            q_a_lora=lora,
            q_b_lora=None,
            kv_a_lora=None,
            kv_b_lora=None,
            o_lora=None,
        )
        layer = SimpleNamespace(
            layer_idx=0,
            self_attention=SimpleNamespace(self_attention=dsa),
            mlp=None,
            moe=None,
        )
        chunk = SimpleNamespace(layers=[layer], mtp=None)

        result = load_lora_adapter_state(
            [chunk],
            {key_a: full_a, key_b: full_b},
            SimpleNamespace(num_hidden_layers=1),
            ParallelState(),
        )

        assert result["loaded_tensors"] == 2
        torch.testing.assert_close(lora.lora_a.full_tensor(), full_a)
        torch.testing.assert_close(lora.lora_b.full_tensor(), full_b)
    finally:
        _cleanup_single_rank_cpu_mesh(torch, created, temp_file)


def test_qwen3_load_lora_adapter_state_copies_into_dtensor_lora_params_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _attn_key,
        load_lora_adapter_state,
    )
    from megatron.lite.primitive.parallel import ParallelState

    mesh, created, temp_file = _init_single_rank_cpu_mesh(torch)
    try:
        q_a_key = _attn_key(0, "q_proj", "lora_A")
        k_a_key = _attn_key(0, "k_proj", "lora_A")
        v_a_key = _attn_key(0, "v_proj", "lora_A")
        q_b_key = _attn_key(0, "q_proj", "lora_B")
        k_b_key = _attn_key(0, "k_proj", "lora_B")
        v_b_key = _attn_key(0, "v_proj", "lora_B")
        full_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        q_b = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
        k_b = torch.tensor([[20.0, 21.0], [22.0, 23.0]])
        v_b = torch.tensor([[30.0, 31.0], [32.0, 33.0]])
        packed_b = torch.cat([q_b, k_b, v_b], dim=0)
        qkv_lora = SimpleNamespace(
            lora_a=_dtensor_param(torch, torch.zeros_like(full_a), mesh),
            lora_b=_dtensor_param(torch, torch.zeros_like(packed_b), mesh),
            rank=2,
            rank_partitioned_a=False,
            output_partitioned_b=False,
        )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=qkv_lora, proj_lora=None),
                    moe=SimpleNamespace(
                        experts=SimpleNamespace(
                            fc1_lora=None,
                            fc2_lora=None,
                            num_local_experts=0,
                        )
                    ),
                )
            ]
        )
        model_cfg = SimpleNamespace(
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=2,
            num_experts=0,
        )

        result = load_lora_adapter_state(
            [chunk],
            {
                q_a_key: full_a,
                k_a_key: full_a.clone(),
                v_a_key: full_a.clone(),
                q_b_key: q_b,
                k_b_key: k_b,
                v_b_key: v_b,
            },
            model_cfg,
            ParallelState(),
        )

        assert result["loaded_tensors"] == 2
        torch.testing.assert_close(qkv_lora.lora_a.full_tensor(), full_a)
        torch.testing.assert_close(qkv_lora.lora_b.full_tensor(), packed_b)
    finally:
        _cleanup_single_rank_cpu_mesh(torch, created, temp_file)


def test_glm5_load_per_expert_lora_state_copies_into_dtensor_grouped_params_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        _expert_key,
        load_lora_adapter_state,
    )
    from megatron.lite.primitive.parallel import ParallelState

    mesh, created, temp_file = _init_single_rank_cpu_mesh(torch)
    try:
        fc1_a = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        fc1_b = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2) + 100
        fc2_a = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + 200
        fc2_b = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + 300
        fc1_lora = SimpleNamespace(
            lora_a=_dtensor_param(torch, torch.zeros_like(fc1_a), mesh),
            lora_b=_dtensor_param(torch, torch.zeros_like(fc1_b), mesh),
            shared_across_experts=False,
        )
        fc2_lora = SimpleNamespace(
            lora_a=_dtensor_param(torch, torch.zeros_like(fc2_a), mesh),
            lora_b=_dtensor_param(torch, torch.zeros_like(fc2_b), mesh),
            shared_across_experts=False,
        )
        experts = SimpleNamespace(num_local_experts=2, fc1_lora=fc1_lora, fc2_lora=fc2_lora)
        layer = SimpleNamespace(
            layer_idx=0,
            self_attention=SimpleNamespace(self_attention=_fake_dsa_without_lora()),
            mlp=None,
            moe=SimpleNamespace(
                experts=experts,
                shared_expert=SimpleNamespace(gate_up_lora=None, down_lora=None),
            ),
        )
        state = {}
        for expert_idx in range(2):
            state[_expert_key(0, expert_idx, "gate_proj", "lora_A")] = fc1_a[expert_idx]
            state[_expert_key(0, expert_idx, "up_proj", "lora_A")] = fc1_a[expert_idx].clone()
            state[_expert_key(0, expert_idx, "gate_proj", "lora_B")] = fc1_b[expert_idx, :3]
            state[_expert_key(0, expert_idx, "up_proj", "lora_B")] = fc1_b[expert_idx, 3:]
            state[_expert_key(0, expert_idx, "down_proj", "lora_A")] = fc2_a[expert_idx]
            state[_expert_key(0, expert_idx, "down_proj", "lora_B")] = fc2_b[expert_idx]

        result = load_lora_adapter_state(
            [SimpleNamespace(layers=[layer], mtp=None)],
            state,
            SimpleNamespace(num_hidden_layers=1, num_experts=2),
            ParallelState(),
        )

        assert result["loaded_tensors"] == 12
        torch.testing.assert_close(fc1_lora.lora_a.full_tensor(), fc1_a)
        torch.testing.assert_close(fc1_lora.lora_b.full_tensor(), fc1_b)
        torch.testing.assert_close(fc2_lora.lora_a.full_tensor(), fc2_a)
        torch.testing.assert_close(fc2_lora.lora_b.full_tensor(), fc2_b)
    finally:
        _cleanup_single_rank_cpu_mesh(torch, created, temp_file)


def test_qwen3_load_per_expert_lora_state_copies_into_dtensor_grouped_params_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _expert_key,
        load_lora_adapter_state,
    )
    from megatron.lite.primitive.parallel import ParallelState

    mesh, created, temp_file = _init_single_rank_cpu_mesh(torch)
    try:
        fc1_a = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        fc1_b = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2) + 100
        fc2_a = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + 200
        fc2_b = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + 300
        fc1_lora = SimpleNamespace(
            lora_a=_dtensor_param(torch, torch.zeros_like(fc1_a), mesh),
            lora_b=_dtensor_param(torch, torch.zeros_like(fc1_b), mesh),
            shared_across_experts=False,
        )
        fc2_lora = SimpleNamespace(
            lora_a=_dtensor_param(torch, torch.zeros_like(fc2_a), mesh),
            lora_b=_dtensor_param(torch, torch.zeros_like(fc2_b), mesh),
            shared_across_experts=False,
        )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=None, proj_lora=None),
                    moe=SimpleNamespace(
                        experts=SimpleNamespace(
                            fc1_lora=fc1_lora,
                            fc2_lora=fc2_lora,
                            num_local_experts=2,
                        )
                    ),
                )
            ]
        )
        state = {}
        for expert_idx in range(2):
            state[_expert_key(0, expert_idx, "gate_proj", "lora_A")] = fc1_a[expert_idx]
            state[_expert_key(0, expert_idx, "up_proj", "lora_A")] = fc1_a[expert_idx].clone()
            state[_expert_key(0, expert_idx, "gate_proj", "lora_B")] = fc1_b[expert_idx, :3]
            state[_expert_key(0, expert_idx, "up_proj", "lora_B")] = fc1_b[expert_idx, 3:]
            state[_expert_key(0, expert_idx, "down_proj", "lora_A")] = fc2_a[expert_idx]
            state[_expert_key(0, expert_idx, "down_proj", "lora_B")] = fc2_b[expert_idx]

        result = load_lora_adapter_state(
            [chunk],
            state,
            SimpleNamespace(
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=2,
                num_experts=2,
            ),
            ParallelState(),
        )

        assert result["loaded_tensors"] == 8
        torch.testing.assert_close(fc1_lora.lora_a.full_tensor(), fc1_a)
        torch.testing.assert_close(fc1_lora.lora_b.full_tensor(), fc1_b)
        torch.testing.assert_close(fc2_lora.lora_a.full_tensor(), fc2_a)
        torch.testing.assert_close(fc2_lora.lora_b.full_tensor(), fc2_b)
    finally:
        _cleanup_single_rank_cpu_mesh(torch, created, temp_file)


def test_lora_adapter_config_rejects_inconsistent_a_b_ranks_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        _validate_adapter_config as validate_glm5_config,
    )
    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _validate_adapter_config as validate_qwen_config,
    )

    glm5_state = {
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.zeros(2, 4),
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.zeros(4, 3),
    }
    with pytest.raises(ValueError, match="inconsistent LoRA ranks"):
        validate_glm5_config(
            [],
            glm5_state,
            {"peft_type": "LORA", "r": 2, "target_modules": "q_a_proj"},
        )

    qwen_state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4, 3),
    }
    with pytest.raises(ValueError, match="inconsistent LoRA ranks"):
        validate_qwen_config(
            [],
            qwen_state,
            {"peft_type": "LORA", "r": 2, "target_modules": "q_proj"},
        )


def test_lora_adapter_config_requires_identity_fields_for_nonempty_state_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        _validate_adapter_config as validate_glm5_config,
    )
    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _validate_adapter_config as validate_qwen_config,
    )

    glm5_state = {
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.zeros(2, 4),
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.zeros(4, 2),
    }
    glm5_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "local/glm5-tiny",
        "r": 2,
        "target_modules": "q_a_proj",
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "use_rslora": False,
    }

    def without(config, key):
        out = dict(config)
        out.pop(key)
        return out

    with pytest.raises(ValueError, match="peft_type='LORA' is required"):
        validate_glm5_config([], glm5_state, without(glm5_config, "peft_type"))
    with pytest.raises(TypeError, match="peft_type must be a string"):
        validate_glm5_config([], glm5_state, {**glm5_config, "peft_type": 123})
    with pytest.raises(ValueError, match="base_model_name_or_path is required"):
        validate_glm5_config([], glm5_state, without(glm5_config, "base_model_name_or_path"))
    with pytest.raises(TypeError, match="base_model_name_or_path must be a string"):
        validate_glm5_config([], glm5_state, {**glm5_config, "base_model_name_or_path": 123})
    with pytest.raises(ValueError, match="base_model_name_or_path must be non-empty"):
        validate_glm5_config([], glm5_state, {**glm5_config, "base_model_name_or_path": " "})
    with pytest.raises(ValueError, match="Adapter config r is required"):
        validate_glm5_config([], glm5_state, without(glm5_config, "r"))
    with pytest.raises(ValueError, match="Adapter config target_modules is required"):
        validate_glm5_config([], glm5_state, without(glm5_config, "target_modules"))
    with pytest.raises(ValueError, match="Adapter config lora_alpha is required"):
        validate_glm5_config([], glm5_state, without(glm5_config, "lora_alpha"))
    with pytest.raises(ValueError, match="Adapter config use_rslora is required"):
        validate_glm5_config([], glm5_state, without(glm5_config, "use_rslora"))
    with pytest.raises(ValueError, match="Adapter config lora_dropout is required"):
        validate_glm5_config([], glm5_state, without(glm5_config, "lora_dropout"))
    validate_glm5_config(
        [],
        {},
        {"peft_type": "LORA", "bias": "none", "modules_to_save": ()},
    )

    qwen_state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4, 2),
    }
    qwen_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "local/qwen3-moe-tiny",
        "r": 2,
        "target_modules": "q_proj",
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "use_rslora": False,
    }
    with pytest.raises(ValueError, match="peft_type='LORA' is required"):
        validate_qwen_config([], qwen_state, without(qwen_config, "peft_type"))
    with pytest.raises(TypeError, match="peft_type must be a string"):
        validate_qwen_config([], qwen_state, {**qwen_config, "peft_type": 123})
    with pytest.raises(ValueError, match="base_model_name_or_path is required"):
        validate_qwen_config([], qwen_state, without(qwen_config, "base_model_name_or_path"))
    with pytest.raises(TypeError, match="base_model_name_or_path must be a string"):
        validate_qwen_config([], qwen_state, {**qwen_config, "base_model_name_or_path": 123})
    with pytest.raises(ValueError, match="base_model_name_or_path must be non-empty"):
        validate_qwen_config([], qwen_state, {**qwen_config, "base_model_name_or_path": ""})
    with pytest.raises(ValueError, match="Adapter config r is required"):
        validate_qwen_config([], qwen_state, without(qwen_config, "r"))
    with pytest.raises(ValueError, match="Adapter config target_modules is required"):
        validate_qwen_config([], qwen_state, without(qwen_config, "target_modules"))
    with pytest.raises(ValueError, match="Adapter config lora_alpha is required"):
        validate_qwen_config([], qwen_state, without(qwen_config, "lora_alpha"))
    with pytest.raises(ValueError, match="Adapter config use_rslora is required"):
        validate_qwen_config([], qwen_state, without(qwen_config, "use_rslora"))
    with pytest.raises(ValueError, match="Adapter config lora_dropout is required"):
        validate_qwen_config([], qwen_state, without(qwen_config, "lora_dropout"))


def test_qwen3_lora_adapter_rejects_non_adapter_only_configs_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _validate_adapter_config,
        load_lora_adapter,
    )
    from megatron.lite.primitive.parallel import ParallelState

    def expect_config_reject(
        updates, match: str, exception: type[Exception] = ValueError
    ) -> None:
        config = {"peft_type": "LORA", **updates}
        with pytest.raises(exception, match=match):
            _validate_adapter_config([], {}, config)

    expect_config_reject({"bias": "all"}, "bias='all'")
    expect_config_reject(
        {"bias": 123},
        "bias must be a string",
        exception=TypeError,
    )
    expect_config_reject(
        {"peft_type": 123},
        "peft_type must be a string",
        exception=TypeError,
    )
    expect_config_reject({"fan_in_fan_out": True}, "fan_in_fan_out=True")
    expect_config_reject({"modules_to_save": ["lm_head"]}, "modules_to_save")
    expect_config_reject({"modules_to_save": ("lm_head",)}, "modules_to_save")
    expect_config_reject({"task_type": "SEQ_CLS"}, "task_type='SEQ_CLS'")
    expect_config_reject(
        {"task_type": ["CAUSAL_LM"]},
        "task_type must be a string",
        exception=TypeError,
    )
    expect_config_reject(
        {"inference_mode": "true"},
        "inference_mode must be a boolean",
        exception=TypeError,
    )
    expect_config_reject(
        {"init_lora_weights": " "},
        "init_lora_weights string must be non-empty",
    )
    _validate_adapter_config([], {}, {"peft_type": "LORA", "inference_mode": True})
    _validate_adapter_config(
        [],
        {},
        {"peft_type": "LORA", "bias": "none", "modules_to_save": ()},
    )
    expect_config_reject(
        {"target_modules": ["eh_proj"]},
        "Unsupported Qwen3-MoE LoRA target module",
    )
    expect_config_reject(
        {"target_modules": {"linear_fc1": True}},
        "target_modules must be a string or sequence of strings",
        exception=TypeError,
    )
    expect_config_reject(
        {"target_modules": MappingProxyType({"linear_fc1": True})},
        "target_modules must be a string or sequence of strings",
        exception=TypeError,
    )
    expect_config_reject(
        {"target_modules": ["linear_fc1", 3]},
        "target_modules entries must be strings",
        exception=TypeError,
    )
    expect_config_reject(
        {"r": "2"},
        "Adapter config r must be an integer",
        exception=TypeError,
    )
    expect_config_reject(
        {"lora_alpha": "4"},
        "Adapter config lora_alpha must be a finite number",
        exception=TypeError,
    )
    expect_config_reject(
        {"lora_dropout": [0.0]},
        "Adapter config lora_dropout must be a finite number",
        exception=TypeError,
    )
    expect_config_reject(
        {"lora_dropout": -0.1},
        "Adapter config lora_dropout must be between 0 and 1",
    )
    expect_config_reject(
        {"lora_dropout": 1.5},
        "Adapter config lora_dropout must be between 0 and 1",
    )
    expect_config_reject(
        {"use_rslora": "true"},
        "use_rslora must be a boolean",
        exception=TypeError,
    )

    adapter_dir = tmp_path / "qwen_bad_adapter"
    adapter_dir.mkdir()
    safetensors_torch.save_file(
        {"base_model.model.model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2)},
        str(adapter_dir / "adapter_model.safetensors"),
    )
    model_cfg = SimpleNamespace(
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
    )
    with pytest.raises(ValueError, match="non-adapter tensor keys"):
        load_lora_adapter(
            [],
            adapter_dir,
            model_cfg,
            ParallelState(),
            lora_config={
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "target_modules": "q_proj",
            },
        )

    sidecar_dir = tmp_path / "qwen_bad_sidecar"
    sidecar_dir.mkdir()
    safetensors_torch.save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 2),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(2, 2),
        },
        str(sidecar_dir / "adapter_model.safetensors"),
    )
    (sidecar_dir / "adapter_config.json").write_text("[]\n")
    with pytest.raises(TypeError, match="adapter_config.json must be a JSON object"):
        load_lora_adapter([], sidecar_dir, model_cfg, ParallelState())

    missing_tensor_bad_config_dir = tmp_path / "qwen_bad_sidecar_without_adapter_model"
    missing_tensor_bad_config_dir.mkdir()
    (missing_tensor_bad_config_dir / "adapter_config.json").write_text("[]\n")
    with pytest.raises(TypeError, match="adapter_config.json must be a JSON object"):
        load_lora_adapter([], missing_tensor_bad_config_dir, model_cfg, ParallelState())

    directory_config_dir = tmp_path / "qwen_directory_config_without_adapter_model"
    directory_config_dir.mkdir()
    (directory_config_dir / "adapter_config.json").mkdir()
    with pytest.raises(IsADirectoryError, match="adapter_config.json path must be a file"):
        load_lora_adapter([], directory_config_dir, model_cfg, ParallelState())

    missing_tensor_bad_constant_config_dir = (
        tmp_path / "qwen_bad_constant_config_without_adapter_model"
    )
    missing_tensor_bad_constant_config_dir.mkdir()
    (missing_tensor_bad_constant_config_dir / "adapter_config.json").write_text(
        '{"peft_type": "LORA", "lora_alpha": NaN}\n'
    )
    with pytest.raises(ValueError, match="adapter_config.json must be standard JSON"):
        load_lora_adapter([], missing_tensor_bad_constant_config_dir, model_cfg, ParallelState())

    missing_tensor_bad_meta_dir = tmp_path / "qwen_bad_meta_without_adapter_model"
    missing_tensor_bad_meta_dir.mkdir()
    (missing_tensor_bad_meta_dir / "adapter_config.json").write_text(
        """{
  "peft_type": "LORA",
  "base_model_name_or_path": "local/qwen-bad-meta",
  "r": 2,
  "target_modules": ["q_proj"],
  "lora_alpha": 4,
  "use_rslora": false,
  "lora_dropout": 0.0
}
"""
    )
    (missing_tensor_bad_meta_dir / "megatron.lite_adapter_meta.json").write_text(
        '"bad-meta"\n'
    )
    with pytest.raises(TypeError, match="megatron.lite_adapter_meta.json must be a JSON object"):
        load_lora_adapter([], missing_tensor_bad_meta_dir, model_cfg, ParallelState())

    directory_meta_dir = tmp_path / "qwen_directory_meta_without_adapter_model"
    directory_meta_dir.mkdir()
    (directory_meta_dir / "adapter_config.json").write_text(
        """{
  "peft_type": "LORA",
  "base_model_name_or_path": "local/qwen-bad-meta",
  "r": 2,
  "target_modules": ["q_proj"],
  "lora_alpha": 4,
  "use_rslora": false,
  "lora_dropout": 0.0
}
"""
    )
    (directory_meta_dir / "megatron.lite_adapter_meta.json").mkdir()
    with pytest.raises(
        IsADirectoryError, match="megatron.lite_adapter_meta.json path must be a file"
    ):
        load_lora_adapter([], directory_meta_dir, model_cfg, ParallelState())

    missing_tensor_bad_constant_meta_dir = (
        tmp_path / "qwen_bad_constant_meta_without_adapter_model"
    )
    missing_tensor_bad_constant_meta_dir.mkdir()
    (missing_tensor_bad_constant_meta_dir / "adapter_config.json").write_text(
        """{
  "peft_type": "LORA",
  "base_model_name_or_path": "local/qwen-bad-meta",
  "r": 2,
  "target_modules": ["q_proj"],
  "lora_alpha": 4,
  "use_rslora": false,
  "lora_dropout": 0.0
}
"""
    )
    (missing_tensor_bad_constant_meta_dir / "megatron.lite_adapter_meta.json").write_text(
        '{"format": Infinity}\n'
    )
    with pytest.raises(ValueError, match="megatron.lite_adapter_meta.json must be standard JSON"):
        load_lora_adapter([], missing_tensor_bad_constant_meta_dir, model_cfg, ParallelState())


def test_qwen3_lora_adapter_rejects_unexported_target_modules_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import save_lora_adapter
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    qkv_lora = LinearLoRA(
        in_features=2,
        out_features=6,
        rank=2,
        alpha=4,
        dropout=0.0,
        use_rslora=True,
    )
    chunk = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_idx=0,
                attn=SimpleNamespace(qkv_lora=qkv_lora, proj_lora=None),
                moe=SimpleNamespace(
                    experts=SimpleNamespace(
                        fc1_lora=None,
                        fc2_lora=None,
                        num_local_experts=0,
                    )
                ),
            )
        ]
    )
    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=0,
        moe_intermediate_size=0,
    )
    adapter_dir = tmp_path / "qwen_partial_all_linear_adapter"

    with pytest.raises(ValueError, match="target_modules do not match exported adapter tensors"):
        save_lora_adapter(
            [chunk],
            model_cfg,
            ParallelState(),
            adapter_dir,
            lora_config={
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "target_modules": "all-linear",
                "use_rslora": True,
            },
        )

    assert not adapter_dir.exists()


def test_qwen3_lora_adapter_save_load_validates_metadata_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    import json

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        load_lora_adapter,
        save_lora_adapter,
    )
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    def make_chunk():
        qkv_lora = LinearLoRA(
            in_features=2,
            out_features=6,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        with torch.no_grad():
            qkv_lora.lora_a.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            qkv_lora.lora_b.copy_(
                torch.tensor(
                    [
                        [10.0, 11.0],
                        [12.0, 13.0],
                        [20.0, 21.0],
                        [22.0, 23.0],
                        [30.0, 31.0],
                        [32.0, 33.0],
                    ]
                )
            )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=qkv_lora, proj_lora=None),
                    moe=SimpleNamespace(
                        experts=SimpleNamespace(
                            fc1_lora=None,
                            fc2_lora=None,
                            num_local_experts=0,
                        )
                    ),
                )
            ]
        )
        return chunk, qkv_lora

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=0,
        moe_intermediate_size=0,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": "linear_qkv",
        "use_rslora": True,
    }
    source_chunk, source_lora = make_chunk()
    ps = ParallelState()
    adapter_dir = tmp_path / "qwen_qkv_adapter"

    meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/qwen3-moe-tiny",
        lora_config=lora_config,
        metadata={"test_case": "qwen_qkv_metadata"},
    )

    assert meta["format"] == "megatron.lite_qwen3_moe_lora_peft_v1"
    assert meta["base_model_name_or_path"] == "local/qwen3-moe-tiny"
    assert meta["expert_lora_representation"] == "per_expert"
    assert meta["lora"]["rank"] == 2
    assert meta["lora"]["alpha"] == 4
    assert meta["lora"]["dropout"] == 0
    assert meta["lora"]["use_rslora"] is True
    assert meta["lora"]["scaling_convention"] == "alpha_over_sqrt_rank"
    assert math.isclose(meta["lora"]["scale"], 4 / math.sqrt(2), rel_tol=0.0, abs_tol=1e-6)
    assert meta["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj"]
    assert meta["metadata"]["init_lora_weights"] is True
    assert meta["metadata"]["test_case"] == "qwen_qkv_metadata"

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert adapter_config["init_lora_weights"] is True
    assert adapter_config["target_modules"] == ["q_proj", "k_proj", "v_proj"]
    adapter_meta = json.loads((adapter_dir / "megatron.lite_adapter_meta.json").read_text())
    assert adapter_meta["base_model_name_or_path"] == "local/qwen3-moe-tiny"
    assert adapter_meta["lora"] == meta["lora"]

    mapping_metadata_dir = tmp_path / "qwen_qkv_mapping_metadata_adapter"
    mapping_lora_config = MappingProxyType(lora_config)
    mapping_meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        mapping_metadata_dir,
        base_model_name_or_path="local/qwen3-moe-tiny",
        lora_config=mapping_lora_config,
        metadata=MappingProxyType({"test_case": "qwen_qkv_mapping_metadata"}),
    )
    assert mapping_meta["metadata"]["test_case"] == "qwen_qkv_mapping_metadata"
    mapping_sidecar = json.loads(
        (mapping_metadata_dir / "megatron.lite_adapter_meta.json").read_text()
    )
    assert mapping_sidecar["metadata"]["test_case"] == "qwen_qkv_mapping_metadata"

    spaced_base_dir = tmp_path / "qwen_qkv_spaced_base_adapter"
    spaced_base_meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        spaced_base_dir,
        base_model_name_or_path="  local/qwen3-moe-tiny  ",
        lora_config=lora_config,
        metadata={"test_case": "qwen_spaced_base"},
    )
    assert spaced_base_meta["base_model_name_or_path"] == "local/qwen3-moe-tiny"
    spaced_base_config = json.loads((spaced_base_dir / "adapter_config.json").read_text())
    spaced_base_sidecar = json.loads(
        (spaced_base_dir / "megatron.lite_adapter_meta.json").read_text()
    )
    assert spaced_base_config["base_model_name_or_path"] == "local/qwen3-moe-tiny"
    assert spaced_base_sidecar["base_model_name_or_path"] == "local/qwen3-moe-tiny"

    nested_mapping_metadata_dir = tmp_path / "qwen_qkv_nested_mapping_metadata_adapter"
    nested_mapping_meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        nested_mapping_metadata_dir,
        base_model_name_or_path="local/qwen3-moe-tiny",
        lora_config=lora_config,
        metadata=MappingProxyType(
            {
                "nested": MappingProxyType({"source": "mapping"}),
                "items": [MappingProxyType({"idx": 1})],
            }
        ),
    )
    assert nested_mapping_meta["metadata"]["nested"] == {"source": "mapping"}
    assert nested_mapping_meta["metadata"]["items"] == [{"idx": 1}]
    nested_mapping_sidecar = json.loads(
        (nested_mapping_metadata_dir / "megatron.lite_adapter_meta.json").read_text()
    )
    assert nested_mapping_sidecar["metadata"]["nested"] == {"source": "mapping"}
    assert nested_mapping_sidecar["metadata"]["items"] == [{"idx": 1}]

    serving_config = dict(adapter_config)
    serving_config["task_type"] = "causal_lm"
    serving_config["inference_mode"] = True
    serving_config["modules_to_save"] = []
    (adapter_dir / "adapter_config.json").write_text(json.dumps(serving_config, indent=2) + "\n")
    serving_target_chunk, serving_target_lora = make_chunk()
    with torch.no_grad():
        serving_target_lora.lora_a.zero_()
        serving_target_lora.lora_b.zero_()
    serving_result = load_lora_adapter(
        [serving_target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=mapping_lora_config,
    )
    assert serving_result["loaded_tensors"] == 2
    torch.testing.assert_close(serving_target_lora.lora_a, source_lora.lora_a)
    torch.testing.assert_close(serving_target_lora.lora_b, source_lora.lora_b)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n")

    metadata_path = adapter_dir / "megatron.lite_adapter_meta.json"
    metadata_text = metadata_path.read_text()
    metadata_path.unlink()
    peft_only_target_chunk, peft_only_target_lora = make_chunk()
    with torch.no_grad():
        peft_only_target_lora.lora_a.zero_()
        peft_only_target_lora.lora_b.zero_()
    peft_only_result = load_lora_adapter(
        [peft_only_target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )
    assert peft_only_result["loaded_tensors"] == 2
    torch.testing.assert_close(peft_only_target_lora.lora_a, source_lora.lora_a)
    torch.testing.assert_close(peft_only_target_lora.lora_b, source_lora.lora_b)
    metadata_path.write_text(metadata_text)

    sentinel_path = adapter_dir / "operator-note.txt"
    sentinel_path.write_text("keep me")
    overwrite_meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/qwen3-moe-tiny-v2",
        lora_config=lora_config,
        metadata={"test_case": "qwen_qkv_metadata_v2"},
    )
    assert sentinel_path.read_text() == "keep me"
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    adapter_meta = json.loads((adapter_dir / "megatron.lite_adapter_meta.json").read_text())
    assert adapter_config["base_model_name_or_path"] == "local/qwen3-moe-tiny-v2"
    assert adapter_meta["base_model_name_or_path"] == "local/qwen3-moe-tiny-v2"
    assert adapter_meta["metadata"]["test_case"] == "qwen_qkv_metadata_v2"
    assert overwrite_meta["metadata"]["test_case"] == "qwen_qkv_metadata_v2"

    old_model_bytes = (adapter_dir / "adapter_model.safetensors").read_bytes()
    old_config_text = (adapter_dir / "adapter_config.json").read_text()
    old_meta_text = (adapter_dir / "megatron.lite_adapter_meta.json").read_text()
    original_replace = Path.replace

    def fail_config_install(self, target):
        if self.name == "adapter_config.json" and ".qwen_qkv_adapter.tmp-" in str(self):
            raise RuntimeError("synthetic qwen artifact replace failure")
        return original_replace(self, target)

    Path.replace = fail_config_install
    try:
        with pytest.raises(RuntimeError, match="synthetic qwen artifact replace failure"):
            save_lora_adapter(
                [source_chunk],
                model_cfg,
                ps,
                adapter_dir,
                base_model_name_or_path="local/qwen3-moe-tiny-v3",
                lora_config=lora_config,
                metadata={"test_case": "qwen_qkv_metadata_v3"},
            )
    finally:
        Path.replace = original_replace
    assert (adapter_dir / "adapter_model.safetensors").read_bytes() == old_model_bytes
    assert (adapter_dir / "adapter_config.json").read_text() == old_config_text
    assert (adapter_dir / "megatron.lite_adapter_meta.json").read_text() == old_meta_text
    assert sentinel_path.read_text() == "keep me"
    assert not list(tmp_path.glob(".qwen_qkv_adapter.tmp-*"))
    assert not list(tmp_path.glob(".qwen_qkv_adapter.bak-*"))

    bad_metadata_dir = tmp_path / "qwen_bad_metadata"
    with pytest.raises(TypeError, match="Adapter metadata metadata must be an object"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_metadata_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
            metadata=[("test_case", "qwen_bad_metadata")],
        )
    assert not bad_metadata_dir.exists()

    bad_json_metadata_dir = tmp_path / "qwen_bad_json_metadata"
    with pytest.raises(TypeError, match="JSON-serializable"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_json_metadata_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
            metadata={"bad": {"not", "json"}},
        )
    assert not bad_json_metadata_dir.exists()

    bad_nan_metadata_dir = tmp_path / "qwen_bad_nan_metadata"
    with pytest.raises(TypeError, match="JSON-serializable"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_nan_metadata_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
            metadata={"bad": [math.nan, math.inf]},
        )
    assert not bad_nan_metadata_dir.exists()

    bad_metadata_key_dir = tmp_path / "qwen_bad_metadata_key"
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_metadata_key_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
            metadata={"nested": [{1: "bad"}]},
        )
    assert not bad_metadata_key_dir.exists()

    bad_model_json_dir = tmp_path / "qwen_bad_model_json"
    bad_model_cfg = SimpleNamespace(**vars(model_cfg))
    bad_model_cfg.hidden_size = math.inf
    with pytest.raises(TypeError, match="megatron.lite_adapter_meta.json"):
        save_lora_adapter(
            [source_chunk],
            bad_model_cfg,
            ps,
            bad_model_json_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
        )
    assert not bad_model_json_dir.exists()

    failed_write_dir = tmp_path / "qwen_failed_atomic_write"
    original_save_file = safetensors_torch.save_file

    def fail_after_partial_file(state, path):
        Path(path).write_bytes(b"partial adapter")
        raise RuntimeError("synthetic qwen adapter write failure")

    safetensors_torch.save_file = fail_after_partial_file
    try:
        with pytest.raises(RuntimeError, match="synthetic qwen adapter write failure"):
            save_lora_adapter(
                [source_chunk],
                model_cfg,
                ps,
                failed_write_dir,
                base_model_name_or_path="local/qwen3-moe-tiny",
                lora_config=lora_config,
            )
    finally:
        safetensors_torch.save_file = original_save_file
    assert not failed_write_dir.exists()
    assert not list(tmp_path.glob(".qwen_failed_atomic_write.tmp-*"))

    output_file = tmp_path / "qwen_output_file"
    output_file.write_text("keep this file")
    with pytest.raises(FileExistsError, match="not a directory"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            output_file,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
        )
    assert output_file.read_text() == "keep this file"
    assert not list(tmp_path.glob(".qwen_output_file.tmp-*"))

    blocked_artifact_dir = tmp_path / "qwen_blocked_artifact_dir"
    blocked_artifact_dir.mkdir()
    (blocked_artifact_dir / "adapter_config.json").mkdir()
    with pytest.raises(IsADirectoryError, match="Adapter artifact path must be a file"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            blocked_artifact_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
        )
    assert (blocked_artifact_dir / "adapter_config.json").is_dir()
    assert not (blocked_artifact_dir / "adapter_model.safetensors").exists()
    assert not list(tmp_path.glob(".qwen_blocked_artifact_dir.tmp-*"))

    bad_init_dir = tmp_path / "qwen_bad_init"
    with pytest.raises(TypeError, match="init_lora_weights"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_init_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
            init_lora_weights=123,
        )
    assert not bad_init_dir.exists()

    bad_empty_init_dir = tmp_path / "qwen_bad_empty_init"
    with pytest.raises(ValueError, match="init_lora_weights string must be non-empty"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_empty_init_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
            init_lora_weights=" ",
        )
    assert not bad_empty_init_dir.exists()

    bad_metadata_init_dir = tmp_path / "qwen_bad_metadata_init"
    with pytest.raises(TypeError, match="init_lora_weights"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_metadata_init_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
            metadata={"init_lora_weights": ["olora_tail"]},
        )
    assert not bad_metadata_init_dir.exists()

    bad_empty_metadata_init_dir = tmp_path / "qwen_bad_empty_metadata_init"
    with pytest.raises(ValueError, match="init_lora_weights string must be non-empty"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_empty_metadata_init_dir,
            base_model_name_or_path="local/qwen3-moe-tiny",
            lora_config=lora_config,
            metadata={"init_lora_weights": ""},
        )
    assert not bad_empty_metadata_init_dir.exists()

    target_chunk, target_lora = make_chunk()
    with torch.no_grad():
        target_lora.lora_a.zero_()
        target_lora.lora_b.zero_()

    result = load_lora_adapter(
        [target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )

    assert result["loaded_tensors"] == 2
    torch.testing.assert_close(target_lora.lora_a, source_lora.lora_a)
    torch.testing.assert_close(target_lora.lora_b, source_lora.lora_b)

    metadata_path = adapter_dir / "megatron.lite_adapter_meta.json"
    base_metadata = json.loads(metadata_path.read_text())

    def expect_metadata_reject(
        updater, match: str, exception: type[Exception] = ValueError
    ) -> None:
        metadata = json.loads(json.dumps(base_metadata))
        updater(metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        fresh_chunk, _ = make_chunk()
        with pytest.raises(exception, match=match):
            load_lora_adapter(
                [fresh_chunk],
                adapter_dir,
                model_cfg,
                ps,
                lora_config=lora_config,
            )

    expect_metadata_reject(
        lambda metadata: metadata.update({"format": "unexpected_format"}),
        "metadata format",
    )
    for required_field in (
        "base_model_name_or_path",
        "num_tensors",
        "num_parameters",
        "expert_lora_representation",
        "lora",
        "parallel",
        "model",
        "metadata",
    ):
        expect_metadata_reject(
            lambda metadata, field=required_field: metadata.pop(field),
            f"Adapter metadata {required_field} is required",
        )
    for object_field, match in (
        ("lora", "Adapter metadata lora must be an object"),
        ("parallel", "Adapter metadata parallel must be an object"),
        ("model", "Adapter metadata model must be an object"),
        ("metadata", "Adapter metadata metadata must be an object"),
    ):
        expect_metadata_reject(
            lambda metadata, field=object_field: metadata.update({field: None}),
            match,
            exception=TypeError,
        )
        expect_metadata_reject(
            lambda metadata, field=object_field: metadata.update({field: []}),
            match,
            exception=TypeError,
        )
    for required_field in (
        "rank",
        "alpha",
        "dropout",
        "use_rslora",
        "scaling_convention",
        "scale",
        "target_modules",
    ):
        expect_metadata_reject(
            lambda metadata, field=required_field: metadata["lora"].pop(field),
            f"Adapter metadata lora.{required_field} is required",
        )
    for required_field in ("tp", "ep", "etp", "pp"):
        expect_metadata_reject(
            lambda metadata, field=required_field: metadata["parallel"].pop(field),
            f"Adapter metadata parallel.{required_field} is required",
        )
    for required_field in (
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_experts",
        "moe_intermediate_size",
    ):
        expect_metadata_reject(
            lambda metadata, field=required_field: metadata["model"].pop(field),
            f"Adapter metadata model.{required_field} is required",
        )
    expect_metadata_reject(
        lambda metadata: metadata.update({"base_model_name_or_path": "local/qwen3-other"}),
        "base_model_name_or_path",
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"base_model_name_or_path": []}),
        "base_model_name_or_path must be a string",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"num_tensors": 999}),
        "num_tensors",
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"num_tensors": str(base_metadata["num_tensors"])}),
        "Adapter metadata num_tensors must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"num_parameters": 999}),
        "num_parameters",
    )
    expect_metadata_reject(
        lambda metadata: metadata.update(
            {"num_parameters": str(base_metadata["num_parameters"])}
        ),
        "Adapter metadata num_parameters must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata.update(
            {"expert_lora_representation": "shared_local_expert_group"}
        ),
        "expert_lora_representation",
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"tp": 2}),
        "parallel.tp=2",
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"tp": "1"}),
        "Adapter metadata parallel.tp must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"ep": "1"}),
        "Adapter metadata parallel.ep must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"ep": True}),
        "Adapter metadata parallel.ep must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"pp": True}),
        "Adapter metadata parallel.pp must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["model"].update({"num_experts": 1}),
        "model.num_experts",
    )
    expect_metadata_reject(
        lambda metadata: metadata["model"].update({"hidden_size": 2.0}),
        "Adapter metadata model.hidden_size must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["model"].update({"num_experts": True}),
        "Adapter metadata model.num_experts must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"rank": 3}),
        "lora.rank",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"alpha": 5}),
        "lora.alpha",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"dropout": 0.25}),
        "lora.dropout",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"use_rslora": "true"}),
        "metadata.lora.use_rslora must be a boolean",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"target_modules": ["q_proj"]}),
        "lora.target_modules",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"scaling_convention": "alpha_over_rank"}),
        "lora.scaling_convention",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"scale": 2.0}),
        "lora.scale",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"rank": "2"}),
        "Adapter metadata lora.rank must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"alpha": "4"}),
        "Adapter metadata lora.alpha must be a finite number",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"dropout": "0.0"}),
        "Adapter metadata lora.dropout must be a finite number",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"dropout": -0.1}),
        "Adapter metadata lora.dropout must be between 0 and 1",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"scale": "2.0"}),
        "Adapter metadata lora.scale must be a finite number",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"metadata": []}),
        "metadata must be an object",
        exception=TypeError,
    )

    def expect_config_metadata_reject(updater, match: str) -> None:
        metadata_path.write_text(json.dumps(base_metadata, indent=2) + "\n")
        config_path = adapter_dir / "adapter_config.json"
        config = json.loads(config_path.read_text())
        updater(config)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        fresh_chunk, _ = make_chunk()
        with pytest.raises(ValueError, match=match):
            load_lora_adapter([fresh_chunk], adapter_dir, model_cfg, ps)
        config_path.write_text(json.dumps(adapter_config, indent=2) + "\n")

    expect_config_metadata_reject(
        lambda config: config.update({"base_model_name_or_path": "local/qwen3-other"}),
        "base_model_name_or_path",
    )
    expect_config_metadata_reject(
        lambda config: config.update({"r": 3}),
        "Adapter config rank r=3",
    )
    expect_config_metadata_reject(
        lambda config: config.update({"lora_alpha": 5}),
        "Adapter config lora_alpha=5",
    )
    expect_config_metadata_reject(
        lambda config: config.update({"lora_dropout": 0.25}),
        "Adapter config lora_dropout=0.25",
    )
    expect_config_metadata_reject(
        lambda config: config.update({"use_rslora": False}),
        "use_rslora=False",
    )
    expect_config_metadata_reject(
        lambda config: config.update({"target_modules": ["q_proj"]}),
        "target_modules",
    )
    expect_config_metadata_reject(
        lambda config: config.update({"init_lora_weights": "olora_tail"}),
        "does not match metadata init_lora_weights",
    )

    metadata_path.write_text("[]\n")
    fresh_chunk, _ = make_chunk()
    with pytest.raises(TypeError, match="megatron.lite_adapter_meta.json must be a JSON object"):
        load_lora_adapter([fresh_chunk], adapter_dir, model_cfg, ps, lora_config=lora_config)


def test_qwen3_lora_adapter_protocol_wrappers_round_trip_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.qwen3_moe.lite import protocol as qwen_protocol
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    def make_chunk():
        qkv_lora = LinearLoRA(
            in_features=2,
            out_features=6,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        with torch.no_grad():
            qkv_lora.lora_a.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            qkv_lora.lora_b.copy_(
                torch.tensor(
                    [
                        [10.0, 11.0],
                        [12.0, 13.0],
                        [20.0, 21.0],
                        [22.0, 23.0],
                        [30.0, 31.0],
                        [32.0, 33.0],
                    ]
                )
            )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=qkv_lora, proj_lora=None),
                    moe=SimpleNamespace(
                        experts=SimpleNamespace(
                            fc1_lora=None,
                            fc2_lora=None,
                            num_local_experts=0,
                        )
                    ),
                )
            ]
        )
        return chunk, qkv_lora

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=0,
        moe_intermediate_size=0,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": "linear_qkv",
        "use_rslora": True,
    }
    ps = ParallelState()
    source_chunk, source_lora = make_chunk()

    state = qwen_protocol.export_lora_adapter_state([source_chunk], model_cfg, ps)
    assert sorted(state) == [
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
    ]

    state_target_chunk, state_target_lora = make_chunk()
    with torch.no_grad():
        state_target_lora.lora_a.zero_()
        state_target_lora.lora_b.zero_()
    state_result = qwen_protocol.load_lora_adapter_state(
        [state_target_chunk], state, model_cfg, ps
    )
    assert state_result["loaded_tensors"] == 2
    torch.testing.assert_close(state_target_lora.lora_a, source_lora.lora_a)
    torch.testing.assert_close(state_target_lora.lora_b, source_lora.lora_b)

    adapter_dir = tmp_path / "qwen_protocol_adapter"
    meta = qwen_protocol.save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/qwen3-moe-protocol",
        lora_config=lora_config,
        metadata={"test_case": "qwen_protocol_lifecycle"},
    )
    assert meta["metadata"]["test_case"] == "qwen_protocol_lifecycle"

    dir_target_chunk, dir_target_lora = make_chunk()
    with torch.no_grad():
        dir_target_lora.lora_a.zero_()
        dir_target_lora.lora_b.zero_()
    dir_result = qwen_protocol.load_lora_adapter(
        [dir_target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )
    assert dir_result["loaded_tensors"] == 2
    torch.testing.assert_close(dir_target_lora.lora_a, source_lora.lora_a)
    torch.testing.assert_close(dir_target_lora.lora_b, source_lora.lora_b)


def test_qwen3_lora_adapter_fsdp_style_wrappers_round_trip_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        export_lora_adapter_state,
        load_lora_adapter,
        load_lora_adapter_state,
        save_lora_adapter,
    )
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    def make_chunk():
        qkv_lora = LinearLoRA(
            in_features=2,
            out_features=6,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        with torch.no_grad():
            qkv_lora.lora_a.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            qkv_lora.lora_b.copy_(
                torch.tensor(
                    [
                        [10.0, 11.0],
                        [12.0, 13.0],
                        [20.0, 21.0],
                        [22.0, 23.0],
                        [30.0, 31.0],
                        [32.0, 33.0],
                    ]
                )
            )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=qkv_lora, proj_lora=None),
                    moe=SimpleNamespace(
                        experts=SimpleNamespace(
                            fc1_lora=None,
                            fc2_lora=None,
                            num_local_experts=0,
                        )
                    ),
                )
            ]
        )
        return chunk, qkv_lora

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=0,
        moe_intermediate_size=0,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": "linear_qkv",
        "use_rslora": True,
    }
    ps = ParallelState()

    source_chunk, source_lora = make_chunk()
    source_wrapped = _fake_fsdp_wrapped_chunk(torch, source_chunk)
    state = export_lora_adapter_state([source_wrapped], model_cfg, ps)
    assert len(state) == 6

    state_target_chunk, state_target_lora = make_chunk()
    state_target_wrapped = _fake_fsdp_wrapped_chunk(torch, state_target_chunk)
    with torch.no_grad():
        state_target_lora.lora_a.zero_()
        state_target_lora.lora_b.zero_()
    state_result = load_lora_adapter_state(
        [state_target_wrapped], state, model_cfg, ps
    )
    assert state_result["loaded_tensors"] == 2
    torch.testing.assert_close(state_target_lora.lora_a, source_lora.lora_a)
    torch.testing.assert_close(state_target_lora.lora_b, source_lora.lora_b)

    adapter_dir = tmp_path / "qwen_fsdp_style_adapter"
    meta = save_lora_adapter(
        [source_wrapped],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/qwen3-moe-fsdp-wrapper",
        lora_config=lora_config,
        metadata={"test_case": "qwen_fsdp_style_adapter"},
    )
    assert meta["metadata"]["test_case"] == "qwen_fsdp_style_adapter"
    assert meta["parallel"] == {"tp": 1, "ep": 1, "etp": 1, "pp": 1}

    dir_target_chunk, dir_target_lora = make_chunk()
    dir_target_wrapped = _fake_fsdp_wrapped_chunk(torch, dir_target_chunk)
    with torch.no_grad():
        dir_target_lora.lora_a.zero_()
        dir_target_lora.lora_b.zero_()
    dir_result = load_lora_adapter(
        [dir_target_wrapped],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )
    assert dir_result["loaded_tensors"] == 2
    torch.testing.assert_close(dir_target_lora.lora_a, source_lora.lora_a)
    torch.testing.assert_close(dir_target_lora.lora_b, source_lora.lora_b)


def test_qwen3_load_lora_adapter_validates_expected_config_without_adapter_config(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        load_lora_adapter,
        save_lora_adapter,
    )
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    def make_chunk():
        qkv_lora = LinearLoRA(
            in_features=2,
            out_features=6,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=qkv_lora, proj_lora=None),
                    moe=SimpleNamespace(
                        experts=SimpleNamespace(
                            fc1_lora=None,
                            fc2_lora=None,
                            num_local_experts=0,
                        )
                    ),
                )
            ]
        )
        return chunk

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=0,
        moe_intermediate_size=0,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": "linear_qkv",
        "use_rslora": True,
    }
    ps = ParallelState()

    missing_adapter_dir = tmp_path / "qwen_missing_adapter_dir"
    with pytest.raises(FileNotFoundError, match="LoRA adapter directory does not exist"):
        load_lora_adapter([make_chunk()], missing_adapter_dir, model_cfg, ps)

    adapter_file = tmp_path / "qwen_adapter_file"
    adapter_file.write_text("not a directory")
    with pytest.raises(NotADirectoryError, match="LoRA adapter path must be a directory"):
        load_lora_adapter([make_chunk()], adapter_file, model_cfg, ps)

    missing_config_no_model_dir = tmp_path / "qwen_missing_config_without_model"
    missing_config_no_model_dir.mkdir()
    with pytest.raises(ValueError, match="caller-provided lora_config"):
        load_lora_adapter([make_chunk()], missing_config_no_model_dir, model_cfg, ps)

    adapter_dir = tmp_path / "qwen_adapter_without_configs"
    save_lora_adapter(
        [make_chunk()],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/qwen3-moe-without-adapter-config",
        lora_config=lora_config,
    )
    (adapter_dir / "adapter_config.json").unlink()
    with pytest.raises(ValueError, match="caller-provided lora_config"):
        load_lora_adapter([make_chunk()], adapter_dir, model_cfg, ps)
    (adapter_dir / "megatron.lite_adapter_meta.json").unlink()

    result = load_lora_adapter([make_chunk()], adapter_dir, model_cfg, ps, lora_config=lora_config)
    assert result["loaded_tensors"] == 2

    missing_model_dir = tmp_path / "qwen_missing_adapter_model"
    save_lora_adapter(
        [make_chunk()],
        model_cfg,
        ps,
        missing_model_dir,
        base_model_name_or_path="local/qwen3-moe-missing-model",
        lora_config=lora_config,
    )
    (missing_model_dir / "adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="LoRA adapter model file is missing"):
        load_lora_adapter([make_chunk()], missing_model_dir, model_cfg, ps)

    directory_model_dir = tmp_path / "qwen_directory_adapter_model"
    save_lora_adapter(
        [make_chunk()],
        model_cfg,
        ps,
        directory_model_dir,
        base_model_name_or_path="local/qwen3-moe-directory-model",
        lora_config=lora_config,
    )
    (directory_model_dir / "adapter_model.safetensors").unlink()
    (directory_model_dir / "adapter_model.safetensors").mkdir()
    with pytest.raises(IsADirectoryError, match="LoRA adapter model path must be a file"):
        load_lora_adapter([make_chunk()], directory_model_dir, model_cfg, ps)

    with pytest.raises(ValueError, match="target modules"):
        load_lora_adapter(
            [make_chunk()],
            adapter_dir,
            model_cfg,
            ps,
            lora_config={**lora_config, "target_modules": "q_proj"},
        )

    with pytest.raises(ValueError, match="alpha"):
        load_lora_adapter(
            [make_chunk()],
            adapter_dir,
            model_cfg,
            ps,
            lora_config={**lora_config, "lora_alpha": 5},
        )

    with pytest.raises(ValueError, match="dropout"):
        load_lora_adapter(
            [make_chunk()],
            adapter_dir,
            model_cfg,
            ps,
            lora_config={**lora_config, "lora_dropout": 0.25},
        )

    with pytest.raises(ValueError, match="use_rslora"):
        load_lora_adapter(
            [make_chunk()],
            adapter_dir,
            model_cfg,
            ps,
            lora_config={**lora_config, "use_rslora": False},
        )

    with pytest.raises(ValueError, match="enabled expected LoRA config"):
        load_lora_adapter(
            [make_chunk()],
            adapter_dir,
            model_cfg,
            ps,
            lora_config={**lora_config, "enabled": False},
        )


def test_qwen3_load_lora_adapter_state_reports_missing_and_unexpected_keys_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _attn_key,
        _expert_key,
        export_lora_adapter_state,
        load_lora_adapter_state,
    )
    from megatron.lite.primitive.modules.lora import LinearLoRA, SharedGroupedLinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    def make_chunk():
        qkv_lora = LinearLoRA(
            in_features=2,
            out_features=6,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        with torch.no_grad():
            qkv_lora.lora_a.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            qkv_lora.lora_b.copy_(
                torch.tensor(
                    [
                        [10.0, 11.0],
                        [12.0, 13.0],
                        [20.0, 21.0],
                        [22.0, 23.0],
                        [30.0, 31.0],
                        [32.0, 33.0],
                    ]
                )
            )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=qkv_lora, proj_lora=None),
                    moe=SimpleNamespace(
                        experts=SimpleNamespace(
                            fc1_lora=None,
                            fc2_lora=None,
                            num_local_experts=0,
                        )
                    ),
                )
            ]
        )
        return chunk, qkv_lora

    model_cfg = SimpleNamespace(
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=0,
    )
    ps = ParallelState()
    source_chunk, source_lora = make_chunk()
    state = export_lora_adapter_state([source_chunk], model_cfg, ps)

    target_chunk, target_lora = make_chunk()
    with torch.no_grad():
        target_lora.lora_a.zero_()
        target_lora.lora_b.zero_()
    result = load_lora_adapter_state([target_chunk], state, model_cfg, ps)
    assert result["loaded_tensors"] == 2
    torch.testing.assert_close(target_lora.lora_a, source_lora.lora_a)
    torch.testing.assert_close(target_lora.lora_b, source_lora.lora_b)

    bad_shape_state = {key: value.clone() for key, value in state.items()}
    bad_shape_state[_attn_key(0, "q_proj", "lora_B")] = torch.zeros(3, 2)
    bad_shape_chunk, _ = make_chunk()
    with pytest.raises(ValueError, match="has shape"):
        load_lora_adapter_state([bad_shape_chunk], bad_shape_state, model_cfg, ps)

    missing_state = {key: value.clone() for key, value in state.items()}
    missing_state.pop(_attn_key(0, "k_proj", "lora_A"))
    missing_state.pop(_attn_key(0, "k_proj", "lora_B"))
    missing_chunk, _ = make_chunk()
    with pytest.raises(KeyError, match="Missing adapter tensor"):
        load_lora_adapter_state([missing_chunk], missing_state, model_cfg, ps)

    extra_state = {key: value.clone() for key, value in state.items()}
    extra_state[_attn_key(0, "o_proj", "lora_A")] = torch.zeros(2, 2)
    extra_state[_attn_key(0, "o_proj", "lora_B")] = torch.zeros(2, 2)
    strict_chunk, _ = make_chunk()
    with pytest.raises(ValueError, match="Unexpected adapter tensor keys"):
        load_lora_adapter_state([strict_chunk], extra_state, model_cfg, ps)

    loose_chunk, _ = make_chunk()
    result = load_lora_adapter_state([loose_chunk], extra_state, model_cfg, ps, strict=False)
    assert result["loaded_tensors"] == 2

    string_false_chunk, _ = make_chunk()
    result = load_lora_adapter_state(
        [string_false_chunk], extra_state, model_cfg, ps, strict="false"
    )
    assert result["loaded_tensors"] == 2

    with pytest.raises(ValueError, match="strict must be a boolean"):
        load_lora_adapter_state([make_chunk()[0]], state, model_cfg, ps, strict="maybe")
    with pytest.raises(TypeError, match="strict must be a boolean"):
        load_lora_adapter_state([make_chunk()[0]], state, model_cfg, ps, strict=1)

    unsupported_lora_state = {key: value.clone() for key, value in state.items()}
    unsupported_lora_state[
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"
    ] = torch.zeros(2, 2)
    unsupported_lora_chunk, _ = make_chunk()
    with pytest.raises(ValueError, match="unsupported adapter tensor keys"):
        load_lora_adapter_state(
            [unsupported_lora_chunk],
            unsupported_lora_state,
            model_cfg,
            ps,
            strict=False,
        )

    non_lora_state = {key: value.clone() for key, value in state.items()}
    non_lora_state["base_model.model.model.layers.0.self_attn.q_proj.weight"] = torch.zeros(2, 2)
    non_lora_chunk, _ = make_chunk()
    with pytest.raises(ValueError, match="non-adapter tensor keys"):
        load_lora_adapter_state(
            [non_lora_chunk],
            non_lora_state,
            model_cfg,
            ps,
            strict=False,
        )

    def make_proj_chunk():
        proj_lora = LinearLoRA(
            in_features=2,
            out_features=2,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=None, proj_lora=proj_lora),
                    moe=SimpleNamespace(
                        experts=SimpleNamespace(
                            fc1_lora=None,
                            fc2_lora=None,
                            num_local_experts=0,
                        )
                    ),
                )
            ]
        )
        return chunk, proj_lora

    proj_chunk, _ = make_proj_chunk()
    proj_state = export_lora_adapter_state([proj_chunk], model_cfg, ps)
    bad_proj_state = {key: value.clone() for key, value in proj_state.items()}
    bad_proj_state[_attn_key(0, "o_proj", "lora_A")] = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="has shape"):
        load_lora_adapter_state([make_proj_chunk()[0]], bad_proj_state, model_cfg, ps)

    def make_shared_expert_chunk():
        fc1_lora = SharedGroupedLinearLoRA(
            2,
            in_features=3,
            out_features=4,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        fc2_lora = SharedGroupedLinearLoRA(
            2,
            in_features=2,
            out_features=3,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        experts = SimpleNamespace(
            fc1_lora=fc1_lora,
            fc2_lora=fc2_lora,
            num_local_experts=2,
        )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=None, proj_lora=None),
                    moe=SimpleNamespace(experts=experts),
                )
            ]
        )
        return chunk, experts

    expert_model_cfg = SimpleNamespace(
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=2,
    )
    expert_chunk, _ = make_shared_expert_chunk()
    expert_state = export_lora_adapter_state([expert_chunk], expert_model_cfg, ps)
    good_expert_result = load_lora_adapter_state(
        [make_shared_expert_chunk()[0]], expert_state, expert_model_cfg, ps
    )
    assert good_expert_result["loaded_tensors"] == 4
    bad_expert_state = {key: value.clone() for key, value in expert_state.items()}
    bad_expert_state[_expert_key(0, 0, "up_proj", "lora_B")] = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="has shape"):
        load_lora_adapter_state([make_shared_expert_chunk()[0]], bad_expert_state, expert_model_cfg, ps)


def test_qwen3_per_expert_grouped_lora_round_trips_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _expert_key,
        export_lora_adapter_state,
        load_lora_adapter,
        load_lora_adapter_state,
        save_lora_adapter,
    )
    from megatron.lite.primitive.modules.lora import GroupedLinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    def make_chunk(*, zero: bool = False):
        fc1_lora = GroupedLinearLoRA(
            num_local_experts=2,
            in_features=2,
            out_features=6,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        fc2_lora = GroupedLinearLoRA(
            num_local_experts=2,
            in_features=3,
            out_features=2,
            rank=2,
            alpha=4,
            dropout=0.0,
            use_rslora=True,
        )
        with torch.no_grad():
            if zero:
                fc1_lora.lora_a.zero_()
                fc1_lora.lora_b.zero_()
                fc2_lora.lora_a.zero_()
                fc2_lora.lora_b.zero_()
            else:
                fc1_lora.lora_a.copy_(torch.arange(8, dtype=torch.float32).view(2, 2, 2))
                fc1_lora.lora_b.copy_(
                    torch.arange(24, dtype=torch.float32).view(2, 6, 2) + 100
                )
                fc2_lora.lora_a.copy_(torch.arange(12, dtype=torch.float32).view(2, 2, 3))
                fc2_lora.lora_b.copy_(
                    torch.arange(8, dtype=torch.float32).view(2, 2, 2) + 200
                )
        chunk = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    attn=SimpleNamespace(qkv_lora=None, proj_lora=None),
                    moe=SimpleNamespace(
                        experts=SimpleNamespace(
                            fc1_lora=fc1_lora,
                            fc2_lora=fc2_lora,
                            num_local_experts=2,
                        )
                    ),
                )
            ]
        )
        return chunk, fc1_lora, fc2_lora

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=2,
        moe_intermediate_size=3,
    )
    ps = ParallelState()
    source_chunk, source_fc1, source_fc2 = make_chunk()
    state = export_lora_adapter_state([source_chunk], model_cfg, ps)

    expected_keys = {
        _expert_key(0, expert_idx, module, suffix)
        for expert_idx in range(2)
        for module in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("lora_A", "lora_B")
    }
    assert set(state) == expected_keys
    torch.testing.assert_close(
        state[_expert_key(0, 1, "gate_proj", "lora_B")],
        source_fc1.lora_b[1, :3],
    )
    torch.testing.assert_close(
        state[_expert_key(0, 1, "up_proj", "lora_B")],
        source_fc1.lora_b[1, 3:],
    )

    target_chunk, target_fc1, target_fc2 = make_chunk(zero=True)
    result = load_lora_adapter_state([target_chunk], state, model_cfg, ps)

    assert result["loaded_tensors"] == 8
    torch.testing.assert_close(target_fc1.lora_a, source_fc1.lora_a)
    torch.testing.assert_close(target_fc1.lora_b, source_fc1.lora_b)
    torch.testing.assert_close(target_fc2.lora_a, source_fc2.lora_a)
    torch.testing.assert_close(target_fc2.lora_b, source_fc2.lora_b)

    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
        "use_rslora": True,
    }
    adapter_dir = tmp_path / "qwen_per_expert_grouped_adapter"
    meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/qwen3-moe-expert-tiny",
        lora_config=lora_config,
        metadata={"test_case": "qwen_per_expert_grouped_lora"},
    )
    assert meta["expert_lora_representation"] == "per_expert"
    assert meta["num_tensors"] == len(expected_keys)
    assert meta["lora"]["target_modules"] == ["gate_proj", "up_proj", "down_proj"]

    dir_target_chunk, dir_target_fc1, dir_target_fc2 = make_chunk(zero=True)
    dir_result = load_lora_adapter(
        [dir_target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )
    assert dir_result["loaded_tensors"] == 8
    torch.testing.assert_close(dir_target_fc1.lora_a, source_fc1.lora_a)
    torch.testing.assert_close(dir_target_fc1.lora_b, source_fc1.lora_b)
    torch.testing.assert_close(dir_target_fc2.lora_a, source_fc2.lora_a)
    torch.testing.assert_close(dir_target_fc2.lora_b, source_fc2.lora_b)


def test_qwen3_lora_adapter_rejects_unsupported_parallel_scopes_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        export_lora_adapter_state,
        load_lora_adapter,
        load_lora_adapter_state,
        save_lora_adapter,
    )
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    qkv_lora = LinearLoRA(
        in_features=2,
        out_features=6,
        rank=2,
        alpha=4,
        dropout=0.0,
        use_rslora=True,
    )
    chunk = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_idx=0,
                attn=SimpleNamespace(qkv_lora=qkv_lora, proj_lora=None),
                moe=SimpleNamespace(
                    experts=SimpleNamespace(
                        fc1_lora=None,
                        fc2_lora=None,
                        num_local_experts=0,
                    )
                ),
            )
        ]
    )
    model_cfg = SimpleNamespace(
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=0,
    )

    with pytest.raises(NotImplementedError, match="pp=1"):
        export_lora_adapter_state([chunk], model_cfg, ParallelState(pp_size=2))

    with pytest.raises(NotImplementedError, match="pp=1"):
        load_lora_adapter_state([chunk], {}, model_cfg, ParallelState(pp_size=2))

    with pytest.raises(NotImplementedError, match="etp=1"):
        export_lora_adapter_state([chunk], model_cfg, ParallelState(etp_size=2))

    with pytest.raises(NotImplementedError, match="etp=1"):
        load_lora_adapter_state([chunk], {}, model_cfg, ParallelState(etp_size=2))

    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": "linear_qkv",
        "use_rslora": True,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "qwen_unsupported_save"
        with pytest.raises(NotImplementedError, match="pp=1"):
            save_lora_adapter(
                [chunk],
                model_cfg,
                ParallelState(pp_size=2),
                save_dir,
                lora_config=lora_config,
            )
        assert not save_dir.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "qwen_missing_base_identity"
        with pytest.raises(ValueError, match="base_model_name_or_path must be non-empty"):
            save_lora_adapter(
                [chunk],
                model_cfg,
                ParallelState(),
                save_dir,
                lora_config=lora_config,
            )
        assert not save_dir.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "qwen_bad_base_identity"
        with pytest.raises(TypeError, match="base_model_name_or_path must be a string"):
            save_lora_adapter(
                [chunk],
                model_cfg,
                ParallelState(),
                save_dir,
                base_model_name_or_path=123,
                lora_config=lora_config,
            )
        assert not save_dir.exists()

    with pytest.raises(NotImplementedError, match="pp=1"):
        load_lora_adapter(
            [chunk],
            "/definitely/missing/qwen_adapter",
            model_cfg,
            ParallelState(pp_size=2),
            lora_config=lora_config,
        )

    with pytest.raises(NotImplementedError, match="etp=1"):
        load_lora_adapter(
            [chunk],
            "/definitely/missing/qwen_adapter",
            model_cfg,
            ParallelState(etp_size=2),
            lora_config=lora_config,
        )


def _tiny_config_kwargs():
    return dict(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=16,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=99,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=2,
        mlp_layer_types=["dense", "dense", "dense"],
    )


def test_olora_tail_factors_use_tail_subspace_without_singular_values():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import _olora_tail_factors

    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))

    lora_b, lora_a = _olora_tail_factors(weight, rank=2)

    assert tuple(lora_b.shape) == (4, 2)
    assert tuple(lora_a.shape) == (2, 4)
    torch.testing.assert_close(
        lora_b @ lora_a,
        torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0])),
    )
    assert not torch.allclose(
        lora_b @ lora_a,
        torch.diag(torch.tensor([0.0, 0.0, 2.0, 1.0])),
    )

    fp16_weight = weight.to(torch.float16)
    fp16_b, fp16_a = _olora_tail_factors(fp16_weight, rank=2)
    assert fp16_b.dtype == torch.float16
    assert fp16_a.dtype == torch.float16

    with pytest.raises(ValueError, match="finite base weight"):
        _olora_tail_factors(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), rank=1)
    with pytest.raises(ValueError, match="finite base weight"):
        _olora_tail_factors(torch.tensor([[1.0, float("inf")], [0.0, 1.0]]), rank=1)
    with pytest.raises(TypeError, match="floating-point base weight"):
        _olora_tail_factors(torch.tensor([[1, 0], [0, 1]], dtype=torch.int64), rank=1)
    with pytest.raises(TypeError, match="floating-point base weight"):
        _olora_tail_factors(torch.tensor([[True, False], [False, True]]), rank=1)


def test_olora_tail_factors_reject_non_finite_svd_factors(monkeypatch):
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import _olora_tail_factors

    def fake_svd(weight, *, full_matrices):
        assert full_matrices is False
        size = min(weight.shape)
        return (
            torch.full((weight.shape[0], size), float("nan")),
            torch.ones(size),
            torch.eye(size, weight.shape[1]),
        )

    monkeypatch.setattr(torch.linalg, "svd", fake_svd)
    with pytest.raises(ValueError, match="non-finite LoRA factors"):
        _olora_tail_factors(torch.eye(2), rank=1)


def test_qwen3_initialize_lora_olora_tail_supports_attention_and_single_local_expert():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
        _olora_tail_factors,
        initialize_lora_olora_tail,
    )
    from megatron.lite.primitive.modules.lora import LinearLoRA, SharedGroupedLinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    torch.manual_seed(1234)

    qkv_weight = torch.nn.Parameter(torch.randn(6, 4))
    proj_weight = torch.nn.Parameter(torch.randn(4, 2))
    fc1_weight = torch.nn.Parameter(torch.randn(8, 4))
    fc2_weight = torch.nn.Parameter(torch.randn(4, 4))
    qkv_lora = LinearLoRA(4, 6, rank=2, alpha=4, dropout=0.0)
    proj_lora = LinearLoRA(2, 4, rank=2, alpha=4, dropout=0.0)
    fc1_lora = SharedGroupedLinearLoRA(1, 4, 8, rank=2, alpha=4, dropout=0.0)
    fc2_lora = SharedGroupedLinearLoRA(1, 4, 4, rank=2, alpha=4, dropout=0.0)
    layer = SimpleNamespace(
        layer_idx=0,
        attn=SimpleNamespace(
            qkv=SimpleNamespace(weight=qkv_weight),
            proj=SimpleNamespace(weight=proj_weight),
            qkv_lora=qkv_lora,
            proj_lora=proj_lora,
        ),
        moe=SimpleNamespace(
            experts=SimpleNamespace(
                fc1=SimpleNamespace(weight0=fc1_weight),
                fc2=SimpleNamespace(weight0=fc2_weight),
                fc1_lora=fc1_lora,
                fc2_lora=fc2_lora,
                num_local_experts=1,
            )
        ),
    )

    result = initialize_lora_olora_tail(
        [SimpleNamespace(layers=[layer])], SimpleNamespace(), ParallelState()
    )

    assert result["init_lora_weights"] == "olora_tail"
    assert result["initialized_module_names"] == [
        "layers.0.self_attn.qkv_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.experts.down_proj",
    ]
    assert result["skipped_modules"] == []

    for lora, weight in (
        (qkv_lora, qkv_weight),
        (proj_lora, proj_weight),
        (fc1_lora, fc1_weight),
        (fc2_lora, fc2_weight),
    ):
        expected_b, expected_a = _olora_tail_factors(weight, rank=2)
        torch.testing.assert_close(lora.lora_a, expected_a)
        torch.testing.assert_close(lora.lora_b, expected_b)
        assert lora.init_lora_weights == "olora_tail"
        assert lora._olora_tail_initialized is True

    with pytest.raises(RuntimeError, match="already has OLoRA-tail initialization"):
        initialize_lora_olora_tail(
            [SimpleNamespace(layers=[layer])], SimpleNamespace(), ParallelState()
        )


def test_qwen3_initialize_lora_olora_tail_skips_multi_local_shared_experts():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.qwen3_moe.lite.lora_adapter import initialize_lora_olora_tail
    from megatron.lite.primitive.modules.lora import SharedGroupedLinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    fc1_lora = SharedGroupedLinearLoRA(2, 4, 8, rank=2, alpha=4, dropout=0.0)
    experts = SimpleNamespace(
        fc1=SimpleNamespace(
            weight0=torch.nn.Parameter(torch.randn(8, 4)),
            weight1=torch.nn.Parameter(torch.randn(8, 4)),
        ),
        fc2=SimpleNamespace(),
        fc1_lora=fc1_lora,
        fc2_lora=None,
        num_local_experts=2,
    )
    layer = SimpleNamespace(
        layer_idx=0,
        attn=SimpleNamespace(qkv_lora=None, proj_lora=None),
        moe=SimpleNamespace(experts=experts),
    )

    result = initialize_lora_olora_tail(
        [SimpleNamespace(layers=[layer])], SimpleNamespace(), ParallelState()
    )

    assert result["initialized_modules"] == 0
    assert result["skipped_modules"] == [
        "layers.0.mlp.experts.gate_up_proj: shared routed expert LoRA spans multiple "
        "local experts; exact OLoRA-tail initialization is undefined"
    ]
    assert not getattr(fc1_lora, "_olora_tail_initialized", False)


def test_initialize_lora_olora_tail_marks_modules_and_blocks_double_init():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import initialize_lora_olora_tail
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    def linear(weight):
        module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            module.weight.copy_(weight)
        return module

    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    lora = LinearLoRA(4, 4, rank=2, alpha=4)
    dsa = SimpleNamespace(
        q_a_proj=linear(weight),
        q_b_proj=linear(torch.eye(4)),
        kv_a_proj_with_mqa=linear(torch.eye(4)),
        kv_b_proj=linear(torch.eye(4)),
        o_proj=linear(torch.eye(4)),
        q_a_lora=lora,
        q_b_lora=None,
        kv_a_lora=None,
        kv_b_lora=None,
        o_lora=None,
    )
    layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=dsa),
        mlp=None,
        moe=None,
    )
    chunk = SimpleNamespace(layers=[layer], mtp=None)

    result = initialize_lora_olora_tail(
        [chunk], SimpleNamespace(num_hidden_layers=1), ParallelState()
    )

    assert result["init_lora_weights"] == "olora_tail"
    assert result["initialized_modules"] == 1
    assert result["initialized_module_names"] == ["layers.0.self_attn.q_a_proj"]
    assert lora.init_lora_weights == "olora_tail"
    assert lora._olora_tail_initialized is True
    torch.testing.assert_close(
        lora.lora_b @ lora.lora_a,
        torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0])),
    )

    with pytest.raises(RuntimeError, match="already has OLoRA-tail initialization"):
        initialize_lora_olora_tail(
            [chunk], SimpleNamespace(num_hidden_layers=1), ParallelState()
        )

    with torch.no_grad():
        dsa.q_a_proj.weight.copy_(torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])))
    forced = initialize_lora_olora_tail(
        [chunk], SimpleNamespace(num_hidden_layers=1), ParallelState(), force=True
    )
    assert forced["initialized_modules"] == 1
    torch.testing.assert_close(
        lora.lora_b @ lora.lora_a,
        torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0])),
    )


def test_glm5_protocol_initialize_lora_olora_tail_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    import json

    from megatron.lite.model.glm5.lite import protocol as glm5_protocol
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    def make_chunk():
        lora = LinearLoRA(4, 4, rank=2, alpha=4)
        dsa = SimpleNamespace(
            q_a_proj=_linear(torch, weight),
            q_b_proj=_linear(torch, torch.eye(4)),
            kv_a_proj_with_mqa=_linear(torch, torch.eye(4)),
            kv_b_proj=_linear(torch, torch.eye(4)),
            o_proj=_linear(torch, torch.eye(4)),
            q_a_lora=lora,
            q_b_lora=None,
            kv_a_lora=None,
            kv_b_lora=None,
            o_lora=None,
        )
        layer = SimpleNamespace(
            layer_idx=0,
            self_attention=SimpleNamespace(self_attention=dsa),
            mlp=None,
            moe=None,
        )
        return SimpleNamespace(layers=[layer], mtp=None), lora

    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    chunk, lora = make_chunk()
    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=0,
        n_shared_experts=0,
        moe_intermediate_size=0,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    ps = ParallelState()

    result = glm5_protocol.initialize_lora_olora_tail([chunk], model_cfg, ps)

    assert result["init_lora_weights"] == "olora_tail"
    assert result["initialized_modules"] == 1
    assert result["initialized_module_names"] == ["layers.0.self_attn.q_a_proj"]
    assert lora.init_lora_weights == "olora_tail"
    torch.testing.assert_close(
        lora.lora_b @ lora.lora_a,
        torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0])),
    )

    adapter_dir = tmp_path / "glm5_protocol_olora_tail_adapter"
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": "q_a",
    }
    meta = glm5_protocol.save_lora_adapter(
        [chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-olora-protocol-tiny",
        lora_config=lora_config,
        init_lora_weights=lora.init_lora_weights,
        metadata={"test_case": "glm5_protocol_olora_tail"},
    )
    assert meta["metadata"]["init_lora_weights"] == "olora_tail"
    assert meta["lora"]["target_modules"] == ["q_a_proj"]
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert adapter_config["init_lora_weights"] == "olora_tail"

    target_chunk, target_lora = make_chunk()
    with torch.no_grad():
        target_lora.lora_a.zero_()
        target_lora.lora_b.zero_()
    load_result = glm5_protocol.load_lora_adapter(
        [target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )
    assert load_result["loaded_tensors"] == 2
    torch.testing.assert_close(target_lora.lora_a, lora.lora_a)
    torch.testing.assert_close(target_lora.lora_b, lora.lora_b)


def test_initialize_lora_olora_tail_reaches_mtp_transformer_layers():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import initialize_lora_olora_tail
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    q_a_lora = LinearLoRA(4, 4, rank=2, alpha=4)
    gate_up_lora = LinearLoRA(4, 4, rank=2, alpha=4)
    down_lora = LinearLoRA(4, 4, rank=2, alpha=4)
    mtp_dsa = SimpleNamespace(
        q_a_proj=_linear(torch, weight),
        q_b_proj=_linear(torch, torch.eye(4)),
        kv_a_proj_with_mqa=_linear(torch, torch.eye(4)),
        kv_b_proj=_linear(torch, torch.eye(4)),
        o_proj=_linear(torch, torch.eye(4)),
        q_a_lora=q_a_lora,
        q_b_lora=None,
        kv_a_lora=None,
        kv_b_lora=None,
        o_lora=None,
    )
    mtp_layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=mtp_dsa),
        mlp=SimpleNamespace(
            gate_up=_linear(torch, weight),
            down=_linear(torch, weight),
            gate_up_lora=gate_up_lora,
            down_lora=down_lora,
        ),
        moe=None,
    )
    main_layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=_fake_dsa_with_base_without_lora(torch)),
        mlp=None,
        moe=None,
    )
    chunk = SimpleNamespace(
        layers=[main_layer],
        mtp=SimpleNamespace(layers=[SimpleNamespace(transformer_layer=mtp_layer)]),
    )

    result = initialize_lora_olora_tail(
        [chunk], SimpleNamespace(num_hidden_layers=2), ParallelState()
    )

    assert result["initialized_modules"] == 3
    assert result["initialized_module_names"] == [
        "layers.2.self_attn.q_a_proj",
        "layers.2.mlp.gate_up_proj",
        "layers.2.mlp.down_proj",
    ]
    expected_delta = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(q_a_lora.lora_b @ q_a_lora.lora_a, expected_delta)
    torch.testing.assert_close(gate_up_lora.lora_b @ gate_up_lora.lora_a, expected_delta)
    torch.testing.assert_close(down_lora.lora_b @ down_lora.lora_a, expected_delta)


def _linear(torch, weight):
    module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        module.weight.copy_(weight)
    return module


def _fake_dsa_with_base_without_lora(torch, width=4):
    weight = torch.eye(width)
    return SimpleNamespace(
        q_a_proj=_linear(torch, weight),
        q_b_proj=_linear(torch, weight),
        kv_a_proj_with_mqa=_linear(torch, weight),
        kv_b_proj=_linear(torch, weight),
        o_proj=_linear(torch, weight),
        q_a_lora=None,
        q_b_lora=None,
        kv_a_lora=None,
        kv_b_lora=None,
        o_lora=None,
    )


def _fake_shared_expert_without_lora(torch, width=4):
    weight = torch.eye(width)
    return SimpleNamespace(
        gate_up=_linear(torch, weight),
        down=_linear(torch, weight),
        gate_up_lora=None,
        down_lora=None,
    )


def _fake_grouped_linear(torch, weights):
    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for idx, weight in enumerate(weights):
                self.register_parameter(
                    f"weight{idx}",
                    torch.nn.Parameter(weight.clone()),
                )

    return FakeGroupedLinear()


def test_initialize_lora_olora_tail_supports_single_local_routed_expert():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import initialize_lora_olora_tail
    from megatron.lite.primitive.modules.lora import SharedGroupedLinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    fc1_lora = SharedGroupedLinearLoRA(1, 4, 4, rank=2, alpha=4)
    fc2_lora = SharedGroupedLinearLoRA(1, 4, 4, rank=2, alpha=4)
    experts = SimpleNamespace(
        num_local_experts=1,
        fc1=_fake_grouped_linear(torch, [weight]),
        fc2=_fake_grouped_linear(torch, [weight]),
        fc1_lora=fc1_lora,
        fc2_lora=fc2_lora,
    )
    layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=_fake_dsa_with_base_without_lora(torch)),
        mlp=None,
        moe=SimpleNamespace(
            experts=experts,
            shared_expert=_fake_shared_expert_without_lora(torch),
        ),
    )
    chunk = SimpleNamespace(layers=[layer], mtp=None)

    result = initialize_lora_olora_tail(
        [chunk], SimpleNamespace(num_hidden_layers=1), ParallelState()
    )

    assert result["initialized_modules"] == 2
    assert result["initialized_module_names"] == [
        "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.experts.down_proj",
    ]
    assert result["skipped_modules"] == []
    expected_delta = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(fc1_lora.lora_b @ fc1_lora.lora_a, expected_delta)
    torch.testing.assert_close(fc2_lora.lora_b @ fc2_lora.lora_a, expected_delta)
    assert fc1_lora.init_lora_weights == "olora_tail"
    assert fc2_lora.init_lora_weights == "olora_tail"

    with pytest.raises(RuntimeError, match="already has OLoRA-tail initialization"):
        initialize_lora_olora_tail(
            [chunk], SimpleNamespace(num_hidden_layers=1), ParallelState()
        )


def test_initialize_lora_olora_tail_supports_per_expert_grouped_lora():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import initialize_lora_olora_tail
    from megatron.lite.primitive.modules.lora import GroupedLinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    weights = [
        torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])),
        torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])),
    ]
    fc1_lora = GroupedLinearLoRA(2, 4, 4, rank=2, alpha=4)
    fc2_lora = GroupedLinearLoRA(2, 4, 4, rank=2, alpha=4)
    experts = SimpleNamespace(
        num_local_experts=2,
        fc1=_fake_grouped_linear(torch, weights),
        fc2=_fake_grouped_linear(torch, weights),
        fc1_lora=fc1_lora,
        fc2_lora=fc2_lora,
    )
    layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=_fake_dsa_with_base_without_lora(torch)),
        mlp=None,
        moe=SimpleNamespace(
            experts=experts,
            shared_expert=_fake_shared_expert_without_lora(torch),
        ),
    )
    chunk = SimpleNamespace(layers=[layer], mtp=None)

    result = initialize_lora_olora_tail(
        [chunk], SimpleNamespace(num_hidden_layers=1), ParallelState()
    )

    assert result["initialized_modules"] == 2
    assert result["skipped_modules"] == []
    expected_first = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    expected_second = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(fc1_lora.lora_b[0] @ fc1_lora.lora_a[0], expected_first)
    torch.testing.assert_close(fc1_lora.lora_b[1] @ fc1_lora.lora_a[1], expected_second)
    torch.testing.assert_close(fc2_lora.lora_b[0] @ fc2_lora.lora_a[0], expected_first)
    torch.testing.assert_close(fc2_lora.lora_b[1] @ fc2_lora.lora_a[1], expected_second)
    assert fc1_lora.init_lora_weights == "olora_tail"
    assert fc2_lora.init_lora_weights == "olora_tail"

    with pytest.raises(RuntimeError, match="already has OLoRA-tail initialization"):
        initialize_lora_olora_tail(
            [chunk], SimpleNamespace(num_hidden_layers=1), ParallelState()
        )


def test_initialize_lora_olora_tail_copies_into_dtensor_linear_lora():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import initialize_lora_olora_tail
    from megatron.lite.primitive.parallel import ParallelState

    mesh, created, temp_file = _init_single_rank_cpu_mesh(torch)
    try:
        weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        lora = SimpleNamespace(
            lora_a=_dtensor_param(torch, torch.zeros(2, 4), mesh),
            lora_b=_dtensor_param(torch, torch.zeros(4, 2), mesh),
            rank_partitioned_a=False,
            output_partitioned_b=False,
        )
        dsa = SimpleNamespace(
            q_a_proj=SimpleNamespace(weight=_dtensor_param(torch, weight, mesh)),
            q_b_proj=_linear(torch, torch.eye(4)),
            kv_a_proj_with_mqa=_linear(torch, torch.eye(4)),
            kv_b_proj=_linear(torch, torch.eye(4)),
            o_proj=_linear(torch, torch.eye(4)),
            q_a_lora=lora,
            q_b_lora=None,
            kv_a_lora=None,
            kv_b_lora=None,
            o_lora=None,
        )
        layer = SimpleNamespace(
            layer_idx=0,
            self_attention=SimpleNamespace(self_attention=dsa),
            mlp=None,
            moe=None,
        )

        result = initialize_lora_olora_tail(
            [SimpleNamespace(layers=[layer], mtp=None)],
            SimpleNamespace(num_hidden_layers=1),
            ParallelState(),
        )

        assert result["initialized_modules"] == 1
        assert lora.init_lora_weights == "olora_tail"
        assert lora._olora_tail_initialized is True
        torch.testing.assert_close(
            lora.lora_b.full_tensor() @ lora.lora_a.full_tensor(),
            torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0])),
        )
    finally:
        _cleanup_single_rank_cpu_mesh(torch, created, temp_file)


def test_initialize_lora_olora_tail_copies_into_dtensor_grouped_lora():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import initialize_lora_olora_tail
    from megatron.lite.primitive.parallel import ParallelState

    mesh, created, temp_file = _init_single_rank_cpu_mesh(torch)
    try:
        weights = [
            torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])),
            torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])),
        ]
        fc1_lora = SimpleNamespace(
            lora_a=_dtensor_param(torch, torch.zeros(2, 2, 4), mesh),
            lora_b=_dtensor_param(torch, torch.zeros(2, 4, 2), mesh),
            num_local_experts=2,
            shared_across_experts=False,
        )
        fc2_lora = SimpleNamespace(
            lora_a=_dtensor_param(torch, torch.zeros(2, 2, 4), mesh),
            lora_b=_dtensor_param(torch, torch.zeros(2, 4, 2), mesh),
            num_local_experts=2,
            shared_across_experts=False,
        )
        experts = SimpleNamespace(
            num_local_experts=2,
            fc1=_fake_grouped_linear(torch, weights),
            fc2=_fake_grouped_linear(torch, weights),
            fc1_lora=fc1_lora,
            fc2_lora=fc2_lora,
        )
        layer = SimpleNamespace(
            layer_idx=0,
            self_attention=SimpleNamespace(self_attention=_fake_dsa_with_base_without_lora(torch)),
            mlp=None,
            moe=SimpleNamespace(
                experts=experts,
                shared_expert=_fake_shared_expert_without_lora(torch),
            ),
        )

        result = initialize_lora_olora_tail(
            [SimpleNamespace(layers=[layer], mtp=None)],
            SimpleNamespace(num_hidden_layers=1),
            ParallelState(),
        )

        assert result["initialized_modules"] == 2
        assert fc1_lora.init_lora_weights == "olora_tail"
        assert fc2_lora.init_lora_weights == "olora_tail"
        expected_first = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0]))
        expected_second = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0]))
        torch.testing.assert_close(
            fc1_lora.lora_b.full_tensor()[0] @ fc1_lora.lora_a.full_tensor()[0],
            expected_first,
        )
        torch.testing.assert_close(
            fc1_lora.lora_b.full_tensor()[1] @ fc1_lora.lora_a.full_tensor()[1],
            expected_second,
        )
        torch.testing.assert_close(
            fc2_lora.lora_b.full_tensor()[0] @ fc2_lora.lora_a.full_tensor()[0],
            expected_first,
        )
        torch.testing.assert_close(
            fc2_lora.lora_b.full_tensor()[1] @ fc2_lora.lora_a.full_tensor()[1],
            expected_second,
        )
    finally:
        _cleanup_single_rank_cpu_mesh(torch, created, temp_file)


def test_initialize_lora_olora_tail_skips_multi_local_shared_routed_experts():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import initialize_lora_olora_tail
    from megatron.lite.primitive.modules.lora import SharedGroupedLinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    weights = [
        torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])),
        torch.diag(torch.tensor([8.0, 7.0, 6.0, 5.0])),
    ]
    experts = SimpleNamespace(
        num_local_experts=2,
        fc1=_fake_grouped_linear(torch, weights),
        fc2=_fake_grouped_linear(torch, weights),
        fc1_lora=SharedGroupedLinearLoRA(2, 4, 4, rank=2, alpha=4),
        fc2_lora=SharedGroupedLinearLoRA(2, 4, 4, rank=2, alpha=4),
    )
    layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=_fake_dsa_with_base_without_lora(torch)),
        mlp=None,
        moe=SimpleNamespace(
            experts=experts,
            shared_expert=_fake_shared_expert_without_lora(torch),
        ),
    )
    chunk = SimpleNamespace(layers=[layer], mtp=None)

    result = initialize_lora_olora_tail(
        [chunk], SimpleNamespace(num_hidden_layers=1), ParallelState()
    )

    assert result["initialized_modules"] == 0
    assert len(result["skipped_modules"]) == 2
    assert "exact OLoRA-tail initialization is undefined" in result["skipped_modules"][0]
    assert (
        result["skipped_reason"]
        == "some routed expert LoRA modules could not be initialized exactly"
    )
    assert not getattr(experts.fc1_lora, "_olora_tail_initialized", False)
    assert not getattr(experts.fc2_lora, "_olora_tail_initialized", False)


def _fake_dsa_without_lora():
    return SimpleNamespace(
        q_a_lora=None,
        q_b_lora=None,
        kv_a_lora=None,
        kv_b_lora=None,
        o_lora=None,
    )


def _fake_shared_routed_expert_chunk(torch, *, alpha=4, dropout=0.0, use_rslora=False):
    from megatron.lite.primitive.modules.lora import SharedGroupedLinearLoRA

    fc1_lora = SharedGroupedLinearLoRA(
        2, 3, 4, rank=2, alpha=alpha, dropout=dropout, use_rslora=use_rslora
    )
    fc2_lora = SharedGroupedLinearLoRA(
        2, 2, 3, rank=2, alpha=alpha, dropout=dropout, use_rslora=use_rslora
    )
    with torch.no_grad():
        fc1_lora.lora_a.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        fc1_lora.lora_b.copy_(
            torch.tensor([[10.0, 11.0], [12.0, 13.0], [20.0, 21.0], [22.0, 23.0]])
        )
        fc2_lora.lora_a.copy_(torch.tensor([[7.0, 8.0], [9.0, 10.0]]))
        fc2_lora.lora_b.copy_(torch.tensor([[30.0, 31.0], [32.0, 33.0], [34.0, 35.0]]))

    experts = SimpleNamespace(
        num_local_experts=2,
        fc1_lora=fc1_lora,
        fc2_lora=fc2_lora,
    )
    layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=_fake_dsa_without_lora()),
        mlp=None,
        moe=SimpleNamespace(
            experts=experts,
            shared_expert=SimpleNamespace(gate_up_lora=None, down_lora=None),
        ),
    )
    return SimpleNamespace(layers=[layer], mtp=None), experts


def _fake_fsdp_wrapped_chunk(torch, chunk):
    class FakeFSDPWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self._fsdp_wrapped_module = wrapped

    return FakeFSDPWrapper(chunk)


def test_initialize_lora_olora_tail_fsdp_style_wrapper_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import initialize_lora_olora_tail
    from megatron.lite.primitive.modules.lora import LinearLoRA
    from megatron.lite.primitive.parallel import ParallelState

    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    lora = LinearLoRA(4, 4, rank=2, alpha=4)
    dsa = SimpleNamespace(
        q_a_proj=_linear(torch, weight),
        q_b_proj=_linear(torch, torch.eye(4)),
        kv_a_proj_with_mqa=_linear(torch, torch.eye(4)),
        kv_b_proj=_linear(torch, torch.eye(4)),
        o_proj=_linear(torch, torch.eye(4)),
        q_a_lora=lora,
        q_b_lora=None,
        kv_a_lora=None,
        kv_b_lora=None,
        o_lora=None,
    )
    layer = SimpleNamespace(
        layer_idx=0,
        self_attention=SimpleNamespace(self_attention=dsa),
        mlp=None,
        moe=None,
    )
    wrapped_chunk = _fake_fsdp_wrapped_chunk(torch, SimpleNamespace(layers=[layer], mtp=None))

    result = initialize_lora_olora_tail(
        [wrapped_chunk], SimpleNamespace(num_hidden_layers=1), ParallelState()
    )

    assert result["init_lora_weights"] == "olora_tail"
    assert result["initialized_modules"] == 1
    assert result["initialized_module_names"] == ["layers.0.self_attn.q_a_proj"]
    assert lora.init_lora_weights == "olora_tail"
    assert lora._olora_tail_initialized is True
    torch.testing.assert_close(
        lora.lora_b @ lora.lora_a,
        torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0])),
    )


def test_glm5_shared_routed_expert_lora_round_trips_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        _attn_key,
        _expert_key,
        export_lora_adapter_state,
        load_lora_adapter_state,
    )
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(num_hidden_layers=1, num_experts=2)
    source_chunk, source_experts = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()

    state = export_lora_adapter_state([source_chunk], model_cfg, ps)

    assert len(state) == 12
    torch.testing.assert_close(
        state[_expert_key(0, 0, "gate_proj", "lora_A")],
        source_experts.fc1_lora.lora_a,
    )
    torch.testing.assert_close(
        state[_expert_key(0, 1, "gate_proj", "lora_A")],
        source_experts.fc1_lora.lora_a,
    )
    torch.testing.assert_close(
        state[_expert_key(0, 0, "gate_proj", "lora_B")],
        source_experts.fc1_lora.lora_b[:2],
    )
    torch.testing.assert_close(
        state[_expert_key(0, 0, "up_proj", "lora_B")],
        source_experts.fc1_lora.lora_b[2:],
    )
    torch.testing.assert_close(
        state[_expert_key(0, 1, "down_proj", "lora_B")],
        source_experts.fc2_lora.lora_b,
    )

    target_chunk, target_experts = _fake_shared_routed_expert_chunk(torch)
    with torch.no_grad():
        target_experts.fc1_lora.lora_a.zero_()
        target_experts.fc1_lora.lora_b.zero_()
        target_experts.fc2_lora.lora_a.zero_()
        target_experts.fc2_lora.lora_b.zero_()

    result = load_lora_adapter_state([target_chunk], state, model_cfg, ps)

    assert result["loaded_tensors"] == len(state)
    torch.testing.assert_close(target_experts.fc1_lora.lora_a, source_experts.fc1_lora.lora_a)
    torch.testing.assert_close(target_experts.fc1_lora.lora_b, source_experts.fc1_lora.lora_b)
    torch.testing.assert_close(target_experts.fc2_lora.lora_a, source_experts.fc2_lora.lora_a)
    torch.testing.assert_close(target_experts.fc2_lora.lora_b, source_experts.fc2_lora.lora_b)

    bad_state = {key: value.clone() for key, value in state.items()}
    bad_state[_expert_key(0, 1, "down_proj", "lora_B")][0, 0] += 1.0
    bad_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="shared expert fc2_lora"):
        load_lora_adapter_state([bad_chunk], bad_state, model_cfg, ps)
    bad_loose_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="shared expert fc2_lora"):
        load_lora_adapter_state([bad_loose_chunk], bad_state, model_cfg, ps, strict=False)

    bad_fc1_gate_a_state = {key: value.clone() for key, value in state.items()}
    bad_fc1_gate_a_state[_expert_key(0, 1, "gate_proj", "lora_A")][0, 0] += 1.0
    bad_fc1_gate_a_state[_expert_key(0, 1, "up_proj", "lora_A")][0, 0] += 1.0
    bad_fc1_gate_a_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="shared expert fc1_lora"):
        load_lora_adapter_state(
            [bad_fc1_gate_a_chunk], bad_fc1_gate_a_state, model_cfg, ps, strict=False
        )

    bad_fc1_gate_b_state = {key: value.clone() for key, value in state.items()}
    bad_fc1_gate_b_state[_expert_key(0, 1, "gate_proj", "lora_B")][0, 0] += 1.0
    bad_fc1_gate_b_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="shared expert fc1_lora"):
        load_lora_adapter_state(
            [bad_fc1_gate_b_chunk], bad_fc1_gate_b_state, model_cfg, ps, strict=False
        )

    bad_gate_up_a_state = {key: value.clone() for key, value in state.items()}
    bad_gate_up_a_state[_expert_key(0, 0, "up_proj", "lora_A")][0, 0] += 1.0
    bad_gate_up_a_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="gate/up lora_A"):
        load_lora_adapter_state(
            [bad_gate_up_a_chunk], bad_gate_up_a_state, model_cfg, ps, strict=False
        )


def test_glm5_load_lora_adapter_state_reports_missing_and_unexpected_keys_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        _attn_key,
        _expert_key,
        export_lora_adapter_state,
        load_lora_adapter_state,
    )
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(num_hidden_layers=1, num_experts=2)
    source_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()
    state = export_lora_adapter_state([source_chunk], model_cfg, ps)

    missing_state = {key: value.clone() for key, value in state.items()}
    missing_state.pop(_expert_key(0, 0, "gate_proj", "lora_A"))
    missing_state.pop(_expert_key(0, 0, "gate_proj", "lora_B"))
    missing_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(KeyError, match="Missing adapter tensor"):
        load_lora_adapter_state([missing_chunk], missing_state, model_cfg, ps)

    extra_state = {key: value.clone() for key, value in state.items()}
    extra_state[_attn_key(0, "q_a_proj", "lora_A")] = torch.zeros(2, 1)
    extra_state[_attn_key(0, "q_a_proj", "lora_B")] = torch.zeros(1, 2)
    strict_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="Unexpected adapter tensor keys"):
        load_lora_adapter_state([strict_chunk], extra_state, model_cfg, ps)

    loose_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    result = load_lora_adapter_state([loose_chunk], extra_state, model_cfg, ps, strict=False)
    assert result["loaded_tensors"] == len(state)

    string_false_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    result = load_lora_adapter_state(
        [string_false_chunk], extra_state, model_cfg, ps, strict="false"
    )
    assert result["loaded_tensors"] == len(state)

    with pytest.raises(ValueError, match="strict must be a boolean"):
        load_lora_adapter_state(
            [_fake_shared_routed_expert_chunk(torch)[0]], state, model_cfg, ps, strict="maybe"
        )
    with pytest.raises(TypeError, match="strict must be a boolean"):
        load_lora_adapter_state(
            [_fake_shared_routed_expert_chunk(torch)[0]], state, model_cfg, ps, strict=1
        )

    unsupported_lora_state = {key: value.clone() for key, value in state.items()}
    unsupported_lora_state["base_model.model.model.layers.0.lm_head.lora_A.weight"] = torch.zeros(
        1, 1
    )
    unsupported_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="unsupported adapter tensor keys"):
        load_lora_adapter_state(
            [unsupported_chunk],
            unsupported_lora_state,
            model_cfg,
            ps,
            strict=False,
        )

    non_lora_state = {key: value.clone() for key, value in state.items()}
    non_lora_state["base_model.model.model.layers.0.self_attn.q_a_proj.weight"] = torch.zeros(1)
    non_lora_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="non-adapter tensor keys"):
        load_lora_adapter_state(
            [non_lora_chunk],
            non_lora_state,
            model_cfg,
            ps,
            strict=False,
        )


def test_glm5_shared_routed_expert_lora_save_load_metadata_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    import json

    from megatron.lite.model.glm5.lite.lora_adapter import (
        _expert_key,
        load_lora_adapter,
        save_lora_adapter,
    )
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    source_chunk, source_experts = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()

    adapter_dir = tmp_path / "shared_routed_expert_adapter"
    meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-tiny",
        lora_config=lora_config,
        init_lora_weights=True,
        metadata={"test_case": "shared_routed_expert"},
    )

    assert meta["format"] == "megatron.lite_glm5_lora_peft_v1"
    assert meta["expert_lora_representation"] == "shared_local_expert_group"
    assert meta["lora"]["rank"] == 2
    assert meta["lora"]["alpha"] == 4
    assert meta["lora"]["use_rslora"] is False
    assert meta["lora"]["scaling_convention"] == "alpha_over_rank"
    assert meta["lora"]["scale"] == 2
    assert meta["lora"]["target_modules"] == ["gate_proj", "up_proj", "down_proj"]
    assert meta["metadata"]["test_case"] == "shared_routed_expert"

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == "local/glm5-tiny"
    assert adapter_config["target_modules"] == ["gate_proj", "up_proj", "down_proj"]
    assert adapter_config["r"] == 2
    assert adapter_config["lora_alpha"] == 4
    adapter_meta = json.loads((adapter_dir / "megatron.lite_adapter_meta.json").read_text())
    assert adapter_meta["base_model_name_or_path"] == "local/glm5-tiny"
    assert adapter_meta["lora"] == meta["lora"]

    mapping_metadata_dir = tmp_path / "glm5_mapping_metadata_adapter"
    mapping_lora_config = MappingProxyType(lora_config)
    mapping_meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        mapping_metadata_dir,
        base_model_name_or_path="local/glm5-tiny",
        lora_config=mapping_lora_config,
        init_lora_weights=True,
        metadata=MappingProxyType({"test_case": "shared_routed_expert_mapping"}),
    )
    assert mapping_meta["metadata"]["test_case"] == "shared_routed_expert_mapping"
    mapping_sidecar = json.loads(
        (mapping_metadata_dir / "megatron.lite_adapter_meta.json").read_text()
    )
    assert mapping_sidecar["metadata"]["test_case"] == "shared_routed_expert_mapping"

    spaced_base_dir = tmp_path / "glm5_spaced_base_adapter"
    spaced_base_meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        spaced_base_dir,
        base_model_name_or_path="  local/glm5-tiny  ",
        lora_config=lora_config,
        init_lora_weights=True,
        metadata={"test_case": "glm5_spaced_base"},
    )
    assert spaced_base_meta["base_model_name_or_path"] == "local/glm5-tiny"
    spaced_base_config = json.loads((spaced_base_dir / "adapter_config.json").read_text())
    spaced_base_sidecar = json.loads(
        (spaced_base_dir / "megatron.lite_adapter_meta.json").read_text()
    )
    assert spaced_base_config["base_model_name_or_path"] == "local/glm5-tiny"
    assert spaced_base_sidecar["base_model_name_or_path"] == "local/glm5-tiny"

    nested_mapping_metadata_dir = tmp_path / "glm5_nested_mapping_metadata_adapter"
    nested_mapping_meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        nested_mapping_metadata_dir,
        base_model_name_or_path="local/glm5-tiny",
        lora_config=lora_config,
        init_lora_weights=True,
        metadata=MappingProxyType(
            {
                "nested": MappingProxyType({"source": "mapping"}),
                "items": [MappingProxyType({"idx": 1})],
            }
        ),
    )
    assert nested_mapping_meta["metadata"]["nested"] == {"source": "mapping"}
    assert nested_mapping_meta["metadata"]["items"] == [{"idx": 1}]
    nested_mapping_sidecar = json.loads(
        (nested_mapping_metadata_dir / "megatron.lite_adapter_meta.json").read_text()
    )
    assert nested_mapping_sidecar["metadata"]["nested"] == {"source": "mapping"}
    assert nested_mapping_sidecar["metadata"]["items"] == [{"idx": 1}]

    serving_config = dict(adapter_config)
    serving_config["task_type"] = "causal_lm"
    serving_config["inference_mode"] = True
    serving_config["modules_to_save"] = []
    (adapter_dir / "adapter_config.json").write_text(json.dumps(serving_config, indent=2) + "\n")
    serving_target_chunk, serving_target_experts = _fake_shared_routed_expert_chunk(torch)
    with torch.no_grad():
        serving_target_experts.fc1_lora.lora_a.zero_()
        serving_target_experts.fc1_lora.lora_b.zero_()
        serving_target_experts.fc2_lora.lora_a.zero_()
        serving_target_experts.fc2_lora.lora_b.zero_()
    serving_result = load_lora_adapter(
        [serving_target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=mapping_lora_config,
    )
    assert serving_result["loaded_tensors"] == 12
    torch.testing.assert_close(
        serving_target_experts.fc1_lora.lora_a, source_experts.fc1_lora.lora_a
    )
    torch.testing.assert_close(
        serving_target_experts.fc1_lora.lora_b, source_experts.fc1_lora.lora_b
    )
    torch.testing.assert_close(
        serving_target_experts.fc2_lora.lora_a, source_experts.fc2_lora.lora_a
    )
    torch.testing.assert_close(
        serving_target_experts.fc2_lora.lora_b, source_experts.fc2_lora.lora_b
    )
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n")

    metadata_path = adapter_dir / "megatron.lite_adapter_meta.json"
    metadata_text = metadata_path.read_text()
    metadata_path.unlink()
    peft_only_target_chunk, peft_only_target_experts = _fake_shared_routed_expert_chunk(torch)
    with torch.no_grad():
        peft_only_target_experts.fc1_lora.lora_a.zero_()
        peft_only_target_experts.fc1_lora.lora_b.zero_()
        peft_only_target_experts.fc2_lora.lora_a.zero_()
        peft_only_target_experts.fc2_lora.lora_b.zero_()
    peft_only_result = load_lora_adapter(
        [peft_only_target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )
    assert peft_only_result["loaded_tensors"] == 12
    torch.testing.assert_close(
        peft_only_target_experts.fc1_lora.lora_a, source_experts.fc1_lora.lora_a
    )
    torch.testing.assert_close(
        peft_only_target_experts.fc1_lora.lora_b, source_experts.fc1_lora.lora_b
    )
    torch.testing.assert_close(
        peft_only_target_experts.fc2_lora.lora_a, source_experts.fc2_lora.lora_a
    )
    torch.testing.assert_close(
        peft_only_target_experts.fc2_lora.lora_b, source_experts.fc2_lora.lora_b
    )
    metadata_path.write_text(metadata_text)

    sentinel_path = adapter_dir / "operator-note.txt"
    sentinel_path.write_text("keep me")
    overwrite_meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-tiny-v2",
        lora_config=lora_config,
        init_lora_weights=True,
        metadata={"test_case": "shared_routed_expert_v2"},
    )
    assert sentinel_path.read_text() == "keep me"
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    adapter_meta = json.loads((adapter_dir / "megatron.lite_adapter_meta.json").read_text())
    assert adapter_config["base_model_name_or_path"] == "local/glm5-tiny-v2"
    assert adapter_meta["base_model_name_or_path"] == "local/glm5-tiny-v2"
    assert adapter_meta["metadata"]["test_case"] == "shared_routed_expert_v2"
    assert overwrite_meta["metadata"]["test_case"] == "shared_routed_expert_v2"

    old_model_bytes = (adapter_dir / "adapter_model.safetensors").read_bytes()
    old_config_text = (adapter_dir / "adapter_config.json").read_text()
    old_meta_text = (adapter_dir / "megatron.lite_adapter_meta.json").read_text()
    original_replace = Path.replace

    def fail_config_install(self, target):
        if self.name == "adapter_config.json" and ".shared_routed_expert_adapter.tmp-" in str(
            self
        ):
            raise RuntimeError("synthetic glm5 artifact replace failure")
        return original_replace(self, target)

    Path.replace = fail_config_install
    try:
        with pytest.raises(RuntimeError, match="synthetic glm5 artifact replace failure"):
            save_lora_adapter(
                [source_chunk],
                model_cfg,
                ps,
                adapter_dir,
                base_model_name_or_path="local/glm5-tiny-v3",
                lora_config=lora_config,
                init_lora_weights=True,
                metadata={"test_case": "shared_routed_expert_v3"},
            )
    finally:
        Path.replace = original_replace
    assert (adapter_dir / "adapter_model.safetensors").read_bytes() == old_model_bytes
    assert (adapter_dir / "adapter_config.json").read_text() == old_config_text
    assert (adapter_dir / "megatron.lite_adapter_meta.json").read_text() == old_meta_text
    assert sentinel_path.read_text() == "keep me"
    assert not list(tmp_path.glob(".shared_routed_expert_adapter.tmp-*"))
    assert not list(tmp_path.glob(".shared_routed_expert_adapter.bak-*"))

    bad_metadata_dir = tmp_path / "glm5_bad_metadata"
    with pytest.raises(TypeError, match="Adapter metadata metadata must be an object"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_metadata_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
            metadata=[("test_case", "glm5_bad_metadata")],
        )
    assert not bad_metadata_dir.exists()

    bad_json_metadata_dir = tmp_path / "glm5_bad_json_metadata"
    with pytest.raises(TypeError, match="JSON-serializable"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_json_metadata_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
            metadata={"bad": {"not", "json"}},
        )
    assert not bad_json_metadata_dir.exists()

    bad_nan_metadata_dir = tmp_path / "glm5_bad_nan_metadata"
    with pytest.raises(TypeError, match="JSON-serializable"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_nan_metadata_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
            metadata={"bad": [math.nan, math.inf]},
        )
    assert not bad_nan_metadata_dir.exists()

    bad_metadata_key_dir = tmp_path / "glm5_bad_metadata_key"
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_metadata_key_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
            metadata={"nested": [{1: "bad"}]},
        )
    assert not bad_metadata_key_dir.exists()

    bad_model_json_dir = tmp_path / "glm5_bad_model_json"
    bad_model_cfg = SimpleNamespace(**vars(model_cfg))
    bad_model_cfg.hidden_size = math.inf
    with pytest.raises(TypeError, match="megatron.lite_adapter_meta.json"):
        save_lora_adapter(
            [source_chunk],
            bad_model_cfg,
            ps,
            bad_model_json_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
        )
    assert not bad_model_json_dir.exists()

    failed_write_dir = tmp_path / "glm5_failed_atomic_write"
    original_save_file = safetensors_torch.save_file

    def fail_after_partial_file(state, path):
        Path(path).write_bytes(b"partial adapter")
        raise RuntimeError("synthetic glm5 adapter write failure")

    safetensors_torch.save_file = fail_after_partial_file
    try:
        with pytest.raises(RuntimeError, match="synthetic glm5 adapter write failure"):
            save_lora_adapter(
                [source_chunk],
                model_cfg,
                ps,
                failed_write_dir,
                base_model_name_or_path="local/glm5-tiny",
                lora_config=lora_config,
            )
    finally:
        safetensors_torch.save_file = original_save_file
    assert not failed_write_dir.exists()
    assert not list(tmp_path.glob(".glm5_failed_atomic_write.tmp-*"))

    output_file = tmp_path / "glm5_output_file"
    output_file.write_text("keep this file")
    with pytest.raises(FileExistsError, match="not a directory"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            output_file,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
        )
    assert output_file.read_text() == "keep this file"
    assert not list(tmp_path.glob(".glm5_output_file.tmp-*"))

    blocked_artifact_dir = tmp_path / "glm5_blocked_artifact_dir"
    blocked_artifact_dir.mkdir()
    (blocked_artifact_dir / "adapter_config.json").mkdir()
    with pytest.raises(IsADirectoryError, match="Adapter artifact path must be a file"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            blocked_artifact_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
        )
    assert (blocked_artifact_dir / "adapter_config.json").is_dir()
    assert not (blocked_artifact_dir / "adapter_model.safetensors").exists()
    assert not list(tmp_path.glob(".glm5_blocked_artifact_dir.tmp-*"))

    bad_init_dir = tmp_path / "glm5_bad_init"
    with pytest.raises(TypeError, match="init_lora_weights"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_init_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
            init_lora_weights=object(),
        )
    assert not bad_init_dir.exists()

    bad_empty_init_dir = tmp_path / "glm5_bad_empty_init"
    with pytest.raises(ValueError, match="init_lora_weights string must be non-empty"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_empty_init_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
            init_lora_weights=" ",
        )
    assert not bad_empty_init_dir.exists()

    bad_metadata_init_dir = tmp_path / "glm5_bad_metadata_init"
    with pytest.raises(TypeError, match="init_lora_weights"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_metadata_init_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
            metadata={"init_lora_weights": {"mode": "olora_tail"}},
        )
    assert not bad_metadata_init_dir.exists()

    bad_empty_metadata_init_dir = tmp_path / "glm5_bad_empty_metadata_init"
    with pytest.raises(ValueError, match="init_lora_weights string must be non-empty"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            bad_empty_metadata_init_dir,
            base_model_name_or_path="local/glm5-tiny",
            lora_config=lora_config,
            metadata={"init_lora_weights": ""},
        )
    assert not bad_empty_metadata_init_dir.exists()

    saved_state = safetensors_torch.load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert set(saved_state) == {
        _expert_key(0, expert_idx, module, suffix)
        for expert_idx in range(2)
        for module, suffix in (
            ("gate_proj", "lora_A"),
            ("gate_proj", "lora_B"),
            ("up_proj", "lora_A"),
            ("up_proj", "lora_B"),
            ("down_proj", "lora_A"),
            ("down_proj", "lora_B"),
        )
    }

    target_chunk, target_experts = _fake_shared_routed_expert_chunk(torch)
    with torch.no_grad():
        target_experts.fc1_lora.lora_a.zero_()
        target_experts.fc1_lora.lora_b.zero_()
        target_experts.fc2_lora.lora_a.zero_()
        target_experts.fc2_lora.lora_b.zero_()

    result = load_lora_adapter(
        [target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )

    assert result["loaded_tensors"] == len(saved_state)
    torch.testing.assert_close(target_experts.fc1_lora.lora_a, source_experts.fc1_lora.lora_a)
    torch.testing.assert_close(target_experts.fc1_lora.lora_b, source_experts.fc1_lora.lora_b)
    torch.testing.assert_close(target_experts.fc2_lora.lora_a, source_experts.fc2_lora.lora_a)
    torch.testing.assert_close(target_experts.fc2_lora.lora_b, source_experts.fc2_lora.lora_b)


def test_glm5_lora_adapter_protocol_wrappers_round_trip_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.glm5.lite import protocol as glm5_protocol
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    ps = ParallelState()
    source_chunk, source_experts = _fake_shared_routed_expert_chunk(torch)

    state = glm5_protocol.export_lora_adapter_state([source_chunk], model_cfg, ps)
    assert len(state) == 12

    state_target_chunk, state_target_experts = _fake_shared_routed_expert_chunk(torch)
    with torch.no_grad():
        state_target_experts.fc1_lora.lora_a.zero_()
        state_target_experts.fc1_lora.lora_b.zero_()
        state_target_experts.fc2_lora.lora_a.zero_()
        state_target_experts.fc2_lora.lora_b.zero_()
    state_result = glm5_protocol.load_lora_adapter_state(
        [state_target_chunk], state, model_cfg, ps
    )
    assert state_result["loaded_tensors"] == len(state)
    torch.testing.assert_close(
        state_target_experts.fc1_lora.lora_a, source_experts.fc1_lora.lora_a
    )
    torch.testing.assert_close(
        state_target_experts.fc1_lora.lora_b, source_experts.fc1_lora.lora_b
    )
    torch.testing.assert_close(
        state_target_experts.fc2_lora.lora_a, source_experts.fc2_lora.lora_a
    )
    torch.testing.assert_close(
        state_target_experts.fc2_lora.lora_b, source_experts.fc2_lora.lora_b
    )

    adapter_dir = tmp_path / "glm5_protocol_adapter"
    meta = glm5_protocol.save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-protocol-tiny",
        lora_config=lora_config,
        metadata={"test_case": "glm5_protocol_lifecycle"},
    )
    assert meta["metadata"]["test_case"] == "glm5_protocol_lifecycle"
    assert meta["expert_lora_representation"] == "shared_local_expert_group"

    dir_target_chunk, dir_target_experts = _fake_shared_routed_expert_chunk(torch)
    with torch.no_grad():
        dir_target_experts.fc1_lora.lora_a.zero_()
        dir_target_experts.fc1_lora.lora_b.zero_()
        dir_target_experts.fc2_lora.lora_a.zero_()
        dir_target_experts.fc2_lora.lora_b.zero_()
    dir_result = glm5_protocol.load_lora_adapter(
        [dir_target_chunk],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )
    assert dir_result["loaded_tensors"] == len(state)
    torch.testing.assert_close(
        dir_target_experts.fc1_lora.lora_a, source_experts.fc1_lora.lora_a
    )
    torch.testing.assert_close(
        dir_target_experts.fc1_lora.lora_b, source_experts.fc1_lora.lora_b
    )
    torch.testing.assert_close(
        dir_target_experts.fc2_lora.lora_a, source_experts.fc2_lora.lora_a
    )
    torch.testing.assert_close(
        dir_target_experts.fc2_lora.lora_b, source_experts.fc2_lora.lora_b
    )


def test_glm5_lora_adapter_fsdp_style_wrappers_round_trip_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        export_lora_adapter_state,
        load_lora_adapter,
        load_lora_adapter_state,
        save_lora_adapter,
    )
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    ps = ParallelState()

    source_chunk, source_experts = _fake_shared_routed_expert_chunk(torch)
    source_wrapped = _fake_fsdp_wrapped_chunk(torch, source_chunk)
    state = export_lora_adapter_state([source_wrapped], model_cfg, ps)
    assert len(state) == 12

    state_target_chunk, state_target_experts = _fake_shared_routed_expert_chunk(torch)
    state_target_wrapped = _fake_fsdp_wrapped_chunk(torch, state_target_chunk)
    with torch.no_grad():
        state_target_experts.fc1_lora.lora_a.zero_()
        state_target_experts.fc1_lora.lora_b.zero_()
        state_target_experts.fc2_lora.lora_a.zero_()
        state_target_experts.fc2_lora.lora_b.zero_()
    state_result = load_lora_adapter_state([state_target_wrapped], state, model_cfg, ps)
    assert state_result["loaded_tensors"] == len(state)
    torch.testing.assert_close(
        state_target_experts.fc1_lora.lora_a, source_experts.fc1_lora.lora_a
    )
    torch.testing.assert_close(
        state_target_experts.fc1_lora.lora_b, source_experts.fc1_lora.lora_b
    )
    torch.testing.assert_close(
        state_target_experts.fc2_lora.lora_a, source_experts.fc2_lora.lora_a
    )
    torch.testing.assert_close(
        state_target_experts.fc2_lora.lora_b, source_experts.fc2_lora.lora_b
    )

    adapter_dir = tmp_path / "glm5_fsdp_style_adapter"
    meta = save_lora_adapter(
        [source_wrapped],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-fsdp-wrapper-tiny",
        lora_config=lora_config,
        metadata={"test_case": "glm5_fsdp_style_adapter"},
    )
    assert meta["metadata"]["test_case"] == "glm5_fsdp_style_adapter"
    assert meta["parallel"] == {"tp": 1, "ep": 1, "etp": 1, "pp": 1}

    dir_target_chunk, dir_target_experts = _fake_shared_routed_expert_chunk(torch)
    dir_target_wrapped = _fake_fsdp_wrapped_chunk(torch, dir_target_chunk)
    with torch.no_grad():
        dir_target_experts.fc1_lora.lora_a.zero_()
        dir_target_experts.fc1_lora.lora_b.zero_()
        dir_target_experts.fc2_lora.lora_a.zero_()
        dir_target_experts.fc2_lora.lora_b.zero_()
    dir_result = load_lora_adapter(
        [dir_target_wrapped],
        adapter_dir,
        model_cfg,
        ps,
        lora_config=lora_config,
    )
    assert dir_result["loaded_tensors"] == len(state)
    torch.testing.assert_close(
        dir_target_experts.fc1_lora.lora_a, source_experts.fc1_lora.lora_a
    )
    torch.testing.assert_close(
        dir_target_experts.fc1_lora.lora_b, source_experts.fc1_lora.lora_b
    )
    torch.testing.assert_close(
        dir_target_experts.fc2_lora.lora_a, source_experts.fc2_lora.lora_a
    )
    torch.testing.assert_close(
        dir_target_experts.fc2_lora.lora_b, source_experts.fc2_lora.lora_b
    )


def test_glm5_shared_routed_expert_rslora_round_trips_metadata_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    import json

    from megatron.lite.model.glm5.lite.lora_adapter import load_lora_adapter, save_lora_adapter
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
        "use_rslora": True,
    }
    source_chunk, source_experts = _fake_shared_routed_expert_chunk(torch, use_rslora=True)
    ps = ParallelState()

    adapter_dir = tmp_path / "shared_routed_expert_rslora_adapter"
    meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-shared-routed-rslora",
        lora_config=lora_config,
    )

    assert meta["lora"]["use_rslora"] is True
    assert meta["lora"]["scaling_convention"] == "alpha_over_sqrt_rank"
    assert math.isclose(meta["lora"]["scale"], 4 / math.sqrt(2), rel_tol=0.0, abs_tol=1e-6)
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert adapter_config["use_rslora"] is True

    target_chunk, target_experts = _fake_shared_routed_expert_chunk(torch, use_rslora=True)
    with torch.no_grad():
        target_experts.fc1_lora.lora_a.zero_()
        target_experts.fc1_lora.lora_b.zero_()
        target_experts.fc2_lora.lora_a.zero_()
        target_experts.fc2_lora.lora_b.zero_()

    result = load_lora_adapter([target_chunk], adapter_dir, model_cfg, ps, lora_config=lora_config)
    assert result["loaded_tensors"] == 12
    torch.testing.assert_close(target_experts.fc1_lora.lora_a, source_experts.fc1_lora.lora_a)
    torch.testing.assert_close(target_experts.fc1_lora.lora_b, source_experts.fc1_lora.lora_b)
    torch.testing.assert_close(target_experts.fc2_lora.lora_a, source_experts.fc2_lora.lora_a)
    torch.testing.assert_close(target_experts.fc2_lora.lora_b, source_experts.fc2_lora.lora_b)


def test_glm5_load_lora_adapter_validates_metadata_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    import json

    from megatron.lite.model.glm5.lite.lora_adapter import load_lora_adapter, save_lora_adapter
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    source_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()

    adapter_dir = tmp_path / "metadata_validation_adapter"
    save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-metadata-validation",
        lora_config=lora_config,
    )
    metadata_path = adapter_dir / "megatron.lite_adapter_meta.json"
    base_metadata = json.loads(metadata_path.read_text())

    def expect_metadata_reject(
        updater,
        match: str,
        exception: type[Exception] = ValueError,
    ) -> None:
        metadata = json.loads(json.dumps(base_metadata))
        updater(metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
        with pytest.raises(exception, match=match):
            load_lora_adapter([target_chunk], adapter_dir, model_cfg, ps, lora_config=lora_config)

    expect_metadata_reject(
        lambda metadata: metadata.update({"format": "unexpected_format"}),
        "metadata format",
    )
    for required_field in (
        "base_model_name_or_path",
        "num_tensors",
        "num_parameters",
        "expert_lora_representation",
        "lora",
        "parallel",
        "model",
        "metadata",
    ):
        expect_metadata_reject(
            lambda metadata, field=required_field: metadata.pop(field),
            f"Adapter metadata {required_field} is required",
        )
    for object_field, match in (
        ("lora", "Adapter metadata lora must be an object"),
        ("parallel", "Adapter metadata parallel must be an object"),
        ("model", "Adapter metadata model must be an object"),
        ("metadata", "Adapter metadata metadata must be an object"),
    ):
        expect_metadata_reject(
            lambda metadata, field=object_field: metadata.update({field: None}),
            match,
            exception=TypeError,
        )
        expect_metadata_reject(
            lambda metadata, field=object_field: metadata.update({field: []}),
            match,
            exception=TypeError,
        )
    for required_field in (
        "rank",
        "alpha",
        "dropout",
        "use_rslora",
        "scaling_convention",
        "scale",
        "target_modules",
    ):
        expect_metadata_reject(
            lambda metadata, field=required_field: metadata["lora"].pop(field),
            f"Adapter metadata lora.{required_field} is required",
        )
    for required_field in ("tp", "ep", "etp", "pp"):
        expect_metadata_reject(
            lambda metadata, field=required_field: metadata["parallel"].pop(field),
            f"Adapter metadata parallel.{required_field} is required",
        )
    for required_field in (
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "num_experts",
        "n_shared_experts",
        "moe_intermediate_size",
        "num_nextn_predict_layers",
        "mtp_use_repeated_layer",
    ):
        expect_metadata_reject(
            lambda metadata, field=required_field: metadata["model"].pop(field),
            f"Adapter metadata model.{required_field} is required",
        )
    expect_metadata_reject(
        lambda metadata: metadata.update({"base_model_name_or_path": "local/glm5-other"}),
        "base_model_name_or_path",
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"base_model_name_or_path": ""}),
        "base_model_name_or_path must be non-empty",
    )
    expect_metadata_reject(
        lambda metadata: metadata["model"].update({"num_experts": 3}),
        "model.num_experts=3",
    )
    expect_metadata_reject(
        lambda metadata: metadata["model"].update({"hidden_size": 3.0}),
        "Adapter metadata model.hidden_size must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["model"].update({"mtp_use_repeated_layer": 0}),
        "Adapter metadata model.mtp_use_repeated_layer must be a boolean",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"tp": 2}),
        "parallel.tp=2",
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"num_tensors": 999}),
        "num_tensors=999",
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"num_tensors": str(base_metadata["num_tensors"])}),
        "Adapter metadata num_tensors must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"num_parameters": 999}),
        "num_parameters=999",
    )
    expect_metadata_reject(
        lambda metadata: metadata.update(
            {"num_parameters": str(base_metadata["num_parameters"])}
        ),
        "Adapter metadata num_parameters must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"expert_lora_representation": "per_expert"}),
        "expert_lora_representation",
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"tp": "1"}),
        "Adapter metadata parallel.tp must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"ep": "1"}),
        "Adapter metadata parallel.ep must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"ep": True}),
        "Adapter metadata parallel.ep must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["parallel"].update({"pp": True}),
        "Adapter metadata parallel.pp must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"rank": 3}),
        "lora.rank=3",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"alpha": 5}),
        "lora.alpha=5",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"target_modules": ["gate_proj"]}),
        "lora.target_modules",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"scaling_convention": "alpha_over_sqrt_rank"}),
        "scaling_convention",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"scale": 3}),
        "lora.scale=3",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"rank": "2"}),
        "Adapter metadata lora.rank must be an integer",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"alpha": "4"}),
        "Adapter metadata lora.alpha must be a finite number",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"dropout": "0.0"}),
        "Adapter metadata lora.dropout must be a finite number",
        exception=TypeError,
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"scale": "2.0"}),
        "Adapter metadata lora.scale must be a finite number",
        exception=TypeError,
    )


def test_glm5_load_lora_adapter_rejects_config_metadata_disagreement_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    import json

    from megatron.lite.model.glm5.lite.lora_adapter import load_lora_adapter, save_lora_adapter
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    source_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()

    adapter_dir = tmp_path / "adapter_config_metadata_disagreement"
    save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-config-metadata-disagreement",
        lora_config=lora_config,
        init_lora_weights=True,
    )
    metadata_path = adapter_dir / "megatron.lite_adapter_meta.json"
    base_metadata = json.loads(metadata_path.read_text())

    def expect_metadata_reject(
        updater,
        match: str,
        exception: type[Exception] = ValueError,
    ) -> None:
        metadata = json.loads(json.dumps(base_metadata))
        updater(metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
        with pytest.raises(exception, match=match):
            load_lora_adapter([target_chunk], adapter_dir, model_cfg, ps)

    expect_metadata_reject(
        lambda metadata: metadata.update({"base_model_name_or_path": "local/glm5-other"}),
        "base_model_name_or_path",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"alpha": 5, "scale": 2.5}),
        "lora_alpha=4",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update({"dropout": 0.25}),
        "lora_dropout=0.0",
    )
    expect_metadata_reject(
        lambda metadata: metadata["lora"].update(
            {
                "use_rslora": True,
                "scaling_convention": "alpha_over_sqrt_rank",
                "scale": 4 / math.sqrt(2),
            }
        ),
        "use_rslora=False",
    )
    expect_metadata_reject(
        lambda metadata: metadata["metadata"].update({"init_lora_weights": "olora_tail"}),
        "init_lora_weights=True",
    )
    expect_metadata_reject(
        lambda metadata: metadata.update({"metadata": []}),
        "metadata must be an object",
        exception=TypeError,
    )


def test_glm5_load_lora_adapter_rejects_non_object_json_sidecars_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.glm5.lite.lora_adapter import load_lora_adapter, save_lora_adapter
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    source_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()

    adapter_dir = tmp_path / "non_object_json_sidecars"
    save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-non-object-sidecars",
        lora_config=lora_config,
    )

    (adapter_dir / "adapter_config.json").write_text("[]\n")
    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(TypeError, match="adapter_config.json must be a JSON object"):
        load_lora_adapter([target_chunk], adapter_dir, model_cfg, ps, lora_config=lora_config)

    missing_tensor_bad_config_dir = tmp_path / "non_object_config_without_adapter_model"
    missing_tensor_bad_config_dir.mkdir()
    (missing_tensor_bad_config_dir / "adapter_config.json").write_text("[]\n")
    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(TypeError, match="adapter_config.json must be a JSON object"):
        load_lora_adapter(
            [target_chunk], missing_tensor_bad_config_dir, model_cfg, ps, lora_config=lora_config
        )

    directory_config_dir = tmp_path / "directory_config_without_adapter_model"
    directory_config_dir.mkdir()
    (directory_config_dir / "adapter_config.json").mkdir()
    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(IsADirectoryError, match="adapter_config.json path must be a file"):
        load_lora_adapter(
            [target_chunk], directory_config_dir, model_cfg, ps, lora_config=lora_config
        )

    missing_tensor_bad_constant_config_dir = (
        tmp_path / "nonstandard_config_without_adapter_model"
    )
    missing_tensor_bad_constant_config_dir.mkdir()
    (missing_tensor_bad_constant_config_dir / "adapter_config.json").write_text(
        '{"peft_type": "LORA", "lora_alpha": NaN}\n'
    )
    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="adapter_config.json must be standard JSON"):
        load_lora_adapter(
            [target_chunk],
            missing_tensor_bad_constant_config_dir,
            model_cfg,
            ps,
            lora_config=lora_config,
        )

    missing_tensor_bad_meta_dir = tmp_path / "non_object_meta_without_adapter_model"
    missing_tensor_bad_meta_dir.mkdir()
    (missing_tensor_bad_meta_dir / "adapter_config.json").write_text(
        """{
  "peft_type": "LORA",
  "base_model_name_or_path": "local/glm5-bad-meta",
  "r": 2,
  "target_modules": ["linear_fc1", "linear_fc2"],
  "lora_alpha": 4,
  "use_rslora": false,
  "lora_dropout": 0.0
}
"""
    )
    (missing_tensor_bad_meta_dir / "megatron.lite_adapter_meta.json").write_text(
        '"bad-meta"\n'
    )
    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(TypeError, match="megatron.lite_adapter_meta.json must be a JSON object"):
        load_lora_adapter(
            [target_chunk], missing_tensor_bad_meta_dir, model_cfg, ps, lora_config=lora_config
        )

    directory_meta_dir = tmp_path / "directory_meta_without_adapter_model"
    directory_meta_dir.mkdir()
    (directory_meta_dir / "adapter_config.json").write_text(
        """{
  "peft_type": "LORA",
  "base_model_name_or_path": "local/glm5-bad-meta",
  "r": 2,
  "target_modules": ["linear_fc1", "linear_fc2"],
  "lora_alpha": 4,
  "use_rslora": false,
  "lora_dropout": 0.0
}
"""
    )
    (directory_meta_dir / "megatron.lite_adapter_meta.json").mkdir()
    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(
        IsADirectoryError, match="megatron.lite_adapter_meta.json path must be a file"
    ):
        load_lora_adapter(
            [target_chunk], directory_meta_dir, model_cfg, ps, lora_config=lora_config
        )

    missing_tensor_bad_constant_meta_dir = tmp_path / "nonstandard_meta_without_adapter_model"
    missing_tensor_bad_constant_meta_dir.mkdir()
    (missing_tensor_bad_constant_meta_dir / "adapter_config.json").write_text(
        """{
  "peft_type": "LORA",
  "base_model_name_or_path": "local/glm5-bad-meta",
  "r": 2,
  "target_modules": ["linear_fc1", "linear_fc2"],
  "lora_alpha": 4,
  "use_rslora": false,
  "lora_dropout": 0.0
}
"""
    )
    (missing_tensor_bad_constant_meta_dir / "megatron.lite_adapter_meta.json").write_text(
        '{"format": Infinity}\n'
    )
    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="megatron.lite_adapter_meta.json must be standard JSON"):
        load_lora_adapter(
            [target_chunk],
            missing_tensor_bad_constant_meta_dir,
            model_cfg,
            ps,
            lora_config=lora_config,
        )

    save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-non-object-sidecars",
        lora_config=lora_config,
    )
    (adapter_dir / "megatron.lite_adapter_meta.json").write_text('"bad-meta"\n')
    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(TypeError, match="megatron.lite_adapter_meta.json must be a JSON object"):
        load_lora_adapter([target_chunk], adapter_dir, model_cfg, ps, lora_config=lora_config)


def test_glm5_load_lora_adapter_rejects_config_mismatches_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    import json

    from megatron.lite.model.glm5.lite.lora_adapter import load_lora_adapter, save_lora_adapter
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    source_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()

    adapter_dir = tmp_path / "adapter_config_mismatch"
    save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-tiny",
        lora_config=lora_config,
    )

    config_path = adapter_dir / "adapter_config.json"
    base_adapter_config = json.loads(config_path.read_text())

    def expect_reject(
        updates,
        match: str,
        exception: type[Exception] = ValueError,
    ) -> None:
        config = dict(base_adapter_config)
        config.update(updates)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
        with pytest.raises(exception, match=match):
            load_lora_adapter(
                [target_chunk],
                adapter_dir,
                model_cfg,
                ps,
                lora_config=lora_config,
            )

    expect_reject({"r": 3}, "rank r=3")
    expect_reject({"lora_alpha": 5}, "lora_alpha=5")
    expect_reject({"lora_dropout": 0.25}, "lora_dropout=0.25")
    expect_reject({"use_rslora": True}, "use_rslora=True")
    expect_reject({"target_modules": ["gate_proj"]}, "target_modules")
    expect_reject({"bias": "all"}, "bias='all'")
    expect_reject({"bias": 123}, "bias must be a string", exception=TypeError)
    expect_reject({"peft_type": 123}, "peft_type must be a string", exception=TypeError)
    expect_reject({"fan_in_fan_out": True}, "fan_in_fan_out=True")
    expect_reject({"modules_to_save": ["lm_head"]}, "modules_to_save")
    expect_reject({"task_type": "SEQ_CLS"}, "task_type='SEQ_CLS'")
    expect_reject(
        {"task_type": ["CAUSAL_LM"]},
        "task_type must be a string",
        exception=TypeError,
    )
    expect_reject(
        {"inference_mode": "true"},
        "inference_mode must be a boolean",
        exception=TypeError,
    )
    expect_reject(
        {"init_lora_weights": 123},
        "init_lora_weights",
        exception=TypeError,
    )
    expect_reject(
        {"init_lora_weights": ""},
        "init_lora_weights string must be non-empty",
    )
    expect_reject(
        {"target_modules": {"linear_fc1": True}},
        "target_modules must be a string or sequence of strings",
        exception=TypeError,
    )
    expect_reject(
        {"target_modules": ["linear_fc1", 3]},
        "target_modules entries must be strings",
        exception=TypeError,
    )
    expect_reject(
        {"r": "2"},
        "Adapter config r must be an integer",
        exception=TypeError,
    )
    expect_reject(
        {"lora_alpha": "4"},
        "Adapter config lora_alpha must be a finite number",
        exception=TypeError,
    )
    expect_reject(
        {"lora_dropout": "0.0"},
        "Adapter config lora_dropout must be a finite number",
        exception=TypeError,
    )
    expect_reject(
        {"lora_dropout": -0.1},
        "Adapter config lora_dropout must be between 0 and 1",
    )
    expect_reject(
        {"lora_dropout": 1.5},
        "Adapter config lora_dropout must be between 0 and 1",
    )
    expect_reject(
        {"use_rslora": "true"},
        "use_rslora must be a boolean",
        exception=TypeError,
    )


def test_glm5_load_lora_adapter_validates_expected_config_without_adapter_config(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    import json

    from megatron.lite.model.glm5.lite.lora_adapter import load_lora_adapter, save_lora_adapter
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    source_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()

    missing_adapter_dir = tmp_path / "glm5_missing_adapter_dir"
    missing_config_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(FileNotFoundError, match="LoRA adapter directory does not exist"):
        load_lora_adapter([missing_config_chunk], missing_adapter_dir, model_cfg, ps)

    adapter_file = tmp_path / "glm5_adapter_file"
    adapter_file.write_text("not a directory")
    missing_config_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(NotADirectoryError, match="LoRA adapter path must be a directory"):
        load_lora_adapter([missing_config_chunk], adapter_file, model_cfg, ps)

    missing_config_no_model_dir = tmp_path / "glm5_missing_config_without_model"
    missing_config_no_model_dir.mkdir()
    missing_config_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="caller-provided lora_config"):
        load_lora_adapter([missing_config_chunk], missing_config_no_model_dir, model_cfg, ps)

    adapter_dir = tmp_path / "adapter_without_configs"
    save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-without-adapter-config",
        lora_config=lora_config,
    )
    (adapter_dir / "adapter_config.json").unlink()
    missing_config_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="caller-provided lora_config"):
        load_lora_adapter([missing_config_chunk], adapter_dir, model_cfg, ps)

    missing_model_dir = tmp_path / "glm5_missing_adapter_model"
    missing_model_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    save_lora_adapter(
        [missing_model_chunk],
        model_cfg,
        ps,
        missing_model_dir,
        base_model_name_or_path="local/glm5-missing-model",
        lora_config=lora_config,
    )
    (missing_model_dir / "adapter_model.safetensors").unlink()
    missing_model_target, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(FileNotFoundError, match="LoRA adapter model file is missing"):
        load_lora_adapter([missing_model_target], missing_model_dir, model_cfg, ps)

    directory_model_dir = tmp_path / "glm5_directory_adapter_model"
    directory_model_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    save_lora_adapter(
        [directory_model_chunk],
        model_cfg,
        ps,
        directory_model_dir,
        base_model_name_or_path="local/glm5-directory-model",
        lora_config=lora_config,
    )
    (directory_model_dir / "adapter_model.safetensors").unlink()
    (directory_model_dir / "adapter_model.safetensors").mkdir()
    directory_model_target, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(IsADirectoryError, match="LoRA adapter model path must be a file"):
        load_lora_adapter([directory_model_target], directory_model_dir, model_cfg, ps)

    metadata_path = adapter_dir / "megatron.lite_adapter_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"] = []
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    bad_metadata_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(TypeError, match="Adapter metadata metadata must be an object"):
        load_lora_adapter(
            [bad_metadata_chunk],
            adapter_dir,
            model_cfg,
            ps,
            lora_config=lora_config,
        )
    (adapter_dir / "megatron.lite_adapter_meta.json").unlink()

    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    result = load_lora_adapter([target_chunk], adapter_dir, model_cfg, ps, lora_config=lora_config)
    assert result["loaded_tensors"] == 12

    bad_target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    bad_target_config = {**lora_config, "target_modules": ["linear_fc1"]}
    with pytest.raises(ValueError, match="target modules"):
        load_lora_adapter(
            [bad_target_chunk],
            adapter_dir,
            model_cfg,
            ps,
            lora_config=bad_target_config,
        )

    bad_rslora_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    bad_rslora_config = {**lora_config, "use_rslora": True}
    with pytest.raises(ValueError, match="use_rslora"):
        load_lora_adapter(
            [bad_rslora_chunk],
            adapter_dir,
            model_cfg,
            ps,
            lora_config=bad_rslora_config,
        )

    disabled_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    disabled_config = {**lora_config, "enabled": False}
    with pytest.raises(ValueError, match="enabled expected LoRA config"):
        load_lora_adapter(
            [disabled_chunk],
            adapter_dir,
            model_cfg,
            ps,
            lora_config=disabled_config,
        )


def test_glm5_load_lora_adapter_rejects_non_adapter_artifact_keys_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    from megatron.lite.model.glm5.lite.lora_adapter import load_lora_adapter, save_lora_adapter
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    source_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()

    adapter_dir = tmp_path / "tainted_adapter"
    save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-tainted-adapter",
        lora_config=lora_config,
    )

    adapter_model_path = adapter_dir / "adapter_model.safetensors"
    state = safetensors_torch.load_file(str(adapter_model_path))
    state["base_model.model.model.layers.0.self_attn.q_a_proj.weight"] = torch.zeros(1)
    safetensors_torch.save_file(state, str(adapter_model_path))

    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="non-adapter tensor keys"):
        load_lora_adapter(
            [target_chunk],
            adapter_dir,
            model_cfg,
            ps,
            strict=False,
            lora_config=lora_config,
        )


def test_glm5_lora_adapter_rejects_unsupported_target_modules_without_te(tmp_path):
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import load_lora_adapter, save_lora_adapter
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    source_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    ps = ParallelState()

    with pytest.raises(ValueError, match="Unsupported GLM5 LoRA target module"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            tmp_path / "bad_save_target",
            lora_config={**lora_config, "target_modules": ["eh_proj"]},
        )
    with pytest.raises(TypeError, match="target_modules must be a string or sequence of strings"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            tmp_path / "bad_save_target_type",
            lora_config={**lora_config, "target_modules": {"linear_fc1": True}},
        )
    with pytest.raises(TypeError, match="target_modules must be a string or sequence of strings"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            tmp_path / "bad_save_target_mapping_type",
            lora_config={
                **lora_config,
                "target_modules": MappingProxyType({"linear_fc1": True}),
            },
        )
    with pytest.raises(TypeError, match="target_modules entries must be strings"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ps,
            tmp_path / "bad_save_target_entry",
            lora_config={**lora_config, "target_modules": ["linear_fc1", 3]},
        )

    adapter_dir = tmp_path / "valid_adapter"
    save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-valid-adapter",
        lora_config=lora_config,
    )
    target_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    with pytest.raises(ValueError, match="Unsupported GLM5 LoRA target module"):
        load_lora_adapter(
            [target_chunk],
            adapter_dir,
            model_cfg,
            ps,
            lora_config={**lora_config, "target_modules": ["self_attn.indexer"]},
        )


def test_glm5_lora_adapter_rejects_unexported_target_modules_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.glm5.lite.lora_adapter import save_lora_adapter
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        num_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=2,
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
    )
    source_chunk, _ = _fake_shared_routed_expert_chunk(torch)
    adapter_dir = tmp_path / "mismatched_targets"

    with pytest.raises(ValueError, match="target_modules do not match exported adapter tensors"):
        save_lora_adapter(
            [source_chunk],
            model_cfg,
            ParallelState(),
            adapter_dir,
            lora_config={
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "target_modules": "all-linear",
            },
        )

    assert not adapter_dir.exists()


def test_glm5_lora_adapter_rejects_unsupported_parallel_scopes_without_te():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        export_lora_adapter_state,
        load_lora_adapter,
        load_lora_adapter_state,
        save_lora_adapter,
    )
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(num_hidden_layers=1, num_experts=2)
    chunk, _ = _fake_shared_routed_expert_chunk(torch)

    with pytest.raises(NotImplementedError, match="tp=1"):
        export_lora_adapter_state([chunk], model_cfg, ParallelState(tp_size=2))

    with pytest.raises(NotImplementedError, match="pp=1"):
        load_lora_adapter_state([chunk], {}, model_cfg, ParallelState(pp_size=2))

    with pytest.raises(NotImplementedError, match="etp=1"):
        export_lora_adapter_state([chunk], model_cfg, ParallelState(etp_size=2))

    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["linear_fc1", "linear_fc2"],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "glm5_unsupported_save"
        with pytest.raises(NotImplementedError, match="tp=1"):
            save_lora_adapter(
                [chunk],
                model_cfg,
                ParallelState(tp_size=2),
                save_dir,
                lora_config=lora_config,
            )
        assert not save_dir.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "glm5_missing_base_identity"
        with pytest.raises(ValueError, match="base_model_name_or_path must be non-empty"):
            save_lora_adapter(
                [chunk],
                model_cfg,
                ParallelState(),
                save_dir,
                lora_config=lora_config,
            )
        assert not save_dir.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "glm5_bad_base_identity"
        with pytest.raises(TypeError, match="base_model_name_or_path must be a string"):
            save_lora_adapter(
                [chunk],
                model_cfg,
                ParallelState(),
                save_dir,
                base_model_name_or_path=[],
                lora_config=lora_config,
            )
        assert not save_dir.exists()

    with pytest.raises(NotImplementedError, match="pp=1"):
        load_lora_adapter(
            [chunk],
            "/definitely/missing/glm5_adapter",
            model_cfg,
            ParallelState(pp_size=2),
            lora_config=lora_config,
        )

    with pytest.raises(NotImplementedError, match="etp=1"):
        load_lora_adapter(
            [chunk],
            "/definitely/missing/glm5_adapter",
            model_cfg,
            ParallelState(etp_size=2),
            lora_config=lora_config,
        )


def _fake_all_linear_layer(torch, *, layer_idx: int, rank: int = 2, use_rslora: bool = True):
    from megatron.lite.primitive.modules.lora import LinearLoRA

    def lora(in_features, out_features):
        return LinearLoRA(
            in_features,
            out_features,
            rank=rank,
            alpha=4,
            dropout=0.0,
            use_rslora=use_rslora,
        )

    dsa = SimpleNamespace(
        q_a_lora=lora(4, 4),
        q_b_lora=lora(4, 4),
        kv_a_lora=lora(4, 4),
        kv_b_lora=lora(4, 4),
        o_lora=lora(4, 4),
    )
    mlp = SimpleNamespace(
        gate_up_lora=lora(4, 8),
        down_lora=lora(4, 4),
    )
    return SimpleNamespace(
        layer_idx=layer_idx,
        self_attention=SimpleNamespace(self_attention=dsa),
        mlp=mlp,
        moe=None,
    )


def _fake_all_linear_mtp_chunk(torch, *, use_rslora: bool = True):
    main_layer = _fake_all_linear_layer(torch, layer_idx=0, use_rslora=use_rslora)
    mtp_layer = _fake_all_linear_layer(torch, layer_idx=0, use_rslora=use_rslora)
    return SimpleNamespace(
        layers=[main_layer],
        mtp=SimpleNamespace(layers=[SimpleNamespace(transformer_layer=mtp_layer)]),
    )


def _fill_lora_params(torch, chunk) -> None:
    with torch.no_grad():
        value = 1.0
        for layer in [*chunk.layers, *(mtp.transformer_layer for mtp in chunk.mtp.layers)]:
            modules = [
                layer.self_attention.self_attention.q_a_lora,
                layer.self_attention.self_attention.q_b_lora,
                layer.self_attention.self_attention.kv_a_lora,
                layer.self_attention.self_attention.kv_b_lora,
                layer.self_attention.self_attention.o_lora,
                layer.mlp.gate_up_lora,
                layer.mlp.down_lora,
            ]
            for module in modules:
                for tensor in (module.lora_a, module.lora_b):
                    tensor.copy_(
                        torch.arange(tensor.numel(), dtype=tensor.dtype).reshape_as(tensor)
                        + value
                    )
                    value += 100.0


def test_glm5_lora_adapter_round_trips_all_linear_with_mtp_without_te(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    from megatron.lite.model.glm5.lite.lora_adapter import (
        export_lora_adapter_state,
        load_lora_adapter,
        save_lora_adapter,
    )
    from megatron.lite.primitive.parallel import ParallelState

    model_cfg = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=4,
        num_attention_heads=1,
        q_lora_rank=4,
        kv_lora_rank=4,
        num_experts=0,
        n_shared_experts=0,
        moe_intermediate_size=0,
        num_nextn_predict_layers=1,
        mtp_use_repeated_layer=False,
    )
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "use_rslora": True,
    }
    source_chunk = _fake_all_linear_mtp_chunk(torch, use_rslora=True)
    _fill_lora_params(torch, source_chunk)
    ps = ParallelState()

    state = export_lora_adapter_state([source_chunk], model_cfg, ps)
    assert "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight" in state
    assert "base_model.model.model.layers.2.self_attn.q_a_proj.lora_A.weight" in state
    assert "base_model.model.model.layers.2.mlp.gate_proj.lora_B.weight" in state
    assert "base_model.model.model.layers.2.mlp.up_proj.lora_B.weight" in state
    assert not any(".eh_proj." in key for key in state)
    assert len(state) == 32

    adapter_dir = tmp_path / "fake_all_linear_mtp_adapter"
    meta = save_lora_adapter(
        [source_chunk],
        model_cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-fake",
        lora_config=lora_config,
        init_lora_weights="olora_tail",
        metadata={"test_case": "fake_all_linear_mtp"},
    )
    assert meta["lora"]["target_modules"] == [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert meta["model"]["num_nextn_predict_layers"] == 1
    assert meta["metadata"]["init_lora_weights"] == "olora_tail"

    target_chunk = _fake_all_linear_mtp_chunk(torch, use_rslora=True)
    result = load_lora_adapter([target_chunk], adapter_dir, model_cfg, ps, lora_config=lora_config)
    assert result["loaded_tensors"] == len(state)

    reexported = export_lora_adapter_state([target_chunk], model_cfg, ps)
    assert set(reexported) == set(state)
    for key, tensor in state.items():
        torch.testing.assert_close(reexported[key], tensor)


def test_glm5_lora_adapter_round_trips_all_linear_with_mtp(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("transformer_engine.pytorch")

    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.lora_adapter import (
        export_lora_adapter_state,
        load_lora_adapter,
        save_lora_adapter,
    )
    from megatron.lite.model.glm5.lite.model import Glm5Model
    from megatron.lite.primitive.parallel import ParallelState

    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "use_rslora": True,
    }
    cfg = Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1)
    ps = ParallelState()
    train_cfg = SimpleNamespace(
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
    )

    model = Glm5Model(cfg, train_cfg, ps, mtp_enable=True, lora_config=lora_config)
    with torch.no_grad():
        value = 1.0
        for name, param in model.named_parameters():
            if "lora" not in name:
                continue
            param.copy_(
                torch.arange(param.numel(), dtype=param.dtype, device=param.device).reshape_as(
                    param
                )
                + value
            )
            value += 1000.0

    state = export_lora_adapter_state([model], cfg, ps)
    assert (
        "base_model.model.model.layers.2.self_attn.q_a_proj.lora_A.weight"
        in state
    )
    assert (
        "base_model.model.model.layers.2.mlp.gate_proj.lora_B.weight"
        in state
    )
    assert (
        "base_model.model.model.layers.2.mlp.up_proj.lora_B.weight"
        in state
    )
    assert not any(".eh_proj." in key for key in state)

    adapter_dir = tmp_path / "adapter"
    meta = save_lora_adapter(
        [model],
        cfg,
        ps,
        adapter_dir,
        base_model_name_or_path="local/glm5-mtp-round-trip",
        lora_config=lora_config,
        init_lora_weights="olora_tail",
        metadata={"test_case": "glm5_lora_mtp_round_trip"},
    )
    assert meta["format"] == "megatron.lite_glm5_lora_peft_v1"
    assert meta["metadata"]["init_lora_weights"] == "olora_tail"
    assert meta["model"]["num_nextn_predict_layers"] == 1

    loaded = Glm5Model(cfg, train_cfg, ps, mtp_enable=True, lora_config=lora_config)
    load_result = load_lora_adapter(
        [loaded],
        adapter_dir,
        cfg,
        ps,
        lora_config=lora_config,
    )
    assert load_result["loaded_tensors"] == len(state)

    reexported = export_lora_adapter_state([loaded], cfg, ps)
    assert set(reexported) == set(state)
    for key, tensor in state.items():
        torch.testing.assert_close(reexported[key], tensor)
