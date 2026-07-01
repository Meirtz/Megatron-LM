# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
import copy
import functools
import os
import subprocess
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import (
    MegatronLiteRuntime,
    _apply_attention_backend_env,
    _build_impl_cfg,
    _last_output_with_payload,
)
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig, RuntimeConfig
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = pytest.mark.mlite


def test_runtime_config_defaults_to_mlite_backend():
    cfg = RuntimeConfig()

    assert cfg.backend == "mlite"
    assert cfg.hf_path == ""
    assert isinstance(cfg.backend_cfg, dict)


def test_runtime_config_accepts_mlite_backend_cfg():
    cfg = RuntimeConfig(
        backend="mlite",
        hf_path="/models/Qwen3",
        backend_cfg={"model_name": "qwen3", "impl": "lite", "tp": 2, "ep": 4},
    )

    assert cfg.backend == "mlite"
    assert cfg.backend_cfg["model_name"] == "qwen3"
    assert cfg.backend_cfg["tp"] == 2


def test_mlite_config_defaults_and_parallel_fields():
    cfg = MegatronLiteConfig(
        model_name="qwen3_moe", parallel=ParallelConfig(tp=4, etp=1, ep=8, pp=2, vpp=2, cp=2)
    )

    assert cfg.model_name == "qwen3_moe"
    assert cfg.impl == "lite"
    assert cfg.parallel.tp == 4
    assert cfg.parallel.ep == 8
    assert cfg.parallel.pp == 2
    assert cfg.parallel.cp == 2


def test_mlite_config_impl_cfg_optimizer_and_load_gate():
    hook = lambda cfg: cfg  # noqa: E731
    cfg = MegatronLiteConfig(
        model_name="qwen3_moe",
        impl_cfg={"recompute": "full", "use_deepep": True},
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0.1, adam_beta1=0.9),
        load_hf_weights=False,
        model_config_hook=hook,
    )

    assert cfg.impl_cfg["recompute"] == "full"
    assert cfg.impl_cfg["use_deepep"] is True
    assert cfg.optimizer.lr == 1e-4
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.load_hf_weights is False
    assert cfg.model_config_hook is hook


def test_mlite_config_from_dict_accepts_optimizer_override_config():
    cfg = MegatronLiteConfig.from_dict(
        "/models/Qwen3",
        {
            "optimizer": {
                "override_optimizer_config": {
                    "fsdp2_use_fp32_master": False,
                    "offload_fraction": 1.0,
                }
            }
        },
    )

    assert cfg.optimizer.override_optimizer_config == {
        "fsdp2_use_fp32_master": False,
        "offload_fraction": 1.0,
    }


def test_mlite_config_from_dict_rejects_num_microbatches():
    with pytest.raises(ValueError, match="num_microbatches"):
        MegatronLiteConfig.from_dict(
            "/models/Qwen3", {"model_name": "qwen3", "tp": 4, "num_microbatches": 2}
        )


@dataclass
class _FakeImplConfig:
    parallel: object
    hf_path: str = ""
    optimizer_config: object = None
    attention_backend_override: str | None = None


def test_build_impl_cfg_backfills_top_level_hf_path_and_runtime_fields():
    proto = type("Proto", (), {"ImplConfig": _FakeImplConfig})
    cfg = MegatronLiteConfig(
        model_name="qwen3", hf_path="/models/top", attention_backend_override="local"
    )

    impl_cfg = _build_impl_cfg(proto, cfg)

    assert impl_cfg.parallel is cfg.parallel
    assert impl_cfg.hf_path == "/models/top"
    assert impl_cfg.optimizer_config is cfg.optimizer
    assert impl_cfg.attention_backend_override == "local"


def test_build_impl_cfg_preserves_explicit_impl_hf_path():
    proto = type("Proto", (), {"ImplConfig": _FakeImplConfig})
    cfg = MegatronLiteConfig(
        model_name="qwen3", hf_path="/models/top", impl_cfg={"hf_path": "/models/impl"}
    )

    impl_cfg = _build_impl_cfg(proto, cfg)

    assert impl_cfg.hf_path == "/models/impl"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("auto", ("1", "1", "1")),
        ("flash", ("1", "0", "0")),
        ("fused", ("0", "1", "0")),
        ("unfused", ("0", "0", "1")),
        ("local", ("0", "0", "0")),
    ],
)
def test_attention_backend_override_sets_expected_env(monkeypatch, backend, expected):
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)

    _apply_attention_backend_env(backend, tag="unit")

    assert (
        os.environ["NVTE_FLASH_ATTN"],
        os.environ["NVTE_FUSED_ATTN"],
        os.environ["NVTE_UNFUSED_ATTN"],
    ) == expected


def test_attention_backend_override_rejects_unknown_backend():
    with pytest.raises(ValueError, match="attention_backend_override"):
        _apply_attention_backend_env("invalid", tag="unit")


def test_pipeline_compact_keeps_forward_payload_only_when_requested():
    from megatron.lite.primitive.parallel.pipeline import _compact_pipeline_output

    logits = torch.ones(1, 2, 3)
    log_probs = torch.zeros(1, 2)
    hidden = torch.empty(2, 1, 4)
    out = {"logits": logits, "log_probs": log_probs, "hidden_states": hidden}

    assert _compact_pipeline_output(out) == {}

    compact = _compact_pipeline_output(out, keep_forward_payload=True)

    assert compact["logits"] is logits
    assert compact["log_probs"] is log_probs
    assert "hidden_states" not in compact


def test_runtime_pp_payload_selection_uses_last_output_with_logits():
    logits = torch.ones(1, 2, 3)
    outputs = [{"loss": 1.0}, {}, {"logits": logits}]

    assert _last_output_with_payload(outputs)["logits"] is logits


class HookedOptimizer:
    def __init__(self):
        self.calls: list[str] = []

    def offload_state_to_cpu(self):
        self.calls.append("offload")

    def load_state_to_device(self):
        self.calls.append("load")


def test_runtime_to_prefers_optimizer_specific_offload_hooks():
    optimizer = HookedOptimizer()
    handle = ModelHandle(model=nn.Linear(2, 2), optimizer=optimizer, _extras={"model_chunks": []})
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    runtime.to(handle, "cpu", model=False, optimizer=True, grad=False)
    runtime.to(handle, "cuda", model=False, optimizer=True, grad=False)

    assert optimizer.calls == ["offload", "load"]


class _FakeStorage:
    def __init__(self, size: int):
        self._size = size
        self.resize_calls: list[int] = []

    def size(self):
        return self._size

    def resize_(self, size: int):
        self.resize_calls.append(size)
        self._size = size
        return self


class _FakeBufferData:
    def __init__(self, size: int):
        self._storage = _FakeStorage(size)
        self.cpu_called = False
        self.pinned = False
        self.copied_from = None
        self.copy_non_blocking = None
        self.zero_calls = 0

    @property
    def data(self):
        return self

    def cpu(self):
        self.cpu_called = True
        return self

    def pin_memory(self):
        self.pinned = True
        return self

    def storage(self):
        return self._storage

    def copy_(self, other, *, non_blocking: bool):
        self.copied_from = other
        self.copy_non_blocking = non_blocking
        return self

    def zero_(self):
        self.zero_calls += 1
        return self


class _FakeBuffer:
    def __init__(self):
        self.param_data = _FakeBufferData(3)
        self.grad_data = _FakeBufferData(5)


class _FakeModule:
    def parameters(self):
        return []


class _FakeMegatronDDP:
    def __init__(self):
        self.buffer = _FakeBuffer()
        self.buffers = [self.buffer]
        self.expert_parallel_buffers = []
        self.module = _FakeModule()
        self.to_calls: list[str] = []

    def to(self, device):
        self.to_calls.append(device)
        raise AssertionError("DDP model chunks must use the buffer offload path")


class _FakeMegatronDDPSubclass(_FakeMegatronDDP):
    pass


class _FakeNativeModel:
    def __init__(self):
        self.calls: list[str] = []

    def to(self, device):
        self.calls.append(device)
        return self


def _install_fake_megatron_ddp(monkeypatch) -> None:
    core = types.ModuleType("megatron.core")
    distributed = types.ModuleType("megatron.core.distributed")
    distributed.DistributedDataParallel = _FakeMegatronDDP
    core.distributed = distributed
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", distributed)


def test_megatron_ddp_detection_accepts_ddp_and_subclasses(monkeypatch):
    from megatron.lite.runtime.megatron_utils import _is_megatron_ddp

    _install_fake_megatron_ddp(monkeypatch)

    assert _is_megatron_ddp(_FakeMegatronDDP()) is True
    assert _is_megatron_ddp(_FakeMegatronDDPSubclass()) is True
    assert _is_megatron_ddp(_FakeNativeModel()) is False


@pytest.mark.parametrize("model_cls", [_FakeMegatronDDP, _FakeMegatronDDPSubclass])
def test_megatron_ddp_model_move_helpers_use_buffer_path(monkeypatch, model_cls):
    from megatron.lite.runtime.megatron_utils import load_model_to_gpu, offload_model_to_cpu

    _install_fake_megatron_ddp(monkeypatch)
    model = model_cls()
    buffer = model.buffer

    offload_model_to_cpu([model])

    assert model.to_calls == []
    assert buffer.param_data.cpu_called is True
    assert buffer.param_data.pinned is True
    assert buffer.param_data_size == 3
    assert buffer.grad_data_size == 5
    assert buffer.param_data.storage().size() == 0
    assert buffer.grad_data.storage().size() == 0

    load_model_to_gpu([model])

    assert model.to_calls == []
    assert buffer.param_data.storage().size() == 3
    assert buffer.grad_data.storage().size() == 5
    assert buffer.param_data.copied_from is buffer.param_data.cpu_data
    assert buffer.param_data.copy_non_blocking is True
    assert buffer.grad_data.zero_calls == 1


def test_native_model_move_helpers_do_not_require_megatron_core(monkeypatch):
    from megatron.lite.runtime.megatron_utils import load_model_to_gpu, offload_model_to_cpu

    monkeypatch.setitem(sys.modules, "megatron.core", None)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", None)
    model = _FakeNativeModel()

    offload_model_to_cpu([model])
    load_model_to_gpu([model])

    assert model.calls == ["cpu", "cuda"]


def test_model_handle_dp_defaults():
    handle = ModelHandle(model=MagicMock())

    assert handle.dp_rank == 0
    assert handle.dp_size == 1
    assert handle.dp_group is None


def test_model_handle_dp_from_parallel_state():
    ps = MagicMock()
    ps.dp_rank = 3
    ps.dp_size = 8
    ps.dp_group = "fake_group"

    handle = ModelHandle(model=MagicMock(), parallel_state=ps)

    assert handle.dp_rank == 3
    assert handle.dp_size == 8
    assert handle.dp_group == "fake_group"


def test_model_handle_cp_range_and_config_properties():
    cfg = {"tp": 8, "ep": 4}
    default_handle = ModelHandle(model=MagicMock())
    configured_handle = ModelHandle(model=MagicMock(), config=cfg, _extras={"cp_range": (1, 8)})

    assert default_handle.cp_range == (1, 1)
    assert configured_handle.cp_range == (1, 8)
    assert configured_handle.config is cfg


def _delta_mem_batch(*, scope: str | None = None) -> PackedBatch:
    extras = {} if scope is None else {"delta_mem_state_scope": scope}
    return PackedBatch(
        input_ids=torch.tensor([1]),
        labels=torch.tensor([1]),
        seq_lens=torch.tensor([1]),
        extras=extras,
    )


def test_mlite_runtime_delta_mem_step_scope_carries_detached_states_across_microbatches():
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    model = nn.Linear(1, 1, bias=False)
    seen_states = []

    def forward_step(module, batch):
        seen_states.append(batch.extras.get("delta_mem_states"))
        loss = module(torch.ones(1, 1)).sum()
        return {"loss": loss, "delta_mem_states": [loss.reshape(1)]}

    handle = ModelHandle(
        model=model,
        parallel_state=types.SimpleNamespace(pp_size=1),
        _extras={
            "forward_step": forward_step,
            "delta_mem_config": types.SimpleNamespace(enabled=True),
        },
    )

    result = runtime.forward_backward(
        handle,
        iter([_delta_mem_batch(scope="step"), _delta_mem_batch()]),
        loss_fn=None,
        num_microbatches=2,
    )

    assert seen_states[0] is None
    assert seen_states[1] is not None
    assert seen_states[1][0].requires_grad is False
    assert result.model_output.delta_mem_states is not None
    assert result.metrics["delta_mem_state_scope"] == "step"
    assert result.metrics["delta_mem_state_capture_count"] == 2
    assert "_delta_mem_runtime_states" not in handle._extras


def test_mlite_runtime_delta_mem_handle_scope_persists_detached_state():
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    model = nn.Linear(1, 1, bias=False)

    def forward_step(module, batch):
        loss = module(torch.ones(1, 1)).sum()
        return {"loss": loss, "delta_mem_states": [loss.reshape(1)]}

    handle = ModelHandle(
        model=model,
        parallel_state=types.SimpleNamespace(pp_size=1),
        _extras={
            "forward_step": forward_step,
            "delta_mem_config": types.SimpleNamespace(enabled=True),
        },
    )

    runtime.forward_backward(
        handle,
        iter([_delta_mem_batch(scope="handle")]),
        loss_fn=None,
        num_microbatches=1,
    )

    states = handle._extras["_delta_mem_runtime_states"]
    assert states[0].requires_grad is False


def test_mlite_runtime_delta_mem_carries_main_and_mtp_states_across_microbatches():
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    model = nn.Linear(1, 1, bias=False)
    seen_main = []
    seen_mtp = []

    def forward_step(module, batch):
        seen_main.append(batch.extras.get("delta_mem_states"))
        seen_mtp.append(batch.extras.get("delta_mem_mtp_states"))
        loss = module(torch.ones(1, 1)).sum()
        return {
            "loss": loss,
            "delta_mem_states": [loss.reshape(1)],
            "delta_mem_mtp_states": [loss.reshape(1) + 1.0],
        }

    handle = ModelHandle(
        model=model,
        parallel_state=types.SimpleNamespace(pp_size=1),
        _extras={
            "forward_step": forward_step,
            "delta_mem_config": types.SimpleNamespace(enabled=True),
        },
    )

    result = runtime.forward_backward(
        handle,
        iter([_delta_mem_batch(scope="step"), _delta_mem_batch()]),
        loss_fn=None,
        num_microbatches=2,
    )

    assert seen_main[0] is None
    assert seen_mtp[0] is None
    assert seen_main[1][0].requires_grad is False
    assert seen_mtp[1][0].requires_grad is False
    assert result.model_output.delta_mem_states is not None
    assert result.model_output.delta_mem_mtp_states is not None
    assert result.metrics["delta_mem_state_layer_count"] == 1
    assert result.metrics["delta_mem_mtp_state_layer_count"] == 1


def test_mlite_runtime_delta_mem_state_explicit_save_load_reset_round_trip(tmp_path):
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    handle = ModelHandle(
        model=nn.Linear(1, 1),
        _extras={"_delta_mem_runtime_states": [torch.tensor([3.0], requires_grad=True)]},
    )

    exported = runtime.export_delta_mem_runtime_state(handle)

    assert exported["format"] == "megatron_lite.delta_mem_runtime_state.v1"
    assert exported["metadata"]["adapter_export_compatible"] is True
    assert exported["states"][0].requires_grad is False

    state_file = runtime.save_delta_mem_runtime_state(handle, tmp_path)
    runtime.reset_delta_mem_runtime_state(handle)
    assert "_delta_mem_runtime_states" not in handle._extras

    loaded = runtime.load_delta_mem_runtime_state(handle, tmp_path)

    assert state_file.name == "delta_mem_runtime_state.pt"
    assert loaded["restored"] is True
    torch.testing.assert_close(handle._extras["_delta_mem_runtime_states"][0], torch.tensor([3.0]))


def test_mlite_runtime_checkpoint_saves_and_loads_delta_mem_sidecar(tmp_path):
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    handle = ModelHandle(
        model=nn.Linear(1, 1),
        parallel_state=types.SimpleNamespace(),
        _extras={"_delta_mem_runtime_states": [torch.tensor([5.0])]},
    )

    with patch("megatron.lite.primitive.ckpt.save_training_checkpoint") as save_checkpoint:
        runtime.save_checkpoint(handle, str(tmp_path), step=7, use_dcp=True)

    save_checkpoint.assert_called_once()
    assert (tmp_path / "step_7" / "delta_mem_runtime_state.pt").exists()

    restored_handle = ModelHandle(model=nn.Linear(1, 1), parallel_state=types.SimpleNamespace())
    with patch("megatron.lite.primitive.ckpt.load_training_checkpoint", return_value=7):
        step = runtime.load_checkpoint(restored_handle, str(tmp_path), use_dcp=True)

    assert step == 7
    torch.testing.assert_close(
        restored_handle._extras["_delta_mem_runtime_states"][0],
        torch.tensor([5.0]),
    )


def test_mlite_runtime_forwards_lora_adapter_helpers_to_protocol(tmp_path):
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    calls = []

    class FakeProtocol:
        __name__ = "fake_protocol"

        @staticmethod
        def export_lora_adapter_state(chunks, model_cfg, ps, **kwargs):
            calls.append(("export", chunks, model_cfg, ps, kwargs))
            return {"adapter.weight": "tensor"}

        @staticmethod
        def save_lora_adapter(chunks, model_cfg, ps, output_dir, **kwargs):
            calls.append(("save", chunks, model_cfg, ps, output_dir, kwargs))
            return {"saved": str(output_dir)}

        @staticmethod
        def load_lora_adapter(chunks, adapter_dir, model_cfg, ps, **kwargs):
            calls.append(("load", chunks, adapter_dir, model_cfg, ps, kwargs))
            return {"loaded": str(adapter_dir)}

    chunks = [MagicMock(name="chunk0"), MagicMock(name="chunk1")]
    model_cfg = MagicMock(name="model_cfg")
    ps = MagicMock(name="parallel_state")
    handle = ModelHandle(
        model=chunks[0],
        parallel_state=ps,
        _extras={
            "model_chunks": chunks,
            "model_cfg": model_cfg,
            "protocol": FakeProtocol,
            "_delta_mem_runtime_states": [torch.tensor([7.0])],
        },
    )

    assert runtime.export_lora_adapter_state(handle, cpu=True) == {"adapter.weight": "tensor"}
    save_dir = tmp_path / "adapter"
    assert runtime.save_lora_adapter(handle, save_dir, lora_config={"rank": 2}) == {
        "saved": str(save_dir)
    }
    assert runtime.load_lora_adapter(handle, save_dir, strict=True) == {"loaded": str(save_dir)}

    assert calls == [
        ("export", chunks, model_cfg, ps, {"cpu": True}),
        ("save", chunks, model_cfg, ps, save_dir, {"lora_config": {"rank": 2}}),
        ("load", chunks, save_dir, model_cfg, ps, {"strict": True}),
    ]


def test_mlite_runtime_lora_adapter_helpers_require_protocol_context():
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    with pytest.raises(ValueError, match="protocol"):
        runtime.export_lora_adapter_state(ModelHandle(model=MagicMock()))

    with pytest.raises(ValueError, match="model_cfg"):
        runtime.save_lora_adapter(
            ModelHandle(model=MagicMock(), _extras={"protocol": types.SimpleNamespace()}),
            "/tmp/adapter",
        )

    with pytest.raises(ValueError, match="non-empty sequence of model_chunks"):
        runtime.export_lora_adapter_state(
            ModelHandle(
                model=MagicMock(),
                parallel_state=MagicMock(),
                _extras={
                    "protocol": types.SimpleNamespace(),
                    "model_cfg": {},
                    "model_chunks": [],
                },
            )
        )

    with pytest.raises(ValueError, match="non-None model_chunks"):
        runtime.export_lora_adapter_state(
            ModelHandle(
                model=MagicMock(),
                parallel_state=MagicMock(),
                _extras={
                    "protocol": types.SimpleNamespace(),
                    "model_cfg": {},
                    "model_chunks": [None],
                },
            )
        )

    with pytest.raises(ValueError, match="parallel_state"):
        runtime.export_lora_adapter_state(
            ModelHandle(
                model=MagicMock(),
                _extras={
                    "protocol": types.SimpleNamespace(),
                    "model_cfg": {},
                    "model_chunks": [MagicMock()],
                },
            )
        )

    with pytest.raises(NotImplementedError, match="load_lora_adapter"):
        runtime.load_lora_adapter(
            ModelHandle(
                model=MagicMock(),
                parallel_state=MagicMock(),
                _extras={
                    "protocol": types.SimpleNamespace(__name__="no_adapter"),
                    "model_cfg": {},
                    "model_chunks": [MagicMock()],
                },
            ),
            "/tmp/adapter",
        )


def test_runtime_dispatch_creates_mlite_backend():
    with patch("megatron.lite.runtime.backends.mlite.create") as mock_create:
        backend = MagicMock()
        mock_create.return_value = backend

        runtime = create_runtime(
            RuntimeConfig(
                backend="mlite", hf_path="/models/test", backend_cfg={"model_name": "qwen3"}
            )
        )

    assert runtime is backend
    mock_create.assert_called_once_with("/models/test", {"model_name": "qwen3"})


def test_runtime_dispatch_unknown_backend_raises():
    with pytest.raises(KeyError):
        create_runtime(RuntimeConfig(backend="nonexistent"))


def _run_verl_sft_dry_run(script: Path, tmp_path: Path, **env_overrides: str) -> str:
    env = {
        **os.environ,
        "MODEL_PATH": "/tmp/mlite-model",
        "TRAIN_FILES": "/tmp/mlite-train.parquet",
        "OUTPUT_ROOT": str(tmp_path),
        "DRY_RUN": "1",
        "NUM_GPUS": "1",
        "NPROC_PER_NODE": "1",
        "TP_SIZE": "1",
        "PP_SIZE": "1",
        "CP_SIZE": "1",
        "EP_SIZE": "1",
        "ETP_SIZE": "1",
        **env_overrides,
    }
    completed = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=True)
    return completed.stdout


def test_verl_sft_script_maps_offload_env_to_backend_args(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_sft.sh"
    )

    command = _run_verl_sft_dry_run(
        script,
        tmp_path,
        PARAM_OFFLOAD="True",
        OPTIMIZER_OFFLOAD="True",
        OPTIMIZER_STATE_OFFLOAD_FRACTION="0.75",
    )

    assert "engine.param_offload=True" in command
    assert "engine.optimizer_offload=True" in command
    assert "+optim.override_optimizer_config.offload_fraction=0.75" in command
    assert "+optim.override_optimizer_config.use_precision_aware_optimizer=True" in command


def test_verl_sft_script_does_not_emit_optimizer_state_offload_when_disabled(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_sft.sh"
    )

    command = _run_verl_sft_dry_run(
        script, tmp_path, PARAM_OFFLOAD="False", OPTIMIZER_OFFLOAD="False"
    )

    assert "engine.param_offload=False" in command
    assert "engine.optimizer_offload=False" in command
    assert "override_optimizer_config.offload_fraction" not in command


def test_verl_sft_script_maps_lora_env_to_impl_cfg(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_sft.sh"
    )

    command = _run_verl_sft_dry_run(
        script,
        tmp_path,
        LORA_RANK="16",
        LORA_ALPHA="32",
        LORA_DROPOUT="0.05",
        LORA_TARGET_MODULES="all-linear",
        LORA_USE_RSLORA="True",
        LORA_INIT="olora_tail",
    )

    assert "+engine.impl_cfg.lora.rank=16" in command
    assert "+engine.impl_cfg.lora.alpha=32" in command
    assert "+engine.impl_cfg.lora.dropout=0.05" in command
    assert "+engine.impl_cfg.lora.target_modules=all-linear" in command
    assert "+engine.impl_cfg.lora.use_rslora=True" in command
    assert "+engine.impl_cfg.lora_init=olora_tail" in command


def test_verl_sft_script_maps_lora_adapter_checkpoint_env(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_sft.sh"
    )

    command = _run_verl_sft_dry_run(
        script,
        tmp_path,
        CHECKPOINT_SAVE_CONTENTS="[model,optimizer,lora_adapter]",
        CHECKPOINT_LOAD_CONTENTS="[lora_adapter]",
        CHECKPOINT_SAVE_LORA_ADAPTER="True",
        LORA_ADAPTER_DIR_NAME="peft_adapter",
    )
    normalized_command = command.replace("\\", "")

    assert "checkpoint.save_contents=[model,optimizer,lora_adapter]" in normalized_command
    assert "checkpoint.load_contents=[lora_adapter]" in normalized_command
    assert "checkpoint.save_lora_adapter=True" in command
    assert "checkpoint.lora_adapter_dir_name=peft_adapter" in command


def _run_verl_grpo_dry_run_process(
    script: Path, tmp_path: Path, **env_overrides: str
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "MODEL_PATH": "/tmp/mlite-model",
        "DATASET_DIR": "/tmp/mlite-gsm8k",
        "OUTPUT_ROOT": str(tmp_path),
        "DRY_RUN": "1",
        "NGPUS_PER_NODE": "1",
        "NPROC_PER_NODE": "1",
        "ACTOR_TP": "1",
        "ACTOR_PP": "1",
        "ACTOR_CP": "1",
        "ACTOR_EP": "1",
        "ACTOR_ETP": "1",
        "ROLLOUT_TP": "1",
        **env_overrides,
    }
    completed = subprocess.run([str(script)], env=env, text=True, capture_output=True)
    return completed


def _run_verl_grpo_dry_run(script: Path, tmp_path: Path, **env_overrides: str) -> str:
    completed = _run_verl_grpo_dry_run_process(script, tmp_path, **env_overrides)
    completed.check_returncode()
    return completed.stdout


def test_verl_grpo_script_maps_lora_env_to_actor_impl_cfg(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_gsm8k_grpo.sh"
    )

    command = _run_verl_grpo_dry_run(
        script,
        tmp_path,
        LORA_RANK="16",
        LORA_ALPHA="32",
        LORA_DROPOUT="0.0",
        LORA_TARGET_MODULES="[linear_qkv,linear_proj,linear_fc1,linear_fc2]",
        LORA_USE_RSLORA="True",
        LORA_INIT="olora_tail",
    )
    normalized_command = command.replace("\\", "")

    assert "+actor_rollout_ref.actor.engine.impl_cfg.lora.rank=16" in command
    assert "+actor_rollout_ref.actor.engine.impl_cfg.lora.alpha=32" in command
    assert "+actor_rollout_ref.actor.engine.impl_cfg.lora.dropout=0.0" in command
    assert (
        "+actor_rollout_ref.actor.engine.impl_cfg.lora.target_modules="
        "[linear_qkv,linear_proj,linear_fc1,linear_fc2]"
        in normalized_command
    )
    assert "+actor_rollout_ref.actor.engine.impl_cfg.lora.use_rslora=True" in command
    assert "+actor_rollout_ref.actor.engine.impl_cfg.lora_init=olora_tail" in command


def test_verl_grpo_script_maps_lora_adapter_checkpoint_env(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_gsm8k_grpo.sh"
    )

    command = _run_verl_grpo_dry_run(
        script,
        tmp_path,
        CHECKPOINT_SAVE_CONTENTS="[lora_adapter]",
        CHECKPOINT_LOAD_CONTENTS="[lora_adapter]",
        CHECKPOINT_SAVE_LORA_ADAPTER="True",
        LORA_ADAPTER_DIR_NAME="peft_adapter",
    )
    normalized_command = command.replace("\\", "")

    assert "checkpoint.save_contents=[lora_adapter]" in normalized_command
    assert "checkpoint.load_contents=[lora_adapter]" in normalized_command
    assert "checkpoint.save_lora_adapter=True" in command
    assert "checkpoint.lora_adapter_dir_name=peft_adapter" in command


def test_verl_grpo_script_maps_r3_router_replay_env_to_rollout_and_mlite(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_gsm8k_grpo.sh"
    )

    command = _run_verl_grpo_dry_run(
        script,
        tmp_path,
        INFER_BACKEND="sglang",
        ROUTER_REPLAY_MODE="R3",
    )

    assert "actor_rollout_ref.rollout.name=sglang" in command
    assert "actor_rollout_ref.model.use_remove_padding=True" in command
    assert "actor_rollout_ref.rollout.enable_rollout_routing_replay=True" in command
    assert "+actor_rollout_ref.actor.engine.router_replay.mode=R3" in command
    assert "+actor_rollout_ref.actor.engine.impl_cfg.router_replay=True" in command


def test_verl_grpo_script_rejects_r3_without_sglang(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_gsm8k_grpo.sh"
    )

    completed = _run_verl_grpo_dry_run_process(
        script,
        tmp_path,
        INFER_BACKEND="vllm",
        ROUTER_REPLAY_MODE="R3",
    )

    assert completed.returncode != 0
    assert "ROUTER_REPLAY_MODE=R3 currently requires INFER_BACKEND=sglang" in completed.stderr


def test_verl_grpo_script_rejects_r2_router_replay_mode(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_gsm8k_grpo.sh"
    )

    completed = _run_verl_grpo_dry_run_process(
        script,
        tmp_path,
        ROUTER_REPLAY_MODE="R2",
    )

    assert completed.returncode != 0
    assert "ROUTER_REPLAY_MODE=R2 is not supported by the MLite GRPO example yet" in completed.stderr


def test_verl_worker_reads_mlite_engine_router_replay_mode_static(tmp_path=None):
    worker_source = (
        Path(__file__).resolve().parents[6] / "verl" / "verl" / "workers" / "engine_workers.py"
    ).read_text()

    assert "def _actor_router_replay_mode" in worker_source
    assert "def _validate_router_replay_symmetry" in worker_source
    assert 'actor_strategy == "mlite"' in worker_source
    assert '_engine_router_replay_mode(_config_get(actor_config, "engine"))' in worker_source
    assert "rr_mode = _validate_router_replay_symmetry" in worker_source
    assert "rollout.enable_rollout_routing_replay=True" in worker_source
    assert "rollout.name='sglang'" in worker_source
    assert 'self.enable_routing_replay = rr_mode != "disabled"' in worker_source


def test_verl_worker_router_replay_mode_helper_resolves_mlite_and_legacy_configs(tmp_path=None):
    worker_path = Path(__file__).resolve().parents[6] / "verl" / "verl" / "workers" / "engine_workers.py"
    tree = ast.parse(worker_path.read_text())
    helper_names = {
        "_config_get",
        "_config_bool_enabled",
        "_engine_router_replay_mode",
        "_actor_router_replay_mode",
        "_rollout_router_replay_enabled",
        "_sequence_packing_enabled",
        "_validate_router_replay_symmetry",
    }
    helper_defs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(worker_path), "exec"), namespace)
    actor_router_replay_mode = namespace["_actor_router_replay_mode"]
    rollout_router_replay_enabled = namespace["_rollout_router_replay_enabled"]
    sequence_packing_enabled = namespace["_sequence_packing_enabled"]
    validate_router_replay_symmetry = namespace["_validate_router_replay_symmetry"]

    dict_config = {
        "strategy": "mlite",
        "engine": {"router_replay": {"mode": "R3"}},
    }
    namespace_config = types.SimpleNamespace(
        strategy="mlite",
        engine=types.SimpleNamespace(router_replay=types.SimpleNamespace(mode="R3")),
    )
    megatron_config = types.SimpleNamespace(
        strategy="megatron",
        megatron=types.SimpleNamespace(router_replay=types.SimpleNamespace(mode="R2")),
    )
    veomni_config = types.SimpleNamespace(
        strategy="veomni",
        veomni=types.SimpleNamespace(router_replay=types.SimpleNamespace(mode="disable")),
    )
    missing_config = types.SimpleNamespace(strategy="mlite", engine=types.SimpleNamespace())

    assert actor_router_replay_mode(dict_config) == "R3"
    assert actor_router_replay_mode(namespace_config) == "R3"
    assert actor_router_replay_mode(megatron_config) == "R2"
    assert actor_router_replay_mode(veomni_config) == "disabled"
    assert actor_router_replay_mode(missing_config) == "disabled"
    assert actor_router_replay_mode({"strategy": "fsdp"}) == "disabled"
    assert rollout_router_replay_enabled({"enable_rollout_routing_replay": "true"}) is True
    assert rollout_router_replay_enabled({"enable_rollout_routing_replay": "disabled"}) is False
    assert sequence_packing_enabled({"use_sequence_packing": "yes"}) is True
    assert sequence_packing_enabled(
        {"engine": {"impl_cfg": {"sequence_packing": True}}}
    ) is True
    assert sequence_packing_enabled(types.SimpleNamespace(sequence_packing="false")) is False

    assert validate_router_replay_symmetry(
        dict_config,
        {"name": "sglang", "enable_rollout_routing_replay": True},
    ) == "R3"
    assert validate_router_replay_symmetry(
        namespace_config,
        types.SimpleNamespace(name="sglang", enable_rollout_routing_replay="yes"),
    ) == "R3"
    assert validate_router_replay_symmetry(
        missing_config,
        types.SimpleNamespace(name="vllm", enable_rollout_routing_replay=False),
    ) == "disabled"

    with pytest.raises(ValueError, match="rollout.name='sglang'"):
        validate_router_replay_symmetry(
            dict_config,
            {"name": "vllm", "enable_rollout_routing_replay": True},
        )
    with pytest.raises(ValueError, match="enable_rollout_routing_replay=True"):
        validate_router_replay_symmetry(
            dict_config,
            {"name": "sglang", "enable_rollout_routing_replay": False},
        )
    with pytest.raises(ValueError, match="requires actor router_replay.mode=R3"):
        validate_router_replay_symmetry(
            missing_config,
            {"name": "sglang", "enable_rollout_routing_replay": True},
        )
    with pytest.raises(NotImplementedError, match="router_replay.mode=R2 is not implemented"):
        validate_router_replay_symmetry(
            {
                "strategy": "mlite",
                "engine": {"router_replay": {"mode": "R2"}},
            },
            {"name": "sglang", "enable_rollout_routing_replay": False},
        )
    with pytest.raises(ValueError, match="Unsupported actor router_replay.mode='R4'"):
        validate_router_replay_symmetry(
            {
                "strategy": "mlite",
                "engine": {"router_replay": {"mode": "R4"}},
            },
            {"name": "sglang", "enable_rollout_routing_replay": False},
        )
    with pytest.raises(ValueError, match="does not support actor sequence packing"):
        validate_router_replay_symmetry(
            {
                "strategy": "mlite",
                "engine": {"router_replay": {"mode": "R3"}},
                "use_sequence_packing": True,
            },
            {"name": "sglang", "enable_rollout_routing_replay": True},
        )
    with pytest.raises(ValueError, match="does not support actor sequence packing"):
        validate_router_replay_symmetry(
            {
                "strategy": "mlite",
                "engine": {
                    "router_replay": {"mode": "R3"},
                    "impl_cfg": {"sequence_packing": "yes"},
                },
            },
            {"name": "sglang", "enable_rollout_routing_replay": True},
        )
    with pytest.raises(ValueError, match="does not support rollout sequence packing"):
        validate_router_replay_symmetry(
            dict_config,
            {
                "name": "sglang",
                "enable_rollout_routing_replay": True,
                "enable_sequence_packing": "on",
            },
        )
    assert (
        validate_router_replay_symmetry(
            dict_config,
            {
                "name": "sglang",
                "enable_rollout_routing_replay": True,
                "sequence_packing": "false",
            },
        )
        == "R3"
    )


def test_verl_worker_routing_replay_decorator_assigns_flag_only_when_enabled(tmp_path=None):
    worker_path = Path(__file__).resolve().parents[6] / "verl" / "verl" / "workers" / "engine_workers.py"
    tree = ast.parse(worker_path.read_text())
    support_names = {
        "_ROUTING_REPLAY_SIDE_CHANNEL_KEYS",
        "_clear_routing_replay_side_channels",
        "_with_routing_replay_flag",
    }
    support_defs = [
        copy.deepcopy(node)
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    ]
    for support_def in support_defs:
        if isinstance(support_def, ast.FunctionDef):
            support_def.returns = None
        for node in ast.walk(support_def):
            if isinstance(node, ast.arg):
                node.annotation = None
            elif isinstance(node, ast.FunctionDef):
                node.returns = None
    module = ast.Module(body=support_defs, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "functools": functools,
        "tu": types.SimpleNamespace(
            assign_non_tensor_data=lambda data, key, value: data.__setitem__(key, value),
            pop=lambda data, key, default=None: data.pop(key, default),
        ),
    }
    exec(compile(module, str(worker_path), "exec"), namespace)
    with_routing_replay_flag = namespace["_with_routing_replay_flag"]

    calls = []

    def endpoint(self, data, *args, **kwargs):
        calls.append((self.enable_routing_replay, dict(data), args, kwargs))
        return "ok"

    enabled_worker = types.SimpleNamespace(enable_routing_replay=True)
    disabled_worker = types.SimpleNamespace(enable_routing_replay=False)

    replay_data = {}
    assert with_routing_replay_flag(True)(endpoint)(
        enabled_worker, replay_data, "arg", key="value"
    ) == "ok"
    assert replay_data["enable_routing_replay"] is True
    assert calls[-1] == (True, {"enable_routing_replay": True}, ("arg",), {"key": "value"})

    ref_data = {
        "input_ids": "keep",
        "routed_experts": "drop",
        "routed_experts_segments": "drop",
        "routed_experts_segment_seq_lens": "drop",
        "routed_experts_num_routers": "drop",
        "router_replay_action": "drop",
        "record_routed_experts": "drop",
        "router_replay_record": "drop",
        "router_replay_layout": "drop",
        "router_replay_trace_collection_path": "drop",
        "router_replay_trace_source": "drop",
        "router_replay_trace_preserved_through_scheduler": "drop",
        "router_replay_postprocess_wrote_trace_to_batch": "drop",
        "router_replay_trace_metadata_conflicts": "drop",
    }
    assert with_routing_replay_flag(False)(endpoint)(enabled_worker, ref_data) == "ok"
    assert ref_data["enable_routing_replay"] is False
    assert ref_data == {"input_ids": "keep", "enable_routing_replay": False}
    assert calls[-1] == (True, {"input_ids": "keep", "enable_routing_replay": False}, (), {})

    untouched_data = {}
    assert with_routing_replay_flag(True)(endpoint)(disabled_worker, untouched_data) == "ok"
    assert untouched_data == {}
    assert calls[-1] == (False, {}, (), {})


def test_verl_mlite_engine_honors_disabled_routing_replay_flag_static(tmp_path=None):
    engine_source = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "verl_mlite"
        / "engine"
        / "mlite_engine.py"
    ).read_text()

    assert 'key="enable_routing_replay"' in engine_source
    assert "if enable_routing_replay is False:" in engine_source
    assert "from megatron.lite.model.protocol_utils import (" in engine_source
    assert "concat_routed_experts_segments" in engine_source
    assert 'key="routed_experts_segments"' in engine_source
    assert 'key="routed_experts_segment_seq_lens"' in engine_source
    assert 'key="routed_experts_num_routers"' in engine_source
    assert "from megatron.lite.primitive.modules.router_replay import get_routed_experts_dtype" in engine_source
    assert "_compact_routed_experts_for_packing" in engine_source
    assert "return None" in engine_source
    assert "return {}" in engine_source


def test_verl_mlite_engine_routing_replay_helpers_honor_disabled_flag(tmp_path=None):
    from megatron.lite.model.protocol_utils import concat_routed_experts_segments
    from megatron.lite.model.protocol_utils import routed_experts_digest
    from megatron.lite.primitive.modules.router_replay import get_routed_experts_dtype

    engine_path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "verl_mlite"
        / "engine"
        / "mlite_engine.py"
    )
    tree = ast.parse(engine_path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronLiteEngine"
    )
    helper_names = {
        "_routed_experts_for_packing",
        "_compact_routed_experts_for_packing",
        "_router_replay_digest_for_extras",
        "_router_replay_extras",
        "_normalize_router_replay_layout",
        "_required_router_replay_trace_metadata",
        "_bool_config_value",
    }
    helper_defs = [
        copy.deepcopy(node)
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    for node in helper_defs:
        node.decorator_list = []
        node.returns = None
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            arg.annotation = None
    module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "Mapping": Mapping,
        "tu": types.SimpleNamespace(
            get_non_tensor_data=lambda data, key, default: data.get(key, default)
        ),
        "concat_routed_experts_segments": concat_routed_experts_segments,
        "routed_experts_digest": routed_experts_digest,
        "get_routed_experts_dtype": get_routed_experts_dtype,
        "MegatronLiteEngine": types.SimpleNamespace(),
        "_ROUTER_REPLAY_TRACE_STRING_EXTRA_KEYS": (
            "router_replay_trace_collection_path",
            "router_replay_trace_source",
        ),
        "_ROUTER_REPLAY_TRACE_BOOL_EXTRA_KEYS": (
            "router_replay_trace_preserved_through_scheduler",
            "router_replay_postprocess_wrote_trace_to_batch",
        ),
        "_ROUTER_REPLAY_LAYOUT_ALIASES": {
            "true": "true_tokens",
            "true_token": "true_tokens",
            "true_tokens": "true_tokens",
            "full": "full_padded",
            "full_padded": "full_padded",
            "cp_local": "cp_local",
            "local": "cp_local",
        },
        "_ROUTER_REPLAY_TRACE_COLLECTION_PATHS": {
            "generate_request",
            "router_generate_request",
            "user_defined_rollout_loop",
            "mlite_engine_bridge",
        },
        "_ROUTER_REPLAY_TRACE_SOURCES": {"routed_experts", "routed_experts_segments"},
        "_ROUTER_REPLAY_DIGEST_EXTRA_KEY": "routed_experts_digest",
        "_ROUTER_REPLAY_TRACE_CONFLICT_EXTRA_KEY": "router_replay_trace_metadata_conflicts",
    }
    exec(compile(module, str(engine_path), "exec"), namespace)
    namespace["MegatronLiteEngine"]._compact_routed_experts_for_packing = namespace[
        "_compact_routed_experts_for_packing"
    ]
    namespace["MegatronLiteEngine"]._bool_config_value = namespace["_bool_config_value"]
    namespace["MegatronLiteEngine"]._normalize_router_replay_layout = namespace[
        "_normalize_router_replay_layout"
    ]
    namespace["MegatronLiteEngine"]._required_router_replay_trace_metadata = namespace[
        "_required_router_replay_trace_metadata"
    ]
    namespace["MegatronLiteEngine"]._router_replay_digest_for_extras = namespace[
        "_router_replay_digest_for_extras"
    ]
    routed_experts_for_packing = namespace["_routed_experts_for_packing"]
    router_replay_extras = namespace["_router_replay_extras"]

    routed_experts = torch.arange(24, dtype=torch.int64).reshape(2, 3, 4).transpose(0, 1)
    disabled_batch = {
        "enable_routing_replay": "false",
        "routed_experts": routed_experts,
        "router_replay_action": "replay_forward",
        "router_replay_layout": "full_padded",
        "record_routed_experts": True,
        "router_replay_record": True,
    }

    assert routed_experts_for_packing(disabled_batch) is None
    assert router_replay_extras(disabled_batch) == {}

    enabled_batch = {
        "enable_routing_replay": True,
        "routed_experts": routed_experts,
        "router_replay_action": "replay_forward",
        "router_replay_layout": "full_padded",
        "record_routed_experts": "false",
        "router_replay_record": "false",
        "router_replay_trace_collection_path": " user_defined_rollout_loop ",
        "router_replay_trace_source": "routed_experts",
        "router_replay_trace_preserved_through_scheduler": "true",
        "router_replay_postprocess_wrote_trace_to_batch": True,
    }
    packed = routed_experts_for_packing(enabled_batch)
    extras = router_replay_extras(enabled_batch)

    assert packed is not None
    assert packed.is_contiguous()
    assert packed.dtype == torch.uint8
    assert torch.equal(packed.to(torch.long), routed_experts.contiguous())
    assert extras == {
        "router_replay_action": "replay_forward",
        "router_replay_layout": "full_padded",
        "record_routed_experts": False,
        "router_replay_record": False,
        "router_replay_trace_collection_path": "user_defined_rollout_loop",
        "router_replay_trace_source": "routed_experts",
        "router_replay_trace_preserved_through_scheduler": True,
        "router_replay_postprocess_wrote_trace_to_batch": True,
    }

    alias_batch = dict(enabled_batch, router_replay_layout="true")
    assert router_replay_extras(alias_batch)["router_replay_layout"] == "true_tokens"

    conflict_batch = dict(enabled_batch, router_replay_trace_metadata_conflicts=["sglang_version"])
    with pytest.raises(ValueError, match="router_replay_trace_metadata_conflicts to be empty"):
        router_replay_extras(conflict_batch)

    with pytest.raises(ValueError, match="requires router_replay_layout"):
        router_replay_extras(
            {
                "enable_routing_replay": True,
                "routed_experts": routed_experts,
                "router_replay_action": "replay_forward",
            }
        )

    with pytest.raises(ValueError, match="Unsupported MegatronLiteEngine router_replay_layout"):
        router_replay_extras(
            {
                "enable_routing_replay": True,
                "routed_experts": routed_experts,
                "router_replay_layout": "diagonal",
            }
        )

    with pytest.raises(ValueError, match="requires router_replay_trace_collection_path"):
        router_replay_extras(
            {
                "enable_routing_replay": True,
                "routed_experts": routed_experts,
                "router_replay_layout": "true_tokens",
            }
        )

    with pytest.raises(ValueError, match="router_replay_trace_source='routed_experts'"):
        router_replay_extras(
            dict(enabled_batch, router_replay_trace_source="routed_experts_segments")
        )

    with pytest.raises(ValueError, match="router_replay_trace_preserved_through_scheduler=True"):
        router_replay_extras(
            dict(enabled_batch, router_replay_trace_preserved_through_scheduler=False)
        )

    with pytest.raises(ValueError, match="router_replay_trace_collection_path must be one of"):
        router_replay_extras(
            dict(enabled_batch, router_replay_trace_collection_path="plain_generate")
        )

    segment_trace_batch = dict(
        enabled_batch,
        routed_experts=None,
        routed_experts_segments=[torch.tensor([[[0, 1]]], dtype=torch.long)],
        router_replay_trace_source="routed_experts_segments",
    )
    segment_trace_batch.pop("routed_experts")
    assert (
        router_replay_extras(segment_trace_batch)["router_replay_trace_source"]
        == "routed_experts_segments"
    )
    segment_extras = router_replay_extras(
        segment_trace_batch, routed_experts=[torch.tensor([[0, 1]], dtype=torch.long)]
    )
    assert segment_extras["router_replay_trace_source"] == "routed_experts_segments"
    assert segment_extras["router_replay_layout"] == "true_tokens"

    segment_no_layout_batch = dict(segment_trace_batch)
    segment_no_layout_batch.pop("router_replay_layout", None)
    segment_no_layout_extras = router_replay_extras(
        segment_no_layout_batch, routed_experts=[torch.tensor([[0, 1]], dtype=torch.long)]
    )
    assert segment_no_layout_extras["router_replay_trace_source"] == "routed_experts_segments"
    assert segment_no_layout_extras["router_replay_layout"] == "true_tokens"

    trace_false_batch = {
        "enable_routing_replay": True,
        "record_routed_experts": "false",
        "router_replay_record": 0,
        "router_replay_trace_preserved_through_scheduler": "false",
        "router_replay_postprocess_wrote_trace_to_batch": 0,
    }
    assert router_replay_extras(trace_false_batch) == {
        "record_routed_experts": False,
        "router_replay_record": False,
        "router_replay_trace_preserved_through_scheduler": False,
        "router_replay_postprocess_wrote_trace_to_batch": False,
    }

    with pytest.raises(ValueError, match="router_replay_trace_preserved_through_scheduler"):
        router_replay_extras(
            {
                "enable_routing_replay": True,
                "router_replay_trace_preserved_through_scheduler": "maybe",
            }
        )

    per_router_batch = {
        "enable_routing_replay": True,
        "routed_experts": [
            torch.tensor([[256, 0], [2, 1]]).transpose(0, 1),
            torch.tensor([[3, 2], [4, 3]]),
        ],
    }
    packed_list = routed_experts_for_packing(per_router_batch)
    assert isinstance(packed_list, list)
    assert all(item.is_contiguous() for item in packed_list)
    assert packed_list[0].dtype == torch.int16
    assert packed_list[1].dtype == torch.uint8
    torch.testing.assert_close(packed_list[0].to(torch.long), torch.tensor([[256, 2], [0, 1]]))
    torch.testing.assert_close(packed_list[1].to(torch.long), torch.tensor([[3, 2], [4, 3]]))

    float_batch = {
        "enable_routing_replay": True,
        "routed_experts": torch.tensor([[0.0, 1.0]]),
    }
    with pytest.raises(TypeError, match="integer tensor dtype"):
        routed_experts_for_packing(float_batch)

    negative_batch = {
        "enable_routing_replay": True,
        "routed_experts": torch.tensor([[0, -1]]),
    }
    with pytest.raises(ValueError, match="non-negative expert ids"):
        routed_experts_for_packing(negative_batch)

    padded_segment = torch.tensor(
        [
            [
                [[1, 0], [4, 3]],
                [[2, 1], [5, 4]],
                [[99, 99], [99, 99]],
            ]
        ],
        dtype=torch.long,
    )
    token_segment = torch.tensor(
        [
            [[8, 7], [6, 5]],
            [[9, 8], [7, 6]],
            [[10, 9], [8, 7]],
        ],
        dtype=torch.long,
    )
    segment_batch = {
        "enable_routing_replay": True,
        "routed_experts_segments": [padded_segment, token_segment],
        "routed_experts_segment_seq_lens": [torch.tensor([2]), None],
        "routed_experts_num_routers": 2,
    }
    packed_segments = routed_experts_for_packing(segment_batch)
    assert isinstance(packed_segments, list)
    torch.testing.assert_close(
        packed_segments[0].to(torch.long),
        torch.tensor([[1, 0], [2, 1], [8, 7], [9, 8], [10, 9]]),
    )
    torch.testing.assert_close(
        packed_segments[1].to(torch.long),
        torch.tensor([[4, 3], [5, 4], [6, 5], [7, 6], [8, 7]]),
    )

    disabled_segment_batch = {
        "enable_routing_replay": "off",
        "routed_experts_segments": [padded_segment],
        "routed_experts_segment_seq_lens": [torch.tensor([2])],
        "routed_experts_num_routers": 2,
        "routed_experts": routed_experts,
    }
    assert routed_experts_for_packing(disabled_segment_batch) is None

    with pytest.raises(ValueError, match="enable_routing_replay"):
        router_replay_extras({"enable_routing_replay": "maybe"})

    negative_segment = padded_segment.clone()
    negative_segment[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="non-negative expert ids"):
        routed_experts_for_packing(
            {
                "enable_routing_replay": True,
                "routed_experts_segments": [negative_segment],
                "routed_experts_segment_seq_lens": [torch.tensor([2])],
                "routed_experts_num_routers": 2,
            }
        )
