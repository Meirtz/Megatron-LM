# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import MegatronLiteEngine, _build_lr_scheduler

_GRPO_LAUNCHER = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "verl"
    / "scripts"
    / "run_qwen3moe_gsm8k_grpo.sh"
)


def _optimizer_config(**override_optimizer_config) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1e-6,
        min_lr=None,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.1,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=10,
        lr_warmup_steps=0,
        lr_warmup_init=0.0,
        lr_decay_steps=None,
        lr_decay_style="constant",
        weight_decay_incr_style="constant",
        lr_wsd_decay_style="exponential",
        lr_wsd_decay_steps=None,
        use_checkpoint_opt_param_scheduler=False,
        betas=(0.9, 0.95),
        override_optimizer_config=override_optimizer_config,
    )


def _engine(
    *,
    engine_config: MegatronLiteEngineConfig,
    optimizer_config: SimpleNamespace | None = None,
) -> MegatronLiteEngine:
    return MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=engine_config,
        optimizer_config=optimizer_config or _optimizer_config(),
        checkpoint_config={},
    )


def _engine_config(**kwargs) -> MegatronLiteEngineConfig:
    values = {"custom_backend_module": None, "impl_cfg": {"use_thd": True}}
    values.update(kwargs)
    return MegatronLiteEngineConfig(**values)


def test_canonical_config_target_import_registers_default_mlite_backend() -> None:
    """Exercise the fresh-process import path used by VERL/Hydra configs."""

    script = r"""
import importlib
import sys

import verl.workers.engine.base as engine_base

EngineRegistry = engine_base.EngineRegistry

assert "verl_mlite.engine.mlite_engine" not in sys.modules
config_module = importlib.import_module("verl_mlite.engine.config")
config = config_module.MegatronLiteEngineConfig(impl_cfg={"use_thd": True})
assert config.custom_backend_module == "verl_mlite.engine.mlite_engine"
assert config.strategy == "mlite"
assert config.weight_sync_target == "hf"
# Legacy/local VERL revisions may query the physical host directly. Isolate this
# registration-only assertion from host device detection; no device work occurs.
engine_base.get_device_name = lambda: "cuda"
engine_cls = EngineRegistry.get_engine_cls(
    model_type="language_model", backend="mlite"
)
assert engine_cls.__module__ == config.custom_backend_module
assert engine_cls.__name__ == "MegatronLiteEngine"
print("CANONICAL_VERL_MLITE_REGISTRY_PASS")
"""
    env = dict(os.environ)
    env["VERL_ENGINE_DEVICE"] = "cuda"
    env.pop("VERL_ENGINE_VENDOR", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"fresh VERL registry process failed\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip() == "CANONICAL_VERL_MLITE_REGISTRY_PASS"


def test_optimizer_offload_enables_full_optimizer_state_offload_by_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(
            use_precision_aware_optimizer=True, decoupled_weight_decay=True
        ),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0
    assert optimizer.use_precision_aware_optimizer is True
    assert optimizer.decoupled_weight_decay is True
    assert optimizer.adam_beta1 == 0.9
    assert optimizer.adam_beta2 == 0.95


def test_explicit_optimizer_offload_fraction_overrides_engine_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(offload_fraction=0.25),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 0.25


def test_optimizer_cpu_offload_alias_maps_to_full_offload_fraction() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=False),
        optimizer_config=_optimizer_config(optimizer_cpu_offload=True),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0


def test_mlite_config_threads_rl_parallel_and_impl_settings() -> None:
    engine = _engine(
        engine_config=_engine_config(
            tp=2,
            ep=8,
            etp=1,
            pp=1,
            cp=1,
            optimizer_offload=True,
            attention_backend_override="flash",
            impl_cfg={"use_thd": True, "deterministic": False},
        )
    )

    config = engine._build_mlite_config()

    assert config.model_name == "qwen3_5"
    assert config.impl == "lite"
    assert config.parallel.tp == 2
    assert config.parallel.ep == 8
    assert config.parallel.etp == 1
    assert config.optimizer.offload_fraction == 1.0
    assert config.attention_backend_override == "flash"
    assert config.impl_cfg["use_thd"] is True
    assert config.impl_cfg["deterministic"] is False


@pytest.mark.parametrize("target", ["hf", "vllm"])
def test_auto_qwen35_weight_sync_uses_configured_consumer_target(target) -> None:
    engine = _engine(engine_config=_engine_config(weight_sync_target=target))
    assert engine.engine_config.model_name == "auto"
    engine._mlite_config = engine._build_mlite_config()
    assert engine._mlite_config.model_name == "qwen3_5"

    captured = {}

    def export_weights(handle, **kwargs):
        captured["handle"] = handle
        captured["kwargs"] = kwargs
        return ["weight-payload"]

    handle = object()
    engine.runtime = SimpleNamespace(export_weights=export_weights)
    engine.handle = handle

    payload, metadata = engine.get_per_tensor_param(limit=3)

    assert payload == ["weight-payload"]
    assert metadata is None
    assert captured["handle"] is handle
    assert captured["kwargs"] == {
        "limit": 3,
        "target": target,
        "export_dtype": "bfloat16",
    }


def test_weight_sync_target_rejects_unknown_consumer() -> None:
    with pytest.raises(ValueError, match="weight_sync_target"):
        _engine_config(weight_sync_target="sglang")


def test_vllm_weight_sync_target_rejects_non_qwen35_model() -> None:
    engine = _engine(engine_config=_engine_config(weight_sync_target="vllm"))
    engine._mlite_config = SimpleNamespace(model_name="deepseek_v4")
    engine.runtime = SimpleNamespace(export_weights=lambda *_args, **_kwargs: [])
    engine.handle = object()

    with pytest.raises(ValueError, match="implemented only for Qwen3.5"):
        engine.get_per_tensor_param()


@pytest.mark.parametrize(
    ("rollout_backend", "expected_target"), [("vllm", "vllm"), ("sglang", "hf")]
)
def test_grpo_launcher_selects_consumer_compatible_weight_keyspace(
    tmp_path, rollout_backend, expected_target
) -> None:
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "INFER_BACKEND": rollout_backend,
        "OUTPUT_ROOT": str(tmp_path / rollout_backend),
    }

    completed = subprocess.run(
        [str(_GRPO_LAUNCHER)], env=env, text=True, capture_output=True, check=True
    )

    assert (
        "actor_rollout_ref.actor.engine.weight_sync_target=" + expected_target
        in completed.stdout
    )


def test_grpo_launcher_rejects_qwen35_trtllm_without_live_reload_evidence(
    tmp_path,
) -> None:
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "INFER_BACKEND": "trtllm",
        "MLITE_MODEL_NAME": "qwen3_5",
        "OUTPUT_ROOT": str(tmp_path / "qwen3_5"),
    }

    completed = subprocess.run(
        [str(_GRPO_LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert completed.returncode != 0
    assert (
        "Qwen3.5 TRT-LLM rollout is blocked for pinned VERL "
        "1ff76cc625e9820d2434dad1b6d9b8e5dd26a359 + TensorRT-LLM "
        "v1.3.0rc19" in completed.stderr
    )
    assert "live dynamic weight reload has not been validated end to end" in (
        completed.stderr
    )
    assert "static mapper/key-normalization analysis is insufficient" in (
        completed.stderr
    )
    assert "deterministically asserts" not in completed.stderr
    assert "actor_rollout_ref.actor.engine.weight_sync_target=" not in completed.stdout


@pytest.mark.parametrize("rollout_backend", ["vllm", "sglang", "trtllm"])
def test_grpo_launcher_rejects_auto_without_guessing_from_non_qwen_model_path(
    tmp_path, rollout_backend
) -> None:
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "INFER_BACKEND": rollout_backend,
        "MODEL_PATH": "/models/DeepSeek-V4-not-a-qwen-model",
        "MLITE_MODEL_NAME": "auto",
        "OUTPUT_ROOT": str(tmp_path / rollout_backend),
    }

    completed = subprocess.run(
        [str(_GRPO_LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert completed.returncode != 0
    assert "MLITE_MODEL_NAME=auto is not safe in this launcher" in completed.stderr
    assert "MODEL_PATH text is not authoritative" in completed.stderr
    assert "Set MLITE_MODEL_NAME explicitly" in completed.stderr
    assert "actor_rollout_ref.actor.engine.weight_sync_target=" not in completed.stdout


@pytest.mark.parametrize("rollout_backend", ["vllm", "trtllm"])
def test_grpo_launcher_keeps_hf_target_for_explicit_non_qwen35_model(
    tmp_path, rollout_backend
) -> None:
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "INFER_BACKEND": rollout_backend,
        "MLITE_MODEL_NAME": "deepseek_v4",
        "OUTPUT_ROOT": str(tmp_path / rollout_backend),
    }

    completed = subprocess.run(
        [str(_GRPO_LAUNCHER)], env=env, text=True, capture_output=True, check=True
    )

    assert "actor_rollout_ref.actor.engine.model_name=deepseek_v4" in completed.stdout
    assert "actor_rollout_ref.actor.engine.weight_sync_target=hf" in completed.stdout
    assert f"actor_rollout_ref.rollout.name={rollout_backend}" in completed.stdout


def test_local_lr_scheduler_warmup_decay_and_state_roundtrip() -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.1}])
    opt = SimpleNamespace(
        total_training_steps=4,
        lr_warmup_steps=1,
        lr_warmup_steps_ratio=0.0,
        lr_warmup_init=0.0,
        lr=1.0,
        min_lr=0.1,
        lr_decay_steps=4,
        lr_decay_style="linear",
        weight_decay=0.1,
        weight_decay_incr_style="constant",
        lr_wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
        use_checkpoint_opt_param_scheduler=False,
    )

    scheduler = _build_lr_scheduler(optimizer, opt)

    assert optimizer.param_groups[0]["lr"] == 0.0
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == 1.0
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.7)

    state = scheduler.state_dict()
    scheduler.step(10)
    scheduler.load_state_dict(state)

    assert scheduler.state_dict() == state
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.7)
