# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import MegatronLiteEngine, _checkpoint_bool, _content_set


class _Scheduler:
    def __init__(self):
        self.loaded_state = None

    def state_dict(self):
        return {"step": 7, "lr": 0.25}

    def load_state_dict(self, state):
        self.loaded_state = state


@pytest.fixture(autouse=True)
def _single_process_dist(monkeypatch):
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: False)


def _optimizer_config() -> SimpleNamespace:
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
        override_optimizer_config={},
    )


def _engine_config(**kwargs) -> MegatronLiteEngineConfig:
    values = {"custom_backend_module": None, "impl_cfg": {"use_thd": True}}
    values.update(kwargs)
    return MegatronLiteEngineConfig(**values)


def _initialized_engine(
    *, checkpoint_config=None, param_offload=False, engine_config_kwargs=None, handle_extras=None
):
    engine_config_kwargs = engine_config_kwargs or {}
    engine = MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=_engine_config(param_offload=param_offload, **engine_config_kwargs),
        optimizer_config=_optimizer_config(),
        checkpoint_config=checkpoint_config or {},
    )

    def placement_fn(name):
        return ["placement", name]

    def expert_classifier(name):
        return name.endswith("expert")

    parallel = SimpleNamespace(tp=1, cp=1, pp=1)
    parallel_state = SimpleNamespace(dp_rank=0)
    module = torch.nn.Linear(2, 2)
    optimizer = object()
    scheduler = _Scheduler()
    engine.module = module
    extras = {
        "protocol": SimpleNamespace(
            PLACEMENT_FN=placement_fn, EXPERT_CLASSIFIER=expert_classifier
        )
    }
    extras.update(handle_extras or {})
    engine.handle = SimpleNamespace(
        _optimizer=optimizer,
        _lr_scheduler=scheduler,
        _config=SimpleNamespace(parallel=parallel),
        _parallel_state=parallel_state,
        _extras=extras,
    )
    engine.runtime = object()
    return (
        engine,
        module,
        optimizer,
        scheduler,
        parallel,
        parallel_state,
        placement_fn,
        expert_classifier,
    )


def test_save_checkpoint_forwards_contents_scheduler_and_param_offload_reload(
    tmp_path, monkeypatch
):
    (
        engine,
        module,
        optimizer,
        scheduler,
        parallel,
        parallel_state,
        placement_fn,
        expert_classifier,
    ) = _initialized_engine(checkpoint_config={"save_contents": ["model"]}, param_offload=True)
    to_calls = []
    save_calls = []
    sync_calls = []
    monkeypatch.setattr(engine, "to", lambda **kwargs: to_calls.append(kwargs))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(tmp_path), global_step=13)

    assert to_calls == [
        {"device": "cuda", "model": True, "optimizer": False, "grad": False},
        {"device": "cpu", "model": True, "optimizer": False, "grad": False},
    ]
    assert sync_calls == [True]
    assert len(save_calls) == 1
    save_args, save_kwargs = save_calls[0]
    assert save_args == (module, optimizer, 13, str(tmp_path), parallel, parallel_state)
    assert save_kwargs["get_placements"] is placement_fn
    assert save_kwargs["is_expert"] is expert_classifier
    assert save_kwargs["save_model"] is True
    assert save_kwargs["save_optimizer"] is False
    assert (
        torch.load(tmp_path / "lr_scheduler.pt", map_location="cpu", weights_only=False)
        == scheduler.state_dict()
    )


def test_save_checkpoint_skips_when_contents_exclude_model_and_optimizer(tmp_path, monkeypatch):
    engine, *_ = _initialized_engine(checkpoint_config={"save_contents": ["extra"]})
    checkpoint_path = tmp_path / "ckpt"
    save_calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(checkpoint_path), global_step=13)

    assert save_calls == []
    assert not checkpoint_path.exists()


def test_checkpoint_content_set_parses_exact_keys_and_rejects_bad_values():
    assert _content_set(None) == set()
    assert _content_set("model") == {"model"}
    assert _content_set("[model, optimizer, lora_adapter]") == {
        "model",
        "optimizer",
        "lora_adapter",
    }
    assert _content_set("[ 'lora_adapter' ]") == {"lora_adapter"}
    assert _content_set("'model'") == {"model"}
    assert _content_set((" 'model' ", '"peft_adapter"')) == {"model", "peft_adapter"}
    assert _content_set(("model", "peft_adapter")) == {"model", "peft_adapter"}
    assert "model" not in _content_set("not_model")

    with pytest.raises(TypeError, match="checkpoint_config.save_contents"):
        _content_set(123, key="checkpoint_config.save_contents")
    with pytest.raises(TypeError, match="checkpoint_config.save_contents"):
        _content_set({"model": True}, key="checkpoint_config.save_contents")
    with pytest.raises(TypeError, match="entries must be strings"):
        _content_set(["model", 123], key="checkpoint_config.save_contents")


def test_checkpoint_bool_parses_strings_and_rejects_bad_values():
    assert _checkpoint_bool(None, key="checkpoint_config.save_lora_adapter") is False
    assert _checkpoint_bool(True, key="checkpoint_config.save_lora_adapter") is True
    assert _checkpoint_bool(False, key="checkpoint_config.save_lora_adapter") is False
    assert _checkpoint_bool("True", key="checkpoint_config.save_lora_adapter") is True
    assert _checkpoint_bool("false", key="checkpoint_config.save_lora_adapter") is False
    assert _checkpoint_bool("1", key="checkpoint_config.save_lora_adapter") is True
    assert _checkpoint_bool("0", key="checkpoint_config.save_lora_adapter") is False

    with pytest.raises(ValueError, match="checkpoint_config.save_lora_adapter"):
        _checkpoint_bool("maybe", key="checkpoint_config.save_lora_adapter")
    with pytest.raises(TypeError, match="checkpoint_config.save_lora_adapter"):
        _checkpoint_bool(1, key="checkpoint_config.save_lora_adapter")


def test_save_checkpoint_uses_exact_content_keys_not_substrings(tmp_path, monkeypatch):
    engine, *_ = _initialized_engine(checkpoint_config={"save_contents": "not_model"})
    checkpoint_path = tmp_path / "ckpt"
    save_calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(checkpoint_path), global_step=13)

    assert save_calls == []
    assert not checkpoint_path.exists()


def test_save_checkpoint_string_false_does_not_enable_lora_adapter(tmp_path, monkeypatch):
    engine, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["extra"], "save_lora_adapter": "False"}
    )
    checkpoint_path = tmp_path / "ckpt"
    save_calls = []
    adapter_calls = []

    class RuntimeWithAdapter:
        def save_lora_adapter(self, *args, **kwargs):
            adapter_calls.append((args, kwargs))

    engine.runtime = RuntimeWithAdapter()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(checkpoint_path), global_step=13)

    assert save_calls == []
    assert adapter_calls == []
    assert not checkpoint_path.exists()


def test_save_checkpoint_accepts_bracketed_lora_adapter_contents_string(tmp_path, monkeypatch):
    lora_config = {"rank": 16, "alpha": 32, "target_modules": "all-linear"}
    engine, *_ = _initialized_engine(
        checkpoint_config={"save_contents": "[lora_adapter]"},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    save_calls = []
    adapter_calls = []

    class RuntimeWithAdapter:
        def save_lora_adapter(self, *args, **kwargs):
            adapter_calls.append((args, kwargs))

    engine.runtime = RuntimeWithAdapter()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(tmp_path), global_step=13)

    assert save_calls == []
    assert len(adapter_calls) == 1
    adapter_args, adapter_kwargs = adapter_calls[0]
    assert adapter_args == (engine.handle, str(tmp_path / "lora_adapter"))
    assert adapter_kwargs["lora_config"] == lora_config
    assert adapter_kwargs["metadata"] == {"global_step": 13}


def test_save_checkpoint_string_true_enables_lora_adapter(tmp_path, monkeypatch):
    lora_config = {"rank": 16, "alpha": 32, "target_modules": "all-linear"}
    engine, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["extra"], "save_lora_adapter": "true"},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    save_calls = []
    adapter_calls = []

    class RuntimeWithAdapter:
        def save_lora_adapter(self, *args, **kwargs):
            adapter_calls.append((args, kwargs))

    engine.runtime = RuntimeWithAdapter()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(tmp_path), global_step=13)

    assert save_calls == []
    assert len(adapter_calls) == 1
    adapter_args, adapter_kwargs = adapter_calls[0]
    assert adapter_args == (engine.handle, str(tmp_path / "lora_adapter"))
    assert adapter_kwargs["lora_config"] == lora_config


def test_lora_adapter_checkpoint_dir_name_stays_inside_checkpoint(tmp_path, monkeypatch):
    del monkeypatch
    engine, *_ = _initialized_engine(
        checkpoint_config={"lora_adapter_dir_name": "adapters/peft"}
    )
    assert engine._lora_adapter_checkpoint_path(str(tmp_path)) == str(tmp_path / "adapters" / "peft")

    for bad_name in ("/tmp/outside", "..", "../outside", "adapters/../outside", "."):
        engine, *_ = _initialized_engine(checkpoint_config={"lora_adapter_dir_name": bad_name})
        with pytest.raises(ValueError, match="inside the checkpoint directory"):
            engine._lora_adapter_checkpoint_path(str(tmp_path))


def test_save_checkpoint_can_write_lora_adapter_sidecar_without_full_checkpoint(
    tmp_path, monkeypatch
):
    lora_config = {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": "all-linear",
        "use_rslora": True,
    }
    engine, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    save_calls = []
    adapter_calls = []

    class RuntimeWithAdapter:
        def save_lora_adapter(self, *args, **kwargs):
            adapter_calls.append((args, kwargs))

    engine.runtime = RuntimeWithAdapter()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(tmp_path), global_step=13)

    assert save_calls == []
    assert len(adapter_calls) == 1
    adapter_args, adapter_kwargs = adapter_calls[0]
    assert adapter_args == (engine.handle, str(tmp_path / "lora_adapter"))
    assert adapter_kwargs["lora_config"] == lora_config
    assert adapter_kwargs["base_model_name_or_path"] == "/tmp/qwen35"
    assert adapter_kwargs["metadata"] == {"global_step": 13}
    assert not (tmp_path / "lr_scheduler.pt").exists()


def test_save_checkpoint_lora_adapter_passes_init_and_user_kwargs(tmp_path, monkeypatch):
    lora_config = {"rank": 8, "alpha": 16, "target_modules": "all-linear"}
    engine, *_ = _initialized_engine(
        checkpoint_config={
            "save_lora_adapter": True,
            "lora_adapter_dir_name": "peft",
            "lora_adapter_kwargs": {
                "metadata": {"source": "unit"},
                "base_model_name_or_path": "override/base",
            },
        },
        engine_config_kwargs={
            "impl_cfg": {
                "use_thd": True,
                "lora": lora_config,
                "lora_init": "olora_tail",
            }
        },
    )
    adapter_calls = []

    class RuntimeWithAdapter:
        def save_lora_adapter(self, *args, **kwargs):
            adapter_calls.append((args, kwargs))

    engine.runtime = RuntimeWithAdapter()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: None,
    )

    engine.save_checkpoint(str(tmp_path), global_step=23)

    assert len(adapter_calls) == 1
    adapter_args, adapter_kwargs = adapter_calls[0]
    assert adapter_args == (engine.handle, str(tmp_path / "peft"))
    assert adapter_kwargs["lora_config"] == lora_config
    assert adapter_kwargs["base_model_name_or_path"] == "override/base"
    assert adapter_kwargs["init_lora_weights"] == "olora_tail"
    assert adapter_kwargs["metadata"] == {"source": "unit", "global_step": 23}


def test_save_checkpoint_lora_adapter_accepts_mapping_kwargs(tmp_path, monkeypatch):
    lora_config = {"rank": 8, "alpha": 16, "target_modules": "all-linear"}
    engine, *_ = _initialized_engine(
        checkpoint_config={
            "save_lora_adapter": True,
            "lora_adapter_kwargs": MappingProxyType({"metadata": {"source": "mapping"}}),
        },
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    adapter_calls = []

    class RuntimeWithAdapter:
        def save_lora_adapter(self, *args, **kwargs):
            adapter_calls.append((args, kwargs))

    engine.runtime = RuntimeWithAdapter()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: None,
    )

    engine.save_checkpoint(str(tmp_path), global_step=29)

    assert len(adapter_calls) == 1
    _, adapter_kwargs = adapter_calls[0]
    assert adapter_kwargs["metadata"] == {"source": "mapping", "global_step": 29}


def test_save_checkpoint_lora_adapter_rejects_non_mapping_kwargs_and_metadata(
    tmp_path, monkeypatch
):
    lora_config = {"rank": 8, "alpha": 16, "target_modules": "all-linear"}

    class RuntimeWithAdapter:
        def save_lora_adapter(self, *args, **kwargs):
            raise AssertionError("adapter save should fail before runtime call")

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: None,
    )

    bad_kwargs_engine, *_ = _initialized_engine(
        checkpoint_config={
            "save_lora_adapter": True,
            "save_contents": ["extra"],
            "lora_adapter_kwargs": [("metadata", {"source": "unit"})],
        },
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    bad_kwargs_engine.runtime = RuntimeWithAdapter()
    with pytest.raises(TypeError, match="lora_adapter_kwargs must be a mapping"):
        bad_kwargs_engine.save_checkpoint(str(tmp_path / "bad_kwargs"), global_step=31)

    bad_metadata_engine, *_ = _initialized_engine(
        checkpoint_config={
            "save_lora_adapter": True,
            "save_contents": ["extra"],
            "lora_adapter_kwargs": {"metadata": [("source", "unit")]},
        },
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    bad_metadata_engine.runtime = RuntimeWithAdapter()
    with pytest.raises(TypeError, match="lora_adapter_kwargs.metadata must be a mapping"):
        bad_metadata_engine.save_checkpoint(str(tmp_path / "bad_metadata"), global_step=37)


def test_save_checkpoint_lora_adapter_validation_runs_before_full_checkpoint_side_effects(
    tmp_path, monkeypatch
):
    lora_config = {"rank": 8, "alpha": 16, "target_modules": "all-linear"}
    save_calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    class RuntimeWithAdapter:
        def save_lora_adapter(self, *args, **kwargs):
            raise AssertionError("adapter save should fail before runtime call")

    bad_kwargs_engine, *_ = _initialized_engine(
        checkpoint_config={
            "save_contents": ["model", "lora_adapter"],
            "lora_adapter_kwargs": [("metadata", {"source": "unit"})],
        },
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    bad_kwargs_engine.runtime = RuntimeWithAdapter()
    checkpoint_path = tmp_path / "bad_kwargs_full_ckpt"
    with pytest.raises(TypeError, match="lora_adapter_kwargs must be a mapping"):
        bad_kwargs_engine.save_checkpoint(str(checkpoint_path), global_step=41)
    assert save_calls == []
    assert not checkpoint_path.exists()

    unsupported_engine, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["model", "lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    unsupported_path = tmp_path / "unsupported_full_ckpt"
    with pytest.raises(NotImplementedError, match="save_lora_adapter"):
        unsupported_engine.save_checkpoint(str(unsupported_path), global_step=43)
    assert save_calls == []
    assert not unsupported_path.exists()

    missing_lora_engine, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["model", "lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True}},
    )
    missing_lora_engine.runtime = RuntimeWithAdapter()
    missing_lora_path = tmp_path / "missing_lora_full_ckpt"
    with pytest.raises(ValueError, match="requires an enabled LoRA config"):
        missing_lora_engine.save_checkpoint(str(missing_lora_path), global_step=47)
    assert save_calls == []
    assert not missing_lora_path.exists()

    disabled_lora_engine, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["model", "lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": {"rank": 0}}},
    )
    disabled_lora_engine.runtime = RuntimeWithAdapter()
    disabled_lora_path = tmp_path / "disabled_lora_full_ckpt"
    with pytest.raises(ValueError, match="requires an enabled LoRA config"):
        disabled_lora_engine.save_checkpoint(str(disabled_lora_path), global_step=53)
    assert save_calls == []
    assert not disabled_lora_path.exists()


def test_save_checkpoint_lora_adapter_only_failure_cleans_new_checkpoint_dir(
    tmp_path, monkeypatch
):
    lora_config = {"rank": 8, "alpha": 16, "target_modules": "all-linear"}
    engine, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full checkpoint should not be saved")
        ),
    )

    class FailingRuntimeWithAdapter:
        def save_lora_adapter(self, handle, adapter_path, **kwargs):
            del handle, kwargs
            adapter_path = Path(adapter_path)
            adapter_path.mkdir(parents=True, exist_ok=True)
            (adapter_path / "partial").write_text("partial")
            raise RuntimeError("adapter save exploded")

    engine.runtime = FailingRuntimeWithAdapter()
    checkpoint_path = tmp_path / "new_adapter_only_ckpt"

    with pytest.raises(RuntimeError, match="adapter save exploded"):
        engine.save_checkpoint(str(checkpoint_path), global_step=59)

    assert not checkpoint_path.exists()


def test_save_checkpoint_lora_adapter_only_failure_preserves_existing_checkpoint_dir(
    tmp_path, monkeypatch
):
    lora_config = {"rank": 8, "alpha": 16, "target_modules": "all-linear"}
    engine, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full checkpoint should not be saved")
        ),
    )
    checkpoint_path = tmp_path / "existing_adapter_only_ckpt"
    checkpoint_path.mkdir()
    sentinel = checkpoint_path / "keep.txt"
    sentinel.write_text("keep")

    class FailingRuntimeWithAdapter:
        def save_lora_adapter(self, *args, **kwargs):
            raise RuntimeError("adapter save exploded")

    engine.runtime = FailingRuntimeWithAdapter()

    with pytest.raises(RuntimeError, match="adapter save exploded"):
        engine.save_checkpoint(str(checkpoint_path), global_step=61)

    assert checkpoint_path.exists()
    assert sentinel.read_text() == "keep"


def test_load_checkpoint_restores_scheduler_and_param_offload_reload(tmp_path, monkeypatch):
    (
        engine,
        module,
        optimizer,
        scheduler,
        parallel,
        parallel_state,
        placement_fn,
        expert_classifier,
    ) = _initialized_engine(param_offload=True)
    torch.save({"step": 23, "lr": 0.125}, tmp_path / "lr_scheduler.pt")
    to_calls = []
    load_calls = []
    sync_calls = []
    monkeypatch.setattr(engine, "to", lambda **kwargs: to_calls.append(kwargs))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    engine.load_checkpoint(str(tmp_path))

    assert to_calls == [
        {"device": "cuda", "model": True, "optimizer": False, "grad": False},
        {"device": "cpu", "model": True, "optimizer": False, "grad": False},
    ]
    assert sync_calls == [True]
    assert scheduler.loaded_state == {"step": 23, "lr": 0.125}
    assert len(load_calls) == 1
    load_args, load_kwargs = load_calls[0]
    assert load_args == (module, optimizer, str(tmp_path), parallel, parallel_state)
    assert load_kwargs["get_placements"] is placement_fn
    assert load_kwargs["is_expert"] is expert_classifier
    assert load_kwargs["load_model"] is True
    assert load_kwargs["load_optimizer"] is True


def test_load_checkpoint_skips_when_contents_exclude_all_mlite_state(tmp_path, monkeypatch):
    engine, *_ = _initialized_engine(
        checkpoint_config={"load_contents": ["extra"]}, param_offload=True
    )
    load_calls = []
    to_calls = []
    sync_calls = []
    monkeypatch.setattr(engine, "to", lambda **kwargs: to_calls.append(kwargs))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    engine.load_checkpoint(str(tmp_path))

    assert load_calls == []
    assert to_calls == []
    assert sync_calls == []


def test_load_checkpoint_can_restore_lora_adapter_without_full_checkpoint(tmp_path, monkeypatch):
    lora_config = {"rank": 16, "alpha": 32, "target_modules": "all-linear"}
    (tmp_path / "lora_adapter").mkdir()
    engine, *_ = _initialized_engine(
        checkpoint_config={"load_contents": ["lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    load_calls = []
    adapter_calls = []

    class RuntimeWithAdapter:
        def load_lora_adapter(self, *args, **kwargs):
            adapter_calls.append((args, kwargs))

    engine.runtime = RuntimeWithAdapter()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    engine.load_checkpoint(str(tmp_path))

    assert load_calls == []
    assert len(adapter_calls) == 1
    adapter_args, adapter_kwargs = adapter_calls[0]
    assert adapter_args == (engine.handle, str(tmp_path / "lora_adapter"))
    assert adapter_kwargs == {"lora_config": lora_config}


def test_load_checkpoint_string_false_does_not_enable_lora_adapter(tmp_path, monkeypatch):
    engine, *_ = _initialized_engine(
        checkpoint_config={"load_contents": ["extra"], "load_lora_adapter": "False"}
    )
    load_calls = []
    adapter_calls = []

    class RuntimeWithAdapter:
        def load_lora_adapter(self, *args, **kwargs):
            adapter_calls.append((args, kwargs))

    engine.runtime = RuntimeWithAdapter()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    engine.load_checkpoint(str(tmp_path))

    assert load_calls == []
    assert adapter_calls == []


def test_load_checkpoint_string_true_enables_lora_adapter(tmp_path, monkeypatch):
    lora_config = {"rank": 16, "alpha": 32, "target_modules": "all-linear"}
    (tmp_path / "lora_adapter").mkdir()
    engine, *_ = _initialized_engine(
        checkpoint_config={"load_contents": ["extra"], "load_lora_adapter": "true"},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    load_calls = []
    adapter_calls = []

    class RuntimeWithAdapter:
        def load_lora_adapter(self, *args, **kwargs):
            adapter_calls.append((args, kwargs))

    engine.runtime = RuntimeWithAdapter()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    engine.load_checkpoint(str(tmp_path))

    assert load_calls == []
    assert len(adapter_calls) == 1
    adapter_args, adapter_kwargs = adapter_calls[0]
    assert adapter_args == (engine.handle, str(tmp_path / "lora_adapter"))
    assert adapter_kwargs == {"lora_config": lora_config}


def test_load_checkpoint_lora_adapter_rejects_non_mapping_kwargs_and_metadata(
    tmp_path, monkeypatch
):
    lora_config = {"rank": 16, "alpha": 32, "target_modules": "all-linear"}

    class RuntimeWithAdapter:
        def load_lora_adapter(self, *args, **kwargs):
            raise AssertionError("adapter load should fail before runtime call")

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: None,
    )

    bad_kwargs_engine, *_ = _initialized_engine(
        checkpoint_config={
            "load_lora_adapter": True,
            "load_contents": ["extra"],
            "lora_adapter_kwargs": [("lora_config", lora_config)],
        },
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    bad_kwargs_engine.runtime = RuntimeWithAdapter()
    with pytest.raises(TypeError, match="lora_adapter_kwargs must be a mapping"):
        bad_kwargs_engine.load_checkpoint(str(tmp_path / "bad_kwargs"))

    bad_metadata_engine, *_ = _initialized_engine(
        checkpoint_config={
            "load_lora_adapter": True,
            "load_contents": ["extra"],
            "lora_adapter_kwargs": {"metadata": [("source", "unit")]},
        },
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    bad_metadata_engine.runtime = RuntimeWithAdapter()
    with pytest.raises(TypeError, match="lora_adapter_kwargs.metadata must be a mapping"):
        bad_metadata_engine.load_checkpoint(str(tmp_path / "bad_metadata"))


def test_load_checkpoint_lora_adapter_validation_runs_before_full_checkpoint_side_effects(
    tmp_path, monkeypatch
):
    lora_config = {"rank": 16, "alpha": 32, "target_modules": "all-linear"}
    load_calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    class RuntimeWithAdapter:
        def load_lora_adapter(self, *args, **kwargs):
            raise AssertionError("adapter load should fail before runtime call")

    bad_kwargs_engine, *_ = _initialized_engine(
        checkpoint_config={
            "load_contents": ["model", "lora_adapter"],
            "lora_adapter_kwargs": [("lora_config", lora_config)],
        },
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    bad_kwargs_engine.runtime = RuntimeWithAdapter()
    with pytest.raises(TypeError, match="lora_adapter_kwargs must be a mapping"):
        bad_kwargs_engine.load_checkpoint(str(tmp_path / "bad_kwargs_full_ckpt"))
    assert load_calls == []

    unsupported_engine, *_ = _initialized_engine(
        checkpoint_config={"load_contents": ["model", "lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    with pytest.raises(NotImplementedError, match="load_lora_adapter"):
        unsupported_engine.load_checkpoint(str(tmp_path / "unsupported_full_ckpt"))
    assert load_calls == []

    missing_adapter_engine, *_ = _initialized_engine(
        checkpoint_config={"load_contents": ["model", "lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    missing_adapter_engine.runtime = RuntimeWithAdapter()
    with pytest.raises(FileNotFoundError, match="LoRA adapter checkpoint directory"):
        missing_adapter_engine.load_checkpoint(str(tmp_path / "missing_adapter_full_ckpt"))
    assert load_calls == []

    file_adapter_root = tmp_path / "file_adapter_full_ckpt"
    file_adapter_root.mkdir()
    (file_adapter_root / "lora_adapter").write_text("not a directory")
    file_adapter_engine, *_ = _initialized_engine(
        checkpoint_config={"load_contents": ["model", "lora_adapter"]},
        engine_config_kwargs={"impl_cfg": {"use_thd": True, "lora": lora_config}},
    )
    file_adapter_engine.runtime = RuntimeWithAdapter()
    with pytest.raises(NotADirectoryError, match="LoRA adapter checkpoint path"):
        file_adapter_engine.load_checkpoint(str(file_adapter_root))
    assert load_calls == []
