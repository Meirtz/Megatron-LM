# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""External VERL engine backed by Megatron Lite runtime primitives."""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Mapping
from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
from megatron.lite.model import resolve_model_type_from_hf
from megatron.lite.model.protocol_utils import (
    concat_routed_experts_segments,
    routed_experts_digest,
)
from megatron.lite.primitive.ckpt import load_training_checkpoint, save_training_checkpoint
from megatron.lite.primitive.modules.lora import normalize_lora_config
from megatron.lite.primitive.modules.router_replay import get_routed_experts_dtype
from megatron.lite.primitive.protocols import default_expert_classifier, default_placement_fn
from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts import LossContext, PackedBatch
from megatron.lite.runtime.contracts.config import OptimizerConfig as MegatronLiteOptimizerConfig
from megatron.lite.runtime.contracts.config import ParallelConfig, RuntimeConfig
from tensordict import TensorDict

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.workers.config import HFModelConfig, OptimizerConfig
from verl_mlite.compat import load_verl_engine_api

from .config import MegatronLiteEngineConfig

BaseEngine, BaseEngineCtx, EngineRegistry, postprocess_batch_func, prepare_micro_batches = (
    load_verl_engine_api()
)

try:
    from verl.utils.dataset.dataset_utils import DatasetPadMode
except ImportError:

    class DatasetPadMode(Enum):
        NO_PADDING = "no_padding"


_LR_SCHEDULER_STATE = "lr_scheduler.pt"
_LORA_ADAPTER_DIR = "lora_adapter"
_LORA_ADAPTER_CONTENT_KEYS = {"adapter", "lora", "lora_adapter", "peft_adapter"}
_CONFIG_MISSING = object()
_ROUTER_REPLAY_TRACE_STRING_EXTRA_KEYS = (
    "router_replay_trace_collection_path",
    "router_replay_trace_source",
)
_ROUTER_REPLAY_TRACE_BOOL_EXTRA_KEYS = (
    "router_replay_trace_preserved_through_scheduler",
    "router_replay_postprocess_wrote_trace_to_batch",
)
_ROUTER_REPLAY_DIGEST_EXTRA_KEY = "routed_experts_digest"
_ROUTER_REPLAY_TRACE_CONFLICT_EXTRA_KEY = "router_replay_trace_metadata_conflicts"
_ROUTER_REPLAY_LAYOUT_ALIASES = {
    "true": "true_tokens",
    "true_token": "true_tokens",
    "true_tokens": "true_tokens",
    "full": "full_padded",
    "full_padded": "full_padded",
    "cp_local": "cp_local",
    "local": "cp_local",
}
_ROUTER_REPLAY_TRACE_COLLECTION_PATHS = {
    "generate_request",
    "router_generate_request",
    "user_defined_rollout_loop",
    "mlite_engine_bridge",
}
_ROUTER_REPLAY_TRACE_SOURCES = {"routed_experts", "routed_experts_segments"}


def _isolate_compile_cache_per_rank() -> None:
    """Avoid torchinductor/triton cache races between local torchrun ranks."""
    rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
    if rank is None:
        return
    for var in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        base = os.environ.get(var)
        if not base:
            continue
        base_var = f"VERL_MLITE_BASE_{var}"
        root = os.environ.setdefault(base_var, base)
        rank_dir = os.path.join(root, f"rank_{rank}")
        os.makedirs(rank_dir, exist_ok=True)
        os.environ[var] = rank_dir


def _is_no_padding_pad_mode(pad_mode: Any) -> bool:
    return (
        pad_mode == DatasetPadMode.NO_PADDING
        or getattr(pad_mode, "name", None) == "NO_PADDING"
        or getattr(pad_mode, "value", None) == "no_padding"
        or str(pad_mode) in {"no_padding", "DatasetPadMode.NO_PADDING"}
    )


def _content_set(contents: Any, *, key: str = "checkpoint contents") -> set[str]:
    def normalize_entry(item: str) -> str:
        return item.strip().strip("'\"").strip()

    if contents is None:
        return set()
    if isinstance(contents, str):
        contents = contents.strip()
        if contents.startswith("[") and contents.endswith("]"):
            contents = contents[1:-1].strip()
        if "," in contents:
            return {entry for item in contents.split(",") if (entry := normalize_entry(item))}
        entry = normalize_entry(contents)
        return {entry} if entry else set()
    if isinstance(contents, Mapping):
        raise TypeError(f"{key} must be None, a string, or a sequence of strings.")
    try:
        iterator = iter(contents)
    except TypeError as exc:
        raise TypeError(f"{key} must be None, a string, or a sequence of strings.") from exc
    result: set[str] = set()
    for item in iterator:
        if not isinstance(item, str):
            raise TypeError(f"{key} entries must be strings, got {type(item)!r}.")
        item = normalize_entry(item)
        if item:
            result.add(item)
    return result


def _checkpoint_bool(value: Any, *, key: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
        raise ValueError(f"{key} must be a boolean or boolean string, got {value!r}.")
    raise TypeError(f"{key} must be a boolean or boolean string, got {type(value)!r}.")


class _MegatronLiteLRScheduler:
    def __init__(
        self,
        optimizer,
        *,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        start_wd: float,
        end_wd: float,
        wd_incr_steps: int,
        wd_incr_style: str,
        wsd_decay_steps: int | None,
        lr_wsd_decay_style: str,
    ):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.lr_warmup_steps = max(lr_warmup_steps, 0)
        self.lr_decay_steps = max(lr_decay_steps, self.lr_warmup_steps + 1)
        self.lr_decay_style = lr_decay_style.lower()
        self.start_wd = start_wd
        self.end_wd = end_wd
        self.wd_incr_steps = max(wd_incr_steps, 1)
        self.wd_incr_style = wd_incr_style.lower()
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style.lower()
        self.num_steps = 0
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.num_steps = int(state.get("num_steps", state.get("step", 0)))
        self._apply()

    def step(self, increment: int = 1) -> None:
        self.num_steps += increment
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def _apply(self) -> None:
        lr = self._get_lr()
        wd = self._get_wd()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            if param_group.get("weight_decay", None) is not None:
                param_group["weight_decay"] = wd

    def _get_lr(self) -> float:
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            ratio = self.num_steps / self.lr_warmup_steps
            return self.init_lr + (self.max_lr - self.init_lr) * ratio

        if self.lr_decay_style == "constant":
            return self.max_lr

        if self.lr_decay_style == "inverse-square-root":
            warmup = max(self.lr_warmup_steps, 1)
            step = max(self.num_steps, 1)
            return max(self.min_lr, self.max_lr * math.sqrt(warmup) / math.sqrt(step))

        if self.lr_decay_style == "wsd":
            return self._get_wsd_lr()

        decay_span = max(self.lr_decay_steps - self.lr_warmup_steps, 1)
        ratio = min(max((self.num_steps - self.lr_warmup_steps) / decay_span, 0.0), 1.0)
        return self._decay(self.max_lr, self.min_lr, ratio, self.lr_decay_style)

    def _get_wsd_lr(self) -> float:
        decay_steps = self.wsd_decay_steps or 0
        decay_start = max(self.lr_decay_steps - decay_steps, self.lr_warmup_steps)
        if decay_steps <= 0 or self.num_steps <= decay_start:
            return self.max_lr
        ratio = min((self.num_steps - decay_start) / max(decay_steps, 1), 1.0)
        return self._decay(self.max_lr, self.min_lr, ratio, self.lr_wsd_decay_style)

    def _get_wd(self) -> float:
        if self.wd_incr_style == "constant":
            return self.end_wd
        ratio = min(max(self.num_steps / self.wd_incr_steps, 0.0), 1.0)
        return self._decay(self.start_wd, self.end_wd, ratio, self.wd_incr_style)

    @staticmethod
    def _decay(start: float, end: float, ratio: float, style: str) -> float:
        if style == "linear":
            return start + (end - start) * ratio
        if style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * ratio) + 1.0)
            return end + (start - end) * coeff
        if style == "exponential":
            if start == 0.0:
                return 0.0
            if end == 0.0:
                return start * (1.0 - ratio)
            return start * ((end / start) ** ratio)
        if style == "constant":
            return start
        raise ValueError(f"Unsupported scheduler decay style: {style!r}")


def _build_lr_scheduler(optimizer, opt: MegatronLiteOptimizerConfig):
    """Build a Megatron-style LR scheduler for Megatron Lite's optimizer."""
    total_steps = opt.total_training_steps
    if total_steps <= 0:
        return None

    warmup_steps = opt.lr_warmup_steps if opt.lr_warmup_steps is not None else -1
    if warmup_steps <= 0 and opt.lr_warmup_steps_ratio > 0:
        warmup_steps = int(opt.lr_warmup_steps_ratio * total_steps)
    warmup_steps = max(warmup_steps, 0)

    decay_steps = opt.lr_decay_steps if opt.lr_decay_steps is not None else total_steps
    min_lr = opt.min_lr if opt.min_lr is not None else 0.0
    for param_group in optimizer.param_groups:
        if param_group.get("min_lr") is None:
            param_group["min_lr"] = min_lr

    return _MegatronLiteLRScheduler(
        optimizer,
        init_lr=opt.lr_warmup_init,
        max_lr=opt.lr,
        min_lr=min_lr,
        lr_warmup_steps=warmup_steps,
        lr_decay_steps=decay_steps,
        lr_decay_style=opt.lr_decay_style,
        start_wd=opt.weight_decay,
        end_wd=opt.weight_decay,
        wd_incr_steps=total_steps,
        wd_incr_style=opt.weight_decay_incr_style,
        wsd_decay_steps=opt.lr_wsd_decay_steps,
        lr_wsd_decay_style=opt.lr_wsd_decay_style,
    )


class _MegatronLiteModeCtx(BaseEngineCtx):
    """Wrap Megatron Lite runtime contexts with VERL's offload behavior."""

    def __init__(self, engine: MegatronLiteEngine, mode: str, **kwargs):
        super().__init__(engine=engine, mode=mode, **kwargs)
        self._runtime_ctx = None

    def __enter__(self):
        super().__enter__()
        assert self.engine.runtime is not None and self.engine.handle is not None
        if self.mode == "train":
            self._runtime_ctx = self.engine.runtime.train_mode(self.engine.handle)
        else:
            self._runtime_ctx = self.engine.runtime.eval_mode(self.engine.handle)
        self._runtime_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._runtime_ctx is not None
        self._runtime_ctx.__exit__(exc_type, exc_val, exc_tb)
        super().__exit__(exc_type, exc_val, exc_tb)
        return False


@EngineRegistry.register(model_type="language_model", backend="mlite", device="cuda")
class MegatronLiteEngine(BaseEngine):
    """VERL BaseEngine implementation that delegates model lifecycle to Megatron Lite."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: MegatronLiteEngineConfig,
        optimizer_config: OptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()
        _isolate_compile_cache_per_rank()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None
        self.device_name = get_device_name()
        self.runtime = None
        self.handle = None
        self.module = None
        self._mlite_config = None
        self._rank = dist.get_rank() if dist.is_initialized() else 0

    @property
    def is_param_offload_enabled(self) -> bool:
        return self.engine_config.param_offload

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self.engine_config.optimizer_offload

    def initialize(self):
        if self.engine_config.full_determinism:
            from verl.workers.engine.utils import enable_full_determinism

            enable_full_determinism(seed=self.engine_config.seed)

        self._mlite_config = self._build_mlite_config()
        self.runtime = create_runtime(
            RuntimeConfig(
                backend="mlite",
                hf_path=self.model_config.local_path,
                backend_cfg=self._mlite_config,
            )
        )
        self.handle = self.runtime.build_model()
        self.module = self._extract_primary_module()

        if self.handle._optimizer is not None and self.handle._lr_scheduler is None:
            self.handle._lr_scheduler = _build_lr_scheduler(
                self.handle._optimizer, self._mlite_config.optimizer
            )

        self.to(
            device="cpu",
            model=self.is_param_offload_enabled,
            optimizer=self.is_optimizer_offload_enabled,
            grad=self.is_param_offload_enabled,
        )

    def train_mode(self, **kwargs):
        self._require_initialized()
        return _MegatronLiteModeCtx(self, mode="train", **kwargs)

    def eval_mode(self, **kwargs):
        self._require_initialized()
        return _MegatronLiteModeCtx(self, mode="eval", **kwargs)

    def optimizer_zero_grad(self):
        self._require_initialized()
        self.runtime.zero_grad(self.handle)

    def optimizer_step(self):
        self._require_initialized()
        _, grad_norm, _ = self.runtime.optimizer_step(self.handle)
        return grad_norm

    def lr_scheduler_step(self):
        self._require_initialized()
        if self.handle._lr_scheduler is not None:
            self.handle._lr_scheduler.step(1)
            return self.handle._optimizer.param_groups[0]["lr"]
        return 0.0

    def forward_backward_batch(
        self, data: TensorDict, loss_function, forward_only: bool = False
    ) -> dict[str, Any]:
        self._require_initialized()
        pad_mode = tu.get_non_tensor_data(
            data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING
        )
        if not _is_no_padding_pad_mode(pad_mode):
            raise NotImplementedError(
                "MegatronLiteEngine only supports pad_mode=no_padding for now."
            )

        tu.assign_non_tensor(data, sp_size=self.engine_config.cp)

        token_mask = data["loss_mask"] if "loss_mask" in data.keys() else data["response_mask"]
        batch_num_tokens = token_mask.sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=self.get_data_parallel_group(),
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        # Megatron drives every forward through the runtime's forward_backward
        # callback; the engine never calls the module directly.
        return self._forward_backward_batch_with_runtime(
            data=data,
            micro_batches=micro_batches,
            indices=indices,
            loss_function=loss_function,
            forward_only=forward_only,
        )

    def get_per_tensor_param(self, **kwargs):
        self._require_initialized()
        if self.is_param_offload_enabled:
            self.to("cuda", model=True, optimizer=False, grad=False)
        export_kwargs = {
            key: kwargs[key]
            for key in ("limit", "include_mtp_only", "include_local_prefixes")
            if key in kwargs
        }
        if self.engine_config.model_name == "qwen3_5":
            export_kwargs["target"] = "vllm"
        if self.engine_config.export_dtype:
            export_kwargs["export_dtype"] = self.engine_config.export_dtype
        return self.runtime.export_weights(self.handle, **export_kwargs), None

    def get_data_parallel_size(self):
        if self.handle is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            return world_size // (
                self.engine_config.tp * self.engine_config.cp * self.engine_config.pp
            )
        return self.handle.dp_size

    def get_data_parallel_rank(self):
        if self.handle is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            dense_dp = self.get_data_parallel_size()
            return (rank // (self.engine_config.tp * self.engine_config.cp)) % dense_dp
        return self.handle.dp_rank

    def get_data_parallel_group(self):
        if self.handle is None:
            if (
                self.engine_config.tp == 1
                and self.engine_config.cp == 1
                and self.engine_config.pp == 1
                and dist.is_initialized()
            ):
                return dist.group.WORLD
            return None
        return self.handle.dp_group

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        self._require_initialized()
        if model or not (optimizer or grad):
            super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        self.runtime.to(self.handle, device, model=model, optimizer=optimizer, grad=grad)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs,
    ) -> None:
        del hdfs_path, max_ckpt_to_keep, kwargs
        self._require_initialized()

        save_contents = self.checkpoint_config.get("save_contents", None)
        save_content_keys = _content_set(save_contents, key="checkpoint_config.save_contents")
        save_model = save_contents is None or "model" in save_content_keys
        save_optimizer = save_contents is None or "optimizer" in save_content_keys
        save_lora_adapter = _checkpoint_bool(
            self.checkpoint_config.get("save_lora_adapter", False),
            key="checkpoint_config.save_lora_adapter",
        ) or bool(save_content_keys & _LORA_ADAPTER_CONTENT_KEYS)
        if not save_model and not save_optimizer and not save_lora_adapter:
            if self._rank == 0:
                print(
                    f"Skipping Megatron Lite checkpoint save at step {global_step}: save_contents={save_contents}"
                )
            if dist.is_initialized():
                dist.barrier()
            return

        lora_adapter_path = None
        lora_adapter_kwargs = None
        save_adapter = None
        if save_lora_adapter:
            save_adapter = getattr(self.runtime, "save_lora_adapter", None)
            if not callable(save_adapter):
                raise NotImplementedError(
                    "Megatron Lite runtime does not support save_lora_adapter."
                )
            lora_adapter_path = self._lora_adapter_checkpoint_path(local_path)
            lora_adapter_kwargs = self._lora_adapter_checkpoint_kwargs(
                require_enabled_lora_config=True
            )

        checkpoint_path_preexisting = os.path.exists(local_path)
        os.makedirs(local_path, exist_ok=True)
        placement_fn, expert_classifier = self._checkpoint_hooks()
        reload_params_for_save = self.is_param_offload_enabled
        if reload_params_for_save:
            self.to(device="cuda", model=True, optimizer=False, grad=False)
            torch.cuda.synchronize()
        try:
            save_full_checkpoint = save_model or save_optimizer
            if save_full_checkpoint:
                save_training_checkpoint(
                    self.module,
                    self.handle._optimizer,
                    global_step,
                    local_path,
                    self.handle._config.parallel,
                    self.handle._parallel_state,
                    get_placements=placement_fn,
                    is_expert=expert_classifier,
                    save_model=save_model,
                    save_optimizer=save_optimizer,
                )
                if self.handle._lr_scheduler is not None and self._rank == 0:
                    torch.save(
                        self.handle._lr_scheduler.state_dict(),
                        os.path.join(local_path, _LR_SCHEDULER_STATE),
                    )
            if save_lora_adapter:
                assert save_adapter is not None
                assert lora_adapter_path is not None
                assert lora_adapter_kwargs is not None
                self._save_lora_adapter_checkpoint(
                    lora_adapter_path,
                    global_step,
                    adapter_kwargs=lora_adapter_kwargs,
                    save_adapter=save_adapter,
                )
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            if (
                save_lora_adapter
                and not save_full_checkpoint
                and not checkpoint_path_preexisting
                and (not dist.is_initialized() or self._rank == 0)
            ):
                shutil.rmtree(local_path, ignore_errors=True)
            raise
        finally:
            if reload_params_for_save:
                self.to(device="cpu", model=True, optimizer=False, grad=False)

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        del_local_after_load: bool = True,
        **kwargs,
    ) -> None:
        del hdfs_path, del_local_after_load, kwargs
        self._require_initialized()

        load_contents = self.checkpoint_config.get("load_contents", None)
        load_content_keys = _content_set(load_contents, key="checkpoint_config.load_contents")
        load_model = load_contents is None or "model" in load_content_keys
        load_optimizer = load_contents is None or "optimizer" in load_content_keys
        load_lora_adapter = _checkpoint_bool(
            self.checkpoint_config.get("load_lora_adapter", False),
            key="checkpoint_config.load_lora_adapter",
        ) or bool(load_content_keys & _LORA_ADAPTER_CONTENT_KEYS)
        if not load_model and not load_optimizer and not load_lora_adapter:
            if self._rank == 0:
                print(
                    f"Skipping Megatron Lite checkpoint load: load_contents={load_contents}"
                )
            if dist.is_initialized():
                dist.barrier()
            return
        lora_adapter_path = None
        lora_adapter_kwargs = None
        load_adapter = None
        if load_lora_adapter:
            load_adapter = getattr(self.runtime, "load_lora_adapter", None)
            if not callable(load_adapter):
                raise NotImplementedError(
                    "Megatron Lite runtime does not support load_lora_adapter."
                )
            lora_adapter_path = self._lora_adapter_checkpoint_path(local_path)
            lora_adapter_kwargs = self._lora_adapter_checkpoint_kwargs()
            lora_adapter_kwargs.pop("metadata", None)
            lora_adapter_kwargs.pop("base_model_name_or_path", None)
            lora_adapter_kwargs.pop("init_lora_weights", None)
            self._validate_lora_adapter_checkpoint_load_path(lora_adapter_path)

        placement_fn, expert_classifier = self._checkpoint_hooks()
        reload_params_for_load = self.is_param_offload_enabled
        if reload_params_for_load:
            self.to(device="cuda", model=True, optimizer=False, grad=False)
            torch.cuda.synchronize()
        try:
            if load_model or load_optimizer:
                load_training_checkpoint(
                    self.module,
                    self.handle._optimizer,
                    local_path,
                    self.handle._config.parallel,
                    self.handle._parallel_state,
                    get_placements=placement_fn,
                    is_expert=expert_classifier,
                    load_model=load_model,
                    load_optimizer=load_optimizer,
                )
                scheduler_path = os.path.join(local_path, _LR_SCHEDULER_STATE)
                if self.handle._lr_scheduler is not None and os.path.exists(scheduler_path):
                    state = torch.load(scheduler_path, map_location="cpu", weights_only=False)
                    self.handle._lr_scheduler.load_state_dict(state)
            if load_lora_adapter:
                assert load_adapter is not None
                assert lora_adapter_path is not None
                assert lora_adapter_kwargs is not None
                self._load_lora_adapter_checkpoint(
                    lora_adapter_path,
                    adapter_kwargs=lora_adapter_kwargs,
                    load_adapter=load_adapter,
                )
            if dist.is_initialized():
                dist.barrier()
        finally:
            if reload_params_for_load:
                self.to(device="cpu", model=True, optimizer=False, grad=False)

    def _lora_adapter_checkpoint_path(self, local_path: str) -> str:
        name = self.checkpoint_config.get("lora_adapter_dir_name", _LORA_ADAPTER_DIR)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("checkpoint_config.lora_adapter_dir_name must be a non-empty string.")
        name = name.strip()
        normalized = os.path.normpath(name)
        raw_parts = name.split(os.sep)
        parts = normalized.split(os.sep)
        if (
            os.path.isabs(name)
            or normalized in {"", ".", os.pardir}
            or any(part == os.pardir for part in raw_parts)
            or any(part == os.pardir for part in parts)
        ):
            raise ValueError(
                "checkpoint_config.lora_adapter_dir_name must be a relative path "
                "inside the checkpoint directory."
            )
        return os.path.join(local_path, normalized)

    @staticmethod
    def _validate_lora_adapter_checkpoint_load_path(adapter_path: str) -> None:
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"LoRA adapter checkpoint directory does not exist: {adapter_path}"
            )
        if not os.path.isdir(adapter_path):
            raise NotADirectoryError(
                f"LoRA adapter checkpoint path must be a directory: {adapter_path}"
            )

    def _lora_adapter_checkpoint_kwargs(
        self, *, require_enabled_lora_config: bool = False
    ) -> dict[str, Any]:
        raw_kwargs = self.checkpoint_config.get("lora_adapter_kwargs", None)
        if raw_kwargs is None:
            adapter_kwargs: dict[str, Any] = {}
        elif isinstance(raw_kwargs, Mapping):
            adapter_kwargs = dict(raw_kwargs)
        else:
            raise TypeError("checkpoint_config.lora_adapter_kwargs must be a mapping.")
        raw_metadata = adapter_kwargs.get("metadata", None)
        if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
            raise TypeError("checkpoint_config.lora_adapter_kwargs.metadata must be a mapping.")
        impl_cfg = self.engine_config.impl_cfg or {}
        if "lora_config" not in adapter_kwargs and impl_cfg.get("lora") is not None:
            adapter_kwargs["lora_config"] = impl_cfg["lora"]
        if require_enabled_lora_config:
            lora_config = adapter_kwargs.get("lora_config", None)
            if lora_config is None:
                raise ValueError(
                    "LoRA adapter sidecar save requires an enabled LoRA config in "
                    "engine.impl_cfg.lora or checkpoint_config.lora_adapter_kwargs.lora_config."
                )
            if not normalize_lora_config(lora_config).enabled:
                raise ValueError(
                    "LoRA adapter sidecar save requires an enabled LoRA config."
                )
        if "base_model_name_or_path" not in adapter_kwargs:
            adapter_kwargs["base_model_name_or_path"] = self.model_config.local_path
        lora_init = self.handle._extras.get("lora_init", impl_cfg.get("lora_init"))
        if lora_init is not None and "init_lora_weights" not in adapter_kwargs:
            adapter_kwargs["init_lora_weights"] = lora_init
        return adapter_kwargs

    def _save_lora_adapter_checkpoint(
        self,
        adapter_path: str,
        global_step: int,
        *,
        adapter_kwargs: dict[str, Any] | None = None,
        save_adapter=None,
    ) -> None:
        if save_adapter is None:
            save_adapter = getattr(self.runtime, "save_lora_adapter", None)
            if not callable(save_adapter):
                raise NotImplementedError(
                    "Megatron Lite runtime does not support save_lora_adapter."
                )
        adapter_kwargs = dict(
            self._lora_adapter_checkpoint_kwargs(require_enabled_lora_config=True)
            if adapter_kwargs is None
            else adapter_kwargs
        )
        raw_metadata = adapter_kwargs.pop("metadata", None)
        if raw_metadata is None:
            metadata: dict[str, Any] = {}
        else:
            metadata = dict(raw_metadata)
        metadata.setdefault("global_step", global_step)
        adapter_kwargs["metadata"] = metadata
        save_adapter(
            self.handle,
            adapter_path,
            **adapter_kwargs,
        )

    def _load_lora_adapter_checkpoint(
        self,
        adapter_path: str,
        *,
        adapter_kwargs: dict[str, Any] | None = None,
        load_adapter=None,
    ) -> None:
        if load_adapter is None:
            load_adapter = getattr(self.runtime, "load_lora_adapter", None)
            if not callable(load_adapter):
                raise NotImplementedError(
                    "Megatron Lite runtime does not support load_lora_adapter."
                )
        if adapter_kwargs is None:
            adapter_kwargs = self._lora_adapter_checkpoint_kwargs()
            adapter_kwargs.pop("metadata", None)
            adapter_kwargs.pop("base_model_name_or_path", None)
            adapter_kwargs.pop("init_lora_weights", None)
        else:
            adapter_kwargs = dict(adapter_kwargs)
        load_adapter(
            self.handle,
            adapter_path,
            **adapter_kwargs,
        )

    def is_mp_src_rank_with_outputs(self):
        if self.handle is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            dense_dp = self.get_data_parallel_size()
            tp_rank = rank % self.engine_config.tp
            cp_rank = (rank // self.engine_config.tp) % self.engine_config.cp
            pp_rank = rank // (self.engine_config.tp * self.engine_config.cp * dense_dp)
            return tp_rank == 0 and cp_rank == 0 and pp_rank == self.engine_config.pp - 1
        return self.runtime.is_mp_src_rank_with_outputs(self.handle)

    def _require_initialized(self) -> None:
        if self.runtime is None or self.handle is None:
            raise RuntimeError("MegatronLiteEngine is not initialized yet.")

    def _build_mlite_config(self) -> MegatronLiteConfig:
        return MegatronLiteConfig(
            model_name=self._resolve_model_name(),
            impl=self.engine_config.impl,
            hf_path=self.model_config.local_path,
            parallel=ParallelConfig(
                tp=self.engine_config.tp,
                etp=self.engine_config.etp or 1,
                ep=self.engine_config.ep,
                pp=self.engine_config.pp,
                vpp=self.engine_config.vpp,
                cp=self.engine_config.cp,
            ),
            optimizer=self._build_mlite_optimizer_config(),
            attention_backend_override=self.engine_config.attention_backend_override,
            router_aux_loss_coef=self.engine_config.router_aux_loss_coef,
            load_hf_weights=self.engine_config.load_hf_weights,
            impl_cfg=self._build_impl_cfg(),
        )

    def _resolve_model_name(self) -> str:
        if self.engine_config.model_name != "auto":
            return self.engine_config.model_name
        return resolve_model_type_from_hf(self.model_config.hf_config)

    def _build_impl_cfg(self) -> dict[str, Any]:
        impl_cfg = dict(self.engine_config.impl_cfg)
        router_replay_mode = self._router_replay_mode()
        if router_replay_mode == "R2":
            raise NotImplementedError(
                "MegatronLiteEngine does not support router_replay.mode=R2 yet. "
                "Use router_replay.mode=R3 with SGLang routed_experts, or disable router replay."
            )
        if router_replay_mode == "R3":
            if self._sequence_packing_enabled(impl_cfg):
                raise ValueError(
                    "MegatronLiteEngine R3 does not support sequence packing. "
                    "Disable sequence packing until routed_experts remapping is implemented."
                )
            router_replay_enabled = impl_cfg.get("router_replay", None)
            if router_replay_enabled is not None and not self._bool_config_value(
                router_replay_enabled, key="engine.impl_cfg.router_replay"
            ):
                raise ValueError(
                    "MegatronLiteEngine engine.router_replay.mode=R3 requires "
                    "engine.impl_cfg.router_replay=True."
                )
            impl_cfg["router_replay"] = True
        if impl_cfg.get("use_thd", True) is not True:
            raise ValueError(
                "MegatronLiteEngine supports only THD/no-padding SFT; set engine.impl_cfg.use_thd=True."
            )
        impl_cfg["use_thd"] = True
        cross_entropy_fusion = getattr(self.engine_config, "cross_entropy_fusion", None)
        if cross_entropy_fusion is None:
            cross_entropy_fusion = getattr(self.engine_config, "use_fused_kernels", False)
        impl_cfg.setdefault("cross_entropy_fusion", bool(cross_entropy_fusion))
        mtp_cfg = getattr(self.model_config, "mtp", None)
        if mtp_cfg is not None:
            mtp_enable = bool(getattr(mtp_cfg, "enable", False))
            mtp_enable_train = mtp_enable and bool(getattr(mtp_cfg, "enable_train", False))
            impl_cfg["mtp_enable"] = mtp_enable
            impl_cfg["mtp_enable_train"] = mtp_enable_train
            impl_cfg["mtp_detach_encoder"] = bool(getattr(mtp_cfg, "detach_encoder", False))
            impl_cfg["mtp_loss_scaling_factor"] = float(
                getattr(mtp_cfg, "mtp_loss_scaling_factor", 0.1)
            )
        if self.engine_config.full_determinism:
            impl_cfg.setdefault("deterministic", True)
        if self.engine_config.forward_only:
            impl_cfg["optimizer"] = None
        return impl_cfg

    def _router_replay_mode(self) -> str:
        router_replay = getattr(self.engine_config, "router_replay", None)
        mode = self._config_get(router_replay, "mode", "disabled")
        if mode in (None, "disable"):
            return "disabled"
        mode = str(mode)
        if mode not in {"disabled", "R2", "R3"}:
            raise ValueError(
                f"Unsupported engine.router_replay.mode={mode!r}; expected disabled, R2, or R3."
            )
        return mode

    @staticmethod
    def _config_get(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        get = getattr(config, "get", None)
        if callable(get):
            try:
                return get(key, default)
            except TypeError:
                pass
        return getattr(config, key, default)

    @classmethod
    def _config_has_key(cls, config: Any, key: str) -> bool:
        if config is None:
            return False
        if isinstance(config, Mapping):
            return key in config
        get = getattr(config, "get", None)
        if callable(get):
            try:
                return get(key, _CONFIG_MISSING) is not _CONFIG_MISSING
            except TypeError:
                pass
        return hasattr(config, key)

    @classmethod
    def _config_like(cls, value: Any) -> bool:
        if isinstance(value, Mapping):
            return True
        get = getattr(value, "get", None)
        if callable(get):
            return True
        return any(
            hasattr(value, key)
            for key in (
                "enable",
                "enabled",
                "use",
                "use_sequence_packing",
                "sequence_packing",
                "algorithm",
                "packing_algorithm",
            )
        )

    @classmethod
    def _sequence_packing_enabled(cls, impl_cfg: Mapping[str, Any]) -> bool:
        for key in (
            "use_sequence_packing",
            "sequence_packing",
            "enable_sequence_packing",
            "sequence_packing_enable",
        ):
            if key in impl_cfg and cls._sequence_packing_value_enabled(
                impl_cfg[key], key=f"engine.impl_cfg.{key}"
            ):
                return True

        for key in ("sequence_packing_algorithm", "packing_algorithm"):
            if key in impl_cfg and cls._sequence_packing_algorithm_enabled(impl_cfg[key]):
                return True

        if "sequence_packing_args" in impl_cfg and cls._sequence_packing_value_enabled(
            impl_cfg["sequence_packing_args"], key="engine.impl_cfg.sequence_packing_args"
        ):
            return True
        return False

    @classmethod
    def _sequence_packing_value_enabled(cls, value: Any, *, key: str) -> bool:
        if cls._config_like(value):
            for subkey in (
                "enable",
                "enabled",
                "use",
                "use_sequence_packing",
                "sequence_packing",
            ):
                if cls._config_has_key(value, subkey) and cls._bool_config_value(
                    cls._config_get(value, subkey),
                    key=f"{key}.{subkey}",
                    unknown_string_is_true=True,
                ):
                    return True
            for subkey in ("algorithm", "packing_algorithm"):
                if cls._config_has_key(
                    value, subkey
                ) and cls._sequence_packing_algorithm_enabled(
                    cls._config_get(value, subkey)
                ):
                    return True
            return False
        return cls._bool_config_value(value, key=key, unknown_string_is_true=True)

    @staticmethod
    def _sequence_packing_algorithm_enabled(value: Any) -> bool:
        if value is None:
            return False
        normalized = str(value).strip().lower()
        return normalized not in {"", "none", "disabled", "false", "0", "no", "off"}

    @staticmethod
    def _bool_config_value(
        value: Any,
        *,
        key: str,
        unknown_string_is_true: bool = False,
    ) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on", "enabled"}:
                return True
            if normalized in {"false", "0", "no", "off", "disabled", "none", "null", ""}:
                return False
            if unknown_string_is_true:
                return True
            raise ValueError(f"{key} must be a boolean or boolean string, got {value!r}.")
        raise TypeError(f"{key} must be a boolean or boolean string, got {type(value)!r}.")

    def _build_mlite_optimizer_config(self) -> MegatronLiteOptimizerConfig:
        optimizer_name = self._normalize_optimizer_name(self.optimizer_config)
        betas = tuple(getattr(self.optimizer_config, "betas", (0.9, 0.999)))
        override = getattr(self.optimizer_config, "override_optimizer_config", {}) or {}
        offload_fraction = override.get(
            "offload_fraction", override.get("optimizer_offload_fraction")
        )
        if offload_fraction is None and override.get("optimizer_cpu_offload"):
            offload_fraction = 1.0
        if offload_fraction is None and self.is_optimizer_offload_enabled:
            offload_fraction = 1.0

        min_lr = getattr(self.optimizer_config, "min_lr", None)
        min_lr_ratio = getattr(self.optimizer_config, "min_lr_ratio", None)
        if min_lr is None:
            min_lr = 0.0 if min_lr_ratio is None else self.optimizer_config.lr * min_lr_ratio

        lr_decay_style = getattr(self.optimizer_config, "lr_decay_style", None)
        if lr_decay_style is None:
            lr_decay_style = getattr(self.optimizer_config, "lr_scheduler_type", "constant")

        return MegatronLiteOptimizerConfig(
            optimizer=optimizer_name,
            lr=self.optimizer_config.lr,
            min_lr=min_lr,
            clip_grad=self.optimizer_config.clip_grad,
            weight_decay=self.optimizer_config.weight_decay,
            lr_warmup_steps_ratio=self.optimizer_config.lr_warmup_steps_ratio,
            total_training_steps=self.optimizer_config.total_training_steps,
            lr_warmup_steps=self.optimizer_config.lr_warmup_steps,
            lr_warmup_init=getattr(self.optimizer_config, "lr_warmup_init", 0.0),
            lr_decay_steps=getattr(self.optimizer_config, "lr_decay_steps", None),
            lr_decay_style=lr_decay_style,
            weight_decay_incr_style=getattr(
                self.optimizer_config, "weight_decay_incr_style", "constant"
            ),
            lr_wsd_decay_style=getattr(self.optimizer_config, "lr_wsd_decay_style", "exponential"),
            lr_wsd_decay_steps=getattr(self.optimizer_config, "lr_wsd_decay_steps", None),
            use_checkpoint_opt_param_scheduler=getattr(
                self.optimizer_config, "use_checkpoint_opt_param_scheduler", False
            ),
            adam_beta1=betas[0],
            adam_beta2=betas[1],
            adam_eps=override.get("adam_eps", override.get("eps")),
            offload_fraction=offload_fraction,
            use_precision_aware_optimizer=override.get("use_precision_aware_optimizer"),
            decoupled_weight_decay=override.get("decoupled_weight_decay"),
        )

    @staticmethod
    def _normalize_optimizer_name(config: OptimizerConfig) -> str:
        optimizer_name = getattr(config, "optimizer", "adam")
        lower = str(optimizer_name).lower()
        if "adam" in lower:
            return "adam"
        raise ValueError(
            f"MegatronLiteEngine only supports Adam-style optimizers today, got {optimizer_name!r}"
        )

    def _extract_primary_module(self):
        model = self.handle._model
        if isinstance(model, list | tuple):
            if not model:
                raise RuntimeError("Megatron Lite runtime returned an empty model chunk list.")
            if len(model) > 1:
                return torch.nn.ModuleList(model)
            return model[0]
        return model

    def _forward_backward_batch_with_runtime(
        self,
        *,
        data: TensorDict,
        micro_batches: list[TensorDict],
        indices,
        loss_function,
        forward_only: bool,
    ) -> dict[str, Any]:
        runtime_batches = []
        num_micro_batches = len(micro_batches)
        batch_num_tokens = tu.get_non_tensor_data(data=data, key="batch_num_tokens", default=None)
        if batch_num_tokens is None:
            raise ValueError(
                "MegatronLiteEngine PP/CP SFT requires batch_num_tokens for VERL-compatible loss scaling."
            )
        if batch_num_tokens <= 0:
            raise ValueError(f"batch_num_tokens must be positive, got {batch_num_tokens}.")
        loss_scale = self.get_data_parallel_size() * num_micro_batches / float(batch_num_tokens)
        for micro_idx, micro_batch in enumerate(micro_batches):
            tu.assign_non_tensor(micro_batch, micro_batch_idx=micro_idx)
            micro_batch = micro_batch.to(get_device_id())
            runtime_batches.append(
                (
                    self._make_runtime_batch(micro_batch),
                    self._make_runtime_loss_context(micro_batch, loss_scale=loss_scale),
                )
            )

        runtime_loss_fn = None
        if loss_function is not None or forward_only:
            runtime_loss_fn = self._make_runtime_loss_fn(loss_function, forward_only=forward_only)

        result = self.runtime.forward_backward(
            self.handle,
            iter(runtime_batches),
            loss_fn=runtime_loss_fn,
            num_microbatches=num_micro_batches,
            forward_only=forward_only,
        )
        metrics = dict(result.metrics)
        micro_outputs = metrics.pop("_micro_outputs", None)
        if micro_outputs is not None and self.is_mp_src_rank_with_outputs():
            return postprocess_batch_func(output_lst=micro_outputs, indices=indices, data=data)
        loss = float(metrics.get("loss", 0.0))
        return {
            "model_output": {},
            "loss": [loss],
            "metrics": {key: [value] for key, value in metrics.items()},
        }

    def _make_runtime_batch(self, micro_batch: TensorDict) -> PackedBatch:
        """Flatten a jagged no-padding batch to a model-agnostic ``PackedBatch``.

        No CP split, no padding, no ``PackedSeqParams`` here: each model's
        protocol owns its pack/unpack pair (zigzag vs contiguous). ``labels`` are
        the unrolled tokens; the protocol rolls them while packing.
        """
        input_ids = micro_batch["input_ids"]
        if not getattr(input_ids, "is_nested", False):
            raise NotImplementedError(
                "MegatronLiteEngine supports only nested no-padding THD batches."
            )
        seq_lens = input_ids.offsets().diff().to(dtype=torch.int64)
        loss_mask = self._loss_mask_for_packing(micro_batch, input_ids)
        routed_experts = self._routed_experts_for_packing(micro_batch)
        return PackedBatch(
            input_ids=input_ids.values().contiguous(),
            labels=input_ids.values().contiguous(),
            loss_mask=None if loss_mask is None else loss_mask.values().contiguous().float(),
            seq_lens=seq_lens,
            routed_experts=routed_experts,
            extras=self._router_replay_extras(
                micro_batch,
                routed_experts=routed_experts,
                seq_lens=seq_lens,
            ),
        )

    def _make_runtime_loss_context(
        self,
        micro_batch: TensorDict,
        *,
        loss_scale: float,
    ) -> LossContext:
        return LossContext(
            temperature=float(self._scalar_temperature(micro_batch)),
            calculate_entropy=bool(
                tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
            ),
            return_log_probs=True,
            loss_scale=loss_scale,
            source_batch=micro_batch,
        )

    @staticmethod
    def _loss_mask_for_packing(
        micro_batch: TensorDict, input_ids: torch.Tensor
    ) -> torch.Tensor | None:
        if "loss_mask" not in micro_batch.keys():
            return None

        loss_mask = micro_batch["loss_mask"]
        if getattr(loss_mask, "is_nested", False):
            return loss_mask

        rows = []
        for seq_ids, row_mask in zip(input_ids.unbind(0), loss_mask, strict=True):
            seq_len = seq_ids.numel()
            response_tokens = int(row_mask.sum().item())
            if response_tokens > seq_len:
                raise ValueError(
                    f"response loss mask has {response_tokens} tokens but packed input sequence has {seq_len} tokens"
                )
            full_mask = torch.zeros(seq_len, dtype=row_mask.dtype, device=row_mask.device)
            if response_tokens:
                full_mask[-response_tokens:] = row_mask[:response_tokens]
            rows.append(full_mask)
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    @staticmethod
    def _routed_experts_for_packing(
        micro_batch: TensorDict,
    ) -> torch.Tensor | list[torch.Tensor] | None:
        enable_routing_replay = tu.get_non_tensor_data(
            data=micro_batch, key="enable_routing_replay", default=None
        )
        if enable_routing_replay is not None:
            enable_routing_replay = MegatronLiteEngine._bool_config_value(
                enable_routing_replay, key="enable_routing_replay"
            )
        if enable_routing_replay is False:
            return None
        router_replay_action = tu.get_non_tensor_data(
            data=micro_batch, key="router_replay_action", default=None
        )
        record_routed_experts = router_replay_action == "record"
        for key in ("record_routed_experts", "router_replay_record"):
            value = tu.get_non_tensor_data(data=micro_batch, key=key, default=None)
            if value is not None:
                record_routed_experts = (
                    record_routed_experts
                    or MegatronLiteEngine._bool_config_value(value, key=key)
                )

        routed_experts_segments = tu.get_non_tensor_data(
            data=micro_batch, key="routed_experts_segments", default=None
        )
        if routed_experts_segments is not None:
            segment_seq_lens = tu.get_non_tensor_data(
                data=micro_batch, key="routed_experts_segment_seq_lens", default=None
            )
            num_routers = tu.get_non_tensor_data(
                data=micro_batch, key="routed_experts_num_routers", default=None
            )
            return concat_routed_experts_segments(
                routed_experts_segments,
                segment_seq_lens,
                num_routers=None if num_routers is None else int(num_routers),
            )

        routed_experts = tu.get_non_tensor_data(
            data=micro_batch, key="routed_experts", default=None
        )
        if routed_experts is None:
            if enable_routing_replay is True and not record_routed_experts:
                raise ValueError(
                    "MegatronLiteEngine R3/replay train path requires routed_experts "
                    "or routed_experts_segments when enable_routing_replay=True. "
                    "Use record_routed_experts=True only for record-mode passes."
                )
            return None

        if isinstance(routed_experts, (list, tuple)):
            replay_data = list(routed_experts)
            for idx, item in enumerate(replay_data):
                if not isinstance(item, torch.Tensor):
                    raise TypeError(
                        "MegatronLiteEngine per-router routed_experts entries must be tensors, "
                        f"got {type(item)!r} at index {idx}."
                    )
                if item.dim() != 2:
                    raise ValueError(
                        "MegatronLiteEngine per-router routed_experts entries must have "
                        f"shape [tokens, topk], got {tuple(item.shape)} at index {idx}."
                    )
            return [
                MegatronLiteEngine._compact_routed_experts_for_packing(
                    item, context=f"per-router routed_experts entry {idx}"
                )
                for idx, item in enumerate(replay_data)
            ]

        if getattr(routed_experts, "is_nested", False):
            return MegatronLiteEngine._compact_routed_experts_for_packing(
                routed_experts.values(), context="nested routed_experts values"
            )
        if not isinstance(routed_experts, torch.Tensor):
            raise TypeError(
                "MegatronLiteEngine routed_experts must be a tensor, nested tensor, "
                "or per-router tensor list; "
                f"got {type(routed_experts)!r}."
            )
        return MegatronLiteEngine._compact_routed_experts_for_packing(
            routed_experts, context="routed_experts"
        )

    @staticmethod
    def _compact_routed_experts_for_packing(
        routed_experts: torch.Tensor, *, context: str
    ) -> torch.Tensor:
        if (
            routed_experts.dtype == torch.bool
            or torch.is_floating_point(routed_experts)
            or torch.is_complex(routed_experts)
        ):
            raise TypeError(
                f"MegatronLiteEngine {context} must use an integer tensor dtype, "
                f"got {routed_experts.dtype}."
            )
        if not routed_experts.numel():
            return routed_experts.contiguous()
        min_expert_id = int(routed_experts.min().item())
        if min_expert_id < 0:
            raise ValueError(
                f"MegatronLiteEngine {context} must contain non-negative expert ids, "
                f"got min={min_expert_id}."
            )
        max_expert_id = int(routed_experts.max().item())
        return routed_experts.to(dtype=get_routed_experts_dtype(max_expert_id)).contiguous()

    @staticmethod
    def _normalize_router_replay_layout(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "MegatronLiteEngine router_replay_layout must be a string, "
                f"got {type(value)!r}."
            )
        normalized = _ROUTER_REPLAY_LAYOUT_ALIASES.get(value.strip().lower())
        if normalized is None:
            valid = ", ".join(sorted(set(_ROUTER_REPLAY_LAYOUT_ALIASES)))
            raise ValueError(
                f"Unsupported MegatronLiteEngine router_replay_layout={value!r}; "
                f"expected one of: {valid}."
            )
        return normalized

    @staticmethod
    def _required_router_replay_trace_metadata(
        micro_batch: TensorDict,
        *,
        expected_source: str,
    ) -> dict[str, Any]:
        if expected_source not in _ROUTER_REPLAY_TRACE_SOURCES:
            raise AssertionError(f"unexpected router replay trace source {expected_source!r}")
        conflicts = tu.get_non_tensor_data(
            data=micro_batch, key=_ROUTER_REPLAY_TRACE_CONFLICT_EXTRA_KEY, default=None
        )
        conflicts_present = False
        if conflicts is not None:
            if isinstance(conflicts, str):
                conflicts_present = bool(conflicts.strip())
            elif isinstance(conflicts, Mapping):
                conflicts_present = bool(conflicts)
            else:
                try:
                    conflicts_present = len(conflicts) > 0
                except TypeError:
                    conflicts_present = bool(conflicts)
        if conflicts_present:
            raise ValueError(
                "MegatronLiteEngine R3/replay train path requires "
                f"{_ROUTER_REPLAY_TRACE_CONFLICT_EXTRA_KEY} to be empty before "
                f"consuming replay payload, got {conflicts!r}."
            )
        collection_path = tu.get_non_tensor_data(
            data=micro_batch, key="router_replay_trace_collection_path", default=None
        )
        if not isinstance(collection_path, str) or not collection_path.strip():
            raise ValueError(
                "MegatronLiteEngine R3/replay train path requires "
                "router_replay_trace_collection_path when replay payload is provided."
            )
        collection_path = collection_path.strip()
        if collection_path not in _ROUTER_REPLAY_TRACE_COLLECTION_PATHS:
            valid = ", ".join(sorted(_ROUTER_REPLAY_TRACE_COLLECTION_PATHS))
            raise ValueError(
                "MegatronLiteEngine router_replay_trace_collection_path must be one of "
                f"{valid}; got {collection_path!r}."
            )

        trace_source = tu.get_non_tensor_data(
            data=micro_batch, key="router_replay_trace_source", default=None
        )
        if trace_source != expected_source:
            raise ValueError(
                "MegatronLiteEngine R3/replay train path requires "
                f"router_replay_trace_source={expected_source!r} for the provided replay "
                f"payload, got {trace_source!r}."
            )

        metadata = {
            "router_replay_trace_collection_path": collection_path,
            "router_replay_trace_source": trace_source,
        }
        for key in _ROUTER_REPLAY_TRACE_BOOL_EXTRA_KEYS:
            value = tu.get_non_tensor_data(data=micro_batch, key=key, default=None)
            if value is None:
                raise ValueError(
                    "MegatronLiteEngine R3/replay train path requires "
                    f"{key}=True when replay payload is provided."
                )
            value = MegatronLiteEngine._bool_config_value(value, key=key)
            if value is not True:
                raise ValueError(
                    "MegatronLiteEngine R3/replay train path requires "
                    f"{key}=True when replay payload is provided."
                )
            metadata[key] = True
        return metadata

    @staticmethod
    def _normalize_router_replay_digest(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "MegatronLiteEngine routed_experts_digest must be a string, "
                f"got {type(value)!r}."
            )
        text = value.strip().lower()
        if text.startswith("sha256:"):
            text = text.removeprefix("sha256:")
        if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
            raise ValueError(
                "MegatronLiteEngine routed_experts_digest must be a sha256 digest "
                "formatted as sha256:<64 hex chars> or <64 hex chars>."
            )
        return f"sha256:{text}"

    @staticmethod
    def _router_replay_digest_for_extras(
        micro_batch: TensorDict,
        *,
        routed_experts: torch.Tensor | list[torch.Tensor],
        seq_lens: torch.Tensor | None,
    ) -> str | None:
        num_routers_value = tu.get_non_tensor_data(
            data=micro_batch, key="routed_experts_num_routers", default=None
        )
        num_routers = None if num_routers_value is None else int(num_routers_value)
        if isinstance(routed_experts, torch.Tensor):
            if routed_experts.dim() == 4:
                return routed_experts_digest(
                    routed_experts,
                    num_routers=num_routers,
                    seq_lens=seq_lens,
                )
            if routed_experts.dim() == 3 and num_routers is None:
                layout_value = tu.get_non_tensor_data(
                    data=micro_batch, key="router_replay_layout", default=None
                )
                layout = (
                    None
                    if layout_value is None
                    else MegatronLiteEngine._normalize_router_replay_layout(layout_value)
                )
                true_tokens = None if seq_lens is None else int(seq_lens.sum().item())
                if layout == "true_tokens" and true_tokens == int(routed_experts.size(0)):
                    num_routers = int(routed_experts.size(1))
                else:
                    return None
        return routed_experts_digest(routed_experts, num_routers=num_routers)

    @staticmethod
    def _router_replay_extras(
        micro_batch: TensorDict,
        *,
        routed_experts: torch.Tensor | list[torch.Tensor] | None = None,
        seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        enable_routing_replay = tu.get_non_tensor_data(
            data=micro_batch, key="enable_routing_replay", default=None
        )
        if enable_routing_replay is not None:
            enable_routing_replay = MegatronLiteEngine._bool_config_value(
                enable_routing_replay, key="enable_routing_replay"
            )
        if enable_routing_replay is False:
            return {}
        extras: dict[str, Any] = {}
        direct_replay_payload_present = (
            tu.get_non_tensor_data(data=micro_batch, key="routed_experts", default=None)
            is not None
        )
        segment_replay_payload_present = (
            tu.get_non_tensor_data(data=micro_batch, key="routed_experts_segments", default=None)
            is not None
        )
        replay_payload_present = routed_experts is not None
        if not replay_payload_present:
            replay_payload_present = (
                direct_replay_payload_present or segment_replay_payload_present
            )
        record_routed_experts = False
        router_replay_action = tu.get_non_tensor_data(
            data=micro_batch, key="router_replay_action", default=None
        )
        if router_replay_action == "record":
            record_routed_experts = True
        for key in ("record_routed_experts", "router_replay_record"):
            value = tu.get_non_tensor_data(data=micro_batch, key=key, default=None)
            if value is not None:
                record_routed_experts = (
                    record_routed_experts
                    or MegatronLiteEngine._bool_config_value(value, key=key)
                )
        layout_value = tu.get_non_tensor_data(
            data=micro_batch, key="router_replay_layout", default=None
        )
        if layout_value is None:
            if segment_replay_payload_present:
                extras["router_replay_layout"] = "true_tokens"
            elif enable_routing_replay is True and replay_payload_present and not record_routed_experts:
                raise ValueError(
                    "MegatronLiteEngine R3/replay train path requires router_replay_layout "
                    "when routed_experts or routed_experts_segments are provided. "
                    "Use one of true_tokens, full_padded, or cp_local so the batch bridge "
                    "records the replay token grain explicitly."
                )
        else:
            extras["router_replay_layout"] = MegatronLiteEngine._normalize_router_replay_layout(
                layout_value
            )
        if segment_replay_payload_present:
            extras["router_replay_layout"] = "true_tokens"
        if enable_routing_replay is True and replay_payload_present and not record_routed_experts:
            expected_source = (
                "routed_experts_segments"
                if segment_replay_payload_present
                else "routed_experts"
            )
            extras.update(
                MegatronLiteEngine._required_router_replay_trace_metadata(
                    micro_batch,
                    expected_source=expected_source,
                )
            )
            provided_digest = tu.get_non_tensor_data(
                data=micro_batch, key=_ROUTER_REPLAY_DIGEST_EXTRA_KEY, default=None
            )
            if provided_digest is not None:
                extras[_ROUTER_REPLAY_DIGEST_EXTRA_KEY] = (
                    MegatronLiteEngine._normalize_router_replay_digest(provided_digest)
                )
            if routed_experts is not None:
                computed_digest = MegatronLiteEngine._router_replay_digest_for_extras(
                    micro_batch,
                    routed_experts=routed_experts,
                    seq_lens=seq_lens,
                )
                if computed_digest is not None:
                    provided_normalized = extras.get(_ROUTER_REPLAY_DIGEST_EXTRA_KEY)
                    if provided_normalized is not None and provided_normalized != computed_digest:
                        raise ValueError(
                            "MegatronLiteEngine routed_experts_digest does not match "
                            "the replay payload content digest."
                        )
                    extras[_ROUTER_REPLAY_DIGEST_EXTRA_KEY] = computed_digest
        for key in (
            "router_replay_action",
            *_ROUTER_REPLAY_TRACE_STRING_EXTRA_KEYS,
        ):
            value = tu.get_non_tensor_data(data=micro_batch, key=key, default=None)
            if value is not None and key not in extras:
                extras[key] = value
        for key in ("record_routed_experts", "router_replay_record"):
            value = tu.get_non_tensor_data(data=micro_batch, key=key, default=None)
            if value is not None:
                extras[key] = MegatronLiteEngine._bool_config_value(value, key=key)
        for key in _ROUTER_REPLAY_TRACE_BOOL_EXTRA_KEYS:
            value = tu.get_non_tensor_data(data=micro_batch, key=key, default=None)
            if value is not None and key not in extras:
                extras[key] = MegatronLiteEngine._bool_config_value(value, key=key)
        return extras

    def _build_verl_model_output(
        self,
        *,
        raw_output: dict[str, torch.Tensor],
        runtime_batch: PackedBatch,
    ) -> dict[str, torch.Tensor]:
        log_probs = raw_output.get("log_probs")
        if log_probs is None:
            raise ValueError("Megatron Lite THD model output must contain token log_probs.")
        proto = self.handle._extras.get("protocol")
        unpack = getattr(proto, "unpack_forward_output", None)
        if unpack is None:
            raise ValueError(
                "Model protocol must expose unpack_forward_output to reverse THD outputs."
            )
        output = {"log_probs": unpack(self.module, runtime_batch, log_probs)}
        entropy = raw_output.get("entropy")
        if entropy is not None:
            output["entropy"] = unpack(self.module, runtime_batch, entropy)
        return output

    def _make_runtime_loss_fn(self, loss_function, *, forward_only: bool):
        def _loss_fn(
            raw_output: dict[str, torch.Tensor],
            runtime_batch: PackedBatch,
            loss_context: LossContext,
        ):
            micro_batch = loss_context.source_batch
            model_output = self._build_verl_model_output(
                raw_output=raw_output, runtime_batch=runtime_batch
            )
            raw_output["_verl_model_output"] = model_output
            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output,
                    data=micro_batch,
                    dp_group=self.get_data_parallel_group(),
                )
            else:
                loss = torch.zeros((), device=get_device_id(), dtype=torch.float32)
                metrics = {}

            if raw_output.get("mtp_loss") is not None:
                metrics = dict(metrics)
                mtp_loss = self._reduce_mtp_metric(raw_output["mtp_loss"])
                metrics["mtp_losses/mtp_1_loss"] = (
                    float(mtp_loss.item()) if mtp_loss.numel() == 1 else mtp_loss.cpu().tolist()
                )

            raw_output["_verl_metrics"] = metrics
            return loss, metrics

        return _loss_fn

    def _mtp_enable_train(self) -> bool:
        mtp_cfg = getattr(self.model_config, "mtp", None)
        return bool(
            mtp_cfg is not None
            and getattr(mtp_cfg, "enable", False)
            and getattr(mtp_cfg, "enable_train", False)
        )

    def _reduce_mtp_metric(self, mtp_loss: torch.Tensor) -> torch.Tensor:
        mtp_loss = mtp_loss.detach().float().clone()
        dp_group = self.get_data_parallel_group()
        if dist.is_initialized() and dp_group is not None:
            dist.all_reduce(mtp_loss, op=dist.ReduceOp.AVG, group=dp_group)
        return mtp_loss

    @staticmethod
    def _scalar_temperature(micro_batch: TensorDict) -> float:
        if "temperature" not in micro_batch.keys():
            return 1.0
        temperature = micro_batch["temperature"]
        if not isinstance(temperature, torch.Tensor):
            return float(temperature)
        values = (
            temperature.values()
            if getattr(temperature, "is_nested", False)
            else temperature.reshape(-1)
        )
        if values.numel() == 0:
            return 1.0
        first = values[0].detach()
        if not torch.all(values.detach() == first).item():
            raise NotImplementedError(
                "MegatronLiteEngine currently supports scalar temperature only."
            )
        return float(first.float().item())

    def _checkpoint_hooks(self):
        proto = self.handle._extras.get("protocol")
        placement_fn = getattr(proto, "PLACEMENT_FN", default_placement_fn)
        expert_classifier = getattr(proto, "EXPERT_CLASSIFIER", default_expert_classifier)
        return placement_fn, expert_classifier
