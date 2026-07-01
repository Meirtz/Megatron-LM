# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA helpers for Megatron Lite native model implementations.

This module is intentionally narrow: it supports the Qwen3-MoE lite path's
Megatron-style sharded linear surfaces, not arbitrary PEFT injection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_TARGET_MODULES = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
_TARGET_ALIASES = {
    "all_linear": "all-linear",
    "qkv": "linear_qkv",
    "proj": "linear_proj",
    "o_proj": "linear_proj",
    "fc1": "linear_fc1",
    "fc2": "linear_fc2",
    "gate_up": "linear_fc1",
    "gate_up_proj": "linear_fc1",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down": "linear_fc2",
    "down_proj": "linear_fc2",
    "q_a": "q_a_proj",
    "q_b": "q_b_proj",
    "kv_a": "kv_a_proj_with_mqa",
    "kv_b": "kv_b_proj",
}
_TARGET_EXPANSIONS = {
    "all-linear": _DEFAULT_TARGET_MODULES,
}


def _normalize_target_modules(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        raise TypeError("LoRA config target_modules must be a string or sequence of strings.")
    try:
        targets = tuple(value)
    except TypeError as exc:
        raise TypeError(
            "LoRA config target_modules must be a string or sequence of strings."
        ) from exc
    invalid = [target for target in targets if not isinstance(target, str)]
    if invalid:
        raise TypeError(
            "LoRA config target_modules entries must be strings, got "
            f"{[type(target).__name__ for target in invalid]}."
        )
    if any(not target.strip() for target in targets):
        raise ValueError("LoRA config target_modules entries must be non-empty strings.")
    return targets


def _type_name(value: Any) -> str:
    return type(value).__name__


def validate_lora_rank_value(value: Any, *, key: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer, got {_type_name(value)}.")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{key} must be {qualifier}, got {value}.")
    return int(value)


def validate_lora_number_value(value: Any, *, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a finite number, got {_type_name(value)}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite, got {value}.")
    return value


def validate_lora_dropout_value(value: Any, *, key: str) -> float:
    value = validate_lora_number_value(value, key=key)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{key} must be between 0 and 1 inclusive, got {value}.")
    return value


def validate_lora_bool_value(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{key} must be a boolean, got {_type_name(value)}.")


def _validate_grouped_lora_splits(
    module_name: str,
    x: torch.Tensor,
    splits: list[int],
    num_local_experts: int,
) -> list[int]:
    if len(splits) != num_local_experts:
        raise ValueError(f"{module_name} expected {num_local_experts} splits, got {len(splits)}.")
    validated = [
        validate_lora_rank_value(size, key=f"{module_name} splits[{idx}]", allow_zero=True)
        for idx, size in enumerate(splits)
    ]
    total = sum(validated)
    if total != x.shape[0]:
        raise ValueError(
            f"{module_name} split sizes sum to {total}, but input has {x.shape[0]} tokens."
        )
    return validated


def _pop_optional_string(values: dict[str, Any], key: str) -> str | None:
    value = values.pop(key, None)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"LoRA config {key} must be a string, got {_type_name(value)}.")
    return value


def _pop_peft_auxiliary_fields(values: dict[str, Any]) -> None:
    base_model_name = values.pop("base_model_name_or_path", None)
    if base_model_name is not None and not isinstance(base_model_name, str):
        raise TypeError(
            "LoRA config base_model_name_or_path must be a string, "
            f"got {_type_name(base_model_name)}."
        )

    peft_type = _pop_optional_string(values, "peft_type")
    if peft_type is not None and peft_type.upper() != "LORA":
        raise ValueError(f"LoRA config peft_type={peft_type!r} is not supported.")

    task_type = _pop_optional_string(values, "task_type")
    if task_type is not None and task_type.upper() != "CAUSAL_LM":
        raise ValueError(f"LoRA config task_type={task_type!r} is not supported.")

    inference_mode = values.pop("inference_mode", None)
    if inference_mode is not None:
        validate_lora_bool_value(inference_mode, key="LoRA config inference_mode")

    bias = _pop_optional_string(values, "bias")
    if bias is not None and bias.lower() != "none":
        raise ValueError(f"LoRA config bias={bias!r} is not supported.")

    fan_in_fan_out = values.pop("fan_in_fan_out", None)
    if fan_in_fan_out is not None and validate_lora_bool_value(
        fan_in_fan_out, key="LoRA config fan_in_fan_out"
    ):
        raise ValueError("LoRA config fan_in_fan_out=True is not supported.")

    modules_to_save = values.pop("modules_to_save", None)
    if modules_to_save not in (None, [], ()):
        raise ValueError("LoRA config modules_to_save is not supported for adapter-only LoRA.")

    init_lora_weights = values.pop("init_lora_weights", None)
    if init_lora_weights is not None and not isinstance(init_lora_weights, (bool, str)):
        raise TypeError(
            "LoRA config init_lora_weights must be a boolean, string, or None, "
            f"got {_type_name(init_lora_weights)}."
        )


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 0
    alpha: int | float | None = None
    dropout: float = 0.0
    target_modules: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_TARGET_MODULES)
    use_rslora: bool = False

    def __post_init__(self) -> None:
        rank = validate_lora_rank_value(self.rank, key="LoRA config rank", allow_zero=True)
        alpha = (
            None
            if self.alpha is None
            else validate_lora_number_value(self.alpha, key="LoRA config alpha")
        )
        dropout = validate_lora_dropout_value(self.dropout, key="LoRA config dropout")
        use_rslora = validate_lora_bool_value(
            self.use_rslora, key="LoRA config use_rslora"
        )
        target_modules = _normalize_target_modules(self.target_modules)
        if rank > 0 and not target_modules:
            raise ValueError("LoRA config target_modules must be non-empty when LoRA is enabled.")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "use_rslora", use_rslora)
        object.__setattr__(self, "target_modules", target_modules)

    @property
    def enabled(self) -> bool:
        return self.rank > 0

    @property
    def scale(self) -> float:
        return lora_scale(self.rank, alpha=self.alpha, use_rslora=self.use_rslora)

    def targets(self) -> set[str]:
        out = set()
        for target in self.target_modules:
            canonical = _TARGET_ALIASES.get(target, target)
            out.update(_TARGET_EXPANSIONS.get(canonical, (canonical,)))
        return out

    def targets_module(self, name: str) -> bool:
        canonical = _TARGET_ALIASES.get(name, name)
        return canonical in self.targets()


def effective_lora_alpha(config: LoraConfig) -> float:
    return float(config.rank if config.alpha is None else config.alpha)


def lora_scale(rank: int, *, alpha: int | float | None = None, use_rslora: bool = False) -> float:
    rank = validate_lora_rank_value(rank, key="LoRA rank", allow_zero=True)
    use_rslora = validate_lora_bool_value(use_rslora, key="LoRA use_rslora")
    alpha_value = (
        float(rank) if alpha is None else validate_lora_number_value(alpha, key="LoRA alpha")
    )
    if rank == 0:
        return 0.0
    denominator = math.sqrt(float(rank)) if use_rslora else float(rank)
    return alpha_value / denominator


def normalize_lora_config(config: LoraConfig | Mapping[str, Any] | None) -> LoraConfig:
    if config is None:
        return LoraConfig()
    if isinstance(config, LoraConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError(
            f"LoRA config must be LoraConfig, mapping, or None, got {type(config)!r}."
        )
    values = dict(config)
    enabled = values.pop("enabled", None)
    if enabled is not None:
        enabled = validate_lora_bool_value(enabled, key="LoRA config enabled")
    if "r" in values and "rank" not in values:
        values["rank"] = values.pop("r")
    else:
        values.pop("r", None)
    if enabled is False:
        if "rank" in values:
            validate_lora_rank_value(values["rank"], key="LoRA config rank", allow_zero=True)
        values["rank"] = 0
    elif enabled is True:
        if "rank" not in values:
            raise ValueError("LoRA config enabled=True requires a positive rank.")
        validate_lora_rank_value(values["rank"], key="LoRA config rank")
    if "lora_alpha" in values and "alpha" not in values:
        values["alpha"] = values.pop("lora_alpha")
    else:
        values.pop("lora_alpha", None)
    if "lora_dropout" in values and "dropout" not in values:
        values["dropout"] = values.pop("lora_dropout")
    else:
        values.pop("lora_dropout", None)
    if "targets" in values and "target_modules" not in values:
        values["target_modules"] = values.pop("targets")
    else:
        values.pop("targets", None)
    if "target_modules" in values:
        values["target_modules"] = _normalize_target_modules(values["target_modules"])
    _pop_peft_auxiliary_fields(values)
    return LoraConfig(**values)


def freeze_non_lora_params(model: nn.Module) -> dict[str, int]:
    """Freeze base parameters and leave adapter parameters trainable."""

    lora_tensors = 0
    lora_numel = 0
    frozen_tensors = 0
    frozen_numel = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "adapter" in name.lower():
            param.requires_grad_(True)
            lora_tensors += 1
            lora_numel += param.numel()
        else:
            param.requires_grad_(False)
            frozen_tensors += 1
            frozen_numel += param.numel()
    return {
        "lora_tensors": lora_tensors,
        "lora_numel": lora_numel,
        "frozen_tensors": frozen_tensors,
        "frozen_numel": frozen_numel,
    }


def trainable_param_stats(model: nn.Module) -> dict[str, int]:
    tensors = 0
    numel = 0
    for param in model.parameters():
        if param.requires_grad:
            tensors += 1
            numel += param.numel()
    return {"trainable_tensors": tensors, "trainable_numel": numel}


def _gather_sequence_parallel(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _AllGatherSequence.apply(x, group)


def _reduce_scatter_sequence_parallel(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _ReduceScatterSequence.apply(x, group)


def _scatter_sequence_parallel(x: torch.Tensor, group, group_rank: int) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _ScatterSequence.apply(x, group, group_rank)


def _all_reduce_sum(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _AllReduceSum.apply(x, group)


class _AllGatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.local_seq = x.shape[0]
        out = torch.empty((x.shape[0] * world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty((ctx.local_seq, *grad.shape[1:]), dtype=grad.dtype, device=grad.device)
        dist.reduce_scatter_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None


class _ReduceScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        if x.shape[0] % world_size != 0:
            raise ValueError(
                f"Cannot reduce-scatter sequence dim {x.shape[0]} over TP={world_size}."
            )
        ctx.group = group
        ctx.world_size = world_size
        out = torch.empty((x.shape[0] // world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty(
            (grad.shape[0] * ctx.world_size, *grad.shape[1:]), dtype=grad.dtype, device=grad.device
        )
        dist.all_gather_into_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None


class _ScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group, group_rank: int) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        if x.shape[0] % world_size != 0:
            raise ValueError(f"Cannot scatter sequence dim {x.shape[0]} over TP={world_size}.")
        ctx.group = group
        ctx.world_size = world_size
        local_seq = x.shape[0] // world_size
        start = int(group_rank) * local_seq
        return x[start : start + local_seq].contiguous()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty(
            (grad.shape[0] * ctx.world_size, *grad.shape[1:]), dtype=grad.dtype, device=grad.device
        )
        dist.all_gather_into_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None, None


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        out = x.contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = grad.contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None


def _all_gather_last_dim(x: torch.Tensor, group, *, reduce_backward: bool = False) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _AllGatherLastDim.apply(x, group, reduce_backward)


class _AllGatherLastDim(torch.autograd.Function):
    """All-gather last dim with Megatron tensor-parallel split backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, reduce_backward: bool) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.local_width = x.shape[-1]
        ctx.group_rank = dist.get_rank(group)
        ctx.reduce_backward = bool(reduce_backward)
        flat = x.movedim(-1, 0).contiguous().view(ctx.local_width, -1)
        gathered = torch.empty(
            (ctx.local_width * world_size, flat.shape[1]), dtype=x.dtype, device=x.device
        )
        dist.all_gather_into_tensor(gathered, flat, group=group)
        return (
            gathered.view(ctx.local_width * world_size, *x.shape[:-1]).movedim(0, -1).contiguous()
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        flat = grad.movedim(-1, 0).contiguous().view(grad.shape[-1], -1)
        start = ctx.group_rank * ctx.local_width
        out = flat.narrow(0, start, ctx.local_width).contiguous()
        if ctx.reduce_backward:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return out.view(ctx.local_width, *grad.shape[:-1]).movedim(0, -1).contiguous(), None, None


class _SequenceParallelRankPartitionedLoRA(torch.autograd.Function):
    """QKV LoRA path that recomputes gathered activations in backward.

    The ordinary composition of all-gather + matmul saves the full
    sequence-parallel gathered input for every layer. For QKV LoRA that input
    is much larger than the low-rank hidden activation. This function saves
    only the local input plus LoRA weights, then repeats the small gather/matmul
    sequence during backward.
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float, group
    ):
        world_size = dist.get_world_size(group) if group is not None else 1
        if world_size > 1:
            gathered = _all_gather_sequence_forward(x, group, world_size)
        else:
            gathered = x
        hidden_local = gathered.matmul(lora_a.t())
        hidden = _all_gather_last_dim_forward(hidden_local, group, world_size)
        out = hidden.matmul(lora_b.t()) * scale
        ctx.save_for_backward(x, lora_a, lora_b)
        ctx.group = group
        ctx.world_size = world_size
        ctx.local_seq = x.shape[0]
        ctx.local_rank_width = hidden_local.shape[-1]
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, lora_a, lora_b = ctx.saved_tensors
        world_size = ctx.world_size
        group = ctx.group
        if world_size > 1:
            gathered = _all_gather_sequence_forward(x, group, world_size)
        else:
            gathered = x
        hidden_local = gathered.matmul(lora_a.t())
        hidden = _all_gather_last_dim_forward(hidden_local, group, world_size)

        grad_out_scaled = grad_out * ctx.scale
        grad_b = (
            grad_out_scaled.reshape(-1, grad_out_scaled.shape[-1])
            .t()
            .matmul(hidden.reshape(-1, hidden.shape[-1]))
        )
        grad_hidden = grad_out_scaled.matmul(lora_b)
        if world_size > 1:
            grad_hidden_local = _split_last_dim(
                grad_hidden, dist.get_rank(group), ctx.local_rank_width
            )
            dist.all_reduce(grad_hidden_local, op=dist.ReduceOp.SUM, group=group)
        else:
            grad_hidden_local = grad_hidden
        grad_a = (
            grad_hidden_local.reshape(-1, grad_hidden_local.shape[-1])
            .t()
            .matmul(gathered.reshape(-1, gathered.shape[-1]))
        )
        grad_gathered = grad_hidden_local.matmul(lora_a)
        if world_size > 1:
            grad_x = _reduce_scatter_sequence_forward(grad_gathered, group, ctx.local_seq)
        else:
            grad_x = grad_gathered
        return grad_x, grad_a, grad_b, None, None


def _all_gather_sequence_forward(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    out = torch.empty((x.shape[0] * world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter_sequence_forward(x: torch.Tensor, group, local_seq: int) -> torch.Tensor:
    out = torch.empty((local_seq, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


def _all_gather_last_dim_forward(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return x
    local_width = x.shape[-1]
    flat = x.movedim(-1, 0).contiguous().view(local_width, -1)
    gathered = torch.empty(
        (local_width * world_size, flat.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(gathered, flat, group=group)
    return gathered.view(local_width * world_size, *x.shape[:-1]).movedim(0, -1).contiguous()


def _split_last_dim(x: torch.Tensor, group_rank: int, local_width: int) -> torch.Tensor:
    start = int(group_rank) * local_width
    return x.narrow(-1, start, local_width).contiguous()


class LinearLoRA(nn.Module):
    """Low-rank delta for a sharded linear layer.

    `a` is replicated unless the caller feeds a row-parallel local input. `b`
    has the local output shard for column-parallel surfaces, and the replicated
    full output for row-parallel surfaces.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | float | None = None,
        dropout: float = 0.0,
        use_rslora: bool = False,
        sequence_parallel_input: bool = False,
        row_parallel_output: bool = False,
        sequence_parallel_scatter_output: bool = False,
        tp_group=None,
        tp_rank: int = 0,
        rank_partition_size: int | None = None,
        rank_partitioned_a: bool = False,
        input_parallel_reduce: bool = False,
        output_partition_size: int | None = None,
        output_partitioned_b: bool = False,
        a_tensor_model_parallel: bool = False,
        b_tensor_model_parallel: bool = False,
    ):
        super().__init__()
        in_features = validate_lora_rank_value(in_features, key="LinearLoRA in_features")
        out_features = validate_lora_rank_value(out_features, key="LinearLoRA out_features")
        self.rank = validate_lora_rank_value(rank, key="LinearLoRA rank")
        self.alpha = (
            float(self.rank)
            if alpha is None
            else validate_lora_number_value(alpha, key="LinearLoRA alpha")
        )
        self.use_rslora = validate_lora_bool_value(use_rslora, key="LinearLoRA use_rslora")
        self.rank_partitioned_a = validate_lora_bool_value(
            rank_partitioned_a, key="LinearLoRA rank_partitioned_a"
        )
        if self.rank_partitioned_a:
            partition_size = (
                validate_lora_rank_value(
                    rank_partition_size, key="LoRA rank partition size"
                )
                if rank_partition_size is not None
                else (dist.get_world_size(tp_group) if tp_group is not None else 1)
            )
            if partition_size <= 0:
                raise ValueError("LoRA rank partition size must be positive.")
            if self.rank % partition_size != 0:
                raise ValueError(
                    f"LoRA rank {self.rank} must be divisible by rank partition size {partition_size}."
                )
            self.rank_partition_size = partition_size
            self.local_rank = self.rank // partition_size
        else:
            self.rank_partition_size = 1
            self.local_rank = self.rank
        self.scale = lora_scale(self.rank, alpha=self.alpha, use_rslora=self.use_rslora)
        self.dropout_p = validate_lora_dropout_value(dropout, key="LinearLoRA dropout")
        self.sequence_parallel_input = validate_lora_bool_value(
            sequence_parallel_input, key="LinearLoRA sequence_parallel_input"
        )
        self.row_parallel_output = validate_lora_bool_value(
            row_parallel_output, key="LinearLoRA row_parallel_output"
        )
        self.sequence_parallel_scatter_output = validate_lora_bool_value(
            sequence_parallel_scatter_output, key="LinearLoRA sequence_parallel_scatter_output"
        )
        if self.row_parallel_output and self.sequence_parallel_scatter_output:
            raise ValueError(
                "Use either row_parallel_output or sequence_parallel_scatter_output, not both."
            )
        self.tp_group = tp_group
        self.tp_rank = validate_lora_rank_value(tp_rank, key="LinearLoRA tp_rank", allow_zero=True)
        self.input_parallel_reduce = validate_lora_bool_value(
            input_parallel_reduce, key="LinearLoRA input_parallel_reduce"
        )
        self.output_partitioned_b = validate_lora_bool_value(
            output_partitioned_b, key="LinearLoRA output_partitioned_b"
        )
        if self.output_partitioned_b:
            partition_size = (
                validate_lora_rank_value(
                    output_partition_size, key="LoRA output partition size"
                )
                if output_partition_size is not None
                else (dist.get_world_size(tp_group) if tp_group is not None else 1)
            )
            if partition_size <= 0:
                raise ValueError("LoRA output partition size must be positive.")
            if out_features % partition_size != 0:
                raise ValueError(
                    f"LoRA output features {out_features} must be divisible by {partition_size}."
                )
            self.output_partition_size = partition_size
            self.local_out_features = out_features // partition_size
        else:
            self.output_partition_size = 1
            self.local_out_features = out_features
        self.lora_a = nn.Parameter(torch.empty(self.local_rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(self.local_out_features, self.rank))
        self.lora_a.tensor_model_parallel = validate_lora_bool_value(
            a_tensor_model_parallel, key="LinearLoRA a_tensor_model_parallel"
        )
        self.lora_b.tensor_model_parallel = validate_lora_bool_value(
            b_tensor_model_parallel, key="LinearLoRA b_tensor_model_parallel"
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel_input and self.rank_partitioned_a and not self.training:
            # Keep eval/inference on the simple path; the memory optimization
            # matters only when autograd needs to retain forward activations.
            pass
        elif (
            self.sequence_parallel_input
            and self.rank_partitioned_a
            and not self.input_parallel_reduce
            and not self.output_partitioned_b
            and not self.row_parallel_output
            and not self.sequence_parallel_scatter_output
            and self.dropout_p == 0.0
        ):
            return _SequenceParallelRankPartitionedLoRA.apply(
                x, self.lora_a, self.lora_b, self.scale, self.tp_group
            )
        if self.sequence_parallel_input:
            x = _gather_sequence_parallel(x, self.tp_group)
        dropped = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p else x
        hidden = dropped.matmul(self.lora_a.t())
        if self.rank_partitioned_a:
            hidden = _all_gather_last_dim(hidden, self.tp_group, reduce_backward=True)
        if self.input_parallel_reduce:
            hidden = _all_reduce_sum(hidden, self.tp_group)
        out = hidden.matmul(self.lora_b.t()) * self.scale
        if self.output_partitioned_b:
            out = _all_gather_last_dim(out, self.tp_group)
        if self.row_parallel_output:
            out = _reduce_scatter_sequence_parallel(out, self.tp_group)
        if self.sequence_parallel_scatter_output:
            out = _scatter_sequence_parallel(out, self.tp_group, self.tp_rank)
        return out

    def materialized_delta_weight(self) -> torch.Tensor:
        """Return the dense LoRA delta weight for weight-space consumers.

        This is intended for modules that use a linear weight analytically
        instead of calling the linear layer directly. It supports the unsharded,
        dropout-free case used by GLM5 DSA's `kv_b_proj` decomposition.
        """

        if self.dropout_p:
            raise NotImplementedError(
                "materialized LoRA delta weights do not support lora_dropout > 0."
            )
        if self.rank_partitioned_a or self.output_partitioned_b:
            raise NotImplementedError("materialized LoRA delta weights require unsharded LoRA tensors.")
        return self.lora_b.matmul(self.lora_a) * self.scale


class GroupedLinearLoRA(nn.Module):
    """Per-local-expert LoRA delta for `te.GroupedLinear` expert surfaces."""

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | float | None = None,
        dropout: float = 0.0,
        use_rslora: bool = False,
    ):
        super().__init__()
        self.num_local_experts = validate_lora_rank_value(
            num_local_experts, key="GroupedLinearLoRA num_local_experts"
        )
        in_features = validate_lora_rank_value(in_features, key="GroupedLinearLoRA in_features")
        out_features = validate_lora_rank_value(out_features, key="GroupedLinearLoRA out_features")
        self.rank = validate_lora_rank_value(rank, key="GroupedLinearLoRA rank")
        self.alpha = (
            float(self.rank)
            if alpha is None
            else validate_lora_number_value(alpha, key="GroupedLinearLoRA alpha")
        )
        self.use_rslora = validate_lora_bool_value(
            use_rslora, key="GroupedLinearLoRA use_rslora"
        )
        self.scale = lora_scale(self.rank, alpha=self.alpha, use_rslora=self.use_rslora)
        self.dropout_p = validate_lora_dropout_value(dropout, key="GroupedLinearLoRA dropout")
        self.lora_a = nn.Parameter(torch.empty(self.num_local_experts, self.rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(self.num_local_experts, out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor, splits: list[int]) -> torch.Tensor:
        splits = _validate_grouped_lora_splits(
            "GroupedLinearLoRA", x, splits, self.num_local_experts
        )
        outputs = []
        offset = 0
        for expert_idx, size in enumerate(splits):
            x_i = x[offset : offset + size]
            if size == 0:
                outputs.append(x_i.new_empty((0, self.lora_b.shape[1])))
            else:
                dropped = (
                    F.dropout(x_i, p=self.dropout_p, training=self.training)
                    if self.dropout_p
                    else x_i
                )
                h_i = dropped.matmul(self.lora_a[expert_idx].t())
                outputs.append(h_i.matmul(self.lora_b[expert_idx].t()) * self.scale)
            offset += size
        return torch.cat(outputs, dim=0) if outputs else x.new_empty((0, self.lora_b.shape[1]))


class SharedGroupedLinearLoRA(nn.Module):
    """LoRA delta shared by all local experts in a GroupedLinear."""

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | float | None = None,
        dropout: float = 0.0,
        use_rslora: bool = False,
    ):
        super().__init__()
        self.num_local_experts = validate_lora_rank_value(
            num_local_experts, key="SharedGroupedLinearLoRA num_local_experts"
        )
        in_features = validate_lora_rank_value(
            in_features, key="SharedGroupedLinearLoRA in_features"
        )
        out_features = validate_lora_rank_value(
            out_features, key="SharedGroupedLinearLoRA out_features"
        )
        self.rank = validate_lora_rank_value(rank, key="SharedGroupedLinearLoRA rank")
        self.alpha = (
            float(self.rank)
            if alpha is None
            else validate_lora_number_value(alpha, key="SharedGroupedLinearLoRA alpha")
        )
        self.use_rslora = validate_lora_bool_value(
            use_rslora, key="SharedGroupedLinearLoRA use_rslora"
        )
        self.scale = lora_scale(self.rank, alpha=self.alpha, use_rslora=self.use_rslora)
        self.dropout_p = validate_lora_dropout_value(
            dropout, key="SharedGroupedLinearLoRA dropout"
        )
        self.shared_across_experts = True
        self.lora_a = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, self.rank))
        self.lora_a.tensor_model_parallel = False
        self.lora_b.tensor_model_parallel = False
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor, splits: list[int]) -> torch.Tensor:
        _validate_grouped_lora_splits(
            "SharedGroupedLinearLoRA", x, splits, self.num_local_experts
        )
        dropped = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p else x
        return dropped.matmul(self.lora_a.t()).matmul(self.lora_b.t()) * self.scale


__all__ = [
    "GroupedLinearLoRA",
    "LinearLoRA",
    "LoraConfig",
    "SharedGroupedLinearLoRA",
    "effective_lora_alpha",
    "freeze_non_lora_params",
    "lora_scale",
    "normalize_lora_config",
    "trainable_param_stats",
]
