# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PEFT adapter import/export for Qwen3-MoE lite native LoRA."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.primitive.modules.lora import (
    LoraConfig,
    effective_lora_alpha,
    normalize_lora_config,
    validate_lora_dropout_value,
    validate_lora_number_value,
    validate_lora_rank_value,
)
from megatron.lite.primitive.parallel import ParallelState

_PEFT_PREFIX = "base_model.model.model"
_ADAPTER_META_FORMAT = "megatron.lite_qwen3_moe_lora_peft_v1"
_OLORA_TAIL_INIT = "olora_tail"
_ADAPTER_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "megatron.lite_adapter_meta.json",
)
_ADAPTER_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_TARGET_MODULE_EXPANSIONS = {
    "all-linear": _ADAPTER_TARGET_MODULES,
    "all_linear": _ADAPTER_TARGET_MODULES,
    "linear_qkv": ("q_proj", "k_proj", "v_proj"),
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "q_proj": ("q_proj",),
    "k_proj": ("k_proj",),
    "v_proj": ("v_proj",),
    "linear_proj": ("o_proj",),
    "proj": ("o_proj",),
    "o_proj": ("o_proj",),
    "linear_fc1": ("gate_proj", "up_proj"),
    "fc1": ("gate_proj", "up_proj"),
    "gate_up": ("gate_proj", "up_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "gate_proj": ("gate_proj", "up_proj"),
    "up_proj": ("gate_proj", "up_proj"),
    "linear_fc2": ("down_proj",),
    "fc2": ("down_proj",),
    "down": ("down_proj",),
    "down_proj": ("down_proj",),
}
_MODEL_METADATA_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_experts",
    "moe_intermediate_size",
)
_LORA_METADATA_FIELDS = (
    "rank",
    "alpha",
    "dropout",
    "use_rslora",
    "scaling_convention",
    "scale",
    "target_modules",
)
_PARALLEL_METADATA_FIELDS = ("tp", "ep", "etp", "pp")


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size(group=None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


_WRAPPED_MODULE_ATTRS = ("module", "_module", "wrapped_module", "_fsdp_wrapped_module")


def _unwrap_model(module: nn.Module) -> nn.Module:
    current = module
    seen: set[int] = set()
    for _ in range(8):
        ident = id(current)
        if ident in seen:
            break
        seen.add(ident)
        next_model = None
        for attr in _WRAPPED_MODULE_ATTRS:
            if not hasattr(current, attr):
                continue
            inner = getattr(current, attr)
            if isinstance(inner, list):
                if len(inner) != 1:
                    continue
                inner = inner[0]
            if inner is not None and inner is not current:
                next_model = inner
                break
        if next_model is None:
            break
        current = next_model
    return current


def _iter_qwen_chunks(chunks: list[nn.Module] | tuple[nn.Module, ...]) -> list[nn.Module]:
    return [_unwrap_model(chunk) for chunk in chunks]


def _expert_lora_representation(chunks: list[nn.Module] | tuple[nn.Module, ...]) -> str:
    has_shared = any(
        _expert_lora_is_shared(getattr(layer.moe.experts, attr))
        for chunk in _iter_qwen_chunks(list(chunks))
        for layer in chunk.layers
        for attr in ("fc1_lora", "fc2_lora")
        if getattr(layer.moe.experts, attr) is not None
    )
    return "shared_local_expert_group" if has_shared else "per_expert"


def _all_gather_cat(tensor: torch.Tensor, group, dim: int) -> torch.Tensor:
    if _world_size(group) == 1:
        return tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim).contiguous()


def _select_tp_replicated(tensor: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    if _world_size(ps.tp_group) == 1:
        return tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(_world_size(ps.tp_group))]
    dist.all_gather(gathered, tensor.contiguous(), group=ps.tp_group)
    return gathered[0].contiguous()


def _is_rank_partitioned_lora_a(lora: Any, ps: ParallelState) -> bool:
    if getattr(lora, "rank_partitioned_a", False):
        return True
    rank = int(getattr(lora, "rank", lora.lora_b.shape[1]))
    return ps.tp_size > 1 and lora.lora_a.shape[0] != rank


def _gather_lora_rank_partition(tensor: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    return _all_gather_cat(tensor, ps.tp_group, dim=0)


def _slice_lora_rank_partition(tensor: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    if ps.tp_size == 1:
        return tensor.contiguous()
    if tensor.shape[0] % ps.tp_size != 0:
        raise ValueError(f"Cannot shard LoRA rank dim {tensor.shape[0]} over TP={ps.tp_size}.")
    local_rank = tensor.shape[0] // ps.tp_size
    start = ps.tp_rank * local_rank
    return tensor[start : start + local_rank].contiguous()


def _is_output_partitioned_lora_b(lora: Any, ps: ParallelState) -> bool:
    if getattr(lora, "output_partitioned_b", False):
        return True
    return ps.tp_size > 1 and lora.lora_b.shape[0] * ps.tp_size == getattr(lora, "out_features", -1)


def _expert_lora_is_shared(lora: Any) -> bool:
    return bool(getattr(lora, "shared_across_experts", False)) or len(lora.lora_a.shape) == 2


def _expand_shared_expert_lora(
    lora: Any, num_local_experts: int, *, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    lora_a = _materialize_export_tensor(
        lora.lora_a, name=f"{name}.lora_A", expected_ndim=None
    )
    lora_b = _materialize_export_tensor(
        lora.lora_b, name=f"{name}.lora_B", expected_ndim=None
    )
    if _expert_lora_is_shared(lora):
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"Shared expert LoRA {name} must materialize to 2-D A/B tensors, "
                f"got A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
            )
        return (
            lora_a.unsqueeze(0).expand(num_local_experts, -1, -1).contiguous(),
            lora_b.unsqueeze(0).expand(num_local_experts, -1, -1).contiguous(),
        )
    if lora_a.ndim != 3 or lora_b.ndim != 3:
        raise ValueError(
            f"Per-expert LoRA {name} must materialize to 3-D A/B tensors, "
            f"got A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
        )
    return lora_a, lora_b


def _split_local_mcore_qkv_b(
    qkv_b: torch.Tensor, *, num_heads_local: int, num_kv_heads_local: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_per_group = num_heads_local // num_kv_heads_local
    group_width = (q_per_group + 2) * head_dim
    packed = qkv_b.view(num_kv_heads_local, group_width, -1)
    q_end = q_per_group * head_dim
    k_end = q_end + head_dim
    q = packed[:, :q_end].reshape(num_heads_local * head_dim, -1)
    k = packed[:, q_end:k_end].reshape(num_kv_heads_local * head_dim, -1)
    v = packed[:, k_end:].reshape(num_kv_heads_local * head_dim, -1)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _pack_local_mcore_qkv_b(
    q_b: torch.Tensor,
    k_b: torch.Tensor,
    v_b: torch.Tensor,
    *,
    num_heads_local: int,
    num_kv_heads_local: int,
    head_dim: int,
) -> torch.Tensor:
    q_per_group = num_heads_local // num_kv_heads_local
    q = q_b.view(num_kv_heads_local, q_per_group * head_dim, -1)
    k = k_b.view(num_kv_heads_local, head_dim, -1)
    v = v_b.view(num_kv_heads_local, head_dim, -1)
    return torch.cat([q, k, v], dim=1).reshape(-1, q_b.shape[-1]).contiguous()


def _layer_prefix(layer_idx: int) -> str:
    return f"{_PEFT_PREFIX}.layers.{layer_idx}"


def _attn_key(layer_idx: int, module: str, suffix: str) -> str:
    return f"{_layer_prefix(layer_idx)}.self_attn.{module}.{suffix}.weight"


def _expert_key(layer_idx: int, expert_idx: int, module: str, suffix: str) -> str:
    return f"{_layer_prefix(layer_idx)}.mlp.experts.{expert_idx}.{module}.{suffix}.weight"


def _has_lora_tensor_suffix(key: str) -> bool:
    return key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")


def _strip_lora_tensor_suffix(key: str) -> str | None:
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _is_supported_adapter_tensor_key(key: str) -> bool:
    stem = _strip_lora_tensor_suffix(key)
    if stem is None:
        return False
    prefix = f"{_PEFT_PREFIX}.layers."
    if not stem.startswith(prefix):
        return False
    parts = stem[len(prefix) :].split(".")
    if not parts or not parts[0].isdigit():
        return False

    attn_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    mlp_targets = {"gate_proj", "up_proj", "down_proj"}
    if len(parts) == 3 and parts[1] == "self_attn":
        return parts[2] in attn_targets
    if (
        len(parts) == 5
        and parts[1] == "mlp"
        and parts[2] == "experts"
        and parts[3].isdigit()
    ):
        return parts[4] in mlp_targets
    return False


def _validate_adapter_state_has_lora_pairs(state: dict[str, torch.Tensor]) -> None:
    missing: list[str] = []
    rank_mismatches: list[str] = []
    seen_stems: set[str] = set()
    for key in sorted(state):
        stem = _strip_lora_tensor_suffix(key)
        if stem is None or stem in seen_stems:
            continue
        seen_stems.add(stem)
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            paired_key = f"{stem}{suffix}"
            if paired_key not in state:
                missing.append(paired_key)
        lora_a_key = f"{stem}.lora_A.weight"
        lora_b_key = f"{stem}.lora_B.weight"
        if lora_a_key in state and lora_b_key in state:
            lora_a_rank = int(state[lora_a_key].shape[0])
            lora_b_rank = int(state[lora_b_key].shape[1])
            if lora_a_rank != lora_b_rank:
                rank_mismatches.append(
                    f"{stem}: lora_A rows={lora_a_rank}, lora_B columns={lora_b_rank}"
                )
    if missing:
        preview = ", ".join(repr(key) for key in missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... and {len(missing) - 8} more"
        raise ValueError(
            f"LoRA adapter state is missing paired LoRA tensor keys: {preview}{suffix}."
        )
    if rank_mismatches:
        preview = "; ".join(rank_mismatches[:8])
        suffix = (
            ""
            if len(rank_mismatches) <= 8
            else f"; ... and {len(rank_mismatches) - 8} more"
        )
        raise ValueError(
            "LoRA adapter state contains inconsistent paired LoRA ranks: "
            f"{preview}{suffix}."
        )


def _dedupe_target_modules(targets) -> list[str]:
    out: list[str] = []
    if isinstance(targets, str):
        targets = (targets,)
    elif isinstance(targets, Mapping):
        raise TypeError("Adapter target_modules must be a string or sequence of strings.")
    else:
        try:
            targets = tuple(targets)
        except TypeError as exc:
            raise TypeError(
                "Adapter target_modules must be a string or sequence of strings."
            ) from exc
    for target in targets:
        if not isinstance(target, str):
            raise TypeError(
                "Adapter target_modules entries must be strings, got "
                f"{type(target).__name__}."
            )
        target = str(target)
        modules = _TARGET_MODULE_EXPANSIONS.get(target)
        if modules is None:
            supported = ", ".join(sorted(_TARGET_MODULE_EXPANSIONS))
            raise ValueError(
                f"Unsupported Qwen3-MoE LoRA target module {target!r}. "
                "Supported adapter targets are attention/MLP projection surfaces only: "
                f"{supported}."
            )
        for module in modules:
            if module not in out:
                out.append(module)
    return out


def _target_modules_from_lora_config(lora_config: LoraConfig) -> list[str]:
    return _dedupe_target_modules(lora_config.target_modules)


def _state_target_modules(state: dict[str, torch.Tensor]) -> set[str]:
    out: set[str] = set()
    for key in state:
        for module in _ADAPTER_TARGET_MODULES:
            if f".{module}." in key:
                out.add(module)
                break
    return out


def _effective_lora_alpha(lora_config: LoraConfig) -> float:
    return effective_lora_alpha(lora_config)


def _numbers_match(left: Any, right: Any) -> bool:
    return math.isclose(
        validate_lora_number_value(left, key="left numeric value"),
        validate_lora_number_value(right, key="right numeric value"),
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
        raise ValueError(f"LoRA adapter strict must be a boolean or boolean string, got {value!r}.")
    raise TypeError(f"LoRA adapter strict must be a boolean or boolean string, got {type(value)!r}.")


def _model_metadata_type_name(value: Any) -> str:
    return type(value).__name__


def _validate_model_metadata_value(value: Any, actual: Any, key: str) -> None:
    if actual is None:
        if value is not None:
            raise ValueError(
                f"Adapter metadata model.{key}={value!r} does not match native model "
                f"{key}=None."
            )
        return
    if isinstance(actual, bool):
        if not isinstance(value, bool):
            raise TypeError(
                f"Adapter metadata model.{key} must be a boolean, "
                f"got {_model_metadata_type_name(value)}."
            )
        if value != actual:
            raise ValueError(
                f"Adapter metadata model.{key}={value!r} does not match "
                f"native model {key}={actual!r}."
            )
        return
    if isinstance(actual, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Adapter metadata model.{key} must be an integer, "
                f"got {_model_metadata_type_name(value)}."
            )
        if value != actual:
            raise ValueError(
                f"Adapter metadata model.{key}={value!r} does not match "
                f"native model {key}={actual!r}."
            )
        return
    if isinstance(actual, float):
        metadata_value = validate_lora_number_value(
            value, key=f"Adapter metadata model.{key}"
        )
        if not _numbers_match(metadata_value, actual):
            raise ValueError(
                f"Adapter metadata model.{key}={value!r} does not match "
                f"native model {key}={actual!r}."
            )
        return
    if value != actual:
        raise ValueError(
            f"Adapter metadata model.{key}={value!r} does not match "
            f"native model {key}={actual!r}."
        )


def _validate_lora_metadata(
    adapter_lora: Any,
    state: dict[str, torch.Tensor],
    lora_config: LoraConfig | Mapping[str, Any] | None,
) -> None:
    if not isinstance(adapter_lora, dict):
        raise TypeError(f"Adapter metadata lora must be an object, got {type(adapter_lora)!r}.")
    _require_nested_metadata_fields(
        adapter_lora, _LORA_METADATA_FIELDS, description="Adapter metadata lora"
    )

    expected = normalize_lora_config(lora_config) if lora_config is not None else None
    state_rank = _infer_state_rank(state)
    metadata_rank = adapter_lora.get("rank")
    if metadata_rank is not None:
        metadata_rank = validate_lora_rank_value(
            metadata_rank, key="Adapter metadata lora.rank"
        )
        if state_rank is not None and metadata_rank != state_rank:
            raise ValueError(
                f"Adapter metadata lora.rank={metadata_rank} does not match tensor rank {state_rank}."
            )
        if expected is not None and metadata_rank != expected.rank:
            raise ValueError(
                "Adapter metadata lora.rank="
                f"{metadata_rank} does not match expected rank {expected.rank}."
            )

    metadata_alpha = adapter_lora.get("alpha")
    if metadata_alpha is not None:
        metadata_alpha = validate_lora_number_value(
            metadata_alpha, key="Adapter metadata lora.alpha"
        )
    if metadata_alpha is not None and expected is not None and not _numbers_match(
        metadata_alpha, _effective_lora_alpha(expected)
    ):
        raise ValueError(
            "Adapter metadata lora.alpha="
            f"{metadata_alpha} does not match expected alpha {_effective_lora_alpha(expected)}."
        )

    metadata_dropout = adapter_lora.get("dropout")
    if metadata_dropout is not None:
        metadata_dropout = validate_lora_dropout_value(
            metadata_dropout, key="Adapter metadata lora.dropout"
        )
    if metadata_dropout is not None and expected is not None and not _numbers_match(
        metadata_dropout, expected.dropout
    ):
        raise ValueError(
            "Adapter metadata lora.dropout="
            f"{metadata_dropout} does not match expected dropout {expected.dropout}."
        )

    metadata_use_rslora = None
    if "use_rslora" in adapter_lora:
        metadata_use_rslora = _adapter_bool(
            adapter_lora.get("use_rslora"), default=False, key="metadata.lora.use_rslora"
        )
        if expected is not None and metadata_use_rslora != expected.use_rslora:
            raise ValueError(
                "Adapter metadata lora.use_rslora="
                f"{metadata_use_rslora} does not match expected use_rslora={expected.use_rslora}."
            )
    elif expected is not None:
        metadata_use_rslora = expected.use_rslora

    metadata_targets = _peft_target_set(adapter_lora.get("target_modules"))
    if metadata_targets is not None:
        state_targets = _state_target_modules(state)
        if metadata_targets != state_targets:
            raise ValueError(
                "Adapter metadata lora.target_modules do not match adapter tensors: "
                f"metadata={sorted(metadata_targets)}, tensors={sorted(state_targets)}."
            )
        if expected is not None:
            expected_targets = set(_target_modules_from_lora_config(expected))
            if metadata_targets != expected_targets:
                raise ValueError(
                    "Adapter metadata lora.target_modules do not match expected LoRA config: "
                    f"metadata={sorted(metadata_targets)}, expected={sorted(expected_targets)}."
                )

    metadata_scaling = adapter_lora.get("scaling_convention")
    if metadata_scaling is not None:
        if metadata_use_rslora is not None:
            expected_scaling = (
                "alpha_over_sqrt_rank" if metadata_use_rslora else "alpha_over_rank"
            )
            if metadata_scaling != expected_scaling:
                raise ValueError(
                    "Adapter metadata lora.scaling_convention="
                    f"{metadata_scaling!r} does not match expected {expected_scaling!r}."
                )
        elif metadata_scaling not in ("alpha_over_sqrt_rank", "alpha_over_rank"):
            raise ValueError(
                f"Adapter metadata lora.scaling_convention={metadata_scaling!r} is not supported."
            )

    metadata_scale = adapter_lora.get("scale")
    if metadata_scale is not None:
        metadata_scale = validate_lora_number_value(
            metadata_scale, key="Adapter metadata lora.scale"
        )
        expected_scale = None
        if expected is not None:
            expected_scale = expected.scale
        elif metadata_rank is not None:
            alpha = metadata_alpha if metadata_alpha is not None else metadata_rank
            expected_scale = LoraConfig(
                rank=metadata_rank,
                alpha=alpha,
                use_rslora=bool(metadata_use_rslora),
            ).scale
        if expected_scale is not None and not _numbers_match(metadata_scale, expected_scale):
            raise ValueError(
                "Adapter metadata lora.scale="
                f"{metadata_scale} does not match expected scale {expected_scale}."
            )


def _json_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not supported")


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise IsADirectoryError(f"{description} path must be a file: {path}")
    try:
        payload = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise ValueError(f"{description} must be standard JSON.") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{description} must be a JSON object, got {type(payload)!r}.")
    return payload


def _json_dumps_standard(payload: dict[str, Any], *, description: str) -> str:
    try:
        return json.dumps(payload, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{description} must be standard JSON-serializable.") from exc


def _write_adapter_dir(
    output: Path,
    *,
    state: dict[str, torch.Tensor],
    config_text: str,
    meta_text: str,
    save_file,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Adapter output path exists and is not a directory: {output}")
    if output.exists():
        for filename in _ADAPTER_FILENAMES:
            path = output / filename
            if path.exists() and not path.is_file():
                raise IsADirectoryError(f"Adapter artifact path must be a file: {path}")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.tmp-", dir=str(output.parent)) as tmp:
        tmp_dir = Path(tmp)
        save_file(state, str(tmp_dir / "adapter_model.safetensors"))
        (tmp_dir / "adapter_config.json").write_text(config_text)
        (tmp_dir / "megatron.lite_adapter_meta.json").write_text(meta_text)
        if output.exists():
            with tempfile.TemporaryDirectory(
                prefix=f".{output.name}.bak-", dir=str(output.parent)
            ) as backup:
                backup_dir = Path(backup)
                installed: list[str] = []
                backed_up: list[str] = []
                try:
                    for filename in _ADAPTER_FILENAMES:
                        target = output / filename
                        backup_path = backup_dir / filename
                        if target.exists():
                            target.replace(backup_path)
                            backed_up.append(filename)
                        (tmp_dir / filename).replace(target)
                        installed.append(filename)
                except Exception:
                    for filename in reversed(installed):
                        target = output / filename
                        if target.exists():
                            target.unlink()
                    for filename in reversed(backed_up):
                        backup_path = backup_dir / filename
                        if backup_path.exists():
                            backup_path.replace(output / filename)
                    raise
        else:
            tmp_dir.rename(output)


def _validate_adapter_root(adapter_dir: str | Path) -> Path:
    adapter_root = Path(adapter_dir)
    if not adapter_root.exists():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_root}")
    if not adapter_root.is_dir():
        raise NotADirectoryError(f"LoRA adapter path must be a directory: {adapter_root}")
    return adapter_root


def _validate_adapter_model_path(adapter_root: Path) -> Path:
    path = adapter_root / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter model file is missing: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"LoRA adapter model path must be a file: {path}")
    return path


def _validate_metadata_json_keys(value: Any, *, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Adapter metadata metadata keys must be strings; "
                    f"{path} contains key {key!r} of type {type(key)!r}."
                )
            _validate_metadata_json_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for idx, child in enumerate(value):
            _validate_metadata_json_keys(child, path=f"{path}[{idx}]")


def _copy_json_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_json_metadata_value(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_copy_json_metadata_value(child) for child in value]
    return value


def _normalize_user_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Adapter metadata metadata must be an object, got {type(metadata)!r}.")
    _validate_metadata_json_keys(metadata)
    metadata = _copy_json_metadata_value(metadata)
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("Adapter metadata metadata must be JSON-serializable.") from exc
    return metadata


def _require_adapter_metadata_fields(adapter_meta: dict[str, Any]) -> None:
    required_fields = (
        "base_model_name_or_path",
        "num_tensors",
        "num_parameters",
        "expert_lora_representation",
        "lora",
        "parallel",
        "model",
        "metadata",
    )
    for field in required_fields:
        if field not in adapter_meta:
            raise ValueError(
                f"Adapter metadata {field} is required for Megatron Lite LoRA adapter metadata."
            )


def _require_nested_metadata_fields(
    payload: dict[str, Any],
    fields: tuple[str, ...],
    *,
    description: str,
) -> None:
    for field in fields:
        if field not in payload:
            raise ValueError(
                f"{description}.{field} is required for Megatron Lite LoRA adapter metadata."
            )


def _validate_base_model_name_or_path(value: Any, *, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{key} base_model_name_or_path must be a string, got {type(value)!r}."
        )
    value = value.strip()
    if not value:
        raise ValueError(f"{key} base_model_name_or_path must be non-empty.")
    return value


def _peft_target_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    return set(_dedupe_target_modules(value))


def _adapter_bool(value: Any, *, default: bool, key: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise TypeError(f"Adapter config {key} must be a boolean, got {type(value)!r}.")


def _infer_state_rank(state: dict[str, torch.Tensor]) -> int | None:
    ranks: set[int] = set()
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
            continue
        if key.endswith(".lora_A.weight"):
            ranks.add(int(tensor.shape[0]))
        elif key.endswith(".lora_B.weight"):
            ranks.add(int(tensor.shape[1]))
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError(
            "Adapter contains inconsistent LoRA ranks across lora_A rows and "
            f"lora_B columns: {sorted(ranks)}."
        )
    return next(iter(ranks))


def _iter_native_lora_modules(chunks: list[nn.Module] | tuple[nn.Module, ...]):
    for chunk in _iter_qwen_chunks(list(chunks)):
        for layer in chunk.layers:
            attn = layer.attn
            if attn.qkv_lora is not None:
                yield attn.qkv_lora
            if attn.proj_lora is not None:
                yield attn.proj_lora
            experts = layer.moe.experts
            if experts.fc1_lora is not None:
                yield experts.fc1_lora
            if experts.fc2_lora is not None:
                yield experts.fc2_lora


def _infer_native_alpha(chunks: list[nn.Module] | tuple[nn.Module, ...]) -> float | None:
    alphas: set[float] = set()
    for module in _iter_native_lora_modules(chunks):
        rank = getattr(module, "rank", None)
        scale = getattr(module, "scale", None)
        if rank is None or scale is None:
            continue
        denominator = (
            math.sqrt(float(rank)) if getattr(module, "use_rslora", False) else float(rank)
        )
        alphas.add(float(scale) * denominator)
    if not alphas:
        return None
    rounded = {round(alpha, 6) for alpha in alphas}
    if len(rounded) != 1:
        raise ValueError(
            f"Native LoRA modules have inconsistent effective alpha values: {sorted(alphas)}."
        )
    return next(iter(alphas))


def _infer_native_use_rslora(chunks: list[nn.Module] | tuple[nn.Module, ...]) -> bool | None:
    values = {
        bool(getattr(module, "use_rslora", False)) for module in _iter_native_lora_modules(chunks)
    }
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("Native LoRA modules have inconsistent use_rslora settings.")
    return next(iter(values))


def _infer_native_dropout(chunks: list[nn.Module] | tuple[nn.Module, ...]) -> float | None:
    values = {
        float(getattr(module, "dropout_p", 0.0)) for module in _iter_native_lora_modules(chunks)
    }
    if not values:
        return None
    rounded = {round(value, 6) for value in values}
    if len(rounded) != 1:
        raise ValueError(f"Native LoRA modules have inconsistent dropout values: {sorted(values)}.")
    return next(iter(values))


def _validate_init_lora_weights(value: Any) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("Adapter config init_lora_weights string must be non-empty.")
        return
    raise TypeError(
        "Adapter config init_lora_weights must be a boolean, string, or None, "
        f"got {type(value)!r}."
    )


def _validate_adapter_only_config(adapter_config: dict[str, Any]) -> None:
    task_type = adapter_config.get("task_type")
    if task_type is not None:
        if not isinstance(task_type, str):
            raise TypeError(
                f"Adapter config task_type must be a string, got {type(task_type)!r}."
            )
        if task_type.upper() != "CAUSAL_LM":
            raise ValueError(
                f"Adapter config task_type={task_type!r} is not supported; expected 'CAUSAL_LM'."
            )

    inference_mode = adapter_config.get("inference_mode")
    if inference_mode is not None and not isinstance(inference_mode, bool):
        raise TypeError(
            "Adapter config inference_mode must be a boolean, "
            f"got {type(inference_mode)!r}."
        )

    bias = adapter_config.get("bias")
    if bias is not None:
        if not isinstance(bias, str):
            raise TypeError(f"Adapter config bias must be a string, got {type(bias)!r}.")
        if bias.lower() != "none":
            raise ValueError(f"Adapter config bias={bias!r} is not supported; expected 'none'.")

    fan_in_fan_out = _adapter_bool(
        adapter_config.get("fan_in_fan_out"), default=False, key="fan_in_fan_out"
    )
    if fan_in_fan_out:
        raise ValueError("Adapter config fan_in_fan_out=True is not supported.")

    modules_to_save = adapter_config.get("modules_to_save")
    if modules_to_save not in (None, [], ()):
        raise ValueError(
            "Adapter config modules_to_save is not supported for adapter-only Qwen3-MoE LoRA."
        )


def _validate_adapter_config(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    state: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
    *,
    lora_config: LoraConfig | Mapping[str, Any] | None = None,
) -> None:
    state_rank = _infer_state_rank(state)
    state_targets = _state_target_modules(state)
    has_adapter_state = state_rank is not None or bool(state_targets)

    peft_type = adapter_config.get("peft_type")
    if peft_type is None and has_adapter_state:
        raise ValueError(
            "Adapter config peft_type='LORA' is required for non-empty LoRA adapter state."
        )
    if peft_type is not None:
        if not isinstance(peft_type, str):
            raise TypeError(
                f"Adapter config peft_type must be a string, got {type(peft_type)!r}."
            )
        if peft_type.upper() != "LORA":
            raise ValueError(
                f"Expected PEFT adapter_config peft_type='LORA', got {peft_type!r}."
            )
    _validate_adapter_only_config(adapter_config)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if has_adapter_state and base_model_name is None:
        raise ValueError(
            "Adapter config base_model_name_or_path is required for non-empty LoRA adapter state."
        )
    if base_model_name is not None:
        _validate_base_model_name_or_path(base_model_name, key="Adapter config")

    if state_rank is not None and adapter_config.get("r") is None:
        raise ValueError("Adapter config r is required for non-empty LoRA adapter state.")
    config_rank = adapter_config.get("r")
    if config_rank is not None:
        config_rank = validate_lora_rank_value(config_rank, key="Adapter config r")
        if state_rank is not None and config_rank != state_rank:
            raise ValueError(
                f"Adapter config rank r={config_rank} does not match tensor rank {state_rank}."
            )

    if state_targets and adapter_config.get("target_modules") is None:
        raise ValueError(
            "Adapter config target_modules is required for non-empty LoRA adapter state."
        )
    config_targets = _peft_target_set(adapter_config.get("target_modules"))
    if config_targets is not None and config_targets != state_targets:
        raise ValueError(
            "Adapter config target_modules do not match adapter tensors: "
            f"config={sorted(config_targets)}, tensors={sorted(state_targets)}."
        )

    config_alpha = adapter_config.get("lora_alpha")
    if has_adapter_state and config_alpha is None:
        raise ValueError(
            "Adapter config lora_alpha is required for non-empty LoRA adapter state."
        )
    if config_alpha is not None:
        config_alpha = validate_lora_number_value(
            config_alpha, key="Adapter config lora_alpha"
        )
    native_alpha = _infer_native_alpha(chunks)
    if (
        config_alpha is not None
        and native_alpha is not None
        and not _numbers_match(config_alpha, native_alpha)
    ):
        raise ValueError(
            f"Adapter config lora_alpha={config_alpha} does not match native model alpha={native_alpha}."
        )
    if has_adapter_state and "use_rslora" not in adapter_config:
        raise ValueError(
            "Adapter config use_rslora is required for non-empty LoRA adapter state."
        )
    config_use_rslora = _adapter_bool(
        adapter_config.get("use_rslora"), default=False, key="use_rslora"
    )
    native_use_rslora = _infer_native_use_rslora(chunks)
    if native_use_rslora is not None and config_use_rslora != native_use_rslora:
        raise ValueError(
            "Adapter config use_rslora="
            f"{config_use_rslora} does not match native model use_rslora={native_use_rslora}."
        )

    config_dropout = adapter_config.get("lora_dropout")
    if has_adapter_state and config_dropout is None:
        raise ValueError(
            "Adapter config lora_dropout is required for non-empty LoRA adapter state."
        )
    if config_dropout is not None:
        config_dropout = validate_lora_dropout_value(
            config_dropout, key="Adapter config lora_dropout"
        )
    native_dropout = _infer_native_dropout(chunks)
    if config_dropout is not None and native_dropout is not None and not _numbers_match(
        config_dropout, native_dropout
    ):
        raise ValueError(
            "Adapter config lora_dropout="
            f"{config_dropout} does not match native model dropout={native_dropout}."
        )

    if "init_lora_weights" in adapter_config:
        _validate_init_lora_weights(adapter_config.get("init_lora_weights"))

    if lora_config is not None:
        expected = normalize_lora_config(lora_config)
        expected_targets = set(_target_modules_from_lora_config(expected))
        if config_rank is not None and config_rank != expected.rank:
            raise ValueError(
                f"Adapter config rank r={config_rank} does not match expected rank {expected.rank}."
            )
        if config_alpha is not None and not _numbers_match(
            config_alpha, _effective_lora_alpha(expected)
        ):
            raise ValueError(
                "Adapter config lora_alpha="
                f"{config_alpha} does not match expected alpha {_effective_lora_alpha(expected)}."
            )
        if config_use_rslora != expected.use_rslora:
            raise ValueError(
                "Adapter config use_rslora="
                f"{config_use_rslora} does not match expected use_rslora={expected.use_rslora}."
            )
        if config_dropout is not None and not _numbers_match(config_dropout, expected.dropout):
            raise ValueError(
                "Adapter config lora_dropout="
                f"{config_dropout} does not match expected dropout={expected.dropout}."
            )
        if config_targets is not None and config_targets != expected_targets:
            raise ValueError(
                "Adapter config target_modules do not match expected LoRA config: "
                f"config={sorted(config_targets)}, expected={sorted(expected_targets)}."
            )


def _validate_expected_lora_config(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    state: dict[str, torch.Tensor],
    lora_config: LoraConfig | Mapping[str, Any],
) -> None:
    expected = normalize_lora_config(lora_config)
    if not expected.enabled:
        raise ValueError("Adapter import requires an enabled expected LoRA config.")
    state_rank = _infer_state_rank(state)
    if state_rank is not None and state_rank != expected.rank:
        raise ValueError(
            f"Adapter tensor rank {state_rank} does not match expected rank {expected.rank}."
        )

    state_targets = _state_target_modules(state)
    expected_targets = set(_target_modules_from_lora_config(expected))
    if state_targets != expected_targets:
        raise ValueError(
            "Adapter tensors target modules do not match expected LoRA config: "
            f"tensors={sorted(state_targets)}, expected={sorted(expected_targets)}."
        )

    native_alpha = _infer_native_alpha(chunks)
    if native_alpha is not None and not _numbers_match(
        native_alpha, _effective_lora_alpha(expected)
    ):
        raise ValueError(
            f"Native model alpha={native_alpha} does not match expected alpha {_effective_lora_alpha(expected)}."
        )

    native_use_rslora = _infer_native_use_rslora(chunks)
    if native_use_rslora is not None and native_use_rslora != expected.use_rslora:
        raise ValueError(
            "Native model use_rslora="
            f"{native_use_rslora} does not match expected use_rslora={expected.use_rslora}."
        )

    native_dropout = _infer_native_dropout(chunks)
    if native_dropout is not None and not _numbers_match(native_dropout, expected.dropout):
        raise ValueError(
            f"Native model dropout={native_dropout} does not match expected dropout={expected.dropout}."
        )


def _validate_adapter_metadata(
    adapter_meta: dict[str, Any],
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    state: dict[str, torch.Tensor],
    lora_config: LoraConfig | Mapping[str, Any] | None,
) -> None:
    metadata_format = adapter_meta.get("format")
    if metadata_format != _ADAPTER_META_FORMAT:
        raise ValueError(
            f"Expected Megatron Lite Qwen3-MoE adapter metadata format {_ADAPTER_META_FORMAT!r}, "
            f"got {metadata_format!r}."
        )
    _require_adapter_metadata_fields(adapter_meta)

    if "base_model_name_or_path" in adapter_meta:
        _validate_base_model_name_or_path(
            adapter_meta["base_model_name_or_path"], key="Adapter metadata"
        )

    metadata_num_tensors = adapter_meta.get("num_tensors")
    if metadata_num_tensors is not None:
        metadata_num_tensors = validate_lora_rank_value(
            metadata_num_tensors,
            key="Adapter metadata num_tensors",
            allow_zero=True,
        )
        if metadata_num_tensors != len(state):
            raise ValueError(
                f"Adapter metadata num_tensors={metadata_num_tensors} does not match "
                f"adapter state tensor count {len(state)}."
            )
    metadata_num_parameters = adapter_meta.get("num_parameters")
    if metadata_num_parameters is not None:
        metadata_num_parameters = validate_lora_rank_value(
            metadata_num_parameters,
            key="Adapter metadata num_parameters",
            allow_zero=True,
        )
        actual_num_parameters = int(sum(tensor.numel() for tensor in state.values()))
        if metadata_num_parameters != actual_num_parameters:
            raise ValueError(
                f"Adapter metadata num_parameters={metadata_num_parameters} does not match "
                f"adapter state parameter count {actual_num_parameters}."
            )

    metadata_representation = adapter_meta.get("expert_lora_representation")
    if metadata_representation is not None:
        expected_representation = _expert_lora_representation(chunks)
        if metadata_representation != expected_representation:
            raise ValueError(
                "Adapter metadata expert_lora_representation="
                f"{metadata_representation!r} does not match native representation "
                f"{expected_representation!r}."
            )

    parallel = adapter_meta.get("parallel")
    if not isinstance(parallel, dict):
        raise TypeError(f"Adapter metadata parallel must be an object, got {type(parallel)!r}.")
    _require_nested_metadata_fields(
        parallel, _PARALLEL_METADATA_FIELDS, description="Adapter metadata parallel"
    )
    parallel_fields = {
        "tp": ps.tp_size,
        "ep": ps.ep_size,
        "etp": ps.etp_size,
        "pp": ps.pp_size,
    }
    for key, expected in parallel_fields.items():
        value = parallel.get(key)
        value = validate_lora_rank_value(value, key=f"Adapter metadata parallel.{key}")
        if value != expected:
            raise ValueError(
                f"Adapter metadata parallel.{key}={value} does not match current "
                f"{key.upper()}={expected}."
            )

    model = adapter_meta.get("model")
    if not isinstance(model, dict):
        raise TypeError(f"Adapter metadata model must be an object, got {type(model)!r}.")
    _require_nested_metadata_fields(
        model, _MODEL_METADATA_FIELDS, description="Adapter metadata model"
    )
    for key in _MODEL_METADATA_FIELDS:
        if not hasattr(model_cfg, key):
            continue
        actual = getattr(model_cfg, key)
        _validate_model_metadata_value(model[key], actual, key)

    metadata = adapter_meta.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"Adapter metadata metadata must be an object, got {type(metadata)!r}.")

    _validate_lora_metadata(adapter_meta.get("lora"), state, lora_config)


def _validate_adapter_config_metadata_consistency(
    adapter_config: dict[str, Any],
    adapter_meta: dict[str, Any],
) -> None:
    if "base_model_name_or_path" in adapter_config and "base_model_name_or_path" in adapter_meta:
        config_base = _validate_base_model_name_or_path(
            adapter_config["base_model_name_or_path"], key="Adapter config"
        )
        metadata_base = _validate_base_model_name_or_path(
            adapter_meta["base_model_name_or_path"], key="Adapter metadata"
        )
        if config_base != metadata_base:
            raise ValueError(
                "Adapter config base_model_name_or_path="
                f"{config_base!r} does not match metadata base_model_name_or_path="
                f"{metadata_base!r}."
            )

    adapter_lora = adapter_meta.get("lora")
    if adapter_lora is not None:
        if not isinstance(adapter_lora, dict):
            raise TypeError(f"Adapter metadata lora must be an object, got {type(adapter_lora)!r}.")
        if "r" in adapter_config and "rank" in adapter_lora:
            config_rank = validate_lora_rank_value(
                adapter_config["r"], key="Adapter config r"
            )
            metadata_rank = validate_lora_rank_value(
                adapter_lora["rank"], key="Adapter metadata lora.rank"
            )
            if config_rank != metadata_rank:
                raise ValueError(
                    "Adapter config r="
                    f"{adapter_config['r']} does not match metadata lora.rank={adapter_lora['rank']}."
                )
        if "lora_alpha" in adapter_config and "alpha" in adapter_lora:
            config_alpha = validate_lora_number_value(
                adapter_config["lora_alpha"], key="Adapter config lora_alpha"
            )
            metadata_alpha = validate_lora_number_value(
                adapter_lora["alpha"], key="Adapter metadata lora.alpha"
            )
            if not _numbers_match(config_alpha, metadata_alpha):
                raise ValueError(
                    "Adapter config lora_alpha="
                    f"{adapter_config['lora_alpha']} does not match metadata lora.alpha="
                    f"{adapter_lora['alpha']}."
                )
        if "lora_dropout" in adapter_config and "dropout" in adapter_lora:
            config_dropout = validate_lora_dropout_value(
                adapter_config["lora_dropout"], key="Adapter config lora_dropout"
            )
            metadata_dropout = validate_lora_dropout_value(
                adapter_lora["dropout"], key="Adapter metadata lora.dropout"
            )
            if not _numbers_match(config_dropout, metadata_dropout):
                raise ValueError(
                    "Adapter config lora_dropout="
                    f"{adapter_config['lora_dropout']} does not match metadata lora.dropout="
                    f"{adapter_lora['dropout']}."
                )
        if "use_rslora" in adapter_lora:
            config_use_rslora = _adapter_bool(
                adapter_config.get("use_rslora"), default=False, key="use_rslora"
            )
            metadata_use_rslora = _adapter_bool(
                adapter_lora.get("use_rslora"), default=False, key="metadata.lora.use_rslora"
            )
            if config_use_rslora != metadata_use_rslora:
                raise ValueError(
                    "Adapter config use_rslora="
                    f"{config_use_rslora} does not match metadata lora.use_rslora="
                    f"{metadata_use_rslora}."
                )
        if "target_modules" in adapter_config and "target_modules" in adapter_lora:
            config_targets = _peft_target_set(adapter_config.get("target_modules"))
            metadata_targets = _peft_target_set(adapter_lora.get("target_modules"))
            if config_targets != metadata_targets:
                raise ValueError(
                    "Adapter config target_modules do not match metadata lora.target_modules: "
                    f"config={sorted(config_targets or [])}, "
                    f"metadata={sorted(metadata_targets or [])}."
                )

    metadata = adapter_meta.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise TypeError(
                f"Adapter metadata metadata must be an object, got {type(metadata)!r}."
            )
        if "init_lora_weights" in adapter_config and "init_lora_weights" in metadata:
            config_init = adapter_config.get("init_lora_weights")
            metadata_init = metadata.get("init_lora_weights")
            _validate_init_lora_weights(metadata_init)
            if config_init != metadata_init:
                raise ValueError(
                    "Adapter config init_lora_weights="
                    f"{config_init!r} does not match metadata init_lora_weights="
                    f"{metadata_init!r}."
                )


def _validate_attention_tp(model_cfg: Qwen3MoEConfig, ps: ParallelState) -> None:
    if ps.tp_size <= 0:
        raise ValueError(f"TP size must be positive, got {ps.tp_size}.")
    if model_cfg.num_attention_heads % ps.tp_size != 0:
        raise ValueError(
            "LoRA adapter import/export requires num_attention_heads "
            f"({model_cfg.num_attention_heads}) to be divisible by TP={ps.tp_size}."
        )
    if model_cfg.num_key_value_heads % ps.tp_size != 0:
        raise ValueError(
            "LoRA adapter import/export requires num_key_value_heads "
            f"({model_cfg.num_key_value_heads}) to be divisible by TP={ps.tp_size}."
        )
    q_heads_local = model_cfg.num_attention_heads // ps.tp_size
    kv_heads_local = model_cfg.num_key_value_heads // ps.tp_size
    if kv_heads_local <= 0:
        raise ValueError(
            "LoRA adapter import/export requires at least one local KV head; "
            f"got num_key_value_heads={model_cfg.num_key_value_heads}, TP={ps.tp_size}."
        )
    if q_heads_local % kv_heads_local != 0:
        raise ValueError(
            "LoRA adapter import/export requires local query heads to be divisible "
            f"by local KV heads, got q={q_heads_local}, kv={kv_heads_local}."
        )


def _validate_export_parallel_scope(ps: ParallelState) -> None:
    if ps.pp_size != 1:
        raise NotImplementedError("LoRA adapter export currently supports pp=1.")
    if ps.etp_size != 1:
        raise NotImplementedError("LoRA adapter export currently supports etp=1.")


def _validate_import_parallel_scope(ps: ParallelState) -> None:
    if ps.pp_size != 1:
        raise NotImplementedError("LoRA adapter import currently supports pp=1.")
    if ps.etp_size != 1:
        raise NotImplementedError("LoRA adapter import currently supports etp=1.")


def _validate_adapter_state_is_lora_only(state: dict[str, torch.Tensor]) -> None:
    if not state:
        raise ValueError("LoRA adapter state is empty; expected at least one LoRA A/B tensor.")
    unexpected = sorted(
        key
        for key in state
        if not _has_lora_tensor_suffix(key)
    )
    if unexpected:
        preview = ", ".join(repr(key) for key in unexpected[:8])
        suffix = "" if len(unexpected) <= 8 else f", ... and {len(unexpected) - 8} more"
        raise ValueError(
            f"LoRA adapter export produced non-adapter tensor keys: {preview}{suffix}."
        )
    unsupported = sorted(key for key in state if not _is_supported_adapter_tensor_key(key))
    if unsupported:
        preview = ", ".join(repr(key) for key in unsupported[:8])
        suffix = "" if len(unsupported) <= 8 else f", ... and {len(unsupported) - 8} more"
        raise ValueError(
            f"LoRA adapter state contains unsupported adapter tensor keys: {preview}{suffix}."
        )
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"LoRA adapter tensor {key!r} must be a torch.Tensor, got {type(tensor)!r}."
            )
        if not torch.is_floating_point(tensor):
            raise TypeError(
                f"LoRA adapter tensor {key!r} must use a floating-point dtype, "
                f"got {tensor.dtype}."
            )
        if tensor.ndim != 2:
            raise ValueError(
                f"LoRA adapter tensor {key!r} must be 2-D, got shape {tuple(tensor.shape)}."
            )
        if any(dim <= 0 for dim in tensor.shape):
            raise ValueError(
                f"LoRA adapter tensor {key!r} must have positive dimensions, "
                f"got shape {tuple(tensor.shape)}."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"LoRA adapter tensor {key!r} must contain only finite values.")
    _validate_adapter_state_has_lora_pairs(state)
    _infer_state_rank(state)


def _materialize_export_tensor(
    tensor: Any,
    *,
    name: str,
    cpu: bool = False,
    expected_ndim: int | None = 2,
) -> torch.Tensor:
    """Return a normal tensor for adapter export, expanding DTensor/FSDP2 params."""

    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
    elif hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"LoRA adapter export tensor {name!r} must materialize to a torch.Tensor, "
            f"got {type(tensor)!r}."
        )
    if expected_ndim is not None and tensor.ndim != expected_ndim:
        raise ValueError(
            f"LoRA adapter export tensor {name!r} must be {expected_ndim}-D after "
            f"materialization, got shape {tuple(tensor.shape)}."
        )
    if cpu:
        tensor = tensor.cpu()
    return tensor.contiguous()


def _validate_exported_target_modules(
    state: dict[str, torch.Tensor],
    lora_config: LoraConfig,
) -> None:
    requested_targets = set(_target_modules_from_lora_config(lora_config))
    state_targets = _state_target_modules(state)
    if requested_targets != state_targets:
        raise ValueError(
            "LoRA config target_modules do not match exported adapter tensors: "
            f"config={sorted(requested_targets)}, tensors={sorted(state_targets)}."
        )


def _base_weight(module: nn.Module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    inner = getattr(module, "linear", None)
    weight = getattr(inner, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    raise TypeError(f"Cannot locate base weight on module {type(module)!r}.")


def _grouped_base_weight(module: nn.Module, local_expert_idx: int) -> torch.Tensor:
    weight = getattr(module, f"weight{local_expert_idx}", None)
    if isinstance(weight, torch.Tensor):
        return weight
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        if weight.ndim == 3:
            return weight[local_expert_idx]
        if weight.ndim == 2 and local_expert_idx == 0:
            return weight
    raise TypeError(
        f"Cannot locate grouped base weight {local_expert_idx} on module {type(module)!r}."
    )


def _olora_tail_factors(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"OLoRA-tail requires a 2-D base weight, got shape {tuple(weight.shape)}.")
    if rank <= 0:
        raise ValueError(f"OLoRA-tail requires a positive rank, got {rank}.")
    if rank > min(weight.shape):
        raise ValueError(
            f"OLoRA-tail rank {rank} exceeds base weight min dimension {min(weight.shape)}."
        )
    target_device = weight.device
    target_dtype = weight.dtype
    svd_weight = weight.detach() if hasattr(weight, "detach") else weight
    if hasattr(svd_weight, "full_tensor"):
        svd_weight = svd_weight.full_tensor()
        if hasattr(svd_weight, "detach"):
            svd_weight = svd_weight.detach()
    elif hasattr(svd_weight, "to_local"):
        svd_weight = svd_weight.to_local()
        if hasattr(svd_weight, "detach"):
            svd_weight = svd_weight.detach()
    if not isinstance(svd_weight, torch.Tensor):
        raise TypeError(
            f"OLoRA-tail base weight must materialize to a torch.Tensor, got {type(svd_weight)!r}."
        )
    if not torch.is_floating_point(svd_weight):
        raise TypeError(f"OLoRA-tail requires a floating-point base weight, got {target_dtype}.")
    svd_weight = svd_weight.float()
    if not torch.isfinite(svd_weight).all():
        raise ValueError("OLoRA-tail requires finite base weight values.")
    u, _, vh = torch.linalg.svd(svd_weight, full_matrices=False)
    lora_b = u[:, -rank:].to(device=target_device, dtype=target_dtype)
    lora_a = vh[-rank:, :].to(device=target_device, dtype=target_dtype)
    if not torch.isfinite(lora_b).all() or not torch.isfinite(lora_a).all():
        raise ValueError("OLoRA-tail SVD produced non-finite LoRA factors.")
    return lora_b, lora_a


def _init_linear_lora_olora_tail(
    lora,
    base_weight: torch.Tensor,
    name: str,
    *,
    force: bool,
) -> bool:
    if lora is None:
        return False
    if getattr(lora, "_olora_tail_initialized", False) and not force:
        raise RuntimeError(
            f"LoRA module {name} already has OLoRA-tail initialization; pass force=True "
            "to reinitialize it explicitly."
        )
    if getattr(lora, "rank_partitioned_a", False) or getattr(lora, "output_partitioned_b", False):
        raise NotImplementedError(
            f"OLoRA-tail for {name} currently supports unsharded LoRA tensors only."
        )
    if lora.lora_a.ndim != 2 or lora.lora_b.ndim != 2:
        raise NotImplementedError(
            f"OLoRA-tail for {name} currently supports ordinary LinearLoRA tensors only."
        )
    rank = int(lora.lora_a.shape[0])
    expected = (lora.lora_b.shape[0], lora.lora_a.shape[1])
    if tuple(base_weight.shape) != expected:
        raise ValueError(
            f"OLoRA-tail base weight shape mismatch for {name}: got {tuple(base_weight.shape)}, "
            f"expected {expected} from LoRA B/A shapes."
        )
    lora_b, lora_a = _olora_tail_factors(base_weight, rank)
    with torch.no_grad():
        _copy_to_lora_param(lora.lora_b, lora_b, key=f"{name}.lora_B")
        _copy_to_lora_param(lora.lora_a, lora_a, key=f"{name}.lora_A")
    lora._olora_tail_initialized = True
    lora.init_lora_weights = _OLORA_TAIL_INIT
    return True


def _init_grouped_lora_olora_tail(
    lora,
    base_module: nn.Module,
    name: str,
    *,
    force: bool,
) -> tuple[bool, str]:
    if lora is None:
        return False, ""
    if getattr(lora, "_olora_tail_initialized", False) and not force:
        raise RuntimeError(
            f"LoRA module {name} already has OLoRA-tail initialization; pass force=True "
            "to reinitialize it explicitly."
        )
    num_local_experts = int(getattr(lora, "num_local_experts", 1))
    if _expert_lora_is_shared(lora):
        if num_local_experts != 1:
            return (
                False,
                "shared routed expert LoRA spans multiple local experts; exact "
                "OLoRA-tail initialization is undefined",
            )
        initialized = _init_linear_lora_olora_tail(
            lora, _grouped_base_weight(base_module, 0), name, force=force
        )
        return initialized, ""

    if lora.lora_a.ndim != 3 or lora.lora_b.ndim != 3:
        raise NotImplementedError(
            f"OLoRA-tail for {name} currently supports grouped LoRA tensors with "
            "shape [experts, rank, in] and [experts, out, rank]."
        )
    if lora.lora_a.shape[0] != num_local_experts or lora.lora_b.shape[0] != num_local_experts:
        raise ValueError(
            f"OLoRA-tail grouped LoRA expert count mismatch for {name}: "
            f"num_local_experts={num_local_experts}, "
            f"lora_A={tuple(lora.lora_a.shape)}, lora_B={tuple(lora.lora_b.shape)}."
        )
    rank = int(lora.lora_a.shape[1])
    expected = (lora.lora_b.shape[1], lora.lora_a.shape[2])
    with torch.no_grad():
        local_lora_b = []
        local_lora_a = []
        for local_idx in range(num_local_experts):
            base_weight = _grouped_base_weight(base_module, local_idx)
            if tuple(base_weight.shape) != expected:
                raise ValueError(
                    f"OLoRA-tail base weight shape mismatch for {name}.expert{local_idx}: "
                    f"got {tuple(base_weight.shape)}, expected {expected} from LoRA B/A shapes."
                )
            lora_b, lora_a = _olora_tail_factors(base_weight, rank)
            local_lora_b.append(lora_b)
            local_lora_a.append(lora_a)
        _copy_to_lora_param(lora.lora_b, torch.stack(local_lora_b, dim=0), key=f"{name}.lora_B")
        _copy_to_lora_param(lora.lora_a, torch.stack(local_lora_a, dim=0), key=f"{name}.lora_A")
    lora._olora_tail_initialized = True
    lora.init_lora_weights = _OLORA_TAIL_INIT
    return True, ""


def initialize_lora_olora_tail(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Initialize supported Qwen3-MoE LoRA modules with OLoRA-tail.

    This follows the same RL-native initializer used by GLM5: LoRA B/A are set
    to the smallest singular-vector subspace of each loaded base weight, without
    singular-value scaling. Qwen3 fused qkv LoRA is initialized against the
    native fused MCore qkv weight; adapter export remains responsible for
    splitting that fused state into PEFT q/k/v keys.
    """

    if ps.tp_size != 1:
        raise NotImplementedError("Qwen3-MoE OLoRA-tail currently supports tp=1.")
    if ps.etp_size != 1:
        raise NotImplementedError("Qwen3-MoE OLoRA-tail currently supports etp=1.")
    del model_cfg
    initialized: list[str] = []
    skipped: list[str] = []

    for chunk in _iter_qwen_chunks(list(chunks)):
        for layer in chunk.layers:
            layer_idx = int(layer.layer_idx)
            attn = layer.attn
            for name, base_module, lora in (
                ("qkv_proj", getattr(attn, "qkv", None), getattr(attn, "qkv_lora", None)),
                ("o_proj", getattr(attn, "proj", None), getattr(attn, "proj_lora", None)),
            ):
                full_name = f"layers.{layer_idx}.self_attn.{name}"
                if lora is not None and _init_linear_lora_olora_tail(
                    lora, _base_weight(base_module), full_name, force=force
                ):
                    initialized.append(full_name)

            experts = layer.moe.experts
            if getattr(experts, "fc1_lora", None) is not None:
                full_name = f"layers.{layer_idx}.mlp.experts.gate_up_proj"
                did_init, skip_reason = _init_grouped_lora_olora_tail(
                    experts.fc1_lora, experts.fc1, full_name, force=force
                )
                if did_init:
                    initialized.append(full_name)
                elif skip_reason:
                    skipped.append(f"{full_name}: {skip_reason}")
            if getattr(experts, "fc2_lora", None) is not None:
                full_name = f"layers.{layer_idx}.mlp.experts.down_proj"
                did_init, skip_reason = _init_grouped_lora_olora_tail(
                    experts.fc2_lora, experts.fc2, full_name, force=force
                )
                if did_init:
                    initialized.append(full_name)
                elif skip_reason:
                    skipped.append(f"{full_name}: {skip_reason}")

    return {
        "init_lora_weights": _OLORA_TAIL_INIT,
        "initialized_modules": len(initialized),
        "initialized_module_names": initialized,
        "skipped_modules": skipped,
        "skipped_reason": (
            "some routed expert LoRA modules could not be initialized exactly"
            if skipped
            else ""
        ),
    }


def export_lora_adapter_state(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    *,
    cpu: bool = True,
) -> dict[str, torch.Tensor]:
    """Export native LoRA tensors to a full PEFT-style adapter state dict.

    All distributed ranks must call this function. Rank 0 returns the full
    adapter state; other ranks return an empty dict.
    """

    _validate_export_parallel_scope(ps)
    _validate_attention_tp(model_cfg, ps)
    state: dict[str, torch.Tensor] = {}
    q_heads_local = model_cfg.num_attention_heads // ps.tp_size
    kv_heads_local = model_cfg.num_key_value_heads // ps.tp_size

    for chunk in _iter_qwen_chunks(list(chunks)):
        for layer in chunk.layers:
            layer_idx = int(layer.layer_idx)
            attn = layer.attn

            if attn.qkv_lora is not None:
                qkv_a_raw = _materialize_export_tensor(
                    attn.qkv_lora.lora_a,
                    name=_attn_key(layer_idx, "q_proj", "lora_A"),
                )
                qkv_b_raw = _materialize_export_tensor(
                    attn.qkv_lora.lora_b,
                    name=(
                        f"{_attn_key(layer_idx, 'q_proj', 'lora_B')}/"
                        f"{_attn_key(layer_idx, 'k_proj', 'lora_B')}/"
                        f"{_attn_key(layer_idx, 'v_proj', 'lora_B')}"
                    ),
                )
                if _is_rank_partitioned_lora_a(attn.qkv_lora, ps):
                    qkv_a = _gather_lora_rank_partition(qkv_a_raw, ps)
                else:
                    qkv_a = _select_tp_replicated(qkv_a_raw, ps)
                q_b_local, k_b_local, v_b_local = _split_local_mcore_qkv_b(
                    qkv_b_raw,
                    num_heads_local=q_heads_local,
                    num_kv_heads_local=kv_heads_local,
                    head_dim=model_cfg.head_dim,
                )
                q_b = _all_gather_cat(q_b_local, ps.tp_group, dim=0)
                k_b = _all_gather_cat(k_b_local, ps.tp_group, dim=0)
                v_b = _all_gather_cat(v_b_local, ps.tp_group, dim=0)
                if _rank() == 0:
                    state[_attn_key(layer_idx, "q_proj", "lora_A")] = qkv_a
                    state[_attn_key(layer_idx, "q_proj", "lora_B")] = q_b
                    state[_attn_key(layer_idx, "k_proj", "lora_A")] = qkv_a.clone()
                    state[_attn_key(layer_idx, "k_proj", "lora_B")] = k_b
                    state[_attn_key(layer_idx, "v_proj", "lora_A")] = qkv_a.clone()
                    state[_attn_key(layer_idx, "v_proj", "lora_B")] = v_b

            if attn.proj_lora is not None:
                proj_a_raw = _materialize_export_tensor(
                    attn.proj_lora.lora_a,
                    name=_attn_key(layer_idx, "o_proj", "lora_A"),
                )
                proj_b_raw = _materialize_export_tensor(
                    attn.proj_lora.lora_b,
                    name=_attn_key(layer_idx, "o_proj", "lora_B"),
                )
                proj_a = _all_gather_cat(proj_a_raw, ps.tp_group, dim=1)
                if _is_output_partitioned_lora_b(attn.proj_lora, ps):
                    proj_b = _all_gather_cat(proj_b_raw, ps.tp_group, dim=0)
                else:
                    proj_b = _select_tp_replicated(proj_b_raw, ps)
                if _rank() == 0:
                    state[_attn_key(layer_idx, "o_proj", "lora_A")] = proj_a
                    state[_attn_key(layer_idx, "o_proj", "lora_B")] = proj_b

            experts = layer.moe.experts
            if experts.fc1_lora is not None:
                fc1_a_local, fc1_b_local = _expand_shared_expert_lora(
                    experts.fc1_lora,
                    experts.num_local_experts,
                    name=f"layers.{layer_idx}.mlp.experts.fc1_lora",
                )
                fc1_a = _all_gather_cat(fc1_a_local, ps.ep_group, dim=0)
                fc1_b = _all_gather_cat(fc1_b_local, ps.ep_group, dim=0)
                gate_b, up_b = fc1_b.chunk(2, dim=1)
                if _rank() == 0:
                    for expert_idx in range(model_cfg.num_experts):
                        state[_expert_key(layer_idx, expert_idx, "gate_proj", "lora_A")] = fc1_a[
                            expert_idx
                        ]
                        state[_expert_key(layer_idx, expert_idx, "gate_proj", "lora_B")] = gate_b[
                            expert_idx
                        ]
                        state[_expert_key(layer_idx, expert_idx, "up_proj", "lora_A")] = fc1_a[
                            expert_idx
                        ].clone()
                        state[_expert_key(layer_idx, expert_idx, "up_proj", "lora_B")] = up_b[
                            expert_idx
                        ]

            if experts.fc2_lora is not None:
                fc2_a_local, fc2_b_local = _expand_shared_expert_lora(
                    experts.fc2_lora,
                    experts.num_local_experts,
                    name=f"layers.{layer_idx}.mlp.experts.fc2_lora",
                )
                fc2_a = _all_gather_cat(fc2_a_local, ps.ep_group, dim=0)
                fc2_b = _all_gather_cat(fc2_b_local, ps.ep_group, dim=0)
                if _rank() == 0:
                    for expert_idx in range(model_cfg.num_experts):
                        state[_expert_key(layer_idx, expert_idx, "down_proj", "lora_A")] = fc2_a[
                            expert_idx
                        ]
                        state[_expert_key(layer_idx, expert_idx, "down_proj", "lora_B")] = fc2_b[
                            expert_idx
                        ]

    if _rank() != 0:
        return {}
    _validate_adapter_state_is_lora_only(state)
    if cpu:
        return {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}
    return {name: tensor.detach().contiguous() for name, tensor in state.items()}


def save_lora_adapter(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    output_dir: str | Path,
    *,
    base_model_name_or_path: str = "",
    lora_config: LoraConfig | Mapping[str, Any] | None = None,
    init_lora_weights: bool | str | None = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a PEFT/Mint-compatible LoRA adapter directory."""

    _validate_export_parallel_scope(ps)
    if lora_config is None:
        raise ValueError("save_lora_adapter requires the LoRA config used to build the model.")
    lora = normalize_lora_config(lora_config)
    if not lora.enabled:
        raise ValueError("save_lora_adapter requires an enabled LoRA config.")
    adapter_metadata = _normalize_user_metadata(metadata)
    init_lora_weights_value = adapter_metadata.pop("init_lora_weights", init_lora_weights)
    _validate_init_lora_weights(init_lora_weights_value)

    from safetensors.torch import save_file

    state = export_lora_adapter_state(chunks, model_cfg, ps, cpu=True)
    _validate_exported_target_modules(state, lora)
    base_model_name_or_path = _validate_base_model_name_or_path(
        base_model_name_or_path, key="save_lora_adapter"
    )
    output = Path(output_dir)
    if _rank() == 0:
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": base_model_name_or_path,
            "inference_mode": False,
            "r": lora.rank,
            "lora_alpha": _json_number(_effective_lora_alpha(lora)),
            "lora_dropout": lora.dropout,
            "use_rslora": lora.use_rslora,
            "target_modules": _target_modules_from_lora_config(lora),
            "bias": "none",
            "fan_in_fan_out": False,
            "init_lora_weights": init_lora_weights_value,
            "modules_to_save": None,
        }
        meta = {
            "format": _ADAPTER_META_FORMAT,
            "base_model_name_or_path": base_model_name_or_path,
            "expert_lora_representation": _expert_lora_representation(chunks),
            "num_tensors": len(state),
            "num_parameters": int(sum(t.numel() for t in state.values())),
            "lora": {
                "rank": lora.rank,
                "alpha": _json_number(_effective_lora_alpha(lora)),
                "dropout": lora.dropout,
                "use_rslora": lora.use_rslora,
                "scaling_convention": (
                    "alpha_over_sqrt_rank" if lora.use_rslora else "alpha_over_rank"
                ),
                "scale": _json_number(lora.scale),
                "target_modules": _target_modules_from_lora_config(lora),
            },
            "parallel": {"tp": ps.tp_size, "ep": ps.ep_size, "etp": ps.etp_size, "pp": ps.pp_size},
            "model": {
                "num_hidden_layers": model_cfg.num_hidden_layers,
                "hidden_size": model_cfg.hidden_size,
                "num_attention_heads": model_cfg.num_attention_heads,
                "num_key_value_heads": model_cfg.num_key_value_heads,
                "head_dim": model_cfg.head_dim,
                "num_experts": model_cfg.num_experts,
                "moe_intermediate_size": model_cfg.moe_intermediate_size,
            },
            "metadata": {
                **adapter_metadata,
                "init_lora_weights": init_lora_weights_value,
            },
        }
        config_text = _json_dumps_standard(config, description="adapter_config.json")
        meta_text = _json_dumps_standard(meta, description="megatron.lite_adapter_meta.json")

        _write_adapter_dir(
            output,
            state=state,
            config_text=config_text,
            meta_text=meta_text,
            save_file=save_file,
        )
        result = {
            "path": str(output),
            "adapter_model": str(output / "adapter_model.safetensors"),
            "adapter_config": str(output / "adapter_config.json"),
            **meta,
        }
    else:
        result = {}
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return result


def _require_tensor(
    state: dict[str, torch.Tensor], key: str, consumed_keys: set[str] | None = None
) -> torch.Tensor:
    try:
        tensor = state[key]
    except KeyError as exc:
        raise KeyError(f"Missing adapter tensor {key!r}") from exc
    if consumed_keys is not None:
        consumed_keys.add(key)
    return tensor


def _require_tensor_shape(
    state: dict[str, torch.Tensor],
    key: str,
    expected_shape: tuple[int, ...],
    consumed_keys: set[str] | None = None,
) -> torch.Tensor:
    tensor = _require_tensor(state, key, consumed_keys)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Adapter tensor {key!r} has shape {tuple(tensor.shape)}, "
            f"expected {expected_shape}."
        )
    return tensor


def _copy_to_lora_param(param: torch.Tensor, value: torch.Tensor, *, key: str) -> None:
    value = value.to(device=param.device, dtype=param.dtype).contiguous()
    try:
        param.data.copy_(value)
        return
    except RuntimeError:
        target = param.data
        mesh = getattr(param, "device_mesh", None) or getattr(target, "device_mesh", None)
        placements = getattr(param, "placements", None) or getattr(target, "placements", None)
        if mesh is None or placements is None:
            raise

    try:
        from torch.distributed.tensor import distribute_tensor
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot load LoRA adapter tensor {key!r} into a DTensor-like parameter "
            "because torch.distributed.tensor is unavailable."
        ) from exc

    try:
        param.data.copy_(distribute_tensor(value, mesh, placements=placements))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load LoRA adapter tensor {key!r} into DTensor-like parameter "
            f"with placements={placements!r}."
        ) from exc


def _slice_tp_output(tensor: torch.Tensor, local_width: int, ps: ParallelState) -> torch.Tensor:
    start = ps.tp_rank * local_width
    return tensor[start : start + local_width].contiguous()


def _slice_tp_input(tensor: torch.Tensor, local_width: int, ps: ParallelState) -> torch.Tensor:
    start = ps.tp_rank * local_width
    return tensor[:, start : start + local_width].contiguous()


def load_lora_adapter_state(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    state: dict[str, torch.Tensor],
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a PEFT adapter state into this rank's native LoRA shards."""

    _validate_import_parallel_scope(ps)
    _validate_attention_tp(model_cfg, ps)
    strict = _strict_bool(strict)
    _validate_adapter_state_is_lora_only(state)
    q_heads_local = model_cfg.num_attention_heads // ps.tp_size
    kv_heads_local = model_cfg.num_key_value_heads // ps.tp_size
    q_width_local = q_heads_local * model_cfg.head_dim
    kv_width_local = kv_heads_local * model_cfg.head_dim
    attn_in_width_local = q_width_local

    loaded = 0
    consumed_keys: set[str] = set()

    def require(key: str) -> torch.Tensor:
        return _require_tensor(state, key, consumed_keys)

    def require_shape(key: str, expected_shape: tuple[int, ...]) -> torch.Tensor:
        return _require_tensor_shape(state, key, expected_shape, consumed_keys)

    for chunk in _iter_qwen_chunks(list(chunks)):
        for layer in chunk.layers:
            layer_idx = int(layer.layer_idx)
            attn = layer.attn
            if attn.qkv_lora is not None:
                qkv_a_shape = tuple(attn.qkv_lora.lora_a.shape)
                if _is_rank_partitioned_lora_a(attn.qkv_lora, ps):
                    qkv_a_shape = (
                        int(getattr(attn.qkv_lora, "rank", attn.qkv_lora.lora_b.shape[1])),
                        attn.qkv_lora.lora_a.shape[1],
                    )
                qkv_b_rank = attn.qkv_lora.lora_b.shape[1]
                q_b_shape = (q_width_local * ps.tp_size, qkv_b_rank)
                kv_b_shape = (kv_width_local * ps.tp_size, qkv_b_rank)
                q_a = require_shape(_attn_key(layer_idx, "q_proj", "lora_A"), qkv_a_shape).to(
                    device=attn.qkv_lora.lora_a.device, dtype=attn.qkv_lora.lora_a.dtype
                )
                k_a = require_shape(_attn_key(layer_idx, "k_proj", "lora_A"), qkv_a_shape).to(q_a)
                v_a = require_shape(_attn_key(layer_idx, "v_proj", "lora_A"), qkv_a_shape).to(q_a)
                if strict and (not torch.equal(q_a, k_a) or not torch.equal(q_a, v_a)):
                    raise ValueError(
                        "Megatron Lite fused qkv_lora requires q/k/v lora_A tensors to match."
                    )
                q_a_local = (
                    _slice_lora_rank_partition(q_a, ps)
                    if _is_rank_partitioned_lora_a(attn.qkv_lora, ps)
                    else q_a.contiguous()
                )
                q_b = _slice_tp_output(
                    require_shape(_attn_key(layer_idx, "q_proj", "lora_B"), q_b_shape).to(
                        device=attn.qkv_lora.lora_b.device, dtype=attn.qkv_lora.lora_b.dtype
                    ),
                    q_width_local,
                    ps,
                )
                k_b = _slice_tp_output(
                    require_shape(_attn_key(layer_idx, "k_proj", "lora_B"), kv_b_shape).to(
                        device=attn.qkv_lora.lora_b.device, dtype=attn.qkv_lora.lora_b.dtype
                    ),
                    kv_width_local,
                    ps,
                )
                v_b = _slice_tp_output(
                    require_shape(_attn_key(layer_idx, "v_proj", "lora_B"), kv_b_shape).to(
                        device=attn.qkv_lora.lora_b.device, dtype=attn.qkv_lora.lora_b.dtype
                    ),
                    kv_width_local,
                    ps,
                )
                _copy_to_lora_param(
                    attn.qkv_lora.lora_a,
                    q_a_local,
                    key=_attn_key(layer_idx, "q_proj", "lora_A"),
                )
                _copy_to_lora_param(
                    attn.qkv_lora.lora_b,
                    _pack_local_mcore_qkv_b(
                        q_b,
                        k_b,
                        v_b,
                        num_heads_local=q_heads_local,
                        num_kv_heads_local=kv_heads_local,
                        head_dim=model_cfg.head_dim,
                    ),
                    key=(
                        f"{_attn_key(layer_idx, 'q_proj', 'lora_B')}/"
                        f"{_attn_key(layer_idx, 'k_proj', 'lora_B')}/"
                        f"{_attn_key(layer_idx, 'v_proj', 'lora_B')}"
                    ),
                )
                loaded += 2

            if attn.proj_lora is not None:
                proj_a_shape = (
                    attn.proj_lora.lora_a.shape[0],
                    attn_in_width_local * ps.tp_size,
                )
                proj_b_shape = (
                    (attn.proj_lora.lora_b.shape[0] * ps.tp_size)
                    if _is_output_partitioned_lora_b(attn.proj_lora, ps)
                    else attn.proj_lora.lora_b.shape[0],
                    attn.proj_lora.lora_b.shape[1],
                )
                proj_a = _slice_tp_input(
                    require_shape(_attn_key(layer_idx, "o_proj", "lora_A"), proj_a_shape).to(
                        device=attn.proj_lora.lora_a.device, dtype=attn.proj_lora.lora_a.dtype
                    ),
                    attn_in_width_local,
                    ps,
                )
                proj_b = require_shape(_attn_key(layer_idx, "o_proj", "lora_B"), proj_b_shape).to(
                    device=attn.proj_lora.lora_b.device, dtype=attn.proj_lora.lora_b.dtype
                )
                if _is_output_partitioned_lora_b(attn.proj_lora, ps):
                    proj_b = _slice_tp_output(proj_b, attn.proj_lora.lora_b.shape[0], ps)
                _copy_to_lora_param(
                    attn.proj_lora.lora_a,
                    proj_a,
                    key=_attn_key(layer_idx, "o_proj", "lora_A"),
                )
                _copy_to_lora_param(
                    attn.proj_lora.lora_b,
                    proj_b,
                    key=_attn_key(layer_idx, "o_proj", "lora_B"),
                )
                loaded += 2

            experts = layer.moe.experts
            expert_start = ps.ep_rank * experts.num_local_experts
            expert_stop = expert_start + experts.num_local_experts
            if experts.fc1_lora is not None:
                if _expert_lora_is_shared(experts.fc1_lora):
                    if experts.fc1_lora.lora_b.shape[0] % 2 != 0:
                        raise ValueError(
                            "Megatron Lite fused shared expert fc1_lora B tensor must have an even "
                            f"output dimension, got {tuple(experts.fc1_lora.lora_b.shape)}."
                        )
                    fc1_a_shape = tuple(experts.fc1_lora.lora_a.shape)
                    fc1_b_shape = (
                        experts.fc1_lora.lora_b.shape[0] // 2,
                        experts.fc1_lora.lora_b.shape[1],
                    )
                    local_gate_a = []
                    local_gate_b = []
                    local_up_b = []
                    for expert_idx in range(expert_start, expert_stop):
                        gate_a = require_shape(
                            _expert_key(layer_idx, expert_idx, "gate_proj", "lora_A"),
                            fc1_a_shape,
                        ).to(
                            device=experts.fc1_lora.lora_a.device,
                            dtype=experts.fc1_lora.lora_a.dtype,
                        )
                        up_a = require_shape(
                            _expert_key(layer_idx, expert_idx, "up_proj", "lora_A"),
                            fc1_a_shape,
                        ).to(gate_a)
                        if strict and not torch.equal(gate_a, up_a):
                            raise ValueError(
                                "Megatron Lite fused fc1_lora requires gate/up lora_A tensors to match."
                            )
                        local_gate_a.append(gate_a)
                        local_gate_b.append(
                            require_shape(
                                _expert_key(layer_idx, expert_idx, "gate_proj", "lora_B"),
                                fc1_b_shape,
                            ).to(
                                device=experts.fc1_lora.lora_b.device,
                                dtype=experts.fc1_lora.lora_b.dtype,
                            )
                        )
                        local_up_b.append(
                            require_shape(
                                _expert_key(layer_idx, expert_idx, "up_proj", "lora_B"),
                                fc1_b_shape,
                            ).to(
                                device=experts.fc1_lora.lora_b.device,
                                dtype=experts.fc1_lora.lora_b.dtype,
                            )
                        )
                    if strict:
                        if any(
                            not torch.equal(local_gate_a[0], value) for value in local_gate_a[1:]
                        ):
                            raise ValueError(
                                "Megatron Lite shared expert fc1_lora can only import PEFT adapters "
                                "whose local expert gate lora_A tensors are identical."
                            )
                        if any(
                            not torch.equal(local_gate_b[0], value) for value in local_gate_b[1:]
                        ):
                            raise ValueError(
                                "Megatron Lite shared expert fc1_lora can only import PEFT adapters "
                                "whose local expert gate lora_B tensors are identical."
                            )
                        if any(not torch.equal(local_up_b[0], value) for value in local_up_b[1:]):
                            raise ValueError(
                                "Megatron Lite shared expert fc1_lora can only import PEFT adapters "
                                "whose local expert up lora_B tensors are identical."
                            )
                    _copy_to_lora_param(
                        experts.fc1_lora.lora_a,
                        local_gate_a[0],
                        key=_expert_key(layer_idx, expert_start, "gate_proj", "lora_A"),
                    )
                    _copy_to_lora_param(
                        experts.fc1_lora.lora_b,
                        torch.cat([local_gate_b[0], local_up_b[0]], dim=0),
                        key=(
                            f"{_expert_key(layer_idx, expert_start, 'gate_proj', 'lora_B')}/"
                            f"{_expert_key(layer_idx, expert_start, 'up_proj', 'lora_B')}"
                        ),
                    )
                    loaded += 2
                else:
                    if experts.fc1_lora.lora_b.shape[1] % 2 != 0:
                        raise ValueError(
                            "Megatron Lite fused expert fc1_lora B tensor must have an even "
                            f"output dimension, got {tuple(experts.fc1_lora.lora_b.shape)}."
                        )
                    fc1_a_shape = tuple(experts.fc1_lora.lora_a.shape[1:])
                    fc1_b_shape = (
                        experts.fc1_lora.lora_b.shape[1] // 2,
                        experts.fc1_lora.lora_b.shape[2],
                    )
                    local_fc1_a = []
                    local_fc1_b = []
                    for expert_idx in range(expert_start, expert_stop):
                        gate_a = require_shape(
                            _expert_key(layer_idx, expert_idx, "gate_proj", "lora_A"),
                            fc1_a_shape,
                        ).to(
                            device=experts.fc1_lora.lora_a.device,
                            dtype=experts.fc1_lora.lora_a.dtype,
                        )
                        up_a = require_shape(
                            _expert_key(layer_idx, expert_idx, "up_proj", "lora_A"),
                            fc1_a_shape,
                        ).to(gate_a)
                        if strict and not torch.equal(gate_a, up_a):
                            raise ValueError(
                                "Megatron Lite fused fc1_lora requires gate/up lora_A tensors to match."
                            )
                        gate_b = require_shape(
                            _expert_key(layer_idx, expert_idx, "gate_proj", "lora_B"),
                            fc1_b_shape,
                        ).to(
                            device=experts.fc1_lora.lora_b.device,
                            dtype=experts.fc1_lora.lora_b.dtype,
                        )
                        up_b = require_shape(
                            _expert_key(layer_idx, expert_idx, "up_proj", "lora_B"),
                            fc1_b_shape,
                        ).to(
                            device=experts.fc1_lora.lora_b.device,
                            dtype=experts.fc1_lora.lora_b.dtype,
                        )
                        local_fc1_a.append(gate_a)
                        local_fc1_b.append(torch.cat([gate_b, up_b], dim=0))
                    _copy_to_lora_param(
                        experts.fc1_lora.lora_a,
                        torch.stack(local_fc1_a, dim=0),
                        key=(
                            f"{_expert_key(layer_idx, expert_start, 'gate_proj', 'lora_A')}.."
                            f"{_expert_key(layer_idx, expert_stop - 1, 'gate_proj', 'lora_A')}"
                        ),
                    )
                    _copy_to_lora_param(
                        experts.fc1_lora.lora_b,
                        torch.stack(local_fc1_b, dim=0),
                        key=(
                            f"{_expert_key(layer_idx, expert_start, 'gate_proj', 'lora_B')}/"
                            f"{_expert_key(layer_idx, expert_start, 'up_proj', 'lora_B')}.."
                            f"{_expert_key(layer_idx, expert_stop - 1, 'gate_proj', 'lora_B')}/"
                            f"{_expert_key(layer_idx, expert_stop - 1, 'up_proj', 'lora_B')}"
                        ),
                    )
                    loaded += 2 * experts.num_local_experts

            if experts.fc2_lora is not None:
                if _expert_lora_is_shared(experts.fc2_lora):
                    fc2_a_shape = tuple(experts.fc2_lora.lora_a.shape)
                    fc2_b_shape = tuple(experts.fc2_lora.lora_b.shape)
                    local_a = []
                    local_b = []
                    for expert_idx in range(expert_start, expert_stop):
                        local_a.append(
                            require_shape(
                                _expert_key(layer_idx, expert_idx, "down_proj", "lora_A"),
                                fc2_a_shape,
                            ).to(
                                device=experts.fc2_lora.lora_a.device,
                                dtype=experts.fc2_lora.lora_a.dtype,
                            )
                        )
                        local_b.append(
                            require_shape(
                                _expert_key(layer_idx, expert_idx, "down_proj", "lora_B"),
                                fc2_b_shape,
                            ).to(
                                device=experts.fc2_lora.lora_b.device,
                                dtype=experts.fc2_lora.lora_b.dtype,
                            )
                        )
                    if strict:
                        if any(not torch.equal(local_a[0], value) for value in local_a[1:]):
                            raise ValueError(
                                "Megatron Lite shared expert fc2_lora can only import PEFT adapters "
                                "whose local expert down lora_A tensors are identical."
                            )
                        if any(not torch.equal(local_b[0], value) for value in local_b[1:]):
                            raise ValueError(
                                "Megatron Lite shared expert fc2_lora can only import PEFT adapters "
                                "whose local expert down lora_B tensors are identical."
                            )
                    _copy_to_lora_param(
                        experts.fc2_lora.lora_a,
                        local_a[0],
                        key=_expert_key(layer_idx, expert_start, "down_proj", "lora_A"),
                    )
                    _copy_to_lora_param(
                        experts.fc2_lora.lora_b,
                        local_b[0],
                        key=_expert_key(layer_idx, expert_start, "down_proj", "lora_B"),
                    )
                    loaded += 2
                else:
                    fc2_a_shape = tuple(experts.fc2_lora.lora_a.shape[1:])
                    fc2_b_shape = tuple(experts.fc2_lora.lora_b.shape[1:])
                    local_fc2_a = []
                    local_fc2_b = []
                    for expert_idx in range(expert_start, expert_stop):
                        local_fc2_a.append(
                            require_shape(
                                _expert_key(layer_idx, expert_idx, "down_proj", "lora_A"),
                                fc2_a_shape,
                            ).to(
                                device=experts.fc2_lora.lora_a.device,
                                dtype=experts.fc2_lora.lora_a.dtype,
                            )
                        )
                        local_fc2_b.append(
                            require_shape(
                                _expert_key(layer_idx, expert_idx, "down_proj", "lora_B"),
                                fc2_b_shape,
                            ).to(
                                device=experts.fc2_lora.lora_b.device,
                                dtype=experts.fc2_lora.lora_b.dtype,
                            )
                        )
                    _copy_to_lora_param(
                        experts.fc2_lora.lora_a,
                        torch.stack(local_fc2_a, dim=0),
                        key=(
                            f"{_expert_key(layer_idx, expert_start, 'down_proj', 'lora_A')}.."
                            f"{_expert_key(layer_idx, expert_stop - 1, 'down_proj', 'lora_A')}"
                        ),
                    )
                    _copy_to_lora_param(
                        experts.fc2_lora.lora_b,
                        torch.stack(local_fc2_b, dim=0),
                        key=(
                            f"{_expert_key(layer_idx, expert_start, 'down_proj', 'lora_B')}.."
                            f"{_expert_key(layer_idx, expert_stop - 1, 'down_proj', 'lora_B')}"
                        ),
                    )
                    loaded += 2 * experts.num_local_experts

    if strict:
        unexpected = sorted(set(state) - consumed_keys)
        if unexpected:
            preview = ", ".join(repr(key) for key in unexpected[:8])
            suffix = "" if len(unexpected) <= 8 else f", ... and {len(unexpected) - 8} more"
            raise ValueError(f"Unexpected adapter tensor keys: {preview}{suffix}.")

    return {"loaded_tensors": loaded}


def load_lora_adapter(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    adapter_dir: str | Path,
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    *,
    strict: bool = True,
    lora_config: LoraConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_import_parallel_scope(ps)
    _validate_attention_tp(model_cfg, ps)
    strict = _strict_bool(strict)

    adapter_root = _validate_adapter_root(adapter_dir)
    config_path = adapter_root / "adapter_config.json"
    adapter_config = None
    if config_path.exists():
        adapter_config = _load_json_object(config_path, description="adapter_config.json")
    elif lora_config is None:
        raise ValueError(
            "load_lora_adapter requires caller-provided lora_config when "
            "adapter_config.json is missing."
        )
    metadata_path = adapter_root / "megatron.lite_adapter_meta.json"
    adapter_meta = None
    if metadata_path.exists():
        adapter_meta = _load_json_object(
            metadata_path, description="megatron.lite_adapter_meta.json"
        )

    from safetensors.torch import load_file

    path = _validate_adapter_model_path(adapter_root)
    state = load_file(str(path), device="cpu")
    _validate_adapter_state_is_lora_only(state)
    if adapter_config is not None:
        _validate_adapter_config(chunks, state, adapter_config, lora_config=lora_config)
    else:
        _validate_expected_lora_config(chunks, state, lora_config)
    if adapter_meta is not None:
        _validate_adapter_metadata(adapter_meta, chunks, model_cfg, ps, state, lora_config)
    if adapter_config is not None and adapter_meta is not None:
        _validate_adapter_config_metadata_consistency(adapter_config, adapter_meta)
    return load_lora_adapter_state(chunks, state, model_cfg, ps, strict=strict)


__all__ = [
    "export_lora_adapter_state",
    "initialize_lora_olora_tail",
    "load_lora_adapter",
    "load_lora_adapter_state",
    "save_lora_adapter",
]
