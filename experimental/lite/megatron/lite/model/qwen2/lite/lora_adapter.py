# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PEFT adapter import/export for dense Qwen2 lite native LoRA.

This is the dense-Qwen2 exact-route lifecycle slice for TP=1/PP=1/CP=1/ETP=1.
It exports fused native LoRA modules to standard Qwen2 PEFT target keys:
``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``, ``gate_proj``, ``up_proj``,
and ``down_proj``.
"""

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

from megatron.lite.model.qwen2.config import Qwen2Config
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
_ADAPTER_META_FORMAT = "megatron.lite_qwen2_lora_peft_v1"
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
    "intermediate_size",
    "vocab_size",
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
_PARALLEL_METADATA_FIELDS = ("tp", "ep", "etp", "pp", "cp")


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


_WRAPPED_MODULE_ATTRS = ("module", "_module", "wrapped_module", "_fsdp_wrapped_module")


def _unwrap_model(module: Any) -> Any:
    current = module
    seen: set[int] = set()
    for _ in range(8):
        ident = id(current)
        if ident in seen:
            break
        seen.add(ident)
        inner_model = getattr(current, "model", None)
        if inner_model is not None and hasattr(inner_model, "layers"):
            return inner_model
        if hasattr(current, "layers"):
            return current
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


def _iter_qwen2_layers(chunks: list[Any] | tuple[Any, ...]):
    for chunk in chunks:
        model = _unwrap_model(chunk)
        for idx, layer in enumerate(getattr(model, "layers", ())):
            yield int(getattr(layer, "layer_idx", idx)), layer


def _layer_prefix(layer_idx: int) -> str:
    return f"{_PEFT_PREFIX}.layers.{layer_idx}"


def _attn_key(layer_idx: int, module: str, suffix: str) -> str:
    return f"{_layer_prefix(layer_idx)}.self_attn.{module}.{suffix}.weight"


def _mlp_key(layer_idx: int, module: str, suffix: str) -> str:
    return f"{_layer_prefix(layer_idx)}.mlp.{module}.{suffix}.weight"


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
    if len(parts) != 3 or not parts[0].isdigit():
        return False
    _, block, module = parts
    return (
        (block == "self_attn" and module in {"q_proj", "k_proj", "v_proj", "o_proj"})
        or (block == "mlp" and module in {"gate_proj", "up_proj", "down_proj"})
    )


def _validate_adapter_state_is_lora_only(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"LoRA adapter state must be a mapping, got {type(state)!r}.")
    if not state:
        raise ValueError("LoRA adapter state must contain at least one LoRA tensor.")
    bad_keys = [key for key in state if not isinstance(key, str) or not _has_lora_tensor_suffix(key)]
    if bad_keys:
        raise ValueError(
            "LoRA adapter state must contain only lora_A/lora_B tensors; "
            f"bad keys={bad_keys[:5]!r}."
        )
    unsupported = [
        key for key in state if isinstance(key, str) and not _is_supported_adapter_tensor_key(key)
    ]
    if unsupported:
        raise ValueError(
            "LoRA adapter state contains unsupported Qwen2 adapter tensor keys: "
            f"{unsupported[:5]!r}."
        )
    non_tensors = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise TypeError(f"LoRA adapter state values must be tensors; bad keys={non_tensors[:5]!r}.")


def _state_target_modules(state: Mapping[str, torch.Tensor]) -> set[str]:
    targets = set()
    for key in state:
        stem = _strip_lora_tensor_suffix(key)
        if stem is None:
            continue
        parts = stem.split(".")
        if parts:
            targets.add(parts[-1])
    return targets


def _dedupe_target_modules(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_targets = (value,)
    elif isinstance(value, Mapping):
        raise TypeError("LoRA target_modules must be a string or sequence of strings.")
    else:
        try:
            raw_targets = tuple(value)
        except TypeError as exc:
            raise TypeError(
                "LoRA target_modules must be a string or sequence of strings."
            ) from exc
    out: list[str] = []
    for target in raw_targets:
        if not isinstance(target, str):
            raise TypeError(
                "LoRA target_modules entries must be strings, "
                f"got {type(target)!r}."
            )
        if not target.strip():
            raise ValueError("LoRA target_modules entries must be non-empty strings.")
        expanded = _TARGET_MODULE_EXPANSIONS.get(target, (target,))
        for item in expanded:
            if item not in _ADAPTER_TARGET_MODULES:
                raise ValueError(f"Unsupported dense Qwen2 LoRA target module: {item!r}.")
            if item not in out:
                out.append(item)
    return out


def _target_modules_from_lora_config(lora: LoraConfig) -> list[str]:
    return _dedupe_target_modules(lora.target_modules)


def _json_number(value: float | int) -> int | float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"LoRA numeric metadata must be finite, got {value}.")
    return int(value) if value.is_integer() else value


def _effective_lora_alpha(config: LoraConfig) -> float:
    return effective_lora_alpha(config)


def _numbers_match(left: float | int, right: float | int) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)


def _materialize_export_tensor(
    tensor: torch.Tensor, *, name: str, expected_ndim: int = 2
) -> torch.Tensor:
    value = tensor.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    value = value.contiguous()
    if value.ndim != expected_ndim:
        raise ValueError(
            f"LoRA tensor {name} must be {expected_ndim}-D, got shape {tuple(value.shape)}."
        )
    return value


def _base_weight(module: nn.Module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    inner = getattr(module, "linear", None)
    weight = getattr(inner, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    raise TypeError(f"Cannot locate base weight on module {type(module)!r}.")


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
            f"Cannot load LoRA adapter tensor {key!r} into a DTensor-like parameter "
            f"with placements={placements!r}."
        ) from exc


def _validate_parallel_scope(ps: ParallelState) -> None:
    unsupported = {
        "tp": ps.tp_size,
        "ep": ps.ep_size,
        "etp": ps.etp_size,
        "pp": ps.pp_size,
        "cp": ps.cp_size,
    }
    bad = {name: value for name, value in unsupported.items() if int(value) != 1}
    if bad:
        raise NotImplementedError(
            "Dense Qwen2 LoRA adapter lifecycle currently supports only "
            f"TP/EP/ETP/PP/CP=1, got {bad}."
        )


def _strict_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y", "on"):
            return True
        if lowered in ("false", "0", "no", "n", "off"):
            return False
    raise TypeError(f"strict must be a boolean or boolean string, got {value!r}.")


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
                    f"Adapter metadata keys must be strings; {path} contains "
                    f"key {key!r} of type {type(key)!r}."
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
        raise TypeError(f"Adapter metadata must be an object, got {type(metadata)!r}.")
    _validate_metadata_json_keys(metadata)
    metadata = _copy_json_metadata_value(metadata)
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("Adapter metadata must be JSON-serializable.") from exc
    return metadata


def _validate_base_model_name_or_path(value: Any, *, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{key} base_model_name_or_path must be a string, got {type(value)!r}.")
    value = value.strip()
    if not value:
        raise ValueError(f"{key} base_model_name_or_path must be non-empty.")
    return value


def _adapter_bool(value: Any, *, default: bool, key: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise TypeError(f"Adapter config {key} must be a boolean, got {type(value)!r}.")


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


def _iter_native_lora_modules(chunks: list[Any] | tuple[Any, ...]):
    for _, layer in _iter_qwen2_layers(chunks):
        attn = layer.self_attn
        for attr in ("qkv_lora", "proj_lora"):
            module = getattr(attn, attr, None)
            if module is not None:
                yield module
        mlp = layer.mlp
        for attr in ("gate_up_lora", "down_lora"):
            module = getattr(mlp, attr, None)
            if module is not None:
                yield module


def _infer_native_alpha(chunks: list[Any] | tuple[Any, ...]) -> float | None:
    alphas: set[float] = set()
    for module in _iter_native_lora_modules(chunks):
        rank = getattr(module, "rank", None)
        scale = getattr(module, "scale", None)
        if rank is None or scale is None:
            continue
        denominator = math.sqrt(float(rank)) if getattr(module, "use_rslora", False) else float(rank)
        alphas.add(float(scale) * denominator)
    if not alphas:
        return None
    rounded = {round(alpha, 6) for alpha in alphas}
    if len(rounded) != 1:
        raise ValueError(
            f"Native LoRA modules have inconsistent effective alpha values: {sorted(alphas)}."
        )
    return next(iter(alphas))


def _infer_native_use_rslora(chunks: list[Any] | tuple[Any, ...]) -> bool | None:
    values = {
        bool(getattr(module, "use_rslora", False)) for module in _iter_native_lora_modules(chunks)
    }
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("Native LoRA modules have inconsistent use_rslora settings.")
    return next(iter(values))


def _infer_native_dropout(chunks: list[Any] | tuple[Any, ...]) -> float | None:
    values = {float(getattr(module, "dropout_p", 0.0)) for module in _iter_native_lora_modules(chunks)}
    if not values:
        return None
    rounded = {round(value, 6) for value in values}
    if len(rounded) != 1:
        raise ValueError(f"Native LoRA modules have inconsistent dropout values: {sorted(values)}.")
    return next(iter(values))


def _infer_state_rank(state: Mapping[str, torch.Tensor]) -> int | None:
    ranks: set[int] = set()
    for key, tensor in state.items():
        if tensor.ndim < 2:
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


def _validate_adapter_only_config(adapter_config: dict[str, Any]) -> None:
    task_type = adapter_config.get("task_type")
    if task_type is not None:
        if not isinstance(task_type, str):
            raise TypeError(f"Adapter config task_type must be a string, got {type(task_type)!r}.")
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

    if _adapter_bool(adapter_config.get("fan_in_fan_out"), default=False, key="fan_in_fan_out"):
        raise ValueError("Adapter config fan_in_fan_out=True is not supported.")
    if adapter_config.get("modules_to_save") not in (None, [], ()):
        raise ValueError("Adapter config modules_to_save is not supported for adapter-only LoRA.")


def _validate_adapter_config(
    chunks: list[Any] | tuple[Any, ...],
    state: Mapping[str, torch.Tensor],
    adapter_config: dict[str, Any],
    *,
    lora_config: LoraConfig | Mapping[str, Any] | None = None,
) -> None:
    state_rank = _infer_state_rank(state)
    state_targets = _state_target_modules(state)

    peft_type = adapter_config.get("peft_type")
    if peft_type is None:
        raise ValueError("Adapter config peft_type='LORA' is required.")
    if not isinstance(peft_type, str) or peft_type.upper() != "LORA":
        raise ValueError(f"Expected PEFT adapter_config peft_type='LORA', got {peft_type!r}.")
    _validate_adapter_only_config(adapter_config)

    _validate_base_model_name_or_path(
        adapter_config.get("base_model_name_or_path"), key="Adapter config"
    )
    config_rank = validate_lora_rank_value(adapter_config.get("r"), key="Adapter config r")
    if state_rank is not None and config_rank != state_rank:
        raise ValueError(
            f"Adapter config rank r={config_rank} does not match tensor rank {state_rank}."
        )
    config_targets = set(_dedupe_target_modules(adapter_config.get("target_modules")))
    if config_targets != state_targets:
        raise ValueError(
            "Adapter config target_modules do not match adapter tensors: "
            f"config={sorted(config_targets)}, tensors={sorted(state_targets)}."
        )
    config_alpha = validate_lora_number_value(
        adapter_config.get("lora_alpha"), key="Adapter config lora_alpha"
    )
    native_alpha = _infer_native_alpha(chunks)
    if native_alpha is not None and not _numbers_match(config_alpha, native_alpha):
        raise ValueError(
            f"Adapter config lora_alpha={config_alpha} does not match native model alpha={native_alpha}."
        )
    config_dropout = validate_lora_dropout_value(
        adapter_config.get("lora_dropout"), key="Adapter config lora_dropout"
    )
    native_dropout = _infer_native_dropout(chunks)
    if native_dropout is not None and not _numbers_match(config_dropout, native_dropout):
        raise ValueError(
            "Adapter config lora_dropout="
            f"{config_dropout} does not match native model dropout={native_dropout}."
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
    if "init_lora_weights" in adapter_config:
        _validate_init_lora_weights(adapter_config.get("init_lora_weights"))

    if lora_config is not None:
        expected = normalize_lora_config(lora_config)
        expected_targets = set(_target_modules_from_lora_config(expected))
        if config_rank != expected.rank:
            raise ValueError(
                f"Adapter config rank r={config_rank} does not match expected rank {expected.rank}."
            )
        if not _numbers_match(config_alpha, _effective_lora_alpha(expected)):
            raise ValueError(
                "Adapter config lora_alpha="
                f"{config_alpha} does not match expected alpha {_effective_lora_alpha(expected)}."
            )
        if config_dropout != expected.dropout:
            raise ValueError(
                f"Adapter config lora_dropout={config_dropout} does not match expected dropout={expected.dropout}."
            )
        if config_use_rslora != expected.use_rslora:
            raise ValueError(
                "Adapter config use_rslora="
                f"{config_use_rslora} does not match expected use_rslora={expected.use_rslora}."
            )
        if config_targets != expected_targets:
            raise ValueError(
                "Adapter config target_modules do not match expected LoRA config: "
                f"config={sorted(config_targets)}, expected={sorted(expected_targets)}."
            )


def _validate_expected_lora_config(
    state: Mapping[str, torch.Tensor],
    lora_config: LoraConfig | Mapping[str, Any] | None,
) -> None:
    if lora_config is None:
        raise ValueError("load_lora_adapter requires lora_config when adapter_config.json is missing.")
    expected = normalize_lora_config(lora_config)
    if not expected.enabled:
        raise ValueError("Expected LoRA config must be enabled for adapter load.")
    state_rank = _infer_state_rank(state)
    if state_rank != expected.rank:
        raise ValueError(f"Adapter tensor rank {state_rank} does not match expected rank {expected.rank}.")
    state_targets = _state_target_modules(state)
    expected_targets = set(_target_modules_from_lora_config(expected))
    if state_targets != expected_targets:
        raise ValueError(
            "Adapter tensor target_modules do not match expected LoRA config: "
            f"tensors={sorted(state_targets)}, expected={sorted(expected_targets)}."
        )


def _validate_adapter_metadata(
    adapter_meta: dict[str, Any],
    model_cfg: Qwen2Config,
    ps: ParallelState,
    state: Mapping[str, torch.Tensor],
    lora_config: LoraConfig | Mapping[str, Any] | None,
) -> None:
    required_fields = (
        "format",
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
            raise ValueError(f"Adapter metadata {field} is required.")
    if adapter_meta["format"] != _ADAPTER_META_FORMAT:
        raise ValueError(
            f"Adapter metadata format={adapter_meta['format']!r} does not match {_ADAPTER_META_FORMAT!r}."
        )
    _validate_base_model_name_or_path(adapter_meta["base_model_name_or_path"], key="Adapter metadata")
    if adapter_meta["expert_lora_representation"] != "dense":
        raise ValueError("Dense Qwen2 adapter metadata expert_lora_representation must be 'dense'.")
    if not isinstance(adapter_meta["num_tensors"], int) or isinstance(adapter_meta["num_tensors"], bool):
        raise TypeError("Adapter metadata num_tensors must be an integer.")
    if not isinstance(adapter_meta["num_parameters"], int) or isinstance(
        adapter_meta["num_parameters"], bool
    ):
        raise TypeError("Adapter metadata num_parameters must be an integer.")
    if adapter_meta["num_tensors"] != len(state):
        raise ValueError("Adapter metadata num_tensors does not match adapter_model tensors.")
    if adapter_meta["num_parameters"] != int(sum(t.numel() for t in state.values())):
        raise ValueError("Adapter metadata num_parameters does not match adapter_model tensors.")

    parallel = adapter_meta["parallel"]
    if not isinstance(parallel, dict):
        raise TypeError("Adapter metadata parallel must be a JSON object.")
    expected_parallel = {"tp": ps.tp_size, "ep": ps.ep_size, "etp": ps.etp_size, "pp": ps.pp_size, "cp": ps.cp_size}
    for key in _PARALLEL_METADATA_FIELDS:
        value = parallel.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"Adapter metadata parallel.{key} must be an integer.")
        if value != expected_parallel[key]:
            raise ValueError(
                f"Adapter metadata parallel.{key}={value} does not match runtime {expected_parallel[key]}."
            )

    model = adapter_meta["model"]
    if not isinstance(model, dict):
        raise TypeError("Adapter metadata model must be a JSON object.")
    for key in _MODEL_METADATA_FIELDS:
        value = model.get(key)
        actual = getattr(model_cfg, key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"Adapter metadata model.{key} must be an integer.")
        if value != actual:
            raise ValueError(
                f"Adapter metadata model.{key}={value} does not match model config {actual}."
            )

    lora = adapter_meta["lora"]
    if not isinstance(lora, dict):
        raise TypeError("Adapter metadata lora must be a JSON object.")
    for key in _LORA_METADATA_FIELDS:
        if key not in lora:
            raise ValueError(f"Adapter metadata lora.{key} is required.")
    expected = normalize_lora_config(lora_config) if lora_config is not None else None
    rank = validate_lora_rank_value(lora["rank"], key="Adapter metadata lora.rank")
    state_rank = _infer_state_rank(state)
    if state_rank is not None and rank != state_rank:
        raise ValueError("Adapter metadata lora.rank does not match adapter tensors.")
    if expected is not None and rank != expected.rank:
        raise ValueError("Adapter metadata lora.rank does not match expected LoRA config.")
    targets = set(_dedupe_target_modules(lora["target_modules"]))
    if targets != _state_target_modules(state):
        raise ValueError("Adapter metadata lora.target_modules does not match adapter tensors.")
    if expected is not None and targets != set(_target_modules_from_lora_config(expected)):
        raise ValueError("Adapter metadata lora.target_modules does not match expected LoRA config.")
    validate_lora_number_value(lora["alpha"], key="Adapter metadata lora.alpha")
    validate_lora_dropout_value(lora["dropout"], key="Adapter metadata lora.dropout")
    if not isinstance(lora["use_rslora"], bool):
        raise TypeError("Adapter metadata lora.use_rslora must be a boolean.")
    if not isinstance(lora["scaling_convention"], str):
        raise TypeError("Adapter metadata lora.scaling_convention must be a string.")
    validate_lora_number_value(lora["scale"], key="Adapter metadata lora.scale")
    if not isinstance(adapter_meta["metadata"], dict):
        raise TypeError("Adapter metadata metadata must be a JSON object.")


def _validate_adapter_config_metadata_consistency(
    adapter_config: dict[str, Any], adapter_meta: dict[str, Any]
) -> None:
    config_base = _validate_base_model_name_or_path(
        adapter_config["base_model_name_or_path"], key="Adapter config"
    )
    meta_base = _validate_base_model_name_or_path(
        adapter_meta["base_model_name_or_path"], key="Adapter metadata"
    )
    if config_base != meta_base:
        raise ValueError(
            "Adapter config base_model_name_or_path does not match Megatron Lite metadata."
        )
    adapter_lora = adapter_meta.get("lora", {})
    if "r" in adapter_config and "rank" in adapter_lora:
        config_rank = validate_lora_rank_value(adapter_config["r"], key="Adapter config r")
        meta_rank = validate_lora_rank_value(adapter_lora["rank"], key="Adapter metadata lora.rank")
        if config_rank != meta_rank:
            raise ValueError("Adapter config rank does not match Megatron Lite metadata rank.")
    config_targets = set(_dedupe_target_modules(adapter_config["target_modules"]))
    meta_targets = set(_dedupe_target_modules(adapter_lora["target_modules"]))
    if config_targets != meta_targets:
        raise ValueError(
            "Adapter config target_modules do not match Megatron Lite metadata target_modules."
        )


def _require_tensor(
    state: Mapping[str, torch.Tensor], key: str, consumed_keys: set[str] | None = None
) -> torch.Tensor:
    try:
        tensor = state[key]
    except KeyError as exc:
        raise KeyError(f"Missing adapter tensor {key!r}") from exc
    if consumed_keys is not None:
        consumed_keys.add(key)
    return tensor


def _require_tensor_shape(
    state: Mapping[str, torch.Tensor],
    key: str,
    expected_shape: tuple[int, ...],
    consumed_keys: set[str] | None = None,
) -> torch.Tensor:
    tensor = _require_tensor(state, key, consumed_keys)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Adapter tensor {key!r} has shape {tuple(tensor.shape)}, expected {expected_shape}."
        )
    return tensor


def export_lora_adapter_state(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    model_cfg: Qwen2Config,
    ps: ParallelState,
    *,
    cpu: bool = True,
) -> dict[str, torch.Tensor]:
    """Export native Qwen2 LoRA tensors to a PEFT-style adapter state dict."""

    _validate_parallel_scope(ps)
    state: dict[str, torch.Tensor] = {}
    for layer_idx, layer in _iter_qwen2_layers(chunks):
        attn = layer.self_attn
        if attn.qkv_lora is not None:
            qkv_a = _materialize_export_tensor(
                attn.qkv_lora.lora_a, name=_attn_key(layer_idx, "q_proj", "lora_A")
            )
            q_b, k_b, v_b = _materialize_export_tensor(
                attn.qkv_lora.lora_b,
                name=(
                    f"{_attn_key(layer_idx, 'q_proj', 'lora_B')}/"
                    f"{_attn_key(layer_idx, 'k_proj', 'lora_B')}/"
                    f"{_attn_key(layer_idx, 'v_proj', 'lora_B')}"
                ),
            ).split([attn.q_size, attn.kv_size, attn.kv_size], dim=0)
            state[_attn_key(layer_idx, "q_proj", "lora_A")] = qkv_a
            state[_attn_key(layer_idx, "q_proj", "lora_B")] = q_b.contiguous()
            state[_attn_key(layer_idx, "k_proj", "lora_A")] = qkv_a.clone()
            state[_attn_key(layer_idx, "k_proj", "lora_B")] = k_b.contiguous()
            state[_attn_key(layer_idx, "v_proj", "lora_A")] = qkv_a.clone()
            state[_attn_key(layer_idx, "v_proj", "lora_B")] = v_b.contiguous()

        if attn.proj_lora is not None:
            state[_attn_key(layer_idx, "o_proj", "lora_A")] = _materialize_export_tensor(
                attn.proj_lora.lora_a, name=_attn_key(layer_idx, "o_proj", "lora_A")
            )
            state[_attn_key(layer_idx, "o_proj", "lora_B")] = _materialize_export_tensor(
                attn.proj_lora.lora_b, name=_attn_key(layer_idx, "o_proj", "lora_B")
            )

        mlp = layer.mlp
        if mlp.gate_up_lora is not None:
            gate_up_a = _materialize_export_tensor(
                mlp.gate_up_lora.lora_a, name=_mlp_key(layer_idx, "gate_proj", "lora_A")
            )
            gate_b, up_b = _materialize_export_tensor(
                mlp.gate_up_lora.lora_b,
                name=(
                    f"{_mlp_key(layer_idx, 'gate_proj', 'lora_B')}/"
                    f"{_mlp_key(layer_idx, 'up_proj', 'lora_B')}"
                ),
            ).chunk(2, dim=0)
            state[_mlp_key(layer_idx, "gate_proj", "lora_A")] = gate_up_a
            state[_mlp_key(layer_idx, "gate_proj", "lora_B")] = gate_b.contiguous()
            state[_mlp_key(layer_idx, "up_proj", "lora_A")] = gate_up_a.clone()
            state[_mlp_key(layer_idx, "up_proj", "lora_B")] = up_b.contiguous()

        if mlp.down_lora is not None:
            state[_mlp_key(layer_idx, "down_proj", "lora_A")] = _materialize_export_tensor(
                mlp.down_lora.lora_a, name=_mlp_key(layer_idx, "down_proj", "lora_A")
            )
            state[_mlp_key(layer_idx, "down_proj", "lora_B")] = _materialize_export_tensor(
                mlp.down_lora.lora_b, name=_mlp_key(layer_idx, "down_proj", "lora_B")
            )

    if _rank() != 0:
        return {}
    _validate_adapter_state_is_lora_only(state)
    if cpu:
        return {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}
    return {name: tensor.detach().contiguous() for name, tensor in state.items()}


def initialize_lora_olora_tail(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    model_cfg: Qwen2Config,
    ps: ParallelState,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Initialize dense Qwen2 LoRA modules from base-weight tail singular vectors."""

    del model_cfg
    _validate_parallel_scope(ps)
    initialized: list[str] = []
    for layer_idx, layer in _iter_qwen2_layers(chunks):
        attn = layer.self_attn
        for name, base_module, lora in (
            ("qkv_proj", getattr(attn, "qkv", None), getattr(attn, "qkv_lora", None)),
            ("o_proj", getattr(attn, "proj", None), getattr(attn, "proj_lora", None)),
        ):
            full_name = f"layers.{layer_idx}.self_attn.{name}"
            if lora is not None and _init_linear_lora_olora_tail(
                lora, _base_weight(base_module), full_name, force=force
            ):
                initialized.append(full_name)

        mlp = layer.mlp
        for name, base_module, lora in (
            ("gate_up_proj", getattr(mlp, "gate_up", None), getattr(mlp, "gate_up_lora", None)),
            ("down_proj", getattr(mlp, "down", None), getattr(mlp, "down_lora", None)),
        ):
            full_name = f"layers.{layer_idx}.mlp.{name}"
            if lora is not None and _init_linear_lora_olora_tail(
                lora, _base_weight(base_module), full_name, force=force
            ):
                initialized.append(full_name)

    return {
        "init_lora_weights": _OLORA_TAIL_INIT,
        "initialized_modules": len(initialized),
        "initialized_module_names": initialized,
        "skipped_modules": [],
        "skipped_reason": "",
    }


def _validate_exported_target_modules(state: Mapping[str, torch.Tensor], lora: LoraConfig) -> None:
    actual = _state_target_modules(state)
    expected = set(_target_modules_from_lora_config(lora))
    if actual != expected:
        raise ValueError(
            "Exported dense Qwen2 adapter target modules do not match LoRA config: "
            f"actual={sorted(actual)}, expected={sorted(expected)}."
        )


def save_lora_adapter(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    model_cfg: Qwen2Config,
    ps: ParallelState,
    output_dir: str | Path,
    *,
    base_model_name_or_path: str = "",
    lora_config: LoraConfig | Mapping[str, Any] | None = None,
    init_lora_weights: bool | str | None = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a PEFT-compatible dense Qwen2 LoRA adapter directory."""

    _validate_parallel_scope(ps)
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
        target_modules = _target_modules_from_lora_config(lora)
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": base_model_name_or_path,
            "inference_mode": False,
            "r": lora.rank,
            "lora_alpha": _json_number(_effective_lora_alpha(lora)),
            "lora_dropout": lora.dropout,
            "use_rslora": lora.use_rslora,
            "target_modules": target_modules,
            "bias": "none",
            "fan_in_fan_out": False,
            "init_lora_weights": init_lora_weights_value,
            "modules_to_save": None,
        }
        meta = {
            "format": _ADAPTER_META_FORMAT,
            "base_model_name_or_path": base_model_name_or_path,
            "expert_lora_representation": "dense",
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
                "target_modules": target_modules,
            },
            "parallel": {
                "tp": ps.tp_size,
                "ep": ps.ep_size,
                "etp": ps.etp_size,
                "pp": ps.pp_size,
                "cp": ps.cp_size,
            },
            "model": {
                "num_hidden_layers": model_cfg.num_hidden_layers,
                "hidden_size": model_cfg.hidden_size,
                "num_attention_heads": model_cfg.num_attention_heads,
                "num_key_value_heads": model_cfg.num_key_value_heads,
                "head_dim": model_cfg.head_dim,
                "intermediate_size": model_cfg.intermediate_size,
                "vocab_size": model_cfg.vocab_size,
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


def load_lora_adapter_state(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    state: dict[str, torch.Tensor],
    model_cfg: Qwen2Config,
    ps: ParallelState,
    *,
    strict: bool | str = True,
) -> dict[str, Any]:
    """Load a PEFT dense Qwen2 adapter state into native fused LoRA modules."""

    del model_cfg
    _validate_parallel_scope(ps)
    strict = _strict_bool(strict)
    _validate_adapter_state_is_lora_only(state)
    loaded = 0
    consumed_keys: set[str] = set()

    for layer_idx, layer in _iter_qwen2_layers(chunks):
        attn = layer.self_attn
        if attn.qkv_lora is not None:
            qkv_a_shape = tuple(attn.qkv_lora.lora_a.shape)
            q_a = _require_tensor_shape(
                state, _attn_key(layer_idx, "q_proj", "lora_A"), qkv_a_shape, consumed_keys
            ).to(device=attn.qkv_lora.lora_a.device, dtype=attn.qkv_lora.lora_a.dtype)
            k_a = _require_tensor_shape(
                state, _attn_key(layer_idx, "k_proj", "lora_A"), qkv_a_shape, consumed_keys
            ).to(q_a)
            v_a = _require_tensor_shape(
                state, _attn_key(layer_idx, "v_proj", "lora_A"), qkv_a_shape, consumed_keys
            ).to(q_a)
            if strict and (not torch.equal(q_a, k_a) or not torch.equal(q_a, v_a)):
                raise ValueError(
                    "Dense Qwen2 fused qkv_lora requires q/k/v lora_A tensors to match."
                )
            q_rank = attn.qkv_lora.lora_b.shape[1]
            q_b = _require_tensor_shape(
                state,
                _attn_key(layer_idx, "q_proj", "lora_B"),
                (attn.q_size, q_rank),
                consumed_keys,
            ).to(device=attn.qkv_lora.lora_b.device, dtype=attn.qkv_lora.lora_b.dtype)
            k_b = _require_tensor_shape(
                state,
                _attn_key(layer_idx, "k_proj", "lora_B"),
                (attn.kv_size, q_rank),
                consumed_keys,
            ).to(q_b)
            v_b = _require_tensor_shape(
                state,
                _attn_key(layer_idx, "v_proj", "lora_B"),
                (attn.kv_size, q_rank),
                consumed_keys,
            ).to(q_b)
            _copy_to_lora_param(
                attn.qkv_lora.lora_a,
                q_a.contiguous(),
                key=_attn_key(layer_idx, "q_proj", "lora_A"),
            )
            _copy_to_lora_param(
                attn.qkv_lora.lora_b,
                torch.cat([q_b, k_b, v_b], dim=0).contiguous(),
                key=(
                    f"{_attn_key(layer_idx, 'q_proj', 'lora_B')}/"
                    f"{_attn_key(layer_idx, 'k_proj', 'lora_B')}/"
                    f"{_attn_key(layer_idx, 'v_proj', 'lora_B')}"
                ),
            )
            loaded += 2

        if attn.proj_lora is not None:
            proj_a = _require_tensor_shape(
                state,
                _attn_key(layer_idx, "o_proj", "lora_A"),
                tuple(attn.proj_lora.lora_a.shape),
                consumed_keys,
            )
            proj_b = _require_tensor_shape(
                state,
                _attn_key(layer_idx, "o_proj", "lora_B"),
                tuple(attn.proj_lora.lora_b.shape),
                consumed_keys,
            )
            _copy_to_lora_param(attn.proj_lora.lora_a, proj_a, key=_attn_key(layer_idx, "o_proj", "lora_A"))
            _copy_to_lora_param(attn.proj_lora.lora_b, proj_b, key=_attn_key(layer_idx, "o_proj", "lora_B"))
            loaded += 2

        mlp = layer.mlp
        if mlp.gate_up_lora is not None:
            gate_up_a_shape = tuple(mlp.gate_up_lora.lora_a.shape)
            gate_a = _require_tensor_shape(
                state, _mlp_key(layer_idx, "gate_proj", "lora_A"), gate_up_a_shape, consumed_keys
            ).to(device=mlp.gate_up_lora.lora_a.device, dtype=mlp.gate_up_lora.lora_a.dtype)
            up_a = _require_tensor_shape(
                state, _mlp_key(layer_idx, "up_proj", "lora_A"), gate_up_a_shape, consumed_keys
            ).to(gate_a)
            if strict and not torch.equal(gate_a, up_a):
                raise ValueError("Dense Qwen2 fused gate_up_lora requires gate/up lora_A tensors to match.")
            rank = mlp.gate_up_lora.lora_b.shape[1]
            gate_b = _require_tensor_shape(
                state,
                _mlp_key(layer_idx, "gate_proj", "lora_B"),
                (mlp.gate_up_lora.lora_b.shape[0] // 2, rank),
                consumed_keys,
            ).to(device=mlp.gate_up_lora.lora_b.device, dtype=mlp.gate_up_lora.lora_b.dtype)
            up_b = _require_tensor_shape(
                state,
                _mlp_key(layer_idx, "up_proj", "lora_B"),
                (mlp.gate_up_lora.lora_b.shape[0] // 2, rank),
                consumed_keys,
            ).to(gate_b)
            _copy_to_lora_param(
                mlp.gate_up_lora.lora_a,
                gate_a.contiguous(),
                key=_mlp_key(layer_idx, "gate_proj", "lora_A"),
            )
            _copy_to_lora_param(
                mlp.gate_up_lora.lora_b,
                torch.cat([gate_b, up_b], dim=0).contiguous(),
                key=(
                    f"{_mlp_key(layer_idx, 'gate_proj', 'lora_B')}/"
                    f"{_mlp_key(layer_idx, 'up_proj', 'lora_B')}"
                ),
            )
            loaded += 2

        if mlp.down_lora is not None:
            down_a = _require_tensor_shape(
                state,
                _mlp_key(layer_idx, "down_proj", "lora_A"),
                tuple(mlp.down_lora.lora_a.shape),
                consumed_keys,
            )
            down_b = _require_tensor_shape(
                state,
                _mlp_key(layer_idx, "down_proj", "lora_B"),
                tuple(mlp.down_lora.lora_b.shape),
                consumed_keys,
            )
            _copy_to_lora_param(mlp.down_lora.lora_a, down_a, key=_mlp_key(layer_idx, "down_proj", "lora_A"))
            _copy_to_lora_param(mlp.down_lora.lora_b, down_b, key=_mlp_key(layer_idx, "down_proj", "lora_B"))
            loaded += 2

    unexpected = sorted(set(state) - consumed_keys)
    if strict and unexpected:
        raise KeyError(f"Unexpected dense Qwen2 adapter tensor keys: {unexpected[:5]!r}.")
    return {
        "loaded_tensors": loaded,
        "unexpected_keys": unexpected,
        "missing_keys": [],
    }


def load_lora_adapter(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    adapter_dir: str | Path,
    model_cfg: Qwen2Config,
    ps: ParallelState,
    *,
    strict: bool | str = True,
    lora_config: LoraConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_parallel_scope(ps)
    strict = _strict_bool(strict)
    adapter_root = _validate_adapter_root(adapter_dir)
    config_path = adapter_root / "adapter_config.json"
    adapter_config = None
    if config_path.exists():
        adapter_config = _load_json_object(config_path, description="adapter_config.json")
    elif lora_config is None:
        raise ValueError(
            "load_lora_adapter requires caller-provided lora_config when adapter_config.json is missing."
        )
    metadata_path = adapter_root / "megatron.lite_adapter_meta.json"
    adapter_meta = None
    if metadata_path.exists():
        adapter_meta = _load_json_object(metadata_path, description="megatron.lite_adapter_meta.json")

    from safetensors.torch import load_file

    state = load_file(str(_validate_adapter_model_path(adapter_root)), device="cpu")
    _validate_adapter_state_is_lora_only(state)
    if adapter_config is not None:
        _validate_adapter_config(chunks, state, adapter_config, lora_config=lora_config)
    else:
        _validate_expected_lora_config(state, lora_config)
    if adapter_meta is not None:
        _validate_adapter_metadata(adapter_meta, model_cfg, ps, state, lora_config)
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
