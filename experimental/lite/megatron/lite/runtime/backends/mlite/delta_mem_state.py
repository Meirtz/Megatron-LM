# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime-owned delta-mem state lifecycle helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist

DeltaMemRuntimeScope = Literal["step", "handle"]

DELTA_MEM_STATE_SCOPE_KEY = "delta_mem_state_scope"
DELTA_MEM_STATE_RESET_KEY = "delta_mem_state_reset"
DELTA_MEM_STATE_DETACH_KEY = "delta_mem_state_detach"
DELTA_MEM_RUNTIME_STATES_KEY = "_delta_mem_runtime_states"
DELTA_MEM_RUNTIME_STATE_FORMAT = "megatron_lite.delta_mem_runtime_state.v1"
DELTA_MEM_RUNTIME_STATE_FILE = "delta_mem_runtime_state.pt"

_STEP_SCOPE_ALIASES = {"step", "microbatch", "microbatches", "train_step", "call"}
_HANDLE_SCOPE_ALIASES = {"handle", "persistent", "session"}
_DISABLED_SCOPE_ALIASES = {"", "none", "off", "false", "disabled"}


def _delta_mem_enabled(handle: Any) -> bool:
    config = getattr(handle, "_extras", {}).get("delta_mem_config")
    return bool(getattr(config, "enabled", False))


def _normalize_scope(value: Any) -> DeltaMemRuntimeScope | None:
    if value is None or value is False:
        return None
    if value is True:
        return "step"
    if not isinstance(value, str):
        raise TypeError("delta_mem_state_scope must be a string, boolean, or None.")
    normalized = value.strip().lower()
    if normalized in _DISABLED_SCOPE_ALIASES:
        return None
    if normalized in _STEP_SCOPE_ALIASES:
        return "step"
    if normalized in _HANDLE_SCOPE_ALIASES:
        return "handle"
    raise ValueError("delta_mem_state_scope must be one of step, handle, or disabled.")


def _bool_extra(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _batch_extras(batch: Any) -> dict[str, Any]:
    extras = getattr(batch, "extras", None)
    if extras is None:
        return {}
    if not isinstance(extras, dict):
        raise TypeError("delta-mem runtime lifecycle expects batch.extras to be a dict.")
    return extras


def _with_extras(batch: Any, extras: dict[str, Any]) -> Any:
    if is_dataclass(batch):
        return replace(batch, extras=extras)
    if hasattr(batch, "extras"):
        batch.extras = extras
        return batch
    raise TypeError("delta-mem runtime lifecycle requires batches with an extras dict.")


def _detach_state(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, list):
        return [_detach_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_state(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _detach_state(item) for key, item in value.items()}
    raise TypeError("delta_mem_states must contain tensors, lists, tuples, dicts, or None.")


def _map_state_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, list):
        return [_map_state_tensors(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_state_tensors(item, fn) for item in value)
    if isinstance(value, Mapping):
        return {key: _map_state_tensors(item, fn) for key, item in value.items()}
    raise TypeError("delta_mem_states must contain tensors, lists, tuples, dicts, or None.")


def _export_tensor(tensor: torch.Tensor, *, cpu: bool, clone: bool) -> torch.Tensor:
    exported = tensor.detach()
    if clone:
        exported = exported.clone()
    if cpu:
        exported = exported.cpu()
    return exported


def _state_layer_count(states: Any) -> int | None:
    if isinstance(states, Sequence) and not isinstance(states, (str, bytes)):
        return len(states)
    if isinstance(states, Mapping):
        return len(states)
    return None


def _split_runtime_states(states: Any) -> tuple[Any, Any]:
    if isinstance(states, Mapping):
        return states.get("main"), states.get("mtp")
    return states, None


def _merge_output_states(output: Mapping[str, Any]) -> Any:
    main_states = output.get("delta_mem_states")
    if "delta_mem_mtp_states" not in output:
        return main_states
    return {"main": main_states, "mtp": output.get("delta_mem_mtp_states")}


def _rank_suffix() -> str | None:
    if dist.is_available() and dist.is_initialized():
        return f"rank_{dist.get_rank():05d}"
    return None


def _state_filename() -> str:
    suffix = _rank_suffix()
    if suffix is None:
        return DELTA_MEM_RUNTIME_STATE_FILE
    return f"delta_mem_runtime_state_{suffix}.pt"


def _looks_like_file(path: Path) -> bool:
    return path.suffix != "" and not path.name.startswith("step_")


def _state_sidecar_for_file(path: Path) -> Path:
    return path.with_name(f"{path.stem}.delta_mem_runtime_state.pt")


def _latest_step_dir(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    step_dirs = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.startswith("step_") and child.name.split("_", 1)[1].isdigit()
    ]
    if not step_dirs:
        return None
    return sorted(step_dirs, key=lambda child: int(child.name.split("_", 1)[1]))[-1]


def delta_mem_runtime_state_file(path: str | os.PathLike[str], *, for_load: bool = False) -> Path:
    """Resolve the delta-mem runtime-state sidecar path for a checkpoint or directory."""

    state_path = Path(path)
    if state_path.name.startswith("delta_mem_runtime_state"):
        return state_path
    if _looks_like_file(state_path):
        return _state_sidecar_for_file(state_path)
    if for_load:
        latest_step = _latest_step_dir(state_path)
        if latest_step is not None:
            state_path = latest_step
    return state_path / _state_filename()


def _torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def has_delta_mem_runtime_state(handle: Any) -> bool:
    return getattr(handle, "_extras", {}).get(DELTA_MEM_RUNTIME_STATES_KEY) is not None


def export_delta_mem_runtime_state(
    handle: Any,
    *,
    cpu: bool = True,
    clone: bool = True,
    require_state: bool = True,
) -> dict[str, Any]:
    """Export runtime-owned mutable delta-mem state without adapter weights."""

    states = getattr(handle, "_extras", {}).get(DELTA_MEM_RUNTIME_STATES_KEY)
    if states is None and require_state:
        raise ValueError("No delta-mem runtime state is available to export.")
    exported = _map_state_tensors(states, lambda tensor: _export_tensor(tensor, cpu=cpu, clone=clone))
    return {
        "format": DELTA_MEM_RUNTIME_STATE_FORMAT,
        "states": exported,
        "metadata": {
            "layer_count": _state_layer_count(exported),
            "rank_suffix": _rank_suffix(),
            "adapter_export_compatible": True,
        },
    }


def import_delta_mem_runtime_state(
    handle: Any,
    payload: Mapping[str, Any],
    *,
    detach: bool = True,
) -> dict[str, Any]:
    """Import a previously exported delta-mem runtime state into a handle."""

    if payload.get("format") != DELTA_MEM_RUNTIME_STATE_FORMAT:
        raise RuntimeError("Unsupported delta-mem runtime state format.")
    states = payload.get("states")
    if detach:
        states = _detach_state(states)
    getattr(handle, "_extras")[DELTA_MEM_RUNTIME_STATES_KEY] = states
    return {
        "format": payload.get("format"),
        "layer_count": _state_layer_count(states),
        "restored": states is not None,
    }


def save_delta_mem_runtime_state(
    handle: Any,
    path: str | os.PathLike[str],
    *,
    require_state: bool = True,
) -> Path | None:
    """Save runtime-owned delta-mem state to a checkpoint sidecar."""

    if not has_delta_mem_runtime_state(handle) and not require_state:
        return None
    payload = export_delta_mem_runtime_state(handle, require_state=require_state)
    state_file = delta_mem_runtime_state_file(path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, state_file)
    return state_file


def load_delta_mem_runtime_state(
    handle: Any,
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    require_exists: bool = True,
) -> dict[str, Any] | None:
    """Load runtime-owned delta-mem state from a checkpoint sidecar."""

    state_file = delta_mem_runtime_state_file(path, for_load=True)
    if not state_file.exists():
        if require_exists:
            raise FileNotFoundError(f"Delta-mem runtime state sidecar not found: {state_file}")
        return None
    payload = _torch_load(state_file, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Delta-mem runtime state sidecar must contain a mapping payload.")
    return import_delta_mem_runtime_state(handle, payload)


def reset_delta_mem_runtime_state(handle: Any) -> None:
    getattr(handle, "_extras", {}).pop(DELTA_MEM_RUNTIME_STATES_KEY, None)


def delta_mem_runtime_lifecycle_requested(batch: Any) -> bool:
    """Return whether a batch explicitly asks the runtime to carry delta-mem state."""

    return _normalize_scope(_batch_extras(batch).get(DELTA_MEM_STATE_SCOPE_KEY)) is not None


class DeltaMemRuntimeState:
    """Own delta-mem state handoff outside the model forward graph."""

    def __init__(self, *, initial_states: Any = None, detach_after_microbatch: bool = True):
        self.states = initial_states
        self.scope: DeltaMemRuntimeScope | None = None
        self.detach_after_microbatch = detach_after_microbatch
        self.capture_count = 0
        self.reset_count = 0

    @property
    def active(self) -> bool:
        return self.scope is not None

    def prepare_batch(self, batch: Any) -> Any:
        extras = _batch_extras(batch)
        requested_scope = _normalize_scope(extras.get(DELTA_MEM_STATE_SCOPE_KEY))
        if requested_scope is not None:
            self.scope = requested_scope
        if self.scope is None:
            return batch

        if DELTA_MEM_STATE_DETACH_KEY in extras:
            self.detach_after_microbatch = _bool_extra(
                extras[DELTA_MEM_STATE_DETACH_KEY],
                name=DELTA_MEM_STATE_DETACH_KEY,
            )
        if _bool_extra(extras.get(DELTA_MEM_STATE_RESET_KEY, False), name=DELTA_MEM_STATE_RESET_KEY):
            self.states = None
            self.reset_count += 1

        forward_extras = dict(extras)
        forward_extras["return_delta_mem_states"] = True
        main_states, mtp_states = _split_runtime_states(self.states)
        if main_states is not None and "delta_mem_states" not in forward_extras:
            forward_extras["delta_mem_states"] = main_states
        if mtp_states is not None and "delta_mem_mtp_states" not in forward_extras:
            forward_extras["delta_mem_mtp_states"] = mtp_states
        return _with_extras(batch, forward_extras)

    def capture_output(self, output: Mapping[str, Any]) -> None:
        if self.scope is None:
            return
        if "delta_mem_states" not in output:
            raise ValueError(
                "delta-mem runtime lifecycle requested return_delta_mem_states, "
                "but model output did not include delta_mem_states."
            )
        states = _merge_output_states(output)
        self.states = _detach_state(states) if self.detach_after_microbatch else states
        self.capture_count += 1

    def finish(self, handle: Any) -> None:
        extras = getattr(handle, "_extras", None)
        if extras is None:
            return
        if self.scope == "handle":
            extras[DELTA_MEM_RUNTIME_STATES_KEY] = self.states
        elif self.scope == "step":
            extras.pop(DELTA_MEM_RUNTIME_STATES_KEY, None)

    def metrics(self) -> dict[str, Any]:
        if self.scope is None:
            return {}
        main_states, mtp_states = _split_runtime_states(self.states)
        return {
            "delta_mem_state_scope": self.scope,
            "delta_mem_state_capture_count": self.capture_count,
            "delta_mem_state_reset_count": self.reset_count,
            "delta_mem_state_layer_count": _state_layer_count(main_states),
            "delta_mem_mtp_state_layer_count": _state_layer_count(mtp_states),
            "delta_mem_state_detached": self.detach_after_microbatch,
        }


def wrap_delta_mem_forward_step(
    forward_step: Callable[[Any, Any], Mapping[str, Any]],
    handle: Any,
) -> tuple[Callable[[Any, Any], Mapping[str, Any]], DeltaMemRuntimeState]:
    """Wrap a model forward_step with opt-in delta-mem state carryover."""

    manager = DeltaMemRuntimeState(
        initial_states=getattr(handle, "_extras", {}).get(DELTA_MEM_RUNTIME_STATES_KEY)
    )

    def _wrapped_forward_step(model: Any, batch: Any) -> Mapping[str, Any]:
        requested = delta_mem_runtime_lifecycle_requested(batch)
        if requested and not _delta_mem_enabled(handle):
            raise ValueError("delta-mem runtime lifecycle requires enabled DeltaMem config.")
        prepared_batch = manager.prepare_batch(batch)
        output = forward_step(model, prepared_batch)
        if not isinstance(output, Mapping):
            raise TypeError("delta-mem runtime lifecycle expects forward_step to return a mapping.")
        manager.capture_output(output)
        return output

    return _wrapped_forward_step, manager


__all__ = [
    "DELTA_MEM_RUNTIME_STATES_KEY",
    "DELTA_MEM_RUNTIME_STATE_FILE",
    "DELTA_MEM_RUNTIME_STATE_FORMAT",
    "DELTA_MEM_STATE_DETACH_KEY",
    "DELTA_MEM_STATE_RESET_KEY",
    "DELTA_MEM_STATE_SCOPE_KEY",
    "DeltaMemRuntimeState",
    "DeltaMemRuntimeScope",
    "delta_mem_runtime_lifecycle_requested",
    "delta_mem_runtime_state_file",
    "export_delta_mem_runtime_state",
    "has_delta_mem_runtime_state",
    "import_delta_mem_runtime_state",
    "load_delta_mem_runtime_state",
    "reset_delta_mem_runtime_state",
    "save_delta_mem_runtime_state",
    "wrap_delta_mem_forward_step",
]
