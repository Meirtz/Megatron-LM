# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
DCP (Distributed Checkpoint) framework for training checkpoints.

Model-agnostic: takes a placement function to describe how each parameter is sharded.
HF weight loading/saving is model-specific and lives in models/<name>/checkpoint.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.distributed.checkpoint as dcp  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from megatron.lite.primitive.ckpt.errors import (
    CheckpointLoadFatalError,
    _checkpoint_exception_started_mutation,
    _is_control_flow_exception,
    _mark_checkpoint_mutation,
    _raise_checkpoint_load_commit_error,
)
from megatron.lite.primitive.ckpt.hf_weights import (
    _distributed_raise_if_error,
    named_persistent_buffers,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.protocols import (
    ExpertClassifierFn,
    PlacementFn,
    default_expert_classifier,
    default_placement_fn,
)
from torch.distributed.checkpoint.metadata import (  # pyright: ignore[reportMissingImports]
    TensorStorageMetadata,
)
from torch.distributed.device_mesh import (  # pyright: ignore[reportMissingImports]
    DeviceMesh,
)
from torch.distributed.tensor import DTensor  # pyright: ignore[reportMissingImports]

_DCP_MODEL_SCHEMA_KEY = "mlite_model_schema_version"
_DCP_MODEL_SCHEMA_VERSION = 2
_CHECKPOINT_MANIFEST = "mlite_checkpoint_manifest.json"
_CHECKPOINT_MANIFEST_FORMAT = "megatron_lite.training_checkpoint.v2"
_LOCAL_TRAINING_FORMAT_V1 = "megatron_lite.local_training.v1"
_LOCAL_TRAINING_FORMAT_V2 = "megatron_lite.local_training.v2"
_LOCAL_TRAINING_FORMAT_V3 = "megatron_lite.local_training.v3"
_LOCAL_TRAINING_COMPLETION = ".mlite_local_checkpoint_complete.json"


class ExtraStateValidator(Protocol):
    """Pure validation callback for a deserialized extra-state value.

    Implementations must not mutate model, optimizer, RNG, another registered
    target, or any other live runtime state, including when they raise.
    """

    def __call__(self, candidate: Any) -> None: ...


class ExtraStateCheckpointTarget(Protocol):
    """Transactional adapter for one live extra-state object.

    ``validate()``, ``snapshot()``, and ``fingerprint()`` must be pure.
    ``validate_step()`` is an optional pure extension with the same contract.
    Only ``apply()`` and ``restore()`` may mutate live state. Registering an
    adapter asserts this contract: in particular, no runtime probe can prove
    that the *first* successful ``fingerprint()`` call did not silently mutate
    state before establishing its baseline. Initial fingerprint exceptions are
    therefore treated as fatal, while silent mutation by that first call is an
    unsupported adapter contract violation.

    Every callback is scoped to exactly one target and must not mutate another
    registered target. Coupled state must be represented by one composite
    target so snapshot/apply/restore remain atomic at this boundary.
    """

    def validate(self, candidate: Any) -> None: ...

    def snapshot(self) -> Any: ...

    def apply(self, candidate: Any) -> None: ...

    def restore(self, snapshot: Any) -> None: ...

    def fingerprint(self) -> Any: ...


def save_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int | str,
    path: str | None = None,
    config=None,
    ps: ParallelState | None = None,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
    *,
    use_dcp: bool | None = True,
    save_rng: bool = True,
    save_model: bool = True,
    save_optimizer: bool = True,
    extra_states: Mapping[str, Any] | None = None,
) -> None:
    """Save training checkpoint using DTensor + DCP for automatic resharding."""
    if path is None and isinstance(step, str):
        path = step
        step = 0
    if path is None:
        raise ValueError("checkpoint path is required")
    if type(step) is not int or step < 0:
        raise TypeError("checkpoint step must be a non-negative integer")
    if use_dcp is None:
        use_dcp = True
    if not use_dcp:
        if not save_model or not save_optimizer:
            raise ValueError(
                "Local-format checkpoints do not support partial component selection; "
                "use_dcp=False requires save_model=True and save_optimizer=True."
            )
        if extra_states:
            raise ValueError("extra_states are supported only for DCP checkpoints")
        _save_local_training_checkpoint(
            model, optimizer, step, path, config=config, ps=ps, save_rng=save_rng
        )
        return
    extra_states = _normalize_extra_states(extra_states)
    if save_optimizer and optimizer is None:
        raise ValueError(
            "save_optimizer=True requires a non-None optimizer; pass save_optimizer=False "
            "for a model-only checkpoint"
        )
    if _supports_dist_opt_distckpt(model, optimizer if save_optimizer else None):
        ckpt_path = os.path.join(path, f"step_{step}")
        manifest = _begin_checkpoint_transaction(
            ckpt_path,
            step=step,
            save_model=save_model,
            save_optimizer=save_optimizer,
            save_rng=save_rng,
            payload_format="distckpt",
            optimizer_storage="distckpt" if save_optimizer else "none",
            extra_state_files=extra_states,
        )
        local_error = None
        try:
            _save_dist_opt_checkpoint(
                model,
                optimizer,
                step,
                ckpt_path,
                save_model=save_model,
                save_optimizer=save_optimizer,
            )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _distributed_raise_if_error(
            local_error, context="dist_opt checkpoint payload save failed"
        )
        if save_rng:
            _save_rank_sidecar_with_consensus(
                lambda: _save_rng_sidecar(ckpt_path), context="RNG sidecar save failed"
            )
        _save_rank0_extra_states_with_consensus(ckpt_path, extra_states)
        _complete_checkpoint_transaction(ckpt_path, manifest)
        log_rank0(f"Saved dist_opt checkpoint at step {step} to {ckpt_path}")
        return
    if config is None or ps is None:
        raise ValueError("DCP checkpointing requires config and ParallelState.")
    if not isinstance(model, nn.Module):
        raise TypeError("DCP checkpointing currently expects a single nn.Module.")
    dense_mesh, expert_mesh = _build_meshes(config)
    state_dict: dict = {"step": step}
    # Pipeline stages own DIFFERENT parameters but their local layers re-index
    # to 0..N, so without a per-stage prefix the DCP FQNs collide across pp ranks
    # (stage0 layer0 and stage1 layer1 both -> "model.0.layers.0..."), corrupting
    # the round-trip. Mirror distckpt's pp-aware keying: disjoint keyspace per stage.
    model_prefix = f"model_pp{ps.pp_rank}" if ps.pp_size > 1 else "model"

    if save_model:
        state_dict[_DCP_MODEL_SCHEMA_KEY] = _DCP_MODEL_SCHEMA_VERSION
        for name, tensor in _named_model_checkpoint_tensors(model):
            placements = get_placements(name)
            mesh = expert_mesh if is_expert(name) else dense_mesh
            state_dict[f"{model_prefix}.{name}"] = _dcp_tensor_from_param(
                tensor, mesh, placements
            )

    ckpt_path = os.path.join(path, f"step_{step}")
    manifest = _begin_checkpoint_transaction(
        ckpt_path,
        step=step,
        save_model=save_model,
        save_optimizer=save_optimizer,
        save_rng=save_rng,
        payload_format="dcp",
        optimizer_storage="rank_sidecar" if save_optimizer else "none",
        extra_state_files=extra_states,
    )
    local_error = None
    try:
        dcp.save(state_dict, checkpoint_id=ckpt_path)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="DCP checkpoint payload save failed"
    )
    if save_optimizer:
        _save_rank_sidecar_with_consensus(
            lambda: _save_optimizer_checkpoint(optimizer, ckpt_path),
            context="optimizer sidecar save failed",
        )
    if save_rng:
        _save_rank_sidecar_with_consensus(
            lambda: _save_rng_sidecar(ckpt_path), context="RNG sidecar save failed"
        )
    _save_rank0_extra_states_with_consensus(ckpt_path, extra_states)
    _complete_checkpoint_transaction(ckpt_path, manifest)
    log_rank0(f"Saved training checkpoint at step {step} to {ckpt_path}")


def load_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    config=None,
    ps: ParallelState | None = None,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
    *,
    use_dcp: bool | None = True,
    load_rng: bool = True,
    load_parameter_state_update_legacy_format: bool = False,
    load_model: bool = True,
    load_optimizer: bool = True,
    allow_legacy_checkpoint: bool = False,
    load_extra_state_files: Iterable[str] | None = None,
    loaded_extra_states: MutableMapping[str, Any] | None = None,
    extra_state_validators: Mapping[str, Callable[[Any], None]] | None = None,
    extra_state_targets: Mapping[str, Any] | None = None,
) -> int:
    """Load a training checkpoint, resharding DCP payloads when required.

    ``allow_legacy_checkpoint`` is a narrow, explicit migration escape hatch.
    It admits local V1/V2 payloads and manifest-less DCP directories only so
    they can be loaded once and immediately re-saved in the current format.
    It never permits a torn V3 local checkpoint without its completion marker,
    nor does it relax validation of a completed current-format checkpoint.
    """
    if use_dcp is None:
        use_dcp = True
    if not use_dcp:
        if not load_model or not load_optimizer:
            raise ValueError(
                "Local-format checkpoints do not support partial component selection; "
                "use_dcp=False requires load_model=True and load_optimizer=True."
            )
        if (
            load_extra_state_files is not None
            or loaded_extra_states is not None
            or extra_state_validators is not None
            or extra_state_targets is not None
        ):
            raise ValueError("extra states are supported only for DCP checkpoints")
        return _load_local_training_checkpoint(
            model,
            optimizer,
            path,
            config=config,
            ps=ps,
            load_rng=load_rng,
            load_parameter_state_update_legacy_format=load_parameter_state_update_legacy_format,
            allow_legacy_checkpoint=allow_legacy_checkpoint,
        )
    if load_optimizer and optimizer is None:
        raise ValueError(
            "load_optimizer=True requires a non-None optimizer; pass load_optimizer=False "
            "for a model-only checkpoint load"
        )
    requested_extra_state_files = _normalize_extra_state_filenames(
        load_extra_state_files
    )
    normalized_extra_state_validators = _normalize_extra_state_validators(
        extra_state_validators, requested=requested_extra_state_files
    )
    normalized_extra_state_targets = _normalize_extra_state_targets(
        extra_state_targets, requested=requested_extra_state_files
    )
    if loaded_extra_states is not None and not isinstance(
        loaded_extra_states, MutableMapping
    ):
        raise TypeError("loaded_extra_states must be a mutable mapping")
    if requested_extra_state_files and loaded_extra_states is None:
        raise ValueError(
            "loaded_extra_states is required when load_extra_state_files is non-empty"
        )
    ckpt_path = _resolve_step_checkpoint_path(
        path, allow_legacy_checkpoint=allow_legacy_checkpoint
    )
    manifest = _validate_checkpoint_manifest(
        ckpt_path,
        load_model=load_model,
        load_optimizer=load_optimizer,
        load_rng=load_rng,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
        load_extra_state_files=requested_extra_state_files,
    )
    expected_checkpoint_step = manifest["step"] if manifest is not None else None
    current_supports_distckpt = False
    requested_optimizer = optimizer if load_optimizer else None
    local_error = None
    if manifest is None:
        try:
            current_supports_distckpt = _supports_dist_opt_distckpt(
                model, requested_optimizer
            )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _distributed_raise_if_error(
            local_error, context="legacy checkpoint backend detection failed"
        )
        _assert_world_consensus(
            current_supports_distckpt,
            context="legacy checkpoint backend detection differs across ranks",
        )
        use_dist_opt_distckpt = current_supports_distckpt
    else:
        use_dist_opt_distckpt = manifest["payload_format"] == "distckpt"
        if use_dist_opt_distckpt and (load_model or load_optimizer):
            try:
                current_supports_distckpt = _supports_dist_opt_distckpt(
                    model, requested_optimizer
                )
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            if local_error is None and not current_supports_distckpt:
                local_error = (
                    "checkpoint payload_format='distckpt' requires an attached "
                    "distckpt model and, when requested, a sharded optimizer"
                )
        _distributed_raise_if_error(
            local_error, context="checkpoint payload backend validation failed"
        )
        _assert_world_consensus(
            use_dist_opt_distckpt,
            context="checkpoint payload backend differs across ranks",
        )
    local_error = None
    if manifest is not None and load_optimizer:
        expected_optimizer_storage = (
            "distckpt" if use_dist_opt_distckpt else "rank_sidecar"
        )
        if manifest["optimizer_storage"] != expected_optimizer_storage:
            local_error = (
                "checkpoint optimizer storage is incompatible with the current "
                f"optimizer: saved={manifest['optimizer_storage']!r}, "
                f"required={expected_optimizer_storage!r}"
            )
    _distributed_raise_if_error(
        local_error, context="checkpoint optimizer storage validation failed"
    )
    optimizer_state, rng_state, extra_state_values = _preload_checkpoint_sidecars(
        ckpt_path,
        optimizer=optimizer,
        load_optimizer=load_optimizer and not use_dist_opt_distckpt,
        allow_legacy_optimizer_state=(allow_legacy_checkpoint and manifest is None),
        load_rng=load_rng,
        rng_required=manifest is not None,
        extra_state_files=requested_extra_state_files,
        extra_state_validators=normalized_extra_state_validators,
        extra_state_targets=normalized_extra_state_targets,
        checkpoint_step=expected_checkpoint_step,
    )
    if use_dist_opt_distckpt:
        step = _load_dist_opt_checkpoint(
            model,
            optimizer,
            ckpt_path,
            load_model=load_model,
            load_optimizer=load_optimizer,
            expected_step=expected_checkpoint_step,
            allow_legacy_checkpoint=(allow_legacy_checkpoint and manifest is None),
        )
        local_exception: Exception | None = None
        if (
            expected_checkpoint_step is not None
            and int(step) != expected_checkpoint_step
        ):
            local_exception = RuntimeError(
                f"checkpoint payload step {step} does not match completion manifest "
                f"step {expected_checkpoint_step}"
            )
        core_mutation_started = _raise_checkpoint_load_commit_error(
            local_exception,
            mutation_started=load_model or load_optimizer,
            context="distckpt returned step validation failed",
        )
        _commit_preloaded_sidecars(
            optimizer=None,
            optimizer_state=None,
            rng_state=rng_state,
            loaded_extra_states=loaded_extra_states,
            extra_state_values=extra_state_values,
            extra_state_targets=normalized_extra_state_targets,
            prior_live_state_mutation=core_mutation_started,
        )
        log_rank0(f"Loaded dist_opt checkpoint from {path} at step {step}")
        return step
    state_dict: dict = {"step": 0}
    model_prefix = ""
    parameter_items: list[tuple[str, torch.Tensor]] = []
    buffer_items: list[tuple[str, torch.Tensor]] = []
    checkpoint_tensor_items: list[tuple[str, torch.Tensor]] = []
    checkpoint_tensor_templates: dict[str, torch.Tensor] = {}

    if load_model:
        topology_identity: dict[str, int] | None = None
        local_error = None
        try:
            if config is None or ps is None:
                raise ValueError("DCP model loading requires config and ParallelState.")
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "DCP model loading currently expects a single nn.Module."
                )
            topology_identity = {
                "tp": int(config.tp or 1),
                "ep": int(config.ep or 1),
                "etp": max(int(config.etp or 1), 1),
                "cp": max(int(config.cp or 1), 1),
                "pp": max(int(config.pp or 1), 1),
                "ps_pp_size": int(ps.pp_size),
            }
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _distributed_raise_if_error(
            local_error, context="DCP model topology preflight failed"
        )
        assert topology_identity is not None
        _assert_world_consensus(
            topology_identity, context="DCP model topology differs across ranks"
        )

        metadata_entries: Mapping[str, Any] = {}
        local_error = None
        try:
            dense_mesh, expert_mesh = _build_meshes(config)
            # Same pp-aware keying as save (see save_training_checkpoint): per-stage
            # disjoint keyspace so pp ranks don't read each other's colliding FQNs.
            model_prefix = f"model_pp{ps.pp_rank}" if ps.pp_size > 1 else "model"
            parameter_items = list(model.named_parameters())
            buffer_items = list(named_persistent_buffers(model))
            checkpoint_tensor_items = parameter_items
            for name, tensor in [*parameter_items, *buffer_items]:
                placements = get_placements(name)
                mesh = expert_mesh if is_expert(name) else dense_mesh
                checkpoint_tensor_templates[f"{model_prefix}.{name}"] = (
                    _empty_dcp_tensor_like_param(tensor, mesh, placements)
                )
            metadata = dcp.FileSystemReader(ckpt_path).read_metadata()
            metadata_entries = metadata.state_dict_metadata
            if not isinstance(metadata_entries, Mapping):
                raise TypeError(
                    "DCP state_dict_metadata must be a mapping, got "
                    f"{type(metadata_entries).__name__}"
                )
            schema_present, include_buffers = _validate_dcp_model_metadata(
                metadata_entries,
                model_prefix=model_prefix,
                parameter_items=parameter_items,
                buffer_items=buffer_items,
                checkpoint_tensor_templates=checkpoint_tensor_templates,
                require_schema=manifest is not None,
            )
            if schema_present:
                state_dict[_DCP_MODEL_SCHEMA_KEY] = 0
            if include_buffers:
                checkpoint_tensor_items = [*parameter_items, *buffer_items]
            else:
                log_rank0(
                    "Loading a legacy MLite DCP checkpoint without persistent buffers; "
                    "buffers retain their initialized values. Re-save the checkpoint to "
                    "upgrade it to schema version 2."
                )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _distributed_raise_if_error(
            local_error, context="DCP model metadata validation failed"
        )

        for name, tensor in checkpoint_tensor_items:
            key = f"{model_prefix}.{name}"
            state_dict[key] = checkpoint_tensor_templates[key]

    local_error = None
    try:
        dcp.load(state_dict, checkpoint_id=ckpt_path)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="DCP staged checkpoint load failed"
    )

    if load_model:
        local_error = None
        if _DCP_MODEL_SCHEMA_KEY in state_dict:
            loaded_schema = int(state_dict[_DCP_MODEL_SCHEMA_KEY])
            if loaded_schema != _DCP_MODEL_SCHEMA_VERSION:
                local_error = (
                    f"unsupported MLite DCP model schema version {loaded_schema}; "
                    f"expected {_DCP_MODEL_SCHEMA_VERSION}"
                )
        _distributed_raise_if_error(
            local_error, context="DCP model schema validation failed"
        )
    step = int(state_dict.get("step", 0))
    local_error = None
    if expected_checkpoint_step is not None and step != expected_checkpoint_step:
        local_error = (
            f"checkpoint payload step {step} does not match completion manifest "
            f"step {expected_checkpoint_step}"
        )
    _distributed_raise_if_error(
        local_error, context="DCP checkpoint step validation failed"
    )
    model_mutation_started = False
    if load_model:
        local_exception = None

        def mark_model_mutation_started() -> None:
            nonlocal model_mutation_started
            model_mutation_started = True

        try:
            for name, tensor in checkpoint_tensor_items:
                key = f"{model_prefix}.{name}"
                if key in state_dict:
                    with torch.no_grad():
                        _copy_tensor_(
                            tensor,
                            state_dict[key],
                            before_copy=mark_model_mutation_started,
                        )
        except BaseException as exc:
            local_exception = exc
        model_mutation_started = _raise_checkpoint_load_commit_error(
            local_exception,
            mutation_started=model_mutation_started,
            context="DCP staged model commit failed",
        )
    _commit_preloaded_sidecars(
        optimizer=optimizer if load_optimizer else None,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
        loaded_extra_states=loaded_extra_states,
        extra_state_values=extra_state_values,
        extra_state_targets=normalized_extra_state_targets,
        prior_live_state_mutation=model_mutation_started,
    )
    log_rank0(f"Loaded training checkpoint from {path} at step {step}")
    return step


def _resolve_step_checkpoint_path(
    path: str, *, allow_legacy_checkpoint: bool = False
) -> str:
    selected = path
    step_dirs: list[tuple[int, str]] = []
    eligible: list[tuple[int, str]] = []
    incomplete: list[str] = []
    legacy: list[str] = []
    selected_manifest_present = False
    selected_manifest: dict[str, Any] | None = None
    local_error = None
    try:
        if not os.path.basename(path).startswith("step_"):
            for dirname in os.listdir(path):
                if not dirname.startswith("step_"):
                    continue
                try:
                    step = int(dirname.removeprefix("step_"))
                except ValueError:
                    continue
                step_dirs.append((step, dirname))

            for step, dirname in sorted(step_dirs):
                candidate = os.path.join(path, dirname)
                manifest_path = _checkpoint_manifest_path(candidate)
                manifest_present = manifest_path.exists()
                if not manifest_present:
                    legacy.append(candidate)
                    if allow_legacy_checkpoint:
                        eligible.append((step, candidate))
                    continue
                manifest = json.loads(manifest_path.read_text())
                _validate_checkpoint_manifest_schema(manifest)
                if manifest["status"] == "complete":
                    eligible.append((step, candidate))
                else:
                    incomplete.append(candidate)

            if eligible:
                selected = max(eligible)[1]
            elif incomplete or legacy:
                raise RuntimeError(
                    f"No strict complete checkpoint is available under {path}; "
                    f"incomplete={incomplete}, legacy={legacy}"
                )

        selected_manifest_path = _checkpoint_manifest_path(selected)
        selected_manifest_present = selected_manifest_path.exists()
        if selected_manifest_present:
            selected_manifest = json.loads(selected_manifest_path.read_text())
            _validate_checkpoint_manifest_schema(selected_manifest)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="checkpoint step resolution failed"
    )

    resolution = {
        "selected_path": _canonical_checkpoint_path(selected),
        "manifest_present": selected_manifest_present,
        "manifest": selected_manifest,
        "eligible": [
            _canonical_checkpoint_path(candidate) for _, candidate in eligible
        ],
        "incomplete": [
            _canonical_checkpoint_path(candidate) for candidate in incomplete
        ],
        "legacy": [_canonical_checkpoint_path(candidate) for candidate in legacy],
    }
    _assert_world_consensus(
        resolution, context="checkpoint step resolution differs across ranks"
    )

    if incomplete:
        log_rank0(
            "Ignoring incomplete checkpoint transactions while resolving latest step: "
            f"{incomplete}"
        )
    if legacy and not allow_legacy_checkpoint:
        log_rank0(
            "Ignoring pre-manifest legacy checkpoints while resolving latest step: "
            f"{legacy}"
        )
    return selected


def _checkpoint_manifest_path(path: str | os.PathLike[str]) -> Path:
    return Path(path) / _CHECKPOINT_MANIFEST


def _canonical_checkpoint_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(os.fspath(path)))


def _raise_checkpoint_phase_error(
    local_exception: BaseException | None,
    *,
    context: str,
    wrap_local_exception: bool = False,
) -> None:
    """Synchronize a phase, optionally adding context to local ordinary errors."""

    if (
        wrap_local_exception
        and local_exception is not None
        and isinstance(local_exception, Exception)
        and not _checkpoint_exception_started_mutation(local_exception)
        and (not dist.is_available() or not dist.is_initialized())
    ):
        raise RuntimeError(
            f"{context}: {type(local_exception).__name__}: {local_exception}"
        ) from local_exception
    _raise_checkpoint_load_commit_error(
        local_exception,
        mutation_started=_checkpoint_exception_started_mutation(local_exception),
        context=context,
    )


def _gather_world_objects(value: Any) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size()
    local_error = None
    try:
        dist.all_gather_object(gathered, value)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="checkpoint WORLD metadata exchange failed"
    )
    return gathered


def _assert_world_consensus(value: Any, *, context: str) -> None:
    gathered = _gather_world_objects(value)
    reference = gathered[0]
    if any(item != reference for item in gathered[1:]):
        raise RuntimeError(f"{context}: rank values={gathered!r}")


def _validate_extra_state_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError("extra-state filenames must be non-empty strings")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(
            f"extra-state filename must be a basename inside the checkpoint: {filename!r}"
        )
    if (
        filename == _CHECKPOINT_MANIFEST
        or filename in {"common.pt", "metadata.json"}
        or filename.startswith(".")
        or filename.endswith(".distcp")
        or filename.startswith("optimizer_rank_")
        or filename.startswith("rng_state_")
    ):
        raise ValueError(f"extra-state filename is reserved by MLite: {filename!r}")
    return filename


def _normalize_extra_state_filenames(
    filenames: Iterable[str] | None,
) -> tuple[str, ...]:
    if filenames is None:
        return ()
    if isinstance(filenames, (str, bytes)):
        raise TypeError(
            "load_extra_state_files must be an iterable of filenames, not a string"
        )
    normalized = tuple(_validate_extra_state_filename(name) for name in filenames)
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"duplicate extra-state filenames are not allowed: {normalized}"
        )
    return normalized


def _normalize_extra_states(extra_states: Mapping[str, Any] | None) -> dict[str, Any]:
    if extra_states is None:
        return {}
    if not isinstance(extra_states, Mapping):
        raise TypeError(
            "extra_states must be a mapping of filename to serializable state"
        )
    return {
        _validate_extra_state_filename(filename): value
        for filename, value in extra_states.items()
    }


def _normalize_extra_state_validators(
    validators: Mapping[str, ExtraStateValidator] | None, *, requested: Iterable[str]
) -> dict[str, ExtraStateValidator]:
    if validators is None:
        return {}
    if not isinstance(validators, Mapping):
        raise TypeError("extra_state_validators must be a mapping")
    normalized: dict[str, ExtraStateValidator] = {}
    for filename, validator in validators.items():
        filename = _validate_extra_state_filename(filename)
        if not callable(validator):
            raise TypeError(f"extra-state validator for {filename!r} must be callable")
        normalized[filename] = validator
    unrequested = sorted(set(normalized) - set(requested))
    if unrequested:
        raise ValueError(
            f"extra-state validators require matching requested files: {unrequested}"
        )
    return normalized


def _normalize_extra_state_targets(
    targets: Mapping[str, ExtraStateCheckpointTarget] | None,
    *,
    requested: Iterable[str],
) -> dict[str, ExtraStateCheckpointTarget]:
    if targets is None:
        return {}
    if not isinstance(targets, Mapping):
        raise TypeError("extra_state_targets must be a mapping")
    normalized: dict[str, ExtraStateCheckpointTarget] = {}
    required_methods = ("validate", "snapshot", "apply", "restore", "fingerprint")
    for filename, target in targets.items():
        filename = _validate_extra_state_filename(filename)
        missing = [
            method
            for method in required_methods
            if not callable(getattr(target, method, None))
        ]
        if missing:
            raise TypeError(
                f"extra-state target for {filename!r} is missing methods: {missing}"
            )
        normalized[filename] = target
    unrequested = sorted(set(normalized) - set(requested))
    if unrequested:
        raise ValueError(
            f"extra-state targets require matching requested files: {unrequested}"
        )
    return normalized


def _checkpoint_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _fsync_file(path: Path) -> None:
    file_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(file_fd)
    finally:
        os.close(file_fd)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_checkpoint_manifest_with_consensus(
    checkpoint_path: str, payload: dict[str, Any], *, context: str
) -> None:
    _assert_world_consensus(
        {
            "checkpoint_path": _canonical_checkpoint_path(checkpoint_path),
            "manifest": payload,
        },
        context=f"{context}: manifest payload differs across ranks",
    )
    local_error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            _atomic_write_json(_checkpoint_manifest_path(checkpoint_path), payload)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context=context)
    readback: dict[str, Any] | None = None
    local_error = None
    try:
        manifest_path = _checkpoint_manifest_path(checkpoint_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest is not visible after rank0 publication: {manifest_path}"
            )
        parsed = json.loads(manifest_path.read_text())
        _validate_checkpoint_manifest_schema(parsed)
        if parsed != payload:
            raise RuntimeError(
                f"manifest readback differs from published payload: {parsed!r}"
            )
        readback = parsed
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context=f"{context}: manifest readback failed"
    )
    _assert_world_consensus(
        {
            "checkpoint_path": _canonical_checkpoint_path(checkpoint_path),
            "manifest": readback,
        },
        context=f"{context}: manifest readback differs across ranks",
    )


def _atomically_reserve_checkpoint_directory(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(destination)
        return
    except FileExistsError as exc:
        if not destination.is_dir():
            raise RuntimeError(
                f"checkpoint destination is not a directory: {destination}"
            ) from exc
        destination_entries = sorted(entry.name for entry in destination.iterdir())
        existing_manifest: dict[str, Any] | None = None
        manifest_path = _checkpoint_manifest_path(destination)
        if manifest_path.exists():
            parsed = json.loads(manifest_path.read_text())
            _validate_checkpoint_manifest_schema(parsed)
            existing_manifest = parsed
        status = (
            existing_manifest.get("status")
            if existing_manifest is not None
            else "unmanifested"
        )
        detail = (
            "checkpoint destination is non-empty and will not be overwritten"
            if destination_entries
            else "checkpoint destination already exists and cannot be atomically reserved"
        )
        raise FileExistsError(
            f"{detail}: {destination} "
            f"(status={status!r}, entries={destination_entries})"
        ) from exc


def _begin_checkpoint_transaction(
    checkpoint_path: str,
    *,
    step: int,
    save_model: bool,
    save_optimizer: bool,
    save_rng: bool,
    payload_format: str,
    optimizer_storage: str,
    extra_state_files: Mapping[str, Any] | Iterable[str] = (),
) -> dict[str, Any]:
    extra_state_files = tuple(sorted(extra_state_files))
    manifest = {
        "format": _CHECKPOINT_MANIFEST_FORMAT,
        "status": "incomplete",
        "step": step,
        "world_size": _checkpoint_world_size(),
        "components": {
            "model": bool(save_model),
            "optimizer": bool(save_optimizer),
            "rng": bool(save_rng),
            "extra_states": bool(extra_state_files),
        },
        "payload_format": payload_format,
        "optimizer_storage": optimizer_storage,
        "extra_state_files": list(extra_state_files),
    }
    local_error = None
    try:
        _validate_checkpoint_manifest_schema(manifest, expected_status="incomplete")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error,
        context="checkpoint transaction reservation request validation failed",
    )

    destination = Path(checkpoint_path)
    reservation = {
        "checkpoint_path": _canonical_checkpoint_path(checkpoint_path),
        "manifest": manifest,
    }
    _assert_world_consensus(
        reservation,
        context="checkpoint transaction reservation request differs across ranks",
    )

    local_error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            _atomically_reserve_checkpoint_directory(destination)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="checkpoint transaction atomic reservation failed"
    )

    destination_entries: list[str] = []
    local_error = None
    try:
        if not destination.is_dir():
            raise RuntimeError(
                "atomically reserved checkpoint directory is not visible: "
                f"{destination}"
            )
        destination_entries = sorted(entry.name for entry in destination.iterdir())
        if destination_entries:
            raise RuntimeError(
                "atomically reserved checkpoint directory was modified before "
                f"manifest publication: {destination_entries}"
            )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="checkpoint transaction reservation visibility failed"
    )
    _assert_world_consensus(
        {
            "checkpoint_path": _canonical_checkpoint_path(checkpoint_path),
            "entries": destination_entries,
        },
        context="checkpoint transaction reservation differs across ranks",
    )
    _write_checkpoint_manifest_with_consensus(
        checkpoint_path,
        manifest,
        context="checkpoint transaction initialization failed",
    )
    return manifest


def _complete_checkpoint_transaction(
    checkpoint_path: str, manifest: dict[str, Any]
) -> None:
    completed = dict(manifest)
    completed["status"] = "complete"
    _validate_checkpoint_manifest_schema(completed, expected_status="complete")
    _write_checkpoint_manifest_with_consensus(
        checkpoint_path,
        completed,
        context="checkpoint completion manifest write failed",
    )


def _save_rank_sidecar_with_consensus(save_fn, *, context: str) -> None:
    local_error = None
    try:
        save_fn()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context=context)


def _save_rank0_extra_states_with_consensus(
    checkpoint_path: str, extra_states: Mapping[str, Any]
) -> None:
    if not extra_states:
        return
    local_error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            for filename, value in extra_states.items():
                _atomic_torch_save(value, Path(checkpoint_path) / filename)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context="rank0 extra-state save failed")


def _validate_checkpoint_manifest_schema(
    manifest: Any, *, expected_status: str | None = None
) -> None:
    """Validate the exact v2 completion-manifest schema without filesystem I/O."""

    if not isinstance(manifest, dict):
        raise RuntimeError("manifest must be a dictionary")
    expected_top_level = {
        "format",
        "status",
        "step",
        "world_size",
        "components",
        "payload_format",
        "optimizer_storage",
        "extra_state_files",
    }
    actual_top_level = set(manifest)
    if actual_top_level != expected_top_level:
        raise RuntimeError(
            "manifest top-level schema mismatch: "
            f"missing={sorted(expected_top_level - actual_top_level)}, "
            f"unexpected={sorted(actual_top_level - expected_top_level)}"
        )
    if manifest["format"] != _CHECKPOINT_MANIFEST_FORMAT:
        raise RuntimeError(f"unsupported format {manifest['format']!r}")
    if manifest["status"] not in {"incomplete", "complete"}:
        raise RuntimeError(f"invalid checkpoint status {manifest['status']!r}")
    if expected_status is not None and manifest["status"] != expected_status:
        raise RuntimeError(f"checkpoint status is {manifest['status']!r}")

    manifest_step = manifest["step"]
    if type(manifest_step) is not int or manifest_step < 0:
        raise RuntimeError("manifest step must be a non-negative integer")
    world_size = manifest["world_size"]
    if type(world_size) is not int or world_size <= 0:
        raise RuntimeError("manifest world_size must be a positive integer")

    components = manifest["components"]
    expected_components = {"model", "optimizer", "rng", "extra_states"}
    if not isinstance(components, dict) or set(components) != expected_components:
        actual_components = set(components) if isinstance(components, dict) else set()
        raise RuntimeError(
            "manifest components schema mismatch: "
            f"missing={sorted(expected_components - actual_components)}, "
            f"unexpected={sorted(actual_components - expected_components)}"
        )
    non_boolean_components = sorted(
        name for name, enabled in components.items() if type(enabled) is not bool
    )
    if non_boolean_components:
        raise RuntimeError(
            f"manifest component values must be booleans: {non_boolean_components}"
        )

    payload_format = manifest["payload_format"]
    if not isinstance(payload_format, str) or payload_format not in {"dcp", "distckpt"}:
        raise RuntimeError(
            f"invalid payload_format {payload_format!r}; expected 'dcp' or 'distckpt'"
        )

    optimizer_storage = manifest["optimizer_storage"]
    valid_optimizer_storage = {"none", "rank_sidecar", "distckpt"}
    if (
        not isinstance(optimizer_storage, str)
        or optimizer_storage not in valid_optimizer_storage
    ):
        raise RuntimeError(
            f"invalid optimizer_storage {optimizer_storage!r}; expected one of "
            f"{sorted(valid_optimizer_storage)}"
        )
    if components["optimizer"] != (optimizer_storage != "none"):
        raise RuntimeError(
            "manifest optimizer component does not match optimizer_storage"
        )
    expected_optimizer_storage = (
        "distckpt" if payload_format == "distckpt" else "rank_sidecar"
    )
    if components["optimizer"] and optimizer_storage != expected_optimizer_storage:
        raise RuntimeError(
            "manifest payload_format is inconsistent with optimizer_storage: "
            f"payload_format={payload_format!r}, optimizer_storage={optimizer_storage!r}"
        )

    extra_state_files = manifest["extra_state_files"]
    if not isinstance(extra_state_files, list):
        raise RuntimeError("manifest extra_state_files must be a list")
    normalized_extra_state_files = _normalize_extra_state_filenames(extra_state_files)
    if components["extra_states"] != bool(normalized_extra_state_files):
        raise RuntimeError(
            "manifest extra_states component does not match extra_state_files"
        )


def _read_checkpoint_manifest_with_consensus(
    checkpoint_path: str,
) -> dict[str, Any] | None:
    """Read one manifest locally, then require identical path/content on WORLD."""

    manifest_path = _checkpoint_manifest_path(checkpoint_path)
    manifest_present = False
    manifest: dict[str, Any] | None = None
    local_error = None
    try:
        manifest_present = manifest_path.exists()
        if manifest_present:
            parsed = json.loads(manifest_path.read_text())
            if not isinstance(parsed, dict):
                raise RuntimeError("manifest JSON root must be a dictionary")
            manifest = parsed
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="checkpoint completion manifest read failed"
    )
    snapshot = {
        "checkpoint_path": _canonical_checkpoint_path(checkpoint_path),
        "manifest_present": manifest_present,
        "manifest": manifest,
    }
    _assert_world_consensus(
        snapshot, context="checkpoint manifest differs across ranks"
    )
    return manifest


def _validate_checkpoint_manifest(
    checkpoint_path: str,
    *,
    load_model: bool,
    load_optimizer: bool,
    load_rng: bool,
    allow_legacy_checkpoint: bool = False,
    load_extra_state_files: Iterable[str] = (),
) -> dict[str, Any] | None:
    requested_extra_state_files = _normalize_extra_state_filenames(
        load_extra_state_files
    )
    manifest = _read_checkpoint_manifest_with_consensus(checkpoint_path)
    if manifest is None:
        if not allow_legacy_checkpoint:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has no completion manifest. "
                "Legacy DCP checkpoints are rejected by default; pass "
                "allow_legacy_checkpoint=True only for an explicit migration load."
            )
        log_rank0(
            f"Checkpoint {checkpoint_path} predates completion manifests; "
            "loading through the explicitly enabled legacy migration path."
        )
        return None

    local_error = None
    try:
        _validate_checkpoint_manifest_schema(manifest, expected_status="complete")
        manifest_step = manifest["step"]
        dirname = Path(checkpoint_path).name
        if dirname.startswith("step_"):
            try:
                directory_step = int(dirname.removeprefix("step_"))
            except ValueError:
                directory_step = None
            if directory_step is not None and manifest_step != directory_step:
                raise RuntimeError(
                    f"manifest step {manifest_step} does not match directory {dirname!r}"
                )
        components = manifest["components"]
        requested = {"model": load_model, "optimizer": load_optimizer, "rng": load_rng}
        missing = [
            name
            for name, enabled in requested.items()
            if enabled and not components[name]
        ]
        if missing:
            raise RuntimeError(
                f"requested checkpoint components were not saved: {missing}"
            )

        declared_extra_state_files = _normalize_extra_state_filenames(
            manifest["extra_state_files"]
        )
        undeclared = sorted(
            set(requested_extra_state_files) - set(declared_extra_state_files)
        )
        if undeclared:
            raise RuntimeError(
                f"requested extra-state files were not saved: {undeclared}"
            )
        for filename in declared_extra_state_files:
            extra_state_path = Path(checkpoint_path) / filename
            if not extra_state_path.exists():
                raise FileNotFoundError(
                    "completed checkpoint is missing declared extra-state file "
                    f"{extra_state_path}"
                )

        saved_world_size = manifest["world_size"]
        current_world_size = _checkpoint_world_size()
        if load_rng and saved_world_size != current_world_size:
            raise RuntimeError(
                "exact RNG resume requires the saved world size: "
                f"saved={saved_world_size}, current={current_world_size}"
            )
        if (
            load_optimizer
            and manifest["optimizer_storage"] == "rank_sidecar"
            and saved_world_size != current_world_size
        ):
            raise RuntimeError(
                "rank-local optimizer sidecars require the saved world size: "
                f"saved={saved_world_size}, current={current_world_size}"
            )
        if load_rng and not _rng_sidecar_file(checkpoint_path).exists():
            raise FileNotFoundError(
                f"completed checkpoint is missing rank-local RNG sidecar "
                f"{_rng_sidecar_file(checkpoint_path)}"
            )
        if (
            load_optimizer
            and manifest["optimizer_storage"] == "rank_sidecar"
            and not Path(_optimizer_checkpoint_path(checkpoint_path)).exists()
        ):
            raise FileNotFoundError(
                "completed checkpoint is missing rank-local optimizer sidecar "
                f"{_optimizer_checkpoint_path(checkpoint_path)}"
            )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="checkpoint completion manifest validation failed"
    )
    return manifest


def _supports_dist_opt_distckpt(
    model: nn.Module | Iterable[nn.Module], optimizer
) -> bool:
    try:
        from megatron.lite.primitive.ckpt.distckpt import supports_dist_opt_distckpt
    except ModuleNotFoundError as exc:
        if exc.name != "megatron.core":
            raise
        return False

    return supports_dist_opt_distckpt(model, optimizer)


def _named_model_checkpoint_tensors(model: nn.Module):
    """Yield parameters plus exactly the buffers PyTorch marks persistent."""
    yield from model.named_parameters()
    yield from named_persistent_buffers(model)


def _validate_dcp_model_metadata(
    metadata_entries: Mapping[str, Any],
    *,
    model_prefix: str,
    parameter_items: list[tuple[str, torch.Tensor]],
    buffer_items: list[tuple[str, torch.Tensor]],
    checkpoint_tensor_templates: Mapping[str, torch.Tensor],
    require_schema: bool,
) -> tuple[bool, bool]:
    """Require an exact current-stage model schema before any DCP mutation.

    PyTorch DCP loads into the dtype and shape of the provided tensor template.
    Without checking the on-disk ``TensorStorageMetadata`` first, that behavior
    silently casts checkpoint tensors (for example FP32 to BF16) instead of
    enforcing an exact-resume contract.
    """
    metadata_keys = set(metadata_entries)
    parameter_keys = {f"{model_prefix}.{name}" for name, _tensor in parameter_items}
    buffer_keys = {f"{model_prefix}.{name}" for name, _tensor in buffer_items}
    expected_model_keys = parameter_keys | buffer_keys
    checkpoint_model_keys = {
        key for key in metadata_keys if key.startswith(f"{model_prefix}.")
    }
    missing_parameters = sorted(parameter_keys - metadata_keys)
    unexpected_model_keys = sorted(checkpoint_model_keys - expected_model_keys)
    present_buffers = buffer_keys & metadata_keys
    schema_present = _DCP_MODEL_SCHEMA_KEY in metadata_keys

    if missing_parameters:
        raise RuntimeError(
            f"checkpoint is missing required model parameters: {missing_parameters}"
        )
    if unexpected_model_keys:
        raise RuntimeError(
            "checkpoint contains unexpected model tensors for the current "
            f"pipeline stage: {unexpected_model_keys}"
        )
    if require_schema and not schema_present:
        raise RuntimeError(
            "completed versioned checkpoint is missing the required model "
            f"schema key {_DCP_MODEL_SCHEMA_KEY!r}"
        )
    if buffer_keys and present_buffers and present_buffers != buffer_keys:
        raise RuntimeError(
            "checkpoint contains only a subset of persistent model buffers: "
            f"missing={sorted(buffer_keys - present_buffers)}"
        )
    if schema_present and present_buffers != buffer_keys:
        raise RuntimeError(
            "versioned checkpoint is missing persistent model buffers: "
            f"{sorted(buffer_keys - present_buffers)}"
        )
    metadata_errors: list[str] = []
    for key in sorted(parameter_keys | present_buffers):
        saved = metadata_entries[key]
        expected = checkpoint_tensor_templates.get(key)
        if expected is None:
            metadata_errors.append(f"{key}: missing live tensor template")
            continue
        if not isinstance(saved, TensorStorageMetadata):
            metadata_errors.append(
                f"{key}: expected TensorStorageMetadata, got {type(saved).__name__}"
            )
            continue
        saved_shape = tuple(saved.size)
        expected_shape = tuple(expected.shape)
        if saved_shape != expected_shape:
            metadata_errors.append(
                f"{key}: global_shape {saved_shape} != {expected_shape}"
            )
        saved_dtype = saved.properties.dtype
        if saved_dtype != expected.dtype:
            metadata_errors.append(f"{key}: dtype {saved_dtype} != {expected.dtype}")
    if metadata_errors:
        raise RuntimeError(
            f"checkpoint model tensor metadata mismatch: {metadata_errors}"
        )
    return schema_present, not buffer_keys or present_buffers == buffer_keys


def _save_dist_opt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int,
    path: str,
    *,
    save_model: bool,
    save_optimizer: bool,
) -> None:
    from megatron.lite.primitive.ckpt.distckpt import save_dist_opt_checkpoint

    save_dist_opt_checkpoint(
        model,
        optimizer,
        step,
        path,
        save_model=save_model,
        save_optimizer=save_optimizer,
    )


def _load_dist_opt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    *,
    load_model: bool,
    load_optimizer: bool,
    expected_step: int | None = None,
    allow_legacy_checkpoint: bool = False,
) -> int:
    from megatron.lite.primitive.ckpt.distckpt import load_dist_opt_checkpoint

    return load_dist_opt_checkpoint(
        model,
        optimizer,
        path,
        load_model=load_model,
        load_optimizer=load_optimizer,
        expected_step=expected_step,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
    )


def _optimizer_checkpoint_path(path: str) -> str:
    rank = dist.get_rank() if dist.is_initialized() else 0
    return os.path.join(path, f"optimizer_rank_{rank}.pt")


def _atomic_torch_save(value: Any, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, tmp_path)
        _fsync_file(tmp_path)
        os.replace(tmp_path, destination)
        _fsync_file(destination)
        _fsync_directory(destination.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _save_optimizer_checkpoint(optimizer, path: str) -> None:
    if optimizer is None:
        log_rank0("Skipping optimizer checkpoint save because optimizer is None")
        return
    state_dict_fn = getattr(optimizer, "state_dict", None)
    if not callable(state_dict_fn):
        raise TypeError(
            f"Optimizer {type(optimizer).__name__} does not provide state_dict()."
        )
    _atomic_torch_save(state_dict_fn(), _optimizer_checkpoint_path(path))


def _read_optimizer_checkpoint(optimizer, path: str) -> Any:
    if optimizer is None:
        raise ValueError("optimizer checkpoint load requires a non-None optimizer")
    ckpt_path = _optimizer_checkpoint_path(path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"optimizer checkpoint requested by load_optimizer=True is missing: {ckpt_path}"
        )
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _apply_optimizer_checkpoint_state(optimizer, state: Any) -> None:
    load_state_dict_fn = getattr(optimizer, "load_state_dict", None)
    if not callable(load_state_dict_fn):
        raise TypeError(
            f"Optimizer {type(optimizer).__name__} does not provide load_state_dict()."
        )
    load_state_dict_fn(state)


def _preflight_optimizer_checkpoint_state(optimizer, state: Any) -> None:
    """Validate optimizer state structurally without mutating live tensors.

    A load/rollback dry-run is not safe at scale: FP32AdamW and similar
    optimizers copy into existing moment/master tensors in place, so a shallow
    state_dict snapshot aliases the mutation while a deep copy can require
    tens of gigabytes per rank. Supported optimizers therefore use a pure
    validator; custom wrappers must expose ``validate_state_dict``.
    """

    try:
        if not callable(getattr(optimizer, "load_state_dict", None)):
            raise TypeError(
                f"Optimizer wrapper {type(optimizer).__name__} does not provide "
                "load_state_dict()."
            )
        _validate_optimizer_checkpoint_state(optimizer, state)
    except Exception as exc:
        raise RuntimeError(
            f"optimizer checkpoint is incompatible: {type(exc).__name__}: {exc}"
        ) from exc


def _migrate_legacy_optimizer_checkpoint_state(optimizer, state: Any) -> Any:
    """Recursively apply an explicitly provided legacy-state migrator."""

    try:
        _validate_optimizer_checkpoint_state(optimizer, state)
        return state
    except Exception:
        # A strict current-format state needs no migration. Invalid/legacy state
        # continues below and is validated again after the opt-in migration.
        pass

    migrator = getattr(optimizer, "migrate_legacy_state_dict", None)
    if callable(migrator):
        migrated = migrator(state)
        if migrated is None:
            raise TypeError(
                f"{type(optimizer).__name__}.migrate_legacy_state_dict returned None"
            )
        return migrated

    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and inner is not optimizer:
        return _migrate_legacy_optimizer_checkpoint_state(inner, state)

    chained = getattr(optimizer, "optimizers", None)
    if isinstance(chained, (list, tuple)) and isinstance(state, dict):
        child_states = state.get("optimizers")
        if isinstance(child_states, list) and len(child_states) == len(chained):
            migrated_state = dict(state)
            migrated_state["optimizers"] = [
                _migrate_legacy_optimizer_checkpoint_state(child, child_state)
                for child, child_state in zip(chained, child_states, strict=True)
            ]
            return migrated_state
    return state


def _prepare_optimizer_checkpoint_state(
    optimizer, state: Any, *, allow_legacy_optimizer_state: bool
) -> Any:
    if not allow_legacy_optimizer_state:
        _preflight_optimizer_checkpoint_state(optimizer, state)
        return state
    try:
        _preflight_optimizer_checkpoint_state(optimizer, state)
        return state
    except RuntimeError:
        migrated = _migrate_legacy_optimizer_checkpoint_state(optimizer, state)
        _preflight_optimizer_checkpoint_state(optimizer, migrated)
        return migrated


def _validate_optimizer_checkpoint_state(optimizer, state: Any) -> None:
    validator = getattr(optimizer, "validate_state_dict", None)
    if callable(validator):
        validator(state)
        return

    mcore_chained = getattr(optimizer, "chained_optimizers", None)
    if isinstance(mcore_chained, (list, tuple)):
        if len(mcore_chained) == 1:
            _validate_optimizer_checkpoint_state(mcore_chained[0], state)
            return
        if not isinstance(state, list) or len(state) != len(mcore_chained):
            raise ValueError("Invalid MCore chained optimizer state_dict cardinality.")
        for child, child_state in zip(mcore_chained, state, strict=True):
            _validate_optimizer_checkpoint_state(child, child_state)
        return

    chained = getattr(optimizer, "optimizers", None)
    if isinstance(chained, (list, tuple)):
        if (
            not isinstance(state, dict)
            or state.get("type") != "chained_torch_optimizer"
        ):
            raise ValueError("Invalid chained optimizer state_dict format.")
        child_states = state.get("optimizers")
        if not isinstance(child_states, list) or len(child_states) != len(chained):
            raise ValueError("Invalid chained optimizer state_dict cardinality.")
        for child, child_state in zip(chained, child_states, strict=True):
            _validate_optimizer_checkpoint_state(child, child_state)
        return

    if (
        callable(getattr(optimizer, "get_parameter_state_dp_zero", None))
        and isinstance(state, dict)
        and "optimizer" in state
    ):
        _validate_mcore_distributed_optimizer_checkpoint_state(optimizer, state)
        return

    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and inner is not optimizer:
        _validate_optimizer_checkpoint_state(inner, state)
        return

    if isinstance(optimizer, torch.optim.Optimizer):
        _validate_torch_optimizer_checkpoint_state(optimizer, state)
        return

    raise TypeError(
        f"Optimizer {type(optimizer).__name__} must implement a non-mutating "
        "validate_state_dict(state) method."
    )


def _validate_mcore_distributed_optimizer_checkpoint_state(
    optimizer, state: dict[str, Any]
) -> None:
    if not isinstance(state, dict) or "optimizer" not in state:
        raise ValueError(
            "MCore distributed optimizer checkpoint must contain optimizer state."
        )
    unexpected_root = sorted(set(state) - {"optimizer", "grad_scaler"})
    if unexpected_root:
        raise ValueError(
            f"MCore distributed optimizer root has unexpected keys: {unexpected_root}"
        )
    grad_scaler = getattr(optimizer, "grad_scaler", None)
    if ("grad_scaler" in state) != (grad_scaler is not None):
        raise ValueError(
            "MCore distributed optimizer grad-scaler presence mismatch: "
            f"checkpoint={'grad_scaler' in state}, runtime={grad_scaler is not None}"
        )
    saved_optimizer = state.get("optimizer")
    if not isinstance(saved_optimizer, dict) or set(saved_optimizer) != {
        "param_groups"
    }:
        raise ValueError("MCore distributed optimizer inner schema mismatch.")
    saved_groups = saved_optimizer.get("param_groups")
    inner_optimizer = getattr(optimizer, "optimizer", None)
    runtime_groups = getattr(inner_optimizer, "param_groups", None)
    if not isinstance(saved_groups, list) or not isinstance(runtime_groups, list):
        raise TypeError("MCore distributed optimizer param_groups must be lists.")
    if len(saved_groups) != len(runtime_groups):
        raise ValueError("MCore distributed optimizer param-group count mismatch.")
    identifier_keys = {
        "wd_mult",
        "lr_mult",
        "is_expert_parallel",
        "is_decoupled_lr",
        "pre_wd_mult",
        "pre_lr_mult",
        "pre_is_expert_parallel",
        "pre_is_decoupled_lr",
    }
    try:
        from megatron.core.optimizer import distrib_optimizer as mcore_distrib_optimizer

        native_fallback_requires_step = not bool(
            getattr(mcore_distrib_optimizer, "HAVE_APEX_OR_TE")
        )
    except (ImportError, AttributeError):
        native_fallback_requires_step = isinstance(
            inner_optimizer, torch.optim.Optimizer
        )
    saved_steps: list[int | float] = []
    for group_index, (saved_group, runtime_group) in enumerate(
        zip(saved_groups, runtime_groups, strict=True)
    ):
        if not isinstance(saved_group, dict) or not isinstance(runtime_group, dict):
            raise TypeError("MCore distributed optimizer param groups must be dicts.")
        runtime_keys = set(runtime_group) - {"params"}
        extra_keys = set(saved_group) - runtime_keys
        required_keys = set(runtime_keys)
        if native_fallback_requires_step:
            required_keys.add("step")
        if not required_keys <= set(saved_group) or extra_keys not in (set(), {"step"}):
            raise ValueError(
                f"MCore distributed optimizer param-group {group_index} schema mismatch."
            )
        for name, saved_value in saved_group.items():
            if name == "step" and name not in runtime_group:
                if (
                    isinstance(saved_value, bool)
                    or not isinstance(saved_value, (int, float))
                    or not math.isfinite(float(saved_value))
                    or saved_value < 0
                ):
                    raise ValueError(
                        "MCore distributed optimizer group step is invalid: "
                        f"{saved_value!r}"
                    )
                continue
            runtime_value = runtime_group[name]
            _validate_torch_optimizer_option_value(
                f"optimizer.param_groups[{group_index}].{name}",
                saved_value,
                runtime_value,
            )
            if name in identifier_keys and (
                type(saved_value) is not type(runtime_value)
                or saved_value != runtime_value
            ):
                raise ValueError(
                    f"MCore distributed optimizer group identifier {name!r} "
                    f"mismatch: checkpoint={saved_value!r}, runtime={runtime_value!r}"
                )
        if "step" in saved_group:
            saved_steps.append(saved_group["step"])
    if native_fallback_requires_step and len(saved_steps) != len(saved_groups):
        raise ValueError(
            "MCore native distributed optimizer checkpoint requires step in every group."
        )
    if saved_steps and any(step != saved_steps[0] for step in saved_steps[1:]):
        raise ValueError(
            f"MCore distributed optimizer group steps are inconsistent: {saved_steps}"
        )
    if grad_scaler is not None:
        _validate_optimizer_aux_state_tree(
            state["grad_scaler"], grad_scaler.state_dict(), "optimizer.grad_scaler"
        )


def _validate_optimizer_aux_state_tree(
    candidate: Any, reference: Any, path: str
) -> None:
    if isinstance(reference, torch.Tensor):
        _validate_torch_optimizer_option_value(path, candidate, reference)
        return
    if isinstance(reference, Mapping):
        if not isinstance(candidate, Mapping) or set(candidate) != set(reference):
            raise ValueError(f"{path} mapping schema mismatch.")
        for key, value in reference.items():
            _validate_optimizer_aux_state_tree(
                candidate[key], value, f"{path}[{key!r}]"
            )
        return
    if isinstance(reference, (list, tuple)):
        if type(candidate) is not type(reference) or len(candidate) != len(reference):
            raise ValueError(f"{path} sequence schema mismatch.")
        for index, (item, value) in enumerate(zip(candidate, reference, strict=True)):
            _validate_optimizer_aux_state_tree(item, value, f"{path}[{index}]")
        return
    _validate_torch_optimizer_option_value(path, candidate, reference)


def _validate_torch_optimizer_checkpoint_state(
    optimizer: torch.optim.Optimizer, state: Any
) -> None:
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("Invalid torch optimizer state_dict schema.")
    saved_groups = state["param_groups"]
    saved_state = state["state"]
    if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
        raise TypeError("Torch optimizer state and param_groups must be dict/list.")
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("Torch optimizer param-group count mismatch.")

    runtime_state_dict = optimizer.state_dict()
    runtime_groups = runtime_state_dict["param_groups"]
    saved_id_to_param: dict[Any, torch.Tensor] = {}
    saved_id_to_group: dict[Any, dict[str, Any]] = {}
    for group_index, (saved_group, live_group, runtime_group) in enumerate(
        zip(saved_groups, optimizer.param_groups, runtime_groups, strict=True)
    ):
        if not isinstance(saved_group, dict) or set(saved_group) != set(runtime_group):
            raise ValueError(
                f"Torch optimizer param-group {group_index} schema mismatch: "
                f"checkpoint={sorted(saved_group) if isinstance(saved_group, dict) else None}, "
                f"runtime={sorted(runtime_group)}."
            )
        saved_params = saved_group.get("params")
        live_params = live_group.get("params")
        if not isinstance(saved_params, list) or not isinstance(live_params, list):
            raise TypeError(
                "Torch optimizer param groups must contain parameter lists."
            )
        if len(saved_params) != len(live_params):
            raise ValueError("Torch optimizer parameter count mismatch.")
        for saved_id, live_param in zip(saved_params, live_params, strict=True):
            if saved_id in saved_id_to_param:
                raise ValueError(
                    f"Duplicate saved optimizer parameter id {saved_id!r}."
                )
            saved_id_to_param[saved_id] = live_param
            saved_id_to_group[saved_id] = saved_group
        for name, reference in runtime_group.items():
            if name == "params":
                continue
            _validate_torch_optimizer_option_value(
                f"param_groups[{group_index}].{name}", saved_group[name], reference
            )
        if (
            "max_lr" in saved_group
            and "min_lr" in saved_group
            and saved_group["max_lr"] < saved_group["min_lr"]
        ):
            raise ValueError(
                f"Torch optimizer param-group {group_index} max_lr must be >= min_lr."
            )
        if (
            "start_wd" in saved_group
            and "end_wd" in saved_group
            and saved_group["end_wd"] < saved_group["start_wd"]
        ):
            raise ValueError(
                f"Torch optimizer param-group {group_index} end_wd must be >= start_wd."
            )

    unknown = sorted(set(saved_state) - set(saved_id_to_param), key=repr)
    if unknown:
        raise ValueError(f"Optimizer state contains unknown parameter ids: {unknown}")
    current_state = optimizer.state
    expected_initialized_ids = {
        saved_id
        for saved_id, live_param in saved_id_to_param.items()
        if current_state.get(live_param)
    }
    missing_initialized_ids = sorted(
        expected_initialized_ids - set(saved_state), key=repr
    )
    if missing_initialized_ids:
        raise ValueError(
            "Optimizer state is missing initialized parameter ids: "
            f"{missing_initialized_ids}"
        )
    for saved_id, param_state in saved_state.items():
        if not isinstance(param_state, dict):
            raise TypeError(
                f"Optimizer state for parameter {saved_id!r} must be a dict."
            )
        live_param = saved_id_to_param[saved_id]
        live_state = current_state.get(live_param, {})
        expected_keys = _expected_torch_optimizer_state_keys(
            optimizer, saved_id_to_group[saved_id]
        )
        if expected_keys is not None and set(param_state) != expected_keys:
            raise ValueError(
                f"Optimizer state key mismatch for parameter {saved_id!r}: "
                f"checkpoint={sorted(param_state)}, expected={sorted(expected_keys)}."
            )
        if live_state and set(param_state) != set(live_state):
            raise ValueError(
                f"Optimizer state key mismatch for parameter {saved_id!r}: "
                f"checkpoint={sorted(param_state)}, runtime={sorted(live_state)}."
            )
        for name, value in param_state.items():
            live_value = live_state.get(name)
            if not isinstance(value, torch.Tensor):
                if live_state:
                    _validate_torch_optimizer_option_value(
                        f"state[{saved_id!r}].{name}", value, live_value
                    )
                continue
            local_value = _to_local_tensor(value)
            if name == "step" and (
                local_value.numel() != 1
                or not torch.isfinite(local_value).all()
                or local_value.item() < 0
            ):
                raise ValueError(
                    f"Optimizer step for parameter {saved_id!r} must be a "
                    "finite non-negative scalar."
                )
            if isinstance(live_value, torch.Tensor):
                local_live = _to_local_tensor(live_value)
                if (
                    local_value.shape != local_live.shape
                    or local_value.dtype != local_live.dtype
                ):
                    raise ValueError(
                        f"Optimizer tensor {name!r} shape/dtype mismatch: "
                        f"checkpoint={tuple(local_value.shape)}/{local_value.dtype}, "
                        f"live={tuple(local_live.shape)}/{local_live.dtype}."
                    )
            elif local_value.numel() != 1:
                local_param = _to_local_tensor(live_param)
                if local_value.shape != local_param.shape:
                    raise ValueError(
                        f"Optimizer tensor {name!r} shape mismatch for parameter "
                        f"{saved_id!r}: checkpoint={tuple(local_value.shape)}, "
                        f"parameter={tuple(local_param.shape)}."
                    )


def _expected_torch_optimizer_state_keys(
    optimizer: torch.optim.Optimizer, param_group: Mapping[str, Any]
) -> set[str] | None:
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        keys = {"step", "exp_avg", "exp_avg_sq"}
        if bool(param_group.get("amsgrad", False)):
            keys.add("max_exp_avg_sq")
        return keys
    if isinstance(optimizer, torch.optim.Adagrad):
        return {"step", "sum"}
    if isinstance(optimizer, torch.optim.RMSprop):
        keys = {"step", "square_avg"}
        if float(param_group.get("momentum", 0.0)) > 0.0:
            keys.add("momentum_buffer")
        if bool(param_group.get("centered", False)):
            keys.add("grad_avg")
        return keys
    if isinstance(optimizer, torch.optim.SGD):
        if float(param_group.get("momentum", 0.0)) > 0.0:
            return {"momentum_buffer"}
        return set()
    return None


def _validate_torch_optimizer_option_value(
    name: str, value: Any, reference: Any
) -> None:
    if isinstance(reference, torch.Tensor):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Torch optimizer {name} must be a tensor.")
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(
                f"Torch optimizer {name} shape/dtype mismatch: "
                f"checkpoint={tuple(value.shape)}/{value.dtype}, "
                f"runtime={tuple(reference.shape)}/{reference.dtype}."
            )
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Torch optimizer {name} must be finite.")
        return
    if isinstance(reference, bool):
        if not isinstance(value, bool):
            raise TypeError(f"Torch optimizer {name} must be bool.")
        return
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Torch optimizer {name} must be numeric.")
        if not math.isfinite(float(value)):
            raise ValueError(f"Torch optimizer {name} must be finite.")
        leaf_name = name.rsplit(".", maxsplit=1)[-1]
        if (
            leaf_name
            in {
                "lr",
                "initial_lr",
                "weight_decay",
                "eps",
                "step",
                "wd_mult",
                "lr_mult",
                "max_lr",
                "min_lr",
                "start_wd",
                "end_wd",
            }
            and value < 0
        ):
            raise ValueError(f"Torch optimizer {name} must be non-negative.")
        return
    if isinstance(reference, tuple):
        if not isinstance(value, tuple) or len(value) != len(reference):
            raise TypeError(f"Torch optimizer {name} must be a matching tuple.")
        for index, (item, expected) in enumerate(zip(value, reference, strict=True)):
            _validate_torch_optimizer_option_value(f"{name}[{index}]", item, expected)
        if name.endswith(".betas") and not all(0.0 <= item < 1.0 for item in value):
            raise ValueError(f"Torch optimizer {name} entries must lie in [0, 1).")
        return
    if isinstance(reference, list):
        if not isinstance(value, list) or len(value) != len(reference):
            raise TypeError(f"Torch optimizer {name} must be a matching list.")
        for index, (item, expected) in enumerate(zip(value, reference, strict=True)):
            _validate_torch_optimizer_option_value(f"{name}[{index}]", item, expected)
        return
    if reference is None:
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise TypeError(
                f"Torch optimizer {name} must be None or a scalar option value."
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Torch optimizer {name} must be finite.")
        return
    if not isinstance(value, type(reference)):
        raise TypeError(
            f"Torch optimizer {name} must have type {type(reference).__name__}."
        )


def _model_chunks(model: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
    if isinstance(model, nn.Module):
        return [model]
    chunks = list(model)
    if not all(isinstance(chunk, nn.Module) for chunk in chunks):
        raise TypeError("checkpoint model chunks must be nn.Module instances.")
    return chunks


def _to_local_tensor(tensor: Any) -> torch.Tensor:
    local_tensor = getattr(tensor, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def _is_dtensor_like(tensor: Any) -> bool:
    return (
        callable(getattr(tensor, "to_local", None))
        and hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
    )


def _dcp_tensor_from_param(
    param: torch.Tensor, mesh: DeviceMesh, placements: list
) -> DTensor:
    if _is_dtensor_like(param):
        return _dtensor_from_dtensor_like_param(param, _to_local_tensor(param).detach())
    return DTensor.from_local(_to_local_tensor(param).detach(), mesh, placements)


def _empty_dcp_tensor_like_param(
    param: torch.Tensor, mesh: DeviceMesh, placements: list
) -> DTensor:
    if _is_dtensor_like(param):
        return _dtensor_from_dtensor_like_param(
            param, torch.empty_like(_to_local_tensor(param))
        )
    return DTensor.from_local(
        torch.empty_like(_to_local_tensor(param)), mesh, placements
    )


def _dtensor_from_dtensor_like_param(
    param: torch.Tensor, local_tensor: torch.Tensor
) -> DTensor:
    return DTensor.from_local(
        local_tensor,
        param.device_mesh,
        param.placements,
        shape=tuple(param.shape),
        stride=tuple(param.stride()),
    )


def _copy_tensor_(
    target: torch.Tensor,
    src: torch.Tensor,
    *,
    before_copy: Callable[[], None] | None = None,
) -> None:
    local_target = _to_local_tensor(target)
    local_src = _to_local_tensor(src).to(
        device=local_target.device, dtype=local_target.dtype
    )
    if before_copy is not None:
        before_copy()
    if isinstance(local_target, torch.Tensor) and local_target is not target:
        local_target.copy_(local_src)
    else:
        target.copy_(local_src)


def _chunk_tensor_state(module: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        state[f"param.{name}"] = _to_local_tensor(param.detach()).cpu().clone()
    for name, buffer in named_persistent_buffers(module):
        state[f"buffer.{name}"] = _to_local_tensor(buffer.detach()).cpu().clone()
    return state


def _preflight_chunk_tensor_state(
    module: nn.Module,
    state: Any,
    *,
    chunk_index: int,
    allow_legacy_nonpersistent_buffers: bool = False,
) -> tuple[str, ...]:
    """Validate one local model chunk without mutating any live tensor."""

    if not isinstance(state, dict):
        raise TypeError(
            f"checkpoint model chunk {chunk_index} must be a dict, got "
            f"{type(state).__name__}"
        )
    persistent_buffers = dict(named_persistent_buffers(module))
    targets = {
        **{f"param.{name}": tensor for name, tensor in module.named_parameters()},
        **{f"buffer.{name}": tensor for name, tensor in persistent_buffers.items()},
    }
    saved_keys = set(state)
    expected_keys = set(targets)
    missing = sorted(expected_keys - saved_keys)
    ignored_legacy_keys: set[str] = set()
    if allow_legacy_nonpersistent_buffers:
        persistent_buffer_names = set(persistent_buffers)
        nonpersistent_buffer_keys = {
            f"buffer.{name}"
            for name, _buffer in module.named_buffers()
            if name not in persistent_buffer_names
        }
        ignored_legacy_keys = (saved_keys - expected_keys) & nonpersistent_buffer_keys
    unexpected = sorted(saved_keys - expected_keys - ignored_legacy_keys, key=repr)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint model chunk {chunk_index} schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key in sorted(ignored_legacy_keys):
        source = state[key]
        if not isinstance(source, torch.Tensor):
            raise TypeError(
                f"legacy checkpoint model chunk {chunk_index} non-persistent buffer "
                f"{key!r} must be torch.Tensor, got {type(source).__name__}"
            )
    for key, target in targets.items():
        source = state[key]
        if not isinstance(source, torch.Tensor):
            raise TypeError(
                f"checkpoint model chunk {chunk_index} tensor {key!r} must be "
                f"torch.Tensor, got {type(source).__name__}"
            )
        local_source = _to_local_tensor(source)
        local_target = _to_local_tensor(target)
        if not isinstance(local_source, torch.Tensor):
            raise TypeError(
                f"checkpoint model chunk {chunk_index} tensor {key!r} did not "
                "materialize as torch.Tensor"
            )
        if (
            tuple(local_source.shape) != tuple(local_target.shape)
            or local_source.dtype != local_target.dtype
        ):
            raise RuntimeError(
                f"checkpoint model chunk {chunk_index} tensor {key!r} "
                "shape/dtype mismatch: "
                f"checkpoint={tuple(local_source.shape)}/{local_source.dtype}, "
                f"target={tuple(local_target.shape)}/{local_target.dtype}"
            )
    return tuple(sorted(ignored_legacy_keys))


def _load_chunk_tensor_state(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    before_copy: Callable[[], None] | None = None,
) -> None:
    targets = {
        **{f"param.{name}": tensor for name, tensor in module.named_parameters()},
        **{
            f"buffer.{name}": tensor
            for name, tensor in named_persistent_buffers(module)
        },
    }
    for key, target in targets.items():
        with torch.no_grad():
            _copy_tensor_(target, state[key], before_copy=before_copy)


def _local_checkpoint_file(path: str | os.PathLike[str]) -> Path:
    ckpt_path = Path(path)
    if (
        _is_distributed_checkpoint_ranked()
        and not ckpt_path.is_dir()
        and ckpt_path.suffix != ""
    ):
        raise ValueError(
            "Distributed local checkpoints require a directory path so every "
            f"rank receives a distinct file; explicit file path is unsafe: {ckpt_path}"
        )
    if ckpt_path.is_dir() or ckpt_path.suffix == "":
        if _is_distributed_checkpoint_ranked():
            return ckpt_path / f"training_state_{_rank_suffix()}.pt"
        return ckpt_path / "training_state.pt"
    return ckpt_path


def _local_checkpoint_parallel_identity(
    config, ps: ParallelState | None
) -> tuple[dict[str, Any] | None, dict[str, int] | None]:
    if config is None or ps is None:
        if _is_distributed_checkpoint_ranked():
            raise ValueError(
                "Distributed local checkpoints require ParallelConfig and "
                "ParallelState to bind each rank shard to its topology."
            )
        return None, None

    world_size = _checkpoint_world_size()
    tp = int(getattr(config, "tp", 1) or 1)
    etp = max(int(getattr(config, "etp", 1) or 1), 1)
    ep = int(getattr(config, "ep", 1) or 1)
    pp = max(int(getattr(config, "pp", 1) or 1), 1)
    vpp = max(int(getattr(config, "vpp", 1) or 1), 1)
    cp = max(int(getattr(config, "cp", 1) or 1), 1)
    if world_size % (tp * cp * pp) != 0 or world_size % (etp * ep * pp) != 0:
        raise RuntimeError(
            "local checkpoint parallel topology does not divide world size: "
            f"world={world_size}, tp={tp}, etp={etp}, ep={ep}, pp={pp}, cp={cp}"
        )
    dense_dp = world_size // (tp * cp * pp)
    expert_dp = world_size // (etp * ep * pp)
    expected_sizes = {
        "tp_size": tp,
        "etp_size": etp,
        "ep_size": ep,
        "pp_size": pp,
        "cp_size": cp,
        "dp_size": dense_dp,
        "expert_dp_size": expert_dp,
    }
    actual_sizes = {name: int(getattr(ps, name)) for name in expected_sizes}
    if actual_sizes != expected_sizes:
        raise RuntimeError(
            "ParallelState sizes do not match local checkpoint config: "
            f"expected={expected_sizes}, actual={actual_sizes}"
        )
    pp_layout = getattr(config, "pp_layout", None)
    pp_layout_json = (
        None
        if pp_layout is None
        else json.dumps(pp_layout, sort_keys=True, separators=(",", ":"))
    )
    topology: dict[str, Any] = {
        "world_size": world_size,
        "tp": tp,
        "etp": etp,
        "ep": ep,
        "pp": pp,
        "vpp": vpp,
        "cp": cp,
        "dp": dense_dp,
        "expert_dp": expert_dp,
        "pp_layout": pp_layout_json,
    }
    coordinate = {
        "global_rank": dist.get_rank() if _is_distributed_checkpoint_ranked() else 0,
        "tp_rank": int(ps.tp_rank),
        "etp_rank": int(ps.etp_rank),
        "ep_rank": int(ps.ep_rank),
        "pp_rank": int(ps.pp_rank),
        "cp_rank": int(ps.cp_rank),
        "dp_rank": int(ps.dp_rank),
        "expert_dp_rank": int(ps.expert_dp_rank),
    }
    bounds = {
        "global_rank": world_size,
        "tp_rank": tp,
        "etp_rank": etp,
        "ep_rank": ep,
        "pp_rank": pp,
        "cp_rank": cp,
        "dp_rank": dense_dp,
        "expert_dp_rank": expert_dp,
    }
    invalid = {
        name: value
        for name, value in coordinate.items()
        if value < 0 or value >= bounds[name]
    }
    if invalid:
        raise RuntimeError(f"invalid local checkpoint rank coordinate: {invalid}")
    return topology, coordinate


def _new_local_checkpoint_generation() -> str:
    generation: str | None = os.urandom(16).hex()
    if dist.is_available() and dist.is_initialized():
        values: list[str | None] = [generation if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(values, src=0)
        generation = values[0]
    if (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(character not in "0123456789abcdef" for character in generation)
    ):
        raise RuntimeError(f"invalid local checkpoint generation: {generation!r}")
    return generation


def _local_checkpoint_completion_path(path: str | os.PathLike[str]) -> Path:
    return Path(path) / _LOCAL_TRAINING_COMPLETION


def _local_checkpoint_completion_payload(
    *, step: int, generation: str, topology: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "format": _LOCAL_TRAINING_FORMAT_V3,
        "step": step,
        "generation": generation,
        "world_size": _checkpoint_world_size(),
        "topology": dict(topology),
    }


def _read_local_checkpoint_completion_with_consensus(
    path: str | os.PathLike[str], *, allow_missing: bool = False
) -> dict[str, Any] | None:
    if not _is_distributed_checkpoint_ranked():
        return None
    marker_path = _local_checkpoint_completion_path(path)
    marker_present = marker_path.is_file()
    _assert_world_consensus(
        marker_present,
        context="local checkpoint completion presence differs across ranks",
    )
    if not marker_present and allow_missing:
        log_rank0(
            "Loading a distributed local checkpoint without a V3 completion marker "
            "through explicit allow_legacy_checkpoint migration mode."
        )
        return None
    marker: dict[str, Any] | None = None
    local_exception: Exception | None = None
    try:
        parsed = json.loads(marker_path.read_text())
        if not isinstance(parsed, dict):
            raise TypeError("local checkpoint completion marker must be a dictionary")
        expected_keys = {"format", "step", "generation", "world_size", "topology"}
        if set(parsed) != expected_keys:
            raise RuntimeError(
                "local checkpoint completion marker schema mismatch: "
                f"missing={sorted(expected_keys - set(parsed))}, "
                f"unexpected={sorted(set(parsed) - expected_keys)}"
            )
        if parsed["format"] != _LOCAL_TRAINING_FORMAT_V3:
            raise RuntimeError("unsupported local checkpoint completion format")
        if type(parsed["step"]) is not int or parsed["step"] < 0:
            raise RuntimeError("invalid local checkpoint completion step")
        generation = parsed["generation"]
        if (
            not isinstance(generation, str)
            or len(generation) != 32
            or any(character not in "0123456789abcdef" for character in generation)
        ):
            raise RuntimeError("invalid local checkpoint completion generation")
        if parsed["world_size"] != _checkpoint_world_size():
            raise RuntimeError(
                "local checkpoint completion world size mismatch: "
                f"saved={parsed['world_size']}, current={_checkpoint_world_size()}"
            )
        if not isinstance(parsed["topology"], dict):
            raise RuntimeError("invalid local checkpoint completion topology")
        marker = parsed
    except Exception as exc:
        local_exception = exc
    _raise_checkpoint_phase_error(
        local_exception, context="local checkpoint completion validation failed"
    )
    assert marker is not None
    _assert_world_consensus(
        marker, context="local checkpoint completion differs across ranks"
    )
    return marker


def _local_optimizer_parameter_state_file(
    ckpt_file: Path, *, generation: str | None = None
) -> Path:
    generation_suffix = f".{generation}" if generation is not None else ""
    return ckpt_file.with_name(
        f"{ckpt_file.stem}.optimizer_parameter_state"
        f"{generation_suffix}{ckpt_file.suffix}"
    )


def _generation_scoped_optimizer_parameter_state_files(
    ckpt_file: Path,
) -> tuple[Path, ...]:
    """Return generation sidecars scoped to one rank-local checkpoint file."""

    prefix = f"{ckpt_file.stem}.optimizer_parameter_state."
    suffix = ckpt_file.suffix
    candidates: list[Path] = []
    for candidate in ckpt_file.parent.glob(f"{prefix}*{suffix}"):
        name = candidate.name
        generation = name[len(prefix) :]
        if suffix:
            generation = generation[: -len(suffix)]
        if len(generation) == 32 and all(
            character in "0123456789abcdef" for character in generation
        ):
            candidates.append(candidate)
    return tuple(sorted(candidates))


def _garbage_collect_optimizer_parameter_state_files(
    candidates: Iterable[Path], *, keep: Path | None
) -> None:
    deleted_parents: set[Path] = set()
    for candidate in candidates:
        if keep is not None and candidate == keep:
            continue
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_file()):
            raise RuntimeError(
                "optimizer parameter-state garbage-collection target must be a "
                f"regular file: {candidate}"
            )
        if candidate.exists():
            candidate.unlink()
            deleted_parents.add(candidate.parent)
    for parent in sorted(deleted_parents):
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _local_transaction_file(destination: Path, phase: str) -> Path:
    for _ in range(1000):
        candidate = destination.with_name(
            f".{destination.name}.mlite-{phase}-{os.getpid()}-{os.urandom(8).hex()}"
        )
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"could not reserve local checkpoint staging path for {destination}"
    )


def _backup_local_checkpoint_file(destination: Path) -> Path | None:
    if not destination.exists():
        return None
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError(
            f"local checkpoint destination must be a regular file: {destination}"
        )
    backup = _local_transaction_file(destination, "backup")
    os.link(destination, backup)
    return backup


def _cleanup_local_checkpoint_transaction(
    prepared: Iterable[tuple[Path, Path, Path | None]], *, remove_backups: bool = True
) -> Exception | None:
    errors: list[str] = []
    for staged, _destination, backup in prepared:
        try:
            staged.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"{staged}: {type(exc).__name__}: {exc}")
        if backup is not None and remove_backups:
            try:
                backup.unlink(missing_ok=True)
            except Exception as exc:
                errors.append(f"{backup}: {type(exc).__name__}: {exc}")
    if errors:
        return RuntimeError("; ".join(errors))
    return None


def _rollback_local_checkpoint_commit(
    committed: Iterable[tuple[Path, Path | None]]
) -> Exception | None:
    errors: list[str] = []
    restored_parents: set[Path] = set()
    for destination, backup in reversed(list(committed)):
        try:
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(backup, destination)
                _fsync_file(destination)
            restored_parents.add(destination.parent)
        except Exception as exc:
            errors.append(
                f"destination={destination}, backup={backup}: "
                f"{type(exc).__name__}: {exc}"
            )
    for parent in sorted(restored_parents):
        try:
            _fsync_directory(parent)
        except Exception as exc:
            errors.append(f"parent={parent}: {type(exc).__name__}: {exc}")
    if errors:
        return RuntimeError("; ".join(errors))
    return None


_UNSUPPORTED_PARAMETER_STATE_TEMPLATE = object()


def _local_file_fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _validate_parameter_state_tree(candidate: Any, template: Any, path: str) -> None:
    if isinstance(template, torch.Tensor):
        if not isinstance(candidate, torch.Tensor):
            raise TypeError(
                f"{path} must be torch.Tensor, got {type(candidate).__name__}"
            )
        if candidate.shape != template.shape or candidate.dtype != template.dtype:
            raise ValueError(
                f"{path} tensor shape/dtype mismatch: "
                f"checkpoint={tuple(candidate.shape)}/{candidate.dtype}, "
                f"target={tuple(template.shape)}/{template.dtype}"
            )
        return
    if isinstance(template, Mapping):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"{path} must be a mapping, got {type(candidate).__name__}")
        if set(candidate) != set(template):
            raise ValueError(
                f"{path} mapping keys mismatch: "
                f"missing={sorted(set(template) - set(candidate), key=repr)}, "
                f"unexpected={sorted(set(candidate) - set(template), key=repr)}"
            )
        for key, expected in template.items():
            _validate_parameter_state_tree(candidate[key], expected, f"{path}[{key!r}]")
        return
    if isinstance(template, (list, tuple)):
        if type(candidate) is not type(template) or len(candidate) != len(template):
            raise TypeError(
                f"{path} must be {type(template).__name__} of length {len(template)}"
            )
        for index, (item, expected) in enumerate(zip(candidate, template, strict=True)):
            _validate_parameter_state_tree(item, expected, f"{path}[{index}]")
        return
    if type(candidate) is not type(template) or candidate != template:
        raise ValueError(
            f"{path} scalar mismatch: checkpoint={candidate!r}, target={template!r}"
        )


def _optimizer_parameter_state_template(optimizer) -> Any:
    chained = getattr(optimizer, "chained_optimizers", None)
    if isinstance(chained, (list, tuple)):
        if not chained:
            return None
        if len(chained) == 1:
            return _optimizer_parameter_state_template(chained[0])
        templates = []
        for child in chained:
            child_template = _optimizer_parameter_state_template(child)
            if child_template is not _UNSUPPORTED_PARAMETER_STATE_TEMPLATE:
                templates.append(child_template)
        if not templates:
            return _UNSUPPORTED_PARAMETER_STATE_TEMPLATE
        return templates if any(item is not None for item in templates) else None

    builder = getattr(optimizer, "get_parameter_state_dp_zero", None)
    if not callable(builder):
        return _UNSUPPORTED_PARAMETER_STATE_TEMPLATE
    return builder(empty_data=True)


def _optimizer_parameter_state_writer(optimizer) -> bool:
    """Return whether this rank owns the MCore DP-zero sidecar file."""

    chained = getattr(optimizer, "chained_optimizers", None)
    if isinstance(chained, (list, tuple)):
        eligible = [
            child
            for child in chained
            if callable(getattr(child, "save_parameter_state", None))
            or callable(getattr(child, "get_parameter_state_dp_zero", None))
        ]
        return any(_optimizer_parameter_state_writer(child) for child in eligible)

    group = getattr(optimizer, "data_parallel_group", None)
    if group is not None:
        group_rank = getattr(group, "rank", None)
        if callable(group_rank):
            return int(group_rank()) == 0
        return dist.get_rank(group) == 0

    inner = getattr(optimizer, "optimizer", None)
    if (
        inner is not None
        and inner is not optimizer
        and callable(getattr(inner, "save_parameter_state", None))
    ):
        return _optimizer_parameter_state_writer(inner)

    # Custom wrappers without an exposed DP group historically write on every
    # rank. Preserve that contract while handling MCore's explicit owner group.
    return True


def _hardlink_parameter_state_for_load(source: Path) -> Path:
    for counter in range(1000):
        staged = source.with_name(f".{source.name}.mlite-load-{os.getpid()}-{counter}")
        try:
            os.link(source, staged)
            return staged
        except FileExistsError:
            continue
    raise FileExistsError(
        f"could not reserve an immutable parameter-state staging link for {source}"
    )


def _atomic_save_optimizer_parameter_state(save_fn, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}"
    )
    try:
        save_fn(str(temporary))
        if temporary.exists():
            _fsync_file(temporary)
            os.replace(temporary, destination)
            _fsync_file(destination)
            _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_local_optimizer_parameter_state(
    optimizer, parameter_state_path: Path, *, update_legacy_format: bool
) -> tuple[Path, Path | None, tuple[int, str] | None]:
    staged_path: Path | None = None
    staged_fingerprint: tuple[int, str] | None = None
    load_path = parameter_state_path
    try:
        is_writer = _optimizer_parameter_state_writer(optimizer)
        if not is_writer:
            if parameter_state_path.exists():
                raise RuntimeError(
                    "non-owner rank unexpectedly has an optimizer parameter-state "
                    f"file: {parameter_state_path}"
                )
            return load_path, None, None
        validator = getattr(optimizer, "validate_parameter_state", None)
        template = _UNSUPPORTED_PARAMETER_STATE_TEMPLATE
        if not callable(validator):
            template = _optimizer_parameter_state_template(optimizer)
            if template is _UNSUPPORTED_PARAMETER_STATE_TEMPLATE:
                raise TypeError(
                    f"Optimizer {type(optimizer).__name__} must implement a pure "
                    "validate_parameter_state(filename, ...) contract or the MCore "
                    "get_parameter_state_dp_zero(empty_data=True) schema builder."
                )
        if callable(validator) or template is not None:
            staged_path = _hardlink_parameter_state_for_load(parameter_state_path)
            load_path = staged_path
            before_fingerprint = _local_file_fingerprint(staged_path)
            if callable(validator):
                validator(str(staged_path), update_legacy_format=update_legacy_format)
            else:
                candidate = torch.load(
                    staged_path, map_location="cpu", weights_only=False
                )
                _validate_parameter_state_tree(
                    candidate, template, "optimizer_parameter_state"
                )
            staged_fingerprint = _local_file_fingerprint(staged_path)
            if staged_fingerprint != before_fingerprint:
                raise RuntimeError(
                    "optimizer parameter-state file changed during preflight: "
                    f"before={before_fingerprint!r}, after={staged_fingerprint!r}"
                )
    except Exception as exc:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        raise RuntimeError(
            "local optimizer parameter-state preflight failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if staged_path is None:
        return load_path, None, None
    assert staged_fingerprint is not None
    return load_path, staged_path, staged_fingerprint


def _revalidate_local_optimizer_parameter_state(
    staged_path: Path | None, expected_fingerprint: tuple[int, str] | None
) -> None:
    try:
        if staged_path is not None:
            current_fingerprint = _local_file_fingerprint(staged_path)
            if current_fingerprint != expected_fingerprint:
                raise RuntimeError(
                    "optimizer parameter-state staging file changed after preflight: "
                    f"expected={expected_fingerprint!r}, current={current_fingerprint!r}"
                )
    except Exception as exc:
        raise RuntimeError(
            "local optimizer parameter-state revalidation failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _cleanup_local_optimizer_parameter_state_staging(
    staged_path: Path | None,
) -> Exception | None:
    try:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
    except Exception as exc:
        return exc
    return None


def _rank_suffix() -> str:
    if dist.is_available() and dist.is_initialized():
        return f"rank_{dist.get_rank():05d}"
    return "rank_00000"


def _is_distributed_checkpoint_ranked() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rng_sidecar_file(path: str | os.PathLike[str]) -> Path:
    return Path(path) / f"rng_state_{_rank_suffix()}.pt"


def _cpu_clone(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()


def _get_cuda_rng_state() -> torch.Tensor | None:
    if not torch.cuda.is_initialized():
        return None
    return _cpu_clone(torch.cuda.get_rng_state())


def _get_cuda_rng_tracker_states() -> dict[str, torch.Tensor]:
    if not torch.cuda.is_initialized():
        return {}

    from megatron.core import tensor_parallel

    tracker = tensor_parallel.get_cuda_rng_tracker()
    states = tracker.get_states()
    converted = {
        name: tensor_parallel.convert_cuda_rng_state(state, to_graphable=False)
        for name, state in states.items()
        if state is not None
    }
    result: dict[str, torch.Tensor] = {}
    for name, state in converted.items():
        if not isinstance(state, torch.Tensor):
            raise TypeError(
                f"CUDA RNG tracker state {name!r} converted to "
                f"{type(state).__name__}, expected torch.Tensor"
            )
        cloned = _cpu_clone(state)
        assert cloned is not None
        result[name] = cloned
    return result


def _get_rng_state() -> dict[str, Any]:
    return {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": _cpu_clone(torch.get_rng_state()),
        "cuda_rng_state": _get_cuda_rng_state(),
        "rng_tracker_states": _get_cuda_rng_tracker_states(),
    }


def _restore_cuda_rng_tracker_states(states: dict[str, torch.Tensor]) -> None:
    if not states or not torch.cuda.is_initialized():
        return
    try:
        from megatron.core import tensor_parallel

        tracker = tensor_parallel.get_cuda_rng_tracker()
        graph_safe = tensor_parallel.is_graph_safe_cuda_rng_tracker(tracker)
        restored = {
            name: tensor_parallel.convert_cuda_rng_state(state, to_graphable=graph_safe)
            for name, state in states.items()
        }
        tracker.set_states(restored)
    except Exception as exc:
        raise RuntimeError(
            "Failed to restore Megatron tensor-parallel RNG tracker state."
        ) from exc


def _validate_rng_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise TypeError(
            f"RNG checkpoint state must be a dictionary, got {type(state).__name__}"
        )
    required = {
        "random_rng_state",
        "np_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "rng_tracker_states",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"RNG checkpoint state is missing required keys: {missing}")
    if not isinstance(state["torch_rng_state"], torch.Tensor):
        raise TypeError("RNG checkpoint torch_rng_state must be a tensor")
    if state["cuda_rng_state"] is not None and not isinstance(
        state["cuda_rng_state"], torch.Tensor
    ):
        raise TypeError("RNG checkpoint cuda_rng_state must be a tensor or None")
    tracker_states = state["rng_tracker_states"]
    if not isinstance(tracker_states, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in tracker_states.items()
    ):
        raise TypeError("RNG checkpoint rng_tracker_states must map strings to tensors")
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    state = _validate_rng_state(state)
    random.setstate(state["random_rng_state"])
    np.random.set_state(state["np_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_state = state.get("cuda_rng_state")
    if cuda_rng_state is not None and torch.cuda.is_initialized():
        torch.cuda.set_rng_state(cuda_rng_state)
    _restore_cuda_rng_tracker_states(state.get("rng_tracker_states", {}))


def _save_rng_sidecar(path: str | os.PathLike[str]) -> None:
    rng_file = _rng_sidecar_file(path)
    _atomic_torch_save(_get_rng_state(), rng_file)


def _read_rng_sidecar(
    path: str | os.PathLike[str], *, required: bool = False
) -> dict[str, Any] | None:
    rng_file = _rng_sidecar_file(path)
    if not rng_file.exists():
        if required:
            raise FileNotFoundError(
                f"RNG sidecar required by completion manifest is missing: {rng_file}"
            )
        log_rank0(f"RNG sidecar not found at {rng_file}; skipping RNG restore.")
        return None
    return _validate_rng_state(
        torch.load(rng_file, map_location="cpu", weights_only=False)
    )


def _rng_state_values_equal(lhs: Any, rhs: Any) -> bool:
    """Compare nested RNG snapshots without ambiguous tensor/array truth values."""

    if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
        return (
            isinstance(lhs, torch.Tensor)
            and isinstance(rhs, torch.Tensor)
            and lhs.dtype == rhs.dtype
            and tuple(lhs.shape) == tuple(rhs.shape)
            and torch.equal(lhs, rhs)
        )
    if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
        return (
            isinstance(lhs, np.ndarray)
            and isinstance(rhs, np.ndarray)
            and lhs.dtype == rhs.dtype
            and lhs.shape == rhs.shape
            and np.array_equal(lhs, rhs)
        )
    if isinstance(lhs, Mapping) or isinstance(rhs, Mapping):
        return (
            isinstance(lhs, Mapping)
            and isinstance(rhs, Mapping)
            and set(lhs) == set(rhs)
            and all(_rng_state_values_equal(lhs[key], rhs[key]) for key in lhs)
        )
    if isinstance(lhs, (tuple, list)) or isinstance(rhs, (tuple, list)):
        return (
            type(lhs) is type(rhs)
            and len(lhs) == len(rhs)
            and all(
                _rng_state_values_equal(left, right)
                for left, right in zip(lhs, rhs, strict=True)
            )
        )
    return type(lhs) is type(rhs) and lhs == rhs


def _preflight_rng_checkpoint_state(state: dict[str, Any] | None) -> None:
    if state is None:
        return
    state = _validate_rng_state(state)
    original_state = _get_rng_state()
    candidate_error: BaseException | None = None
    try:
        _restore_rng_state(state)
    except BaseException as exc:
        candidate_error = exc
    rollback_error: BaseException | None = None
    try:
        _restore_rng_state(original_state)
    except BaseException as exc:
        rollback_error = exc
    if rollback_error is None:
        try:
            restored_state = _get_rng_state()
            if not _rng_state_values_equal(restored_state, original_state):
                raise RuntimeError(
                    "RNG rollback returned without error but the restored state "
                    "does not match the preflight snapshot"
                )
        except BaseException as exc:
            rollback_error = exc
    if rollback_error is not None:
        if _is_control_flow_exception(rollback_error):
            _mark_checkpoint_mutation(rollback_error)
            raise rollback_error
        if _is_control_flow_exception(candidate_error):
            assert candidate_error is not None
            _mark_checkpoint_mutation(candidate_error)
            raise candidate_error
        detail = (
            f"RNG rollback failed or could not be verified: "
            f"{type(rollback_error).__name__}: {rollback_error}"
        )
        if candidate_error is not None:
            detail = (
                f"candidate RNG load failed: {type(candidate_error).__name__}: "
                f"{candidate_error}; {detail}"
            )
        raise CheckpointLoadFatalError(
            "RNG checkpoint preflight mutated live state and could not restore it; "
            f"runtime must be poisoned: {detail}"
        ) from rollback_error
    if candidate_error is not None:
        if _is_control_flow_exception(candidate_error):
            raise candidate_error
        raise RuntimeError(
            f"RNG checkpoint is incompatible: {type(candidate_error).__name__}: "
            f"{candidate_error}"
        ) from candidate_error


def _assert_extra_state_target_fingerprints_match(
    fingerprints: Mapping[str, str], *, context: str
) -> None:
    if not dist.is_initialized():
        return
    gathered: list[dict[str, str] | None] = [None] * dist.get_world_size()
    local_error = None
    try:
        dist.all_gather_object(gathered, dict(fingerprints))
        reference = gathered[0]
        if any(fingerprint != reference for fingerprint in gathered[1:]):
            raise RuntimeError(
                f"extra-state target fingerprints differ by rank: {gathered}"
            )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context=context)


def _restore_extra_state_targets(
    targets: Mapping[str, Any],
    snapshots: Mapping[str, Any],
    baseline_fingerprints: Mapping[str, str],
) -> BaseException | str | None:
    errors: list[str] = []
    control_flow_error: BaseException | None = None
    for filename in reversed(sorted(snapshots)):
        target = targets[filename]
        try:
            target.restore(snapshots[filename])
            restored = _canonical_extra_state_fingerprint(target.fingerprint())
            if restored != baseline_fingerprints[filename]:
                raise RuntimeError(
                    f"fingerprint {restored} != baseline "
                    f"{baseline_fingerprints[filename]}"
                )
        except BaseException as exc:
            if _is_control_flow_exception(exc) and control_flow_error is None:
                _mark_checkpoint_mutation(exc)
                control_flow_error = exc
            else:
                errors.append(f"{filename}: {type(exc).__name__}: {exc}")
    try:
        _assert_extra_state_targets_unchanged(
            targets, baseline_fingerprints, callback="extra-state target rollback"
        )
    except BaseException as exc:
        if _is_control_flow_exception(exc) and control_flow_error is None:
            _mark_checkpoint_mutation(exc)
            control_flow_error = exc
        else:
            errors.append(f"global rollback verification: {type(exc).__name__}: {exc}")
    if control_flow_error is not None:
        return control_flow_error
    return "; ".join(errors) or None


def _canonical_extra_state_fingerprint(value: Any) -> str:
    """Hash a side-effect-free plain-data fingerprint deterministically."""

    def frame(tag: bytes, *parts: bytes) -> bytes:
        payload = bytearray(tag)
        for part in parts:
            payload.extend(len(part).to_bytes(8, "big"))
            payload.extend(part)
        return bytes(payload)

    def encode(item: Any) -> bytes:
        if item is None:
            return b"N"
        if isinstance(item, bool):
            return b"B1" if item else b"B0"
        if isinstance(item, int):
            return frame(b"I", str(item).encode("ascii"))
        if isinstance(item, float):
            return frame(b"F", item.hex().encode("ascii"))
        if isinstance(item, str):
            return frame(b"S", item.encode("utf-8"))
        if isinstance(item, bytes):
            return frame(b"Y", item)
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            raw = tensor.view(torch.uint8).numpy().tobytes()
            return frame(
                b"T",
                str(tensor.dtype).encode("ascii"),
                encode(tuple(tensor.shape)),
                hashlib.sha256(raw).digest(),
            )
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("object-dtype NumPy arrays are not canonical")
            array = np.ascontiguousarray(item)
            return frame(
                b"A",
                array.dtype.str.encode("ascii"),
                encode(tuple(array.shape)),
                hashlib.sha256(array.tobytes()).digest(),
            )
        if isinstance(item, np.generic):
            return frame(b"G", item.dtype.str.encode("ascii"), encode(item.item()))
        if isinstance(item, tuple):
            return frame(b"U", *(encode(element) for element in item))
        if isinstance(item, list):
            return frame(b"L", *(encode(element) for element in item))
        if isinstance(item, Mapping):
            encoded_items = sorted(
                (encode(key), encode(mapped_value))
                for key, mapped_value in item.items()
            )
            return frame(
                b"M",
                *(
                    frame(b"K", key, mapped_value)
                    for key, mapped_value in encoded_items
                ),
            )
        raise TypeError(
            "extra-state fingerprint must contain only canonical plain data; "
            f"got {type(item).__name__}"
        )

    return hashlib.sha256(encode(value)).hexdigest()


def _capture_extra_state_target_fingerprints(
    targets: Mapping[str, ExtraStateCheckpointTarget],
    ordered_filenames: Iterable[str],
    *,
    context: str,
) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for filename in ordered_filenames:
        try:
            fingerprints[filename] = _canonical_extra_state_fingerprint(
                targets[filename].fingerprint()
            )
        except Exception as exc:
            raise RuntimeError(
                f"extra-state target {filename!r} fingerprint failed during "
                f"{context}; live-state purity cannot be established: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return fingerprints


def _assert_extra_state_targets_unchanged(
    targets: Mapping[str, ExtraStateCheckpointTarget],
    baseline_fingerprints: Mapping[str, str],
    *,
    callback: str,
) -> None:
    current_fingerprints = _capture_extra_state_target_fingerprints(
        targets, sorted(targets), context=f"purity check after {callback}"
    )
    for filename, baseline in baseline_fingerprints.items():
        current = current_fingerprints[filename]
        if current != baseline:
            raise RuntimeError(
                f"{callback} mutated live target {filename!r}: "
                f"before={baseline}, after={current}"
            )


def _preflight_extra_state_targets(
    targets: Mapping[str, ExtraStateCheckpointTarget],
    extra_state_values: Mapping[str, Any],
    *,
    validators: Mapping[str, ExtraStateValidator] | None = None,
    checkpoint_step: int | None = None,
) -> None:
    """Run pure callbacks without applying payloads to live target state."""

    validators = validators or {}
    if not targets and not validators:
        return
    local_exception: BaseException | None = None
    mutation_observed = False
    baseline_fingerprints: dict[str, str] = {}
    try:
        ordered_filenames = sorted(targets)
        baseline_fingerprints = _capture_extra_state_target_fingerprints(
            targets, ordered_filenames, context="initial preflight baseline"
        )

        def run_pure_callback(callback, *, label: str) -> None:
            nonlocal mutation_observed
            callback_exception: BaseException | None = None
            try:
                callback()
            except BaseException as exc:
                callback_exception = exc
            try:
                _assert_extra_state_targets_unchanged(
                    targets, baseline_fingerprints, callback=label
                )
            except BaseException:
                mutation_observed = True
                if _is_control_flow_exception(callback_exception):
                    assert callback_exception is not None
                    _mark_checkpoint_mutation(callback_exception)
                    raise callback_exception
                raise
            if callback_exception is not None:
                raise callback_exception

        for filename in sorted(validators):
            run_pure_callback(
                lambda filename=filename: validators[filename](
                    extra_state_values[filename]
                ),
                label=f"extra-state validator {filename!r}",
            )

        for filename in ordered_filenames:
            validate_step = getattr(targets[filename], "validate_step", None)
            if callable(validate_step):
                run_pure_callback(
                    lambda filename=filename, validate_step=validate_step: validate_step(
                        extra_state_values[filename], checkpoint_step
                    ),
                    label=f"extra-state target {filename!r} validate_step()",
                )

        for filename in ordered_filenames:
            target = targets[filename]
            run_pure_callback(
                lambda filename=filename, target=target: target.validate(
                    extra_state_values[filename]
                ),
                label=f"extra-state target {filename!r} validate()",
            )
    except BaseException as exc:
        local_exception = exc
        if targets and not baseline_fingerprints:
            # A target's first fingerprint failed before a trusted baseline was
            # available. It may have changed live state before raising.
            mutation_observed = True
    if mutation_observed:
        _raise_checkpoint_load_commit_error(
            local_exception,
            mutation_started=True,
            context="extra-state target preflight failed",
        )
    else:
        _raise_checkpoint_phase_error(
            local_exception,
            context="extra-state target preflight failed",
            wrap_local_exception=True,
        )


def _commit_extra_state_targets(
    targets: Mapping[str, ExtraStateCheckpointTarget],
    extra_state_values: Mapping[str, Any],
    *,
    prior_live_state_mutation: bool = False,
) -> bool:
    if not targets:
        return prior_live_state_mutation
    snapshots: dict[str, Any] = {}
    baseline_fingerprints: dict[str, str] = {}
    candidate_fingerprints: dict[str, str] = {}
    local_exception: BaseException | None = None
    snapshot_mutation_observed = False
    try:
        ordered_filenames = sorted(targets)
        baseline_fingerprints = _capture_extra_state_target_fingerprints(
            targets, ordered_filenames, context="initial commit baseline"
        )
        for filename in ordered_filenames:
            target = targets[filename]
            try:
                snapshots[filename] = target.snapshot()
            except BaseException as exc:
                snapshot_mutation_observed = True
                if _is_control_flow_exception(exc):
                    _mark_checkpoint_mutation(exc)
                    raise
                raise RuntimeError(
                    f"extra-state target {filename!r} snapshot() failed; "
                    "live-state purity cannot be established: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                _assert_extra_state_targets_unchanged(
                    targets,
                    baseline_fingerprints,
                    callback=f"extra-state target {filename!r} snapshot()",
                )
            except BaseException:
                snapshot_mutation_observed = True
                raise
    except BaseException as exc:
        local_exception = exc
        if not baseline_fingerprints:
            snapshot_mutation_observed = True
    mutation_started = _raise_checkpoint_load_commit_error(
        local_exception,
        mutation_started=prior_live_state_mutation or snapshot_mutation_observed,
        context="extra-state target commit snapshot failed",
    )

    local_apply_started = False
    local_exception = None
    try:
        applied_filenames: set[str] = set()
        for filename in ordered_filenames:
            target = targets[filename]
            local_apply_started = True
            target.apply(extra_state_values[filename])
            live_fingerprints = _capture_extra_state_target_fingerprints(
                targets,
                ordered_filenames,
                context=f"commit verification after applying {filename!r}",
            )
            candidate_fingerprints[filename] = live_fingerprints[filename]
            applied_filenames.add(filename)
            expected_fingerprints = {
                observed_filename: (
                    candidate_fingerprints[observed_filename]
                    if observed_filename in applied_filenames
                    else baseline_fingerprints[observed_filename]
                )
                for observed_filename in ordered_filenames
            }
            for observed_filename, expected in expected_fingerprints.items():
                current = live_fingerprints[observed_filename]
                if current != expected:
                    raise RuntimeError(
                        f"extra-state target {filename!r} apply() mutated target "
                        f"{observed_filename!r} outside its own transaction: "
                        f"expected={expected}, current={current}"
                    )
        final_fingerprints = _capture_extra_state_target_fingerprints(
            targets, ordered_filenames, context="final commit verification"
        )
        if final_fingerprints != candidate_fingerprints:
            raise RuntimeError(
                "extra-state target final live fingerprints differ from the "
                f"applied candidates: expected={candidate_fingerprints}, "
                f"current={final_fingerprints}"
            )
    except BaseException as exc:
        local_exception = exc

    commit_error: CheckpointLoadFatalError | None = None
    try:
        mutation_started = _raise_checkpoint_load_commit_error(
            local_exception,
            mutation_started=mutation_started or local_apply_started,
            context="extra-state target commit failed",
        )
        try:
            _assert_extra_state_target_fingerprints_match(
                final_fingerprints,
                context="extra-state target commit fingerprint validation failed",
            )
        except Exception as exc:
            _raise_checkpoint_load_commit_error(
                exc,
                mutation_started=mutation_started,
                context="extra-state target commit fingerprint validation failed",
            )
    except CheckpointLoadFatalError as exc:
        if exc._peer_control_flow:
            raise
        commit_error = exc
    if commit_error is None:
        return mutation_started

    rollback_error = _restore_extra_state_targets(
        targets, snapshots, baseline_fingerprints
    )
    rollback_exception = (
        rollback_error
        if isinstance(rollback_error, BaseException)
        else RuntimeError(rollback_error) if rollback_error is not None else None
    )
    try:
        _raise_checkpoint_load_commit_error(
            rollback_exception,
            mutation_started=True,
            context="extra-state target rollback failed",
        )
    except CheckpointLoadFatalError as exc:
        raise CheckpointLoadFatalError(
            "extra-state target commit failed after core checkpoint commit; "
            f"runtime must be poisoned; rollback also failed: {exc}"
        ) from commit_error
    raise CheckpointLoadFatalError(
        "extra-state target commit failed after core checkpoint commit; "
        f"runtime must be poisoned: {commit_error}"
    ) from commit_error


def _preload_checkpoint_sidecars(
    checkpoint_path: str,
    *,
    optimizer,
    load_optimizer: bool,
    allow_legacy_optimizer_state: bool = False,
    load_rng: bool,
    rng_required: bool,
    extra_state_files: Iterable[str],
    extra_state_validators: Mapping[str, ExtraStateValidator],
    extra_state_targets: Mapping[str, ExtraStateCheckpointTarget],
    checkpoint_step: int | None,
) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
    """Deserialize and validate all sidecars before any model checkpoint commit."""

    optimizer_state: Any = None
    rng_state: dict[str, Any] | None = None
    extra_state_values: dict[str, Any] = {}
    local_exception: BaseException | None = None
    try:
        if load_optimizer:
            optimizer_state = _read_optimizer_checkpoint(optimizer, checkpoint_path)
        if load_rng:
            rng_state = _read_rng_sidecar(checkpoint_path, required=rng_required)
        for filename in extra_state_files:
            extra_state_values[filename] = torch.load(
                Path(checkpoint_path) / filename, map_location="cpu", weights_only=False
            )

        if load_optimizer:
            optimizer_state = _prepare_optimizer_checkpoint_state(
                optimizer,
                optimizer_state,
                allow_legacy_optimizer_state=allow_legacy_optimizer_state,
            )
        if load_rng:
            _preflight_rng_checkpoint_state(rng_state)
    except BaseException as exc:
        local_exception = exc
    _raise_checkpoint_phase_error(
        local_exception,
        context="checkpoint sidecar preflight failed",
        wrap_local_exception=True,
    )
    _preflight_extra_state_targets(
        extra_state_targets,
        extra_state_values,
        validators=extra_state_validators,
        checkpoint_step=checkpoint_step,
    )
    return optimizer_state, rng_state, extra_state_values


def _commit_preloaded_sidecars(
    *,
    optimizer,
    optimizer_state: Any,
    rng_state: dict[str, Any] | None,
    loaded_extra_states: MutableMapping[str, Any] | None,
    extra_state_values: Mapping[str, Any],
    extra_state_targets: Mapping[str, Any],
    prior_live_state_mutation: bool = False,
) -> None:
    mutation_started = prior_live_state_mutation
    if optimizer is not None:
        local_exception: BaseException | None = None
        optimizer_apply_started = False
        try:
            optimizer_apply_started = True
            _apply_optimizer_checkpoint_state(optimizer, optimizer_state)
        except BaseException as exc:
            local_exception = exc
        mutation_started = _raise_checkpoint_load_commit_error(
            local_exception,
            mutation_started=mutation_started or optimizer_apply_started,
            context="checkpoint optimizer sidecar commit failed",
        )
    if rng_state is not None:
        local_exception = None
        rng_apply_started = False
        try:
            rng_apply_started = True
            _restore_rng_state(rng_state)
        except BaseException as exc:
            local_exception = exc
        mutation_started = _raise_checkpoint_load_commit_error(
            local_exception,
            mutation_started=mutation_started or rng_apply_started,
            context="checkpoint RNG sidecar commit failed",
        )
    mutation_started = _commit_extra_state_targets(
        extra_state_targets,
        extra_state_values,
        prior_live_state_mutation=mutation_started,
    )
    if loaded_extra_states is not None:
        local_exception = None
        publication_started = False
        try:
            publication_started = True
            loaded_extra_states.update(extra_state_values)
        except BaseException as exc:
            local_exception = exc
        _raise_checkpoint_load_commit_error(
            local_exception,
            mutation_started=mutation_started or publication_started,
            context="checkpoint extra-state publication failed",
        )


def _save_local_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int,
    path: str,
    *,
    config=None,
    ps: ParallelState | None = None,
    save_rng: bool = True,
) -> None:
    _assert_world_consensus(
        {"checkpoint_path": _canonical_checkpoint_path(path), "requested_step": step},
        context="local checkpoint save identity differs across ranks",
    )
    topology: dict[str, Any] | None = None
    rank_coordinate: dict[str, int] | None = None
    local_exception: Exception | None = None
    try:
        topology, rank_coordinate = _local_checkpoint_parallel_identity(config, ps)
    except Exception as exc:
        local_exception = exc
    _raise_checkpoint_phase_error(
        local_exception, context="local checkpoint save topology preflight failed"
    )
    if _is_distributed_checkpoint_ranked():
        assert topology is not None
        _assert_world_consensus(
            topology, context="local checkpoint save topology differs across ranks"
        )
    generation = _new_local_checkpoint_generation()
    ckpt_file = _local_checkpoint_file(path)
    prepared: list[tuple[Path, Path, Path | None]] = []
    staged_parameter_state: Path | None = None
    staged_checkpoint: Path | None = None
    optimizer_parameter_state_file: Path | None = None
    save_parameter_state: Callable | None = None
    parameter_state_writer = False
    state: dict[str, Any] | None = None
    previous_parameter_state_files: tuple[Path, ...] = ()

    # Phase 1 is local-only. No rank may enter MCore's DP collectives until
    # every rank has built and validated its model/optimizer payload.
    local_exception = None
    try:
        chunks = _model_chunks(model)
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        previous_parameter_state_files = (
            _generation_scoped_optimizer_parameter_state_files(ckpt_file)
        )
        save_parameter_state = getattr(optimizer, "save_parameter_state", None)
        optimizer_parameter_state_file = (
            _local_optimizer_parameter_state_file(ckpt_file, generation=generation)
            if callable(save_parameter_state)
            else None
        )
        parameter_state_writer = (
            _optimizer_parameter_state_writer(optimizer)
            if optimizer_parameter_state_file is not None
            else False
        )
        state = {
            "format": _LOCAL_TRAINING_FORMAT_V3,
            "step": int(step),
            "generation": generation,
            "topology": topology,
            "rank_coordinate": rank_coordinate,
            "model": [_chunk_tensor_state(chunk) for chunk in chunks],
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "optimizer_parameter_state": (
                optimizer_parameter_state_file.name
                if optimizer_parameter_state_file is not None
                else None
            ),
            "rng_state": _get_rng_state() if save_rng else None,
        }
        if optimizer_parameter_state_file is not None:
            staged_parameter_state = _local_transaction_file(
                optimizer_parameter_state_file, "prepare"
            )
        staged_checkpoint = _local_transaction_file(ckpt_file, "prepare")
    except Exception as exc:
        local_exception = exc
    _raise_checkpoint_phase_error(
        local_exception, context="local checkpoint save payload preparation failed"
    )
    assert state is not None
    assert staged_checkpoint is not None

    # Phase 2 deliberately calls save_parameter_state on every rank. MCore
    # gathers within each DP group but materializes a file only on that group's
    # rank 0; the other ranks keep their rank-local logical filename for load.
    if optimizer_parameter_state_file is not None:
        assert callable(save_parameter_state)
        assert staged_parameter_state is not None
        local_exception = None
        try:
            _atomic_save_optimizer_parameter_state(
                save_parameter_state, staged_parameter_state
            )
            if parameter_state_writer and not staged_parameter_state.is_file():
                raise RuntimeError(
                    "optimizer parameter-state owner did not create its staging file: "
                    f"{staged_parameter_state}"
                )
            if not parameter_state_writer and staged_parameter_state.exists():
                prepared.append(
                    (staged_parameter_state, optimizer_parameter_state_file, None)
                )
                raise RuntimeError(
                    "non-owner rank unexpectedly created optimizer parameter-state "
                    f"staging file: {staged_parameter_state}"
                )
            if parameter_state_writer:
                prepared.append(
                    (staged_parameter_state, optimizer_parameter_state_file, None)
                )
        except Exception as exc:
            local_exception = exc
        try:
            _raise_checkpoint_phase_error(
                local_exception,
                context="local checkpoint optimizer parameter-state save failed",
            )
        except Exception as parameter_state_exception:
            cleanup_exception = _cleanup_local_checkpoint_transaction(prepared)
            try:
                _raise_checkpoint_phase_error(
                    cleanup_exception,
                    context="local checkpoint optimizer parameter-state cleanup failed",
                )
            except Exception as exc:
                raise RuntimeError(
                    "local checkpoint optimizer parameter-state save failed and "
                    f"cleanup also failed: {exc}"
                ) from parameter_state_exception
            raise

    # Phase 3 contains only rank-local file work and is followed by WORLD
    # consensus before any destination is published.
    local_exception = None
    try:
        if parameter_state_writer:
            assert optimizer_parameter_state_file is not None
            assert staged_parameter_state is not None
            prepared[-1] = (
                staged_parameter_state,
                optimizer_parameter_state_file,
                _backup_local_checkpoint_file(optimizer_parameter_state_file),
            )
        _atomic_torch_save(state, staged_checkpoint)
        prepared.append((staged_checkpoint, ckpt_file, None))
        prepared[-1] = (
            staged_checkpoint,
            ckpt_file,
            _backup_local_checkpoint_file(ckpt_file),
        )
    except Exception as exc:
        local_exception = exc

    try:
        _raise_checkpoint_phase_error(
            local_exception, context="local checkpoint save file materialization failed"
        )
    except Exception as preparation_exception:
        cleanup_exception = _cleanup_local_checkpoint_transaction(prepared)
        try:
            _raise_checkpoint_phase_error(
                cleanup_exception,
                context="local checkpoint save preparation cleanup failed",
            )
        except Exception as exc:
            raise RuntimeError(
                "local checkpoint save preparation failed and transaction cleanup "
                f"also failed: {exc}"
            ) from preparation_exception
        raise

    committed: list[tuple[Path, Path | None]] = []
    local_exception = None
    try:
        for staged, destination, backup in prepared:
            os.replace(staged, destination)
            committed.append((destination, backup))
            _fsync_file(destination)
        for parent in sorted(
            {destination.parent for destination, _backup in committed}
        ):
            _fsync_directory(parent)
    except Exception as exc:
        local_exception = exc

    commit_exception: Exception | None = None
    try:
        _raise_checkpoint_phase_error(
            local_exception, context="local checkpoint save commit failed"
        )
    except Exception as exc:
        commit_exception = exc
    if commit_exception is not None:
        rollback_exception = _rollback_local_checkpoint_commit(committed)
        cleanup_exception = _cleanup_local_checkpoint_transaction(
            prepared, remove_backups=rollback_exception is None
        )
        try:
            _raise_checkpoint_phase_error(
                rollback_exception, context="local checkpoint save rollback failed"
            )
        except Exception as exc:
            raise RuntimeError(
                "local checkpoint save commit failed; rollback backups were "
                f"preserved where possible; rollback also failed: {exc}"
            ) from commit_exception
        _raise_checkpoint_phase_error(
            cleanup_exception, context="local checkpoint save rollback cleanup failed"
        )
        raise commit_exception

    if _is_distributed_checkpoint_ranked():
        completion = _local_checkpoint_completion_payload(
            step=step, generation=generation, topology=topology
        )
        completion_path = _local_checkpoint_completion_path(path)
        completion_backup: Path | None = None
        completion_publish_attempted = False
        local_exception = None
        try:
            if dist.get_rank() == 0:
                completion_backup = _backup_local_checkpoint_file(completion_path)
        except Exception as exc:
            local_exception = exc
        completion_exception: Exception | None = None
        try:
            _raise_checkpoint_phase_error(
                local_exception, context="local checkpoint completion backup failed"
            )
        except Exception as exc:
            completion_exception = exc
        if completion_exception is None:
            completion_publish_attempted = True
            local_exception = None
            try:
                if dist.get_rank() == 0:
                    _atomic_write_json(completion_path, completion)
            except Exception as exc:
                local_exception = exc
            try:
                _raise_checkpoint_phase_error(
                    local_exception,
                    context="local checkpoint completion publish failed",
                )
            except Exception as exc:
                completion_exception = exc
        if completion_exception is None:
            local_exception = None
            try:
                readback = _read_local_checkpoint_completion_with_consensus(path)
                if readback != completion:
                    raise RuntimeError(
                        "local checkpoint completion readback mismatch: "
                        f"expected={completion!r}, actual={readback!r}"
                    )
            except Exception as exc:
                local_exception = exc
            try:
                _raise_checkpoint_phase_error(
                    local_exception,
                    context="local checkpoint completion readback failed",
                )
            except Exception as exc:
                completion_exception = exc
        if completion_exception is not None:
            marker_rollback_exception: Exception | None = None
            try:
                if dist.get_rank() == 0 and completion_publish_attempted:
                    if completion_backup is None:
                        completion_path.unlink(missing_ok=True)
                    else:
                        os.replace(completion_backup, completion_path)
                    directory_fd = os.open(completion_path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                elif dist.get_rank() == 0 and completion_backup is not None:
                    completion_backup.unlink(missing_ok=True)
            except Exception as exc:
                marker_rollback_exception = exc
            try:
                _raise_checkpoint_phase_error(
                    marker_rollback_exception,
                    context="local checkpoint completion rollback failed",
                )
            except Exception as exc:
                raise RuntimeError(
                    "local checkpoint completion failed and its previous marker "
                    f"could not be restored: {exc}"
                ) from completion_exception

            rollback_exception = _rollback_local_checkpoint_commit(committed)
            cleanup_exception = _cleanup_local_checkpoint_transaction(
                prepared, remove_backups=rollback_exception is None
            )
            try:
                _raise_checkpoint_phase_error(
                    rollback_exception,
                    context=(
                        "local checkpoint completion failure payload rollback failed"
                    ),
                )
            except Exception as exc:
                raise RuntimeError(
                    "local checkpoint completion publish failed; payload rollback "
                    f"backups were preserved where possible; rollback also failed: {exc}"
                ) from completion_exception
            _raise_checkpoint_phase_error(
                cleanup_exception,
                context="local checkpoint completion failure cleanup failed",
            )
            raise completion_exception
        local_exception = None
        try:
            if dist.get_rank() == 0 and completion_backup is not None:
                completion_backup.unlink(missing_ok=True)
        except Exception as exc:
            local_exception = exc
        _raise_checkpoint_phase_error(
            local_exception, context="local checkpoint completion backup cleanup failed"
        )

    cleanup_exception = _cleanup_local_checkpoint_transaction(prepared)
    _raise_checkpoint_phase_error(
        cleanup_exception, context="local checkpoint save success cleanup failed"
    )
    local_exception = None
    try:
        _garbage_collect_optimizer_parameter_state_files(
            previous_parameter_state_files,
            keep=(optimizer_parameter_state_file if parameter_state_writer else None),
        )
    except Exception as exc:
        local_exception = exc
    _raise_checkpoint_phase_error(
        local_exception,
        context="local checkpoint optimizer parameter-state garbage collection failed",
    )
    log_rank0(f"Saved local training checkpoint at step {step} to {ckpt_file}")


@dataclass(frozen=True)
class _LocalTrainingCheckpointLoadPlan:
    ckpt_file: Path
    checkpoint_format: str
    step: int
    generation: str | None
    topology: dict[str, Any] | None
    rank_coordinate: dict[str, int] | None
    chunks: list[nn.Module]
    chunk_states: list[dict[str, torch.Tensor]]
    optimizer: Any
    saved_optimizer_state: Any
    saved_rng_state: dict[str, Any] | None
    load_rng: bool
    target_parameter_state_loader: Callable | None
    parameter_state_path: Path | None
    parameter_state_load_path: Path | None
    staged_parameter_state_path: Path | None
    parameter_state_fingerprint: tuple[int, str] | None
    load_parameter_state_update_legacy_format: bool
    ignored_legacy_buffer_keys: list[str]


def _preflight_local_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    *,
    load_rng: bool = True,
    load_parameter_state_update_legacy_format: bool = False,
    allow_legacy_checkpoint: bool = False,
) -> _LocalTrainingCheckpointLoadPlan:
    ckpt_file = _local_checkpoint_file(path)
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(
            f"Local checkpoint root must be a dict, got {type(state).__name__}"
        )
    checkpoint_format = state.get("format")
    if checkpoint_format not in {
        _LOCAL_TRAINING_FORMAT_V1,
        _LOCAL_TRAINING_FORMAT_V2,
        _LOCAL_TRAINING_FORMAT_V3,
    }:
        raise RuntimeError(f"Unsupported local checkpoint format in {ckpt_file}")
    if checkpoint_format != _LOCAL_TRAINING_FORMAT_V3 and not allow_legacy_checkpoint:
        raise RuntimeError(
            "Legacy local checkpoints are rejected by default: "
            f"format={checkpoint_format!r}. Pass allow_legacy_checkpoint=True "
            "only for an explicit one-time migration load, then immediately "
            "re-save it as V3."
        )
    if checkpoint_format != _LOCAL_TRAINING_FORMAT_V3 and allow_legacy_checkpoint:
        scope = "distributed " if _is_distributed_checkpoint_ranked() else ""
        log_rank0(
            f"Legacy {scope}local checkpoint load is not crash-generation "
            f"protected (format={checkpoint_format!r}); use only for one-time "
            "migration and re-save as V3 immediately."
        )
    expected_root_keys = {
        "format",
        "step",
        "model",
        "optimizer",
        "optimizer_parameter_state",
        "rng_state",
    }
    if checkpoint_format == _LOCAL_TRAINING_FORMAT_V3:
        expected_root_keys.update({"generation", "topology", "rank_coordinate"})
    if set(state) != expected_root_keys:
        raise RuntimeError(
            "Local checkpoint root schema mismatch: "
            f"missing={sorted(expected_root_keys - set(state))}, "
            f"unexpected={sorted(set(state) - expected_root_keys)}"
        )
    step = state["step"]
    if type(step) is not int or step < 0:
        raise RuntimeError(
            f"Local checkpoint step must be a non-negative integer, got {step!r}"
        )
    generation = state.get("generation")
    if checkpoint_format == _LOCAL_TRAINING_FORMAT_V3 and (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(character not in "0123456789abcdef" for character in generation)
    ):
        raise RuntimeError(f"Local checkpoint generation is invalid: {generation!r}")
    topology = state.get("topology")
    rank_coordinate = state.get("rank_coordinate")
    if checkpoint_format == _LOCAL_TRAINING_FORMAT_V3:
        if topology is not None and not isinstance(topology, dict):
            raise RuntimeError("Local checkpoint topology must be a dictionary or None")
        if rank_coordinate is not None and not isinstance(rank_coordinate, dict):
            raise RuntimeError(
                "Local checkpoint rank_coordinate must be a dictionary or None"
            )
        if _is_distributed_checkpoint_ranked() and (
            not isinstance(topology, dict) or not isinstance(rank_coordinate, dict)
        ):
            raise RuntimeError(
                "Distributed V3 local checkpoint requires topology and rank coordinate"
            )
    chunks = _model_chunks(model)
    chunk_states = state.get("model")
    if not isinstance(chunk_states, list) or len(chunk_states) != len(chunks):
        raise RuntimeError("Checkpoint model chunk count does not match target model.")
    ignored_legacy_buffer_keys: list[str] = []
    for chunk_index, (chunk, chunk_state) in enumerate(
        zip(chunks, chunk_states, strict=True)
    ):
        ignored_legacy_buffer_keys.extend(
            f"chunk{chunk_index}.{key}"
            for key in _preflight_chunk_tensor_state(
                chunk,
                chunk_state,
                chunk_index=chunk_index,
                allow_legacy_nonpersistent_buffers=(
                    checkpoint_format == _LOCAL_TRAINING_FORMAT_V1
                ),
            )
        )
    saved_optimizer_state = state.get("optimizer")
    saved_has_optimizer = saved_optimizer_state is not None
    target_has_optimizer = optimizer is not None
    if saved_has_optimizer != target_has_optimizer:
        raise RuntimeError(
            "Local checkpoint optimizer presence does not match the target runtime: "
            f"saved={saved_has_optimizer}, target={target_has_optimizer}"
        )
    parameter_state_name = state.get("optimizer_parameter_state")
    saved_has_parameter_state = parameter_state_name is not None
    target_parameter_state_loader = getattr(optimizer, "load_parameter_state", None)
    target_accepts_parameter_state = callable(target_parameter_state_loader)
    if saved_has_parameter_state != target_accepts_parameter_state:
        raise RuntimeError(
            "Local checkpoint optimizer parameter-state presence does not match "
            "the target runtime: "
            f"saved={saved_has_parameter_state}, "
            f"target={target_accepts_parameter_state}"
        )
    parameter_state_path: Path | None = None
    if parameter_state_name is not None:
        expected_parameter_state_name = _local_optimizer_parameter_state_file(
            ckpt_file,
            generation=(
                generation if checkpoint_format == _LOCAL_TRAINING_FORMAT_V3 else None
            ),
        ).name
        if (
            not isinstance(parameter_state_name, str)
            or Path(parameter_state_name).name != parameter_state_name
            or parameter_state_name != expected_parameter_state_name
        ):
            raise RuntimeError(
                "Local checkpoint optimizer parameter-state filename is invalid: "
                f"{parameter_state_name!r}"
            )
        parameter_state_path = ckpt_file.with_name(parameter_state_name)
    if optimizer is not None:
        saved_optimizer_state = _prepare_optimizer_checkpoint_state(
            optimizer,
            saved_optimizer_state,
            allow_legacy_optimizer_state=(
                allow_legacy_checkpoint
                and checkpoint_format == _LOCAL_TRAINING_FORMAT_V1
            ),
        )
    saved_rng_state = state.get("rng_state")
    if load_rng:
        if saved_rng_state is None:
            raise RuntimeError(
                "Local checkpoint RNG state requested by load_rng=True is missing"
            )
        _preflight_rng_checkpoint_state(saved_rng_state)
    parameter_state_load_path = parameter_state_path
    staged_parameter_state_path: Path | None = None
    parameter_state_fingerprint: tuple[int, str] | None = None
    if parameter_state_path is not None:
        (
            parameter_state_load_path,
            staged_parameter_state_path,
            parameter_state_fingerprint,
        ) = _preflight_local_optimizer_parameter_state(
            optimizer,
            parameter_state_path,
            update_legacy_format=load_parameter_state_update_legacy_format,
        )
    return _LocalTrainingCheckpointLoadPlan(
        ckpt_file=ckpt_file,
        checkpoint_format=checkpoint_format,
        step=step,
        generation=generation,
        topology=topology,
        rank_coordinate=rank_coordinate,
        chunks=chunks,
        chunk_states=chunk_states,
        optimizer=optimizer,
        saved_optimizer_state=saved_optimizer_state,
        saved_rng_state=saved_rng_state,
        load_rng=load_rng,
        target_parameter_state_loader=(
            target_parameter_state_loader
            if callable(target_parameter_state_loader)
            else None
        ),
        parameter_state_path=parameter_state_path,
        parameter_state_load_path=parameter_state_load_path,
        staged_parameter_state_path=staged_parameter_state_path,
        parameter_state_fingerprint=parameter_state_fingerprint,
        load_parameter_state_update_legacy_format=(
            load_parameter_state_update_legacy_format
        ),
        ignored_legacy_buffer_keys=ignored_legacy_buffer_keys,
    )


def _commit_local_training_checkpoint_core(
    plan: _LocalTrainingCheckpointLoadPlan, *, mark_mutation_started: Callable[[], None]
) -> None:
    for chunk, chunk_state in zip(plan.chunks, plan.chunk_states, strict=True):
        _load_chunk_tensor_state(chunk, chunk_state, before_copy=mark_mutation_started)
    if plan.optimizer is not None:
        mark_mutation_started()
        plan.optimizer.load_state_dict(plan.saved_optimizer_state)
        if plan.parameter_state_load_path is None:
            reload_model_params = getattr(plan.optimizer, "reload_model_params", None)
            if callable(reload_model_params):
                mark_mutation_started()
                reload_model_params()


def _commit_local_training_checkpoint_parameter_state(
    plan: _LocalTrainingCheckpointLoadPlan, *, mark_mutation_started: Callable[[], None]
) -> None:
    if plan.parameter_state_load_path is None:
        return
    assert callable(plan.target_parameter_state_loader)
    mark_mutation_started()
    plan.target_parameter_state_loader(
        str(plan.parameter_state_load_path),
        update_legacy_format=plan.load_parameter_state_update_legacy_format,
    )


def _commit_local_training_checkpoint_rng(
    plan: _LocalTrainingCheckpointLoadPlan, *, mark_mutation_started: Callable[[], None]
) -> None:
    if plan.load_rng:
        mark_mutation_started()
        _restore_rng_state(plan.saved_rng_state)


def _load_local_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    *,
    config=None,
    ps: ParallelState | None = None,
    load_rng: bool = True,
    load_parameter_state_update_legacy_format: bool = False,
    allow_legacy_checkpoint: bool = False,
) -> int:
    _assert_world_consensus(
        _canonical_checkpoint_path(path),
        context="local checkpoint load path differs across ranks",
    )
    current_topology: dict[str, Any] | None = None
    current_rank_coordinate: dict[str, int] | None = None
    local_exception: Exception | None = None
    try:
        current_topology, current_rank_coordinate = _local_checkpoint_parallel_identity(
            config, ps
        )
    except Exception as exc:
        local_exception = exc
    _raise_checkpoint_phase_error(
        local_exception, context="local checkpoint load topology preflight failed"
    )
    if _is_distributed_checkpoint_ranked():
        assert current_topology is not None
        _assert_world_consensus(
            current_topology,
            context="local checkpoint load topology differs across ranks",
        )
    if _is_distributed_checkpoint_ranked():
        _local_checkpoint_file(path)
    completion = _read_local_checkpoint_completion_with_consensus(
        path, allow_missing=allow_legacy_checkpoint
    )
    plan: _LocalTrainingCheckpointLoadPlan | None = None
    local_exception: BaseException | None = None
    try:
        plan = _preflight_local_training_checkpoint(
            model,
            optimizer,
            path,
            load_rng=load_rng,
            load_parameter_state_update_legacy_format=(
                load_parameter_state_update_legacy_format
            ),
            allow_legacy_checkpoint=allow_legacy_checkpoint,
        )
    except BaseException as exc:
        local_exception = exc

    try:
        _raise_checkpoint_phase_error(
            local_exception, context="local checkpoint load preflight failed"
        )
    except BaseException:
        if plan is not None and plan.staged_parameter_state_path is not None:
            plan.staged_parameter_state_path.unlink(missing_ok=True)
        raise
    assert plan is not None
    try:
        _assert_world_consensus(
            {
                "checkpoint_format": plan.checkpoint_format,
                "step": plan.step,
                "generation": plan.generation,
                "topology": plan.topology,
                "optimizer_present": plan.optimizer is not None,
                "parameter_state_present": plan.parameter_state_path is not None,
            },
            context="local checkpoint loaded identity differs across ranks",
        )
    except Exception as identity_exception:
        cleanup_exception = _cleanup_local_optimizer_parameter_state_staging(
            plan.staged_parameter_state_path
        )
        try:
            _raise_checkpoint_phase_error(
                cleanup_exception,
                context="local checkpoint identity failure cleanup failed",
            )
        except Exception as exc:
            raise RuntimeError(
                "local checkpoint identity validation failed and staging cleanup "
                f"also failed: {exc}"
            ) from identity_exception
        raise
    if plan.checkpoint_format == _LOCAL_TRAINING_FORMAT_V3:
        local_exception = None
        if (
            plan.topology != current_topology
            or plan.rank_coordinate != current_rank_coordinate
        ):
            local_exception = RuntimeError(
                "local checkpoint topology/rank coordinate does not match runtime: "
                f"saved_topology={plan.topology}, current_topology={current_topology}, "
                f"saved_rank={plan.rank_coordinate}, "
                f"current_rank={current_rank_coordinate}"
            )
        topology_exception: Exception | None = None
        try:
            _raise_checkpoint_phase_error(
                local_exception, context="local checkpoint topology validation failed"
            )
        except Exception as exc:
            topology_exception = exc
        if topology_exception is not None:
            cleanup_exception = _cleanup_local_optimizer_parameter_state_staging(
                plan.staged_parameter_state_path
            )
            _raise_checkpoint_phase_error(
                cleanup_exception,
                context="local checkpoint topology mismatch cleanup failed",
            )
            raise topology_exception
    if (
        _is_distributed_checkpoint_ranked()
        and plan.checkpoint_format == _LOCAL_TRAINING_FORMAT_V3
        and completion is None
    ):
        cleanup_exception = _cleanup_local_optimizer_parameter_state_staging(
            plan.staged_parameter_state_path
        )
        _raise_checkpoint_phase_error(
            cleanup_exception,
            context="local checkpoint missing completion cleanup failed",
        )
        raise RuntimeError(
            "V3 distributed local checkpoint is missing its required completion "
            "marker; allow_legacy_checkpoint cannot bypass a torn V3 transaction."
        )
    if completion is not None:
        loaded_identity = {
            "format": plan.checkpoint_format,
            "step": plan.step,
            "generation": plan.generation,
            "topology": plan.topology,
        }
        expected_identity = {
            "format": completion["format"],
            "step": completion["step"],
            "generation": completion["generation"],
            "topology": completion["topology"],
        }
        if loaded_identity != expected_identity:
            cleanup_exception = _cleanup_local_optimizer_parameter_state_staging(
                plan.staged_parameter_state_path
            )
            _raise_checkpoint_phase_error(
                cleanup_exception,
                context="local checkpoint completion mismatch cleanup failed",
            )
            raise RuntimeError(
                "local checkpoint payload does not match the completed generation: "
                f"payload={loaded_identity}, completion={expected_identity}"
            )

    local_exception = None
    try:
        if plan.parameter_state_path is not None:
            _revalidate_local_optimizer_parameter_state(
                plan.staged_parameter_state_path, plan.parameter_state_fingerprint
            )
    except Exception as exc:
        local_exception = exc
    try:
        _raise_checkpoint_phase_error(
            local_exception, context="local checkpoint load revalidation failed"
        )
    except Exception:
        if plan.staged_parameter_state_path is not None:
            plan.staged_parameter_state_path.unlink(missing_ok=True)
        raise

    # Commit is split into local, DP-collective, and local phases. Each local
    # phase reaches WORLD consensus before any rank may enter the next phase.
    commit_exception: BaseException | None = None
    mutation_started = False
    commit_phases: list[tuple[str, Callable[[Callable[[], None]], None]]] = [
        (
            "local checkpoint load core commit failed",
            lambda mark: _commit_local_training_checkpoint_core(
                plan, mark_mutation_started=mark
            ),
        )
    ]
    if plan.parameter_state_load_path is not None:
        commit_phases.append(
            (
                "local checkpoint load optimizer parameter-state commit failed",
                lambda mark: _commit_local_training_checkpoint_parameter_state(
                    plan, mark_mutation_started=mark
                ),
            )
        )
    commit_phases.append(
        (
            "local checkpoint load RNG commit failed",
            lambda mark: _commit_local_training_checkpoint_rng(
                plan, mark_mutation_started=mark
            ),
        )
    )
    for context, action in commit_phases:
        local_exception = None
        phase_mutation_started = False

        def mark_phase_mutation_started() -> None:
            nonlocal phase_mutation_started
            phase_mutation_started = True

        try:
            action(mark_phase_mutation_started)
        except BaseException as exc:
            local_exception = exc
        try:
            mutation_started = _raise_checkpoint_load_commit_error(
                local_exception,
                mutation_started=mutation_started or phase_mutation_started,
                context=context,
            )
        except BaseException as exc:
            commit_exception = exc
            break

    if commit_exception is not None:
        if _is_control_flow_exception(commit_exception) or (
            isinstance(commit_exception, CheckpointLoadFatalError)
            and commit_exception._peer_control_flow
        ):
            raise commit_exception
        staging_cleanup_exception = _cleanup_local_optimizer_parameter_state_staging(
            plan.staged_parameter_state_path
        )
        try:
            _raise_checkpoint_load_commit_error(
                staging_cleanup_exception,
                mutation_started=_checkpoint_exception_started_mutation(
                    commit_exception
                ),
                context="local checkpoint load failed-commit staging cleanup failed",
            )
        except CheckpointLoadFatalError as exc:
            raise CheckpointLoadFatalError(
                "local checkpoint load commit failed; runtime must be poisoned; "
                "discard and reinitialize model/optimizer state; staging cleanup "
                f"also failed: {exc}; original commit error: {commit_exception}"
            ) from commit_exception
        except Exception as exc:
            raise RuntimeError(
                "local checkpoint load failed before live-state mutation and "
                f"staging cleanup also failed: {exc}; original error: "
                f"{commit_exception}"
            ) from commit_exception
        if not isinstance(commit_exception, CheckpointLoadFatalError):
            raise commit_exception
        raise CheckpointLoadFatalError(
            "local checkpoint load commit failed; runtime must be poisoned; "
            "discard and reinitialize model/optimizer state before further use: "
            f"{commit_exception}"
        ) from commit_exception

    staging_cleanup_exception = _cleanup_local_optimizer_parameter_state_staging(
        plan.staged_parameter_state_path
    )
    _raise_checkpoint_load_commit_error(
        staging_cleanup_exception,
        mutation_started=mutation_started,
        context="local checkpoint load staging cleanup failed",
    )

    if plan.ignored_legacy_buffer_keys:
        log_rank0(
            "Loaded legacy local checkpoint while ignoring current non-persistent "
            f"runtime buffers: {plan.ignored_legacy_buffer_keys}"
        )
    log_rank0(
        f"Loaded local training checkpoint from {plan.ckpt_file} at step {plan.step}"
    )
    return plan.step


def _build_meshes(config):
    """Build separate meshes for dense and expert parameters.

    Dense mesh  [PP, DP, CP, TP]  — matches init_parallel dense decomposition.
    Expert mesh [PP, EDP, EP, ETP] — matches init_parallel expert decomposition.

    Both meshes use C-order layout so the innermost (rightmost) dimension
    corresponds to the fastest-changing rank index, consistent with
    init_parallel's rank = (...) * inner_size + inner_rank formula.
    """
    ws = dist.get_world_size()
    tp = int(config.tp or 1)
    ep = int(config.ep or 1)
    etp = max(int(config.etp or 1), 1)
    cp = max(int(config.cp or 1), 1)
    pp = max(int(config.pp or 1), 1)

    dense_dp = ws // (tp * cp * pp)
    expert_dp = ws // (etp * ep * pp)

    ranks = torch.arange(ws)
    dense_mesh = DeviceMesh("cuda", ranks.reshape(pp, dense_dp, cp, tp))
    expert_mesh = DeviceMesh("cuda", ranks.reshape(pp, expert_dp, ep, etp))
    return dense_mesh, expert_mesh


def log_rank0(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[megatron.lite] {msg}", flush=True)


# ======================================================================
# QKV / FC1 canonicalize for DCP (interleaved-TP ↔ canonical layout)
# ======================================================================


def _ag(data, size, group, dim=0):
    from megatron.lite.primitive.ckpt.hf_weights import allgather_concat

    return allgather_concat(data, size, group, dim)


def canonicalize_qkv_for_dcp(
    model, num_attention_heads, num_key_value_heads, head_dim, ps
):
    """Rearrange fused QKV from interleaved-TP to canonical (Q|K|V) for DCP save."""
    if ps.tp_size <= 1:
        return
    from megatron.lite.primitive.utils import ensure_divisible

    nq = ensure_divisible(num_attention_heads, ps.tp_size) * head_dim
    nkv = ensure_divisible(num_key_value_heads, ps.tp_size) * head_dim
    for name, param in model.named_parameters():
        if "qkv" not in name or "layer_norm" in name:
            continue
        full = _ag(param.data, ps.tp_size, ps.tp_group)
        cs = param.data.shape[0]
        q, k, v = [], [], []
        for r in range(ps.tp_size):
            s = full[r * cs : (r + 1) * cs]
            q.append(s[:nq])
            k.append(s[nq : nq + nkv])
            v.append(s[nq + nkv :])
        canon = torch.cat([torch.cat(q), torch.cat(k), torch.cat(v)], dim=0)
        param.data.copy_(canon.chunk(ps.tp_size, dim=0)[ps.tp_rank])


def decanon_qkv_after_dcp(
    model, num_attention_heads, num_key_value_heads, head_dim, ps
):
    """Reverse of canonicalize_qkv_for_dcp."""
    if ps.tp_size <= 1:
        return
    qs = num_attention_heads * head_dim
    kvs = num_key_value_heads * head_dim
    for name, param in model.named_parameters():
        if "qkv" not in name or "layer_norm" in name:
            continue
        full = _ag(param.data, ps.tp_size, ps.tp_group)
        ql = full[:qs].chunk(ps.tp_size)[ps.tp_rank]
        kl = full[qs : qs + kvs].chunk(ps.tp_size)[ps.tp_rank]
        vl = full[qs + kvs :].chunk(ps.tp_size)[ps.tp_rank]
        param.data.copy_(torch.cat([ql, kl, vl], dim=0))


def canonicalize_fc1_for_dcp(model, ps):
    """Rearrange fused gate-up FC1 from interleaved-ETP to canonical for DCP save."""
    if ps.etp_size <= 1:
        return
    for name, param in model.named_parameters():
        if "experts" not in name or "fc1" not in name:
            continue
        full = _ag(param.data, ps.etp_size, ps.etp_group)
        cs = param.data.shape[0]
        ffn = cs // 2
        g, u = [], []
        for r in range(ps.etp_size):
            s = full[r * cs : (r + 1) * cs]
            g.append(s[:ffn])
            u.append(s[ffn:])
        canon = torch.cat([torch.cat(g), torch.cat(u)], dim=0)
        param.data.copy_(canon.chunk(ps.etp_size, dim=0)[ps.etp_rank])


def decanon_fc1_after_dcp(model, ps):
    """Reverse of canonicalize_fc1_for_dcp."""
    if ps.etp_size <= 1:
        return
    for name, param in model.named_parameters():
        if "experts" not in name or "fc1" not in name:
            continue
        full = _ag(param.data, ps.etp_size, ps.etp_group)
        ffn = full.shape[0] // 2
        gl = full[:ffn].chunk(ps.etp_size)[ps.etp_rank]
        ul = full[ffn:].chunk(ps.etp_size)[ps.etp_rank]
        param.data.copy_(torch.cat([gl, ul], dim=0))


__all__ = [
    "canonicalize_fc1_for_dcp",
    "canonicalize_qkv_for_dcp",
    "decanon_fc1_after_dcp",
    "decanon_qkv_after_dcp",
    "load_training_checkpoint",
    "save_training_checkpoint",
]
