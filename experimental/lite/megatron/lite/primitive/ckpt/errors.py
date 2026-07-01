# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Public checkpoint failure contracts."""

from __future__ import annotations

import torch.distributed as dist


class CheckpointLoadFatalError(RuntimeError):
    """A checkpoint load failed after live runtime state began changing.

    Catching this exception is not a recovery boundary.  The target model,
    optimizer, RNG, or registered extra state may be only partially restored,
    including when a distributed peer was the rank that failed.  Callers must
    permanently stop using the associated ``ModelHandle`` and build a fresh
    runtime state before retrying.

    Read-only checkpoint discovery, deserialization, metadata validation, and
    other preflight failures deliberately do *not* raise this exception because
    they occur before the final live-state application phase.
    """

    def __init__(self, message: str, *, peer_control_flow: bool = False) -> None:
        super().__init__(message)
        self._peer_control_flow = peer_control_flow


def _is_control_flow_exception(error: BaseException | None) -> bool:
    return error is not None and not isinstance(error, Exception)


def _mark_checkpoint_mutation(error: BaseException) -> None:
    """Mark a control-flow exception whose source rank crossed mutation."""

    error._mlite_checkpoint_mutation_started = True


def _checkpoint_exception_started_mutation(error: BaseException | None) -> bool:
    return isinstance(error, CheckpointLoadFatalError) or bool(
        getattr(error, "_mlite_checkpoint_mutation_started", False)
    )


def _raise_checkpoint_load_commit_error(
    local_exception: BaseException | None, *, mutation_started: bool, context: str
) -> bool:
    """Synchronize one application phase and enforce the fatal-load boundary.

    Returns whether any participating rank has started mutating live state when
    the phase succeeds.  Before that boundary a local exception is preserved;
    after it, every rank raises :class:`CheckpointLoadFatalError`.
    """

    if not dist.is_available() or not dist.is_initialized():
        if local_exception is None:
            return mutation_started
        if not mutation_started:
            raise local_exception
        if _is_control_flow_exception(local_exception):
            _mark_checkpoint_mutation(local_exception)
            raise local_exception
        raise CheckpointLoadFatalError(
            f"{context} after live-state mutation; runtime must be poisoned and "
            "the associated ModelHandle discarded: "
            f"{type(local_exception).__name__}: {local_exception}"
        ) from local_exception

    record = {
        "mutation_started": bool(mutation_started),
        "control_flow": _is_control_flow_exception(local_exception),
        "error": (
            None
            if local_exception is None
            else f"{type(local_exception).__name__}: {local_exception}"
        ),
    }
    records: list[dict[str, object] | None] = [None] * dist.get_world_size()
    try:
        dist.all_gather_object(records, record)
    except BaseException as exc:
        # Once an application phase is in flight, losing rank consensus means
        # this rank cannot prove that no peer crossed the mutation boundary.
        if _is_control_flow_exception(local_exception):
            assert local_exception is not None
            _mark_checkpoint_mutation(local_exception)
            raise local_exception
        if _is_control_flow_exception(exc):
            _mark_checkpoint_mutation(exc)
            raise
        cause = local_exception if local_exception is not None else exc
        raise CheckpointLoadFatalError(
            f"{context} lost distributed commit consensus; live-state mutation "
            "may have started on a peer, so runtime must be poisoned and the "
            f"associated ModelHandle discarded: {type(exc).__name__}: {exc}"
        ) from cause

    normalized = [item or {} for item in records]
    any_mutation_started = any(
        item.get("mutation_started") is True for item in normalized
    )
    first_mutating_error = next(
        (
            item.get("error")
            for item in normalized
            if item.get("mutation_started") is True and item.get("error") is not None
        ),
        None,
    )
    first_control_flow_error = next(
        (
            item.get("error")
            for item in normalized
            if item.get("control_flow") is True and item.get("error") is not None
        ),
        None,
    )
    first_error = (
        first_mutating_error
        or first_control_flow_error
        or next(
            (item.get("error") for item in normalized if item.get("error") is not None),
            None,
        )
    )
    if first_error is None:
        return any_mutation_started
    any_control_flow = any(item.get("control_flow") is True for item in normalized)
    if any_mutation_started or any_control_flow:
        if _is_control_flow_exception(local_exception):
            assert local_exception is not None
            if any_mutation_started:
                _mark_checkpoint_mutation(local_exception)
            raise local_exception
        boundary = (
            "after live-state mutation"
            if any_mutation_started
            else "after a peer control-flow interruption whose safe point cannot be proven"
        )
        raise CheckpointLoadFatalError(
            f"{context} {boundary}; runtime must be poisoned and "
            f"the associated ModelHandle discarded: {first_error}",
            peer_control_flow=any_control_flow,
        ) from local_exception
    if _is_control_flow_exception(local_exception):
        assert local_exception is not None
        raise local_exception
    raise RuntimeError(f"{context}: {first_error}") from local_exception


__all__ = ["CheckpointLoadFatalError"]
