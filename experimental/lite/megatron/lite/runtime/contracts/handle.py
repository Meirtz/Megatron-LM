# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ModelHandle — opaque handle returned by Runtime.build_model()."""

from __future__ import annotations

from typing import Any


class ModelHandle:
    """Opaque handle returned by Runtime.build_model().

    Only documented properties on this class are part of the public contract.
    Internal helpers may still use ``_model`` / ``_optimizer`` / ``_extras``
    while the higher-level runtime API is being stabilized.

    A Megatron Lite checkpoint load that raises
    :class:`~megatron.lite.runtime.CheckpointLoadFatalError` permanently poisons
    this handle. Non-``Exception`` control flow such as ``KeyboardInterrupt`` or
    ``SystemExit`` is propagated unchanged, but also poisons the handle because
    its exact interruption point cannot be proven safe. Ordinary ``Exception``
    failures during read-only checkpoint preflight leave the handle usable.
    Handles are intentionally neither copyable nor serializable: aliases could
    otherwise share live state while carrying independent poison flags.
    """

    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any = None,
        lr_scheduler: Any = None,
        parallel_state: Any = None,
        config: Any = None,
        _extras: dict[str, Any] | None = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._parallel_state = parallel_state
        self._config = config
        self._extras = _extras or {}
        self._checkpoint_load_poisoned = False
        self._checkpoint_load_poison_reason: str | None = None

    @property
    def poisoned(self) -> bool:
        """Whether a fatal checkpoint load permanently invalidated this handle."""

        return self._checkpoint_load_poisoned

    @property
    def poison_reason(self) -> str | None:
        """The first fatal checkpoint-load error, or ``None`` when usable."""

        return self._checkpoint_load_poison_reason

    def _poison_after_checkpoint_load(self, error: BaseException) -> None:
        """Permanently record the first post-mutation checkpoint failure."""

        if not self._checkpoint_load_poisoned:
            self._checkpoint_load_poisoned = True
            self._checkpoint_load_poison_reason = f"{type(error).__name__}: {error}"

    @staticmethod
    def _raise_opaque_copy_error() -> None:
        raise TypeError(
            "ModelHandle is an opaque runtime capability and cannot be copied or "
            "serialized; retain the original handle or build a fresh one."
        )

    def __copy__(self):
        self._raise_opaque_copy_error()

    def __deepcopy__(self, memo):
        del memo
        self._raise_opaque_copy_error()

    def __reduce_ex__(self, protocol):
        del protocol
        self._raise_opaque_copy_error()

    @property
    def dp_rank(self) -> int:
        ps = self._parallel_state
        if ps is None:
            return 0
        return getattr(ps, "dp_rank", 0)

    @property
    def dp_size(self) -> int:
        ps = self._parallel_state
        if ps is None:
            return 1
        return getattr(ps, "dp_size", 1)

    @property
    def dp_group(self):
        ps = self._parallel_state
        if ps is None:
            return None
        return getattr(ps, "dp_group", None)

    @property
    def cp_range(self) -> tuple[int, int]:
        return self._extras.get("cp_range", (1, 1))

    @property
    def config(self) -> Any:
        """Backend config captured when this handle was built."""
        return self._config


__all__ = ["ModelHandle"]
