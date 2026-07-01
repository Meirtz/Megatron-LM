# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Delta-mem writable-state primitive.

This module implements the local tensor contract for the Scaling PEFT
delta-mem update:

    S_t = Diag(lambda_t) S_{t-1}
        + Diag(beta_t) (v_t - S_{t-1} k_t) k_t^T

It owns the state read/write math plus a small hidden-state write projector.
Attention correction wiring, checkpoint lifecycle, and evaluation recipes remain
model-level responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

DeltaMemGranularity = Literal["token", "sequence", "multi"]


_DELTA_MEM_POLICY_ALIASES = {
    "token": "token",
    "token_state": "token",
    "token-state": "token",
    "sequence": "sequence",
    "sequence_state": "sequence",
    "sequence-state": "sequence",
    "multi": "multi",
    "multi_state": "multi",
    "multi-state": "multi",
    "msw": "multi",
}


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _validate_state(state: torch.Tensor, *, rank: int) -> tuple[int, ...]:
    if state.ndim < 2:
        raise ValueError("delta-mem state must have shape (..., rank, rank).")
    if state.shape[-2:] != (rank, rank):
        raise ValueError(
            "delta-mem state trailing shape must be "
            f"({rank}, {rank}), got {tuple(state.shape[-2:])}."
        )
    if not torch.is_floating_point(state):
        raise TypeError("delta-mem state must be a floating-point tensor.")
    return tuple(state.shape[:-2])


def _validate_vector(
    tensor: torch.Tensor,
    *,
    name: str,
    rank: int,
    prefix: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    expected_shape = (*prefix, rank)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"delta-mem {name} must have shape {expected_shape}, got {tuple(tensor.shape)}.")
    if tensor.dtype != dtype:
        raise TypeError(f"delta-mem {name} dtype {tensor.dtype} must match state dtype {dtype}.")
    if tensor.device != device:
        raise ValueError(f"delta-mem {name} device {tensor.device} must match state device {device}.")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"delta-mem {name} must be a floating-point tensor.")


def _validate_write_inputs(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retention: torch.Tensor,
    write_strength: torch.Tensor,
    *,
    rank: int,
) -> tuple[int, ...]:
    prefix = _validate_state(state, rank=rank)
    for name, tensor in (
        ("key", key),
        ("value", value),
        ("retention", retention),
        ("write_strength", write_strength),
    ):
        _validate_vector(
            tensor,
            name=name,
            rank=rank,
            prefix=prefix,
            dtype=state.dtype,
            device=state.device,
        )
    return prefix


def _validate_hidden_size(value: int, *, name: str) -> int:
    return _validate_positive_int(value, name=name)


def _validate_bool(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _normalize_granularity(value: str, *, name: str) -> DeltaMemGranularity:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = _DELTA_MEM_POLICY_ALIASES.get(value.strip().lower())
    if normalized is None:
        raise ValueError(f"{name} must be one of token, sequence, or multi.")
    return normalized  # type: ignore[return-value]


def _validate_write_tensor_set(
    key: torch.Tensor,
    value: torch.Tensor,
    retention: torch.Tensor,
    write_strength: torch.Tensor,
    *,
    rank: int,
) -> tuple[int, ...]:
    if key.ndim < 1:
        raise ValueError("delta-mem key must have shape (..., rank).")
    if key.shape[-1] != rank:
        raise ValueError(f"delta-mem key trailing dimension must be {rank}, got {key.shape[-1]}.")
    if not torch.is_floating_point(key):
        raise TypeError("delta-mem key must be a floating-point tensor.")
    prefix = tuple(key.shape[:-1])
    for name, tensor in (
        ("value", value),
        ("retention", retention),
        ("write_strength", write_strength),
    ):
        _validate_vector(
            tensor,
            name=name,
            rank=rank,
            prefix=prefix,
            dtype=key.dtype,
            device=key.device,
        )
    return prefix


def _validate_read_key(
    read_key: torch.Tensor | None,
    *,
    key: torch.Tensor,
    rank: int,
    prefix: tuple[int, ...],
) -> None:
    if read_key is None:
        return
    _validate_vector(
        read_key,
        name="read_key",
        rank=rank,
        prefix=prefix,
        dtype=key.dtype,
        device=key.device,
    )


@dataclass(frozen=True)
class DeltaMemConfig:
    enabled: bool = False
    rank: int = 0
    write_bias: bool = True
    correction_bias: bool = False
    state_policy: DeltaMemStatePolicy | Mapping[str, Any] | str | None = None

    def __post_init__(self) -> None:
        enabled = _validate_bool(self.enabled, name="DeltaMem config enabled")
        rank = self.rank
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("DeltaMem config rank must be an integer.")
        if enabled and rank <= 0:
            raise ValueError("DeltaMem config enabled=True requires a positive rank.")
        if not enabled and rank < 0:
            raise ValueError("DeltaMem config rank must be non-negative.")
        _validate_bool(self.write_bias, name="DeltaMem config write_bias")
        _validate_bool(self.correction_bias, name="DeltaMem config correction_bias")
        state_policy = normalize_delta_mem_state_policy(self.state_policy)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "rank", int(rank))
        object.__setattr__(self, "state_policy", state_policy)


@dataclass(frozen=True)
class DeltaMemWriteTensors:
    key: torch.Tensor
    value: torch.Tensor
    retention: torch.Tensor
    write_strength: torch.Tensor


@dataclass(frozen=True)
class DeltaMemStatePolicy:
    granularity: DeltaMemGranularity = "token"
    num_states: int = 1
    detach_after_write: bool = False

    def __post_init__(self) -> None:
        granularity = _normalize_granularity(self.granularity, name="DeltaMem state policy granularity")
        num_states = _validate_positive_int(self.num_states, name="DeltaMem state policy num_states")
        detach_after_write = _validate_bool(
            self.detach_after_write,
            name="DeltaMem state policy detach_after_write",
        )
        if granularity != "multi" and num_states != 1:
            raise ValueError("DeltaMem token/sequence state policies require num_states=1.")
        object.__setattr__(self, "granularity", granularity)
        object.__setattr__(self, "num_states", num_states)
        object.__setattr__(self, "detach_after_write", detach_after_write)


@dataclass(frozen=True)
class DeltaMemAttentionCorrectionOutput:
    query_delta: torch.Tensor
    output_delta: torch.Tensor
    readout: torch.Tensor
    next_state: torch.Tensor
    write_inputs: DeltaMemWriteTensors
    state_history: torch.Tensor | None = None
    state_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class DeltaMemStatePolicyOutput:
    readout: torch.Tensor
    next_state: torch.Tensor
    state_history: torch.Tensor | None = None
    state_indices: torch.Tensor | None = None


def normalize_delta_mem_config(config: DeltaMemConfig | Mapping[str, Any] | None) -> DeltaMemConfig:
    if config is None:
        return DeltaMemConfig()
    if isinstance(config, DeltaMemConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("DeltaMem config must be DeltaMemConfig, mapping, or None.")
    values = dict(config)
    if "r" in values and "rank" not in values:
        values["rank"] = values.pop("r")
    if "rank" in values and "enabled" not in values:
        values["enabled"] = True
    if "policy" in values and "state_policy" not in values:
        values["state_policy"] = values.pop("policy")
    policy_values: dict[str, Any] = {}
    for key in ("granularity", "mode", "kind", "num_states", "states", "detach_after_write", "detach"):
        if key in values:
            policy_values[key] = values.pop(key)
    if policy_values:
        if "state_policy" in values:
            raise ValueError("DeltaMem config state_policy cannot be combined with top-level state policy keys.")
        values["state_policy"] = policy_values
    allowed = {"enabled", "rank", "write_bias", "correction_bias", "state_policy"}
    unknown = sorted(key for key in values if key not in allowed)
    if unknown:
        raise ValueError(f"Unsupported DeltaMem config keys: {unknown}.")
    return DeltaMemConfig(**values)


def normalize_delta_mem_state_policy(
    policy: DeltaMemStatePolicy | Mapping[str, Any] | str | None,
) -> DeltaMemStatePolicy:
    if policy is None:
        return DeltaMemStatePolicy()
    if isinstance(policy, DeltaMemStatePolicy):
        return policy
    if isinstance(policy, str):
        return DeltaMemStatePolicy(granularity=policy)
    if not isinstance(policy, Mapping):
        raise TypeError("DeltaMem state policy must be DeltaMemStatePolicy, mapping, string, or None.")
    values = dict(policy)
    if "mode" in values and "granularity" not in values:
        values["granularity"] = values.pop("mode")
    if "kind" in values and "granularity" not in values:
        values["granularity"] = values.pop("kind")
    if "states" in values and "num_states" not in values:
        values["num_states"] = values.pop("states")
    if "detach" in values and "detach_after_write" not in values:
        values["detach_after_write"] = values.pop("detach")
    allowed = {"granularity", "num_states", "detach_after_write"}
    unknown = sorted(key for key in values if key not in allowed)
    if unknown:
        raise ValueError(f"Unsupported DeltaMem state policy keys: {unknown}.")
    return DeltaMemStatePolicy(**values)


def delta_mem_read(state: torch.Tensor, key: torch.Tensor, *, rank: int | None = None) -> torch.Tensor:
    """Read a low-rank delta-mem state as ``S @ key``."""

    inferred_rank = state.shape[-1] if rank is None and state.ndim >= 2 else rank
    rank = _validate_positive_int(inferred_rank, name="delta-mem rank")
    prefix = _validate_state(state, rank=rank)
    _validate_vector(
        key,
        name="key",
        rank=rank,
        prefix=prefix,
        dtype=state.dtype,
        device=state.device,
    )
    return torch.matmul(state, key.unsqueeze(-1)).squeeze(-1)


def delta_mem_write(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retention: torch.Tensor,
    write_strength: torch.Tensor,
    *,
    rank: int | None = None,
) -> torch.Tensor:
    """Apply the delta-rule write update to an explicit ``(..., rank, rank)`` state."""

    inferred_rank = state.shape[-1] if rank is None and state.ndim >= 2 else rank
    rank = _validate_positive_int(inferred_rank, name="delta-mem rank")
    _validate_write_inputs(
        state,
        key,
        value,
        retention,
        write_strength,
        rank=rank,
    )
    projected = delta_mem_read(state, key, rank=rank)
    residual = value - projected
    retained = retention.unsqueeze(-1) * state
    write = (write_strength * residual).unsqueeze(-1) * key.unsqueeze(-2)
    return retained + write


class DeltaMemState(nn.Module):
    """Shape-checked delta-mem state read/write wrapper."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = _validate_positive_int(rank, name="DeltaMemState rank")

    def initial_state(
        self,
        *batch_shape: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        for index, dim in enumerate(batch_shape):
            _validate_positive_int(dim, name=f"DeltaMemState batch_shape[{index}]")
        return torch.zeros(
            *batch_shape,
            self.rank,
            self.rank,
            dtype=dtype if dtype is not None else torch.float32,
            device=device,
        )

    def read(self, state: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return delta_mem_read(state, key, rank=self.rank)

    def write(
        self,
        state: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        retention: torch.Tensor,
        write_strength: torch.Tensor,
    ) -> torch.Tensor:
        return delta_mem_write(
            state,
            key,
            value,
            retention,
            write_strength,
            rank=self.rank,
        )

    def forward(
        self,
        state: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        retention: torch.Tensor,
        write_strength: torch.Tensor,
        read_key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = self.write(state, key, value, retention, write_strength)
        readout = self.read(next_state, key if read_key is None else read_key)
        return readout, next_state


class DeltaMemStateManager(nn.Module):
    """Policy-owned delta-mem state manager for token, sequence, and multi-state writes."""

    def __init__(self, rank: int, policy: DeltaMemStatePolicy | Mapping[str, Any] | str | None = None):
        super().__init__()
        self.rank = _validate_positive_int(rank, name="DeltaMemStateManager rank")
        self.policy = normalize_delta_mem_state_policy(policy)
        self.state = DeltaMemState(self.rank)

    def initial_state_for_write(
        self,
        key: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        prefix = _validate_write_tensor_set(
            key,
            torch.zeros_like(key),
            torch.ones_like(key),
            torch.ones_like(key),
            rank=self.rank,
        )
        if self.policy.granularity == "token":
            state_prefix = prefix
        else:
            if len(prefix) == 0:
                raise ValueError("DeltaMem sequence/multi policies require a leading time dimension.")
            _validate_positive_int(prefix[0], name="DeltaMem write time dimension")
            state_prefix = prefix[1:]
            if self.policy.granularity == "multi":
                state_prefix = (self.policy.num_states, *state_prefix)
        return torch.zeros(
            *state_prefix,
            self.rank,
            self.rank,
            dtype=dtype if dtype is not None else key.dtype,
            device=device if device is not None else key.device,
        )

    def detach_state(self, state: torch.Tensor) -> torch.Tensor:
        _validate_state(state, rank=self.rank)
        return state.detach()

    def forward(
        self,
        state: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        retention: torch.Tensor,
        write_strength: torch.Tensor,
        *,
        read_key: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        return_state_history: bool = False,
    ) -> DeltaMemStatePolicyOutput:
        if self.policy.granularity == "token":
            return self._write_token_state(
                state,
                key,
                value,
                retention,
                write_strength,
                read_key=read_key,
                state_indices=state_indices,
                return_state_history=return_state_history,
            )
        if self.policy.granularity == "sequence":
            return self._write_sequence_state(
                state,
                key,
                value,
                retention,
                write_strength,
                read_key=read_key,
                state_indices=state_indices,
                return_state_history=return_state_history,
            )
        return self._write_multi_state(
            state,
            key,
            value,
            retention,
            write_strength,
            read_key=read_key,
            state_indices=state_indices,
            return_state_history=return_state_history,
        )

    def _write_token_state(
        self,
        state: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        retention: torch.Tensor,
        write_strength: torch.Tensor,
        *,
        read_key: torch.Tensor | None,
        state_indices: torch.Tensor | None,
        return_state_history: bool,
    ) -> DeltaMemStatePolicyOutput:
        if state_indices is not None:
            raise ValueError("DeltaMem token-state policy does not accept state_indices.")
        if return_state_history:
            raise ValueError("DeltaMem token-state policy does not produce recurrent state history.")
        prefix = _validate_write_tensor_set(key, value, retention, write_strength, rank=self.rank)
        _validate_read_key(read_key, key=key, rank=self.rank, prefix=prefix)
        _validate_write_inputs(state, key, value, retention, write_strength, rank=self.rank)
        next_state = self.state.write(state, key, value, retention, write_strength)
        readout = self.state.read(next_state, key if read_key is None else read_key)
        if self.policy.detach_after_write:
            next_state = next_state.detach()
        return DeltaMemStatePolicyOutput(readout=readout, next_state=next_state)

    def _write_sequence_state(
        self,
        state: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        retention: torch.Tensor,
        write_strength: torch.Tensor,
        *,
        read_key: torch.Tensor | None,
        state_indices: torch.Tensor | None,
        return_state_history: bool,
    ) -> DeltaMemStatePolicyOutput:
        if state_indices is not None:
            raise ValueError("DeltaMem sequence-state policy does not accept state_indices.")
        prefix = _validate_write_tensor_set(key, value, retention, write_strength, rank=self.rank)
        _validate_read_key(read_key, key=key, rank=self.rank, prefix=prefix)
        time_steps, state_prefix = self._validate_recurrent_prefix(prefix)
        state_leading = _validate_state(state, rank=self.rank)
        if state_leading != state_prefix:
            raise ValueError(
                "DeltaMem sequence-state initial state must have shape "
                f"{(*state_prefix, self.rank, self.rank)}, got {tuple(state.shape)}."
            )
        current = state
        readouts: list[torch.Tensor] = []
        history: list[torch.Tensor] = []
        for step in range(time_steps):
            current = self.state.write(
                current,
                key[step],
                value[step],
                retention[step],
                write_strength[step],
            )
            readouts.append(self.state.read(current, key[step] if read_key is None else read_key[step]))
            if return_state_history:
                history.append(current)
            if self.policy.detach_after_write:
                current = current.detach()
        return DeltaMemStatePolicyOutput(
            readout=torch.stack(readouts, dim=0),
            next_state=current,
            state_history=torch.stack(history, dim=0) if return_state_history else None,
        )

    def _write_multi_state(
        self,
        state: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        retention: torch.Tensor,
        write_strength: torch.Tensor,
        *,
        read_key: torch.Tensor | None,
        state_indices: torch.Tensor | None,
        return_state_history: bool,
    ) -> DeltaMemStatePolicyOutput:
        prefix = _validate_write_tensor_set(key, value, retention, write_strength, rank=self.rank)
        _validate_read_key(read_key, key=key, rank=self.rank, prefix=prefix)
        time_steps, state_prefix = self._validate_recurrent_prefix(prefix)
        expected_state_prefix = (self.policy.num_states, *state_prefix)
        state_leading = _validate_state(state, rank=self.rank)
        if state_leading != expected_state_prefix:
            raise ValueError(
                "DeltaMem multi-state initial state must have shape "
                f"{(*expected_state_prefix, self.rank, self.rank)}, got {tuple(state.shape)}."
            )
        indices = self._normalize_state_indices(state_indices, time_steps=time_steps, device=key.device)
        current = state
        readouts: list[torch.Tensor] = []
        history: list[torch.Tensor] = []
        for step in range(time_steps):
            state_index = int(indices[step].item())
            updated = self.state.write(
                current[state_index],
                key[step],
                value[step],
                retention[step],
                write_strength[step],
            )
            banks = [current[index] for index in range(self.policy.num_states)]
            banks[state_index] = updated
            current = torch.stack(banks, dim=0)
            readouts.append(self.state.read(updated, key[step] if read_key is None else read_key[step]))
            if return_state_history:
                history.append(current)
            if self.policy.detach_after_write:
                current = current.detach()
        return DeltaMemStatePolicyOutput(
            readout=torch.stack(readouts, dim=0),
            next_state=current,
            state_history=torch.stack(history, dim=0) if return_state_history else None,
            state_indices=indices,
        )

    def _validate_recurrent_prefix(self, prefix: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        if len(prefix) == 0:
            raise ValueError("DeltaMem sequence/multi policies require a leading time dimension.")
        time_steps = _validate_positive_int(prefix[0], name="DeltaMem write time dimension")
        return time_steps, prefix[1:]

    def _normalize_state_indices(
        self,
        state_indices: torch.Tensor | None,
        *,
        time_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if state_indices is None:
            return torch.arange(time_steps, device=device, dtype=torch.long) % self.policy.num_states
        if not isinstance(state_indices, torch.Tensor):
            raise TypeError("DeltaMem multi-state state_indices must be a tensor.")
        if state_indices.shape != (time_steps,):
            raise ValueError(f"DeltaMem multi-state state_indices must have shape ({time_steps},).")
        if state_indices.dtype not in (torch.int32, torch.int64, torch.long):
            raise TypeError("DeltaMem multi-state state_indices must use an integer dtype.")
        if state_indices.device != device:
            raise ValueError("DeltaMem multi-state state_indices device must match write tensors.")
        if state_indices.numel() and (
            int(state_indices.min().item()) < 0 or int(state_indices.max().item()) >= self.policy.num_states
        ):
            raise ValueError("DeltaMem multi-state state_indices are out of range.")
        return state_indices.to(dtype=torch.long)


class DeltaMemWriteProjection(nn.Module):
    """Trainable hidden-state projection for delta-mem write inputs."""

    def __init__(self, hidden_size: int, rank: int, *, bias: bool = True):
        super().__init__()
        self.hidden_size = _validate_hidden_size(hidden_size, name="DeltaMemWriteProjection hidden_size")
        self.rank = _validate_positive_int(rank, name="DeltaMemWriteProjection rank")
        if not isinstance(bias, bool):
            raise TypeError("DeltaMemWriteProjection bias must be a boolean.")
        self.proj = nn.Linear(self.hidden_size, 4 * self.rank, bias=bias)

    def forward(self, hidden: torch.Tensor) -> DeltaMemWriteTensors:
        if hidden.ndim < 1:
            raise ValueError("delta-mem projection input must have shape (..., hidden_size).")
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                "delta-mem projection input trailing dimension must be "
                f"{self.hidden_size}, got {hidden.shape[-1]}."
            )
        if not torch.is_floating_point(hidden):
            raise TypeError("delta-mem projection input must be a floating-point tensor.")
        key, value, retention_logits, write_strength_logits = self.proj(hidden).chunk(4, dim=-1)
        return DeltaMemWriteTensors(
            key=key,
            value=value,
            retention=torch.sigmoid(retention_logits),
            write_strength=torch.sigmoid(write_strength_logits),
        )

    def write(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor,
        *,
        read_key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self(hidden)
        next_state = delta_mem_write(
            state,
            inputs.key,
            inputs.value,
            inputs.retention,
            inputs.write_strength,
            rank=self.rank,
        )
        readout = delta_mem_read(
            next_state,
            inputs.key if read_key is None else read_key,
            rank=self.rank,
        )
        return readout, next_state


class DeltaMemAttentionCorrection(nn.Module):
    """Trainable bridge from delta-mem readout to attention query/output deltas."""

    def __init__(
        self,
        hidden_size: int,
        query_size: int,
        output_size: int,
        rank: int,
        *,
        write_bias: bool = True,
        correction_bias: bool = False,
        state_policy: DeltaMemStatePolicy | Mapping[str, Any] | str | None = None,
    ):
        super().__init__()
        self.hidden_size = _validate_hidden_size(hidden_size, name="DeltaMemAttentionCorrection hidden_size")
        self.query_size = _validate_hidden_size(query_size, name="DeltaMemAttentionCorrection query_size")
        self.output_size = _validate_hidden_size(output_size, name="DeltaMemAttentionCorrection output_size")
        self.rank = _validate_positive_int(rank, name="DeltaMemAttentionCorrection rank")
        if not isinstance(correction_bias, bool):
            raise TypeError("DeltaMemAttentionCorrection correction_bias must be a boolean.")
        self.write_projection = DeltaMemWriteProjection(
            self.hidden_size,
            self.rank,
            bias=write_bias,
        )
        self.state_manager = DeltaMemStateManager(self.rank, policy=state_policy)
        self.query_delta_proj = nn.Linear(self.rank, self.query_size, bias=correction_bias)
        self.output_delta_proj = nn.Linear(self.rank, self.output_size, bias=correction_bias)

    def initial_state(self, hidden: torch.Tensor) -> torch.Tensor:
        write_inputs = self.write_projection(hidden)
        return self.state_manager.initial_state_for_write(
            write_inputs.key,
            dtype=write_inputs.key.dtype,
            device=write_inputs.key.device,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        read_key: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        return_state_history: bool = False,
    ) -> DeltaMemAttentionCorrectionOutput:
        write_inputs = self.write_projection(hidden)
        if state is None:
            state = self.state_manager.initial_state_for_write(
                write_inputs.key,
                dtype=write_inputs.key.dtype,
                device=write_inputs.key.device,
            )
        policy_output = self.state_manager(
            state,
            write_inputs.key,
            write_inputs.value,
            write_inputs.retention,
            write_inputs.write_strength,
            read_key=read_key,
            state_indices=state_indices,
            return_state_history=return_state_history,
        )
        return DeltaMemAttentionCorrectionOutput(
            query_delta=self.query_delta_proj(policy_output.readout),
            output_delta=self.output_delta_proj(policy_output.readout),
            readout=policy_output.readout,
            next_state=policy_output.next_state,
            write_inputs=write_inputs,
            state_history=policy_output.state_history,
            state_indices=policy_output.state_indices,
        )


__all__ = [
    "DeltaMemConfig",
    "DeltaMemAttentionCorrection",
    "DeltaMemAttentionCorrectionOutput",
    "DeltaMemGranularity",
    "DeltaMemState",
    "DeltaMemStateManager",
    "DeltaMemStatePolicy",
    "DeltaMemStatePolicyOutput",
    "DeltaMemWriteProjection",
    "DeltaMemWriteTensors",
    "delta_mem_read",
    "delta_mem_write",
    "normalize_delta_mem_config",
    "normalize_delta_mem_state_policy",
]
