# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MinT-style adapter lifecycle records for Megatron Lite.

This module is deliberately metadata-only. It separates policy identity from
compute residency, tracks exported LoRA adapter revisions, and records serving
residency as cool stored, warm CPU cache, or hot GPU batch state. In short,
policy identity is separate from compute residency. It does not load weights,
start serving workers, or claim any paper evaluation result.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

COOL_STORED = "cool"
WARM_CPU = "warm"
HOT_GPU = "hot"

AdapterResidencyTier = Literal["cool", "warm", "hot"]

_VALID_RESIDENCY_TIERS = {COOL_STORED, WARM_CPU, HOT_GPU}


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return int(value)


def _positive_int(value: Any, *, name: str) -> int:
    value = _non_negative_int(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(value).__name__}.")
    return MappingProxyType(dict(value))


def _normalize_target_modules(value: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        modules = (value,)
    elif isinstance(value, Sequence):
        modules = tuple(value)
    else:
        raise TypeError("target_modules must be a string or a sequence of strings.")
    if not modules:
        raise ValueError("target_modules must be non-empty.")
    normalized = tuple(_non_empty_string(module, name="target_modules entry") for module in modules)
    return normalized


def _normalize_residency_tier(value: Any) -> AdapterResidencyTier:
    value = _non_empty_string(value, name="residency tier").lower()
    if value not in _VALID_RESIDENCY_TIERS:
        raise ValueError("residency tier must be one of cool, warm, or hot.")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class BaseDeployment:
    """Resident frozen base model identity shared by many policies."""

    base_deployment_id: str
    model_name: str
    model_version: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "base_deployment_id", _non_empty_string(self.base_deployment_id, name="base_deployment_id")
        )
        object.__setattr__(self, "model_name", _non_empty_string(self.model_name, name="model_name"))
        object.__setattr__(self, "model_version", _non_empty_string(self.model_version, name="model_version"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_deployment_id": self.base_deployment_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AdapterRevision:
    """Fixed exported PEFT/LoRA adapter revision in serving layout."""

    revision_id: str
    policy_id: str
    base_deployment_id: str
    adapter_path: str
    rank: int
    target_modules: tuple[str, ...]
    parent_revision_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision_id", _non_empty_string(self.revision_id, name="revision_id"))
        object.__setattr__(self, "policy_id", _non_empty_string(self.policy_id, name="policy_id"))
        object.__setattr__(
            self, "base_deployment_id", _non_empty_string(self.base_deployment_id, name="base_deployment_id")
        )
        object.__setattr__(self, "adapter_path", _non_empty_string(self.adapter_path, name="adapter_path"))
        object.__setattr__(self, "rank", _positive_int(self.rank, name="rank"))
        object.__setattr__(self, "target_modules", _normalize_target_modules(self.target_modules))
        if self.parent_revision_id is not None:
            object.__setattr__(
                self,
                "parent_revision_id",
                _non_empty_string(self.parent_revision_id, name="parent_revision_id"),
            )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "policy_id": self.policy_id,
            "base_deployment_id": self.base_deployment_id,
            "adapter_path": self.adapter_path,
            "rank": self.rank,
            "target_modules": list(self.target_modules),
            "parent_revision_id": self.parent_revision_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PolicyRecord:
    """Persistent policy identity independent of where its adapter is resident."""

    policy_id: str
    base_deployment_id: str
    rank: int
    target_modules: tuple[str, ...]
    adapter_revision_ids: tuple[str, ...] = ()
    active_revision_id: str | None = None
    rollout_record_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _non_empty_string(self.policy_id, name="policy_id"))
        object.__setattr__(
            self, "base_deployment_id", _non_empty_string(self.base_deployment_id, name="base_deployment_id")
        )
        object.__setattr__(self, "rank", _positive_int(self.rank, name="rank"))
        object.__setattr__(self, "target_modules", _normalize_target_modules(self.target_modules))
        object.__setattr__(
            self,
            "adapter_revision_ids",
            tuple(_non_empty_string(item, name="adapter_revision_id") for item in self.adapter_revision_ids),
        )
        if self.active_revision_id is not None:
            active = _non_empty_string(self.active_revision_id, name="active_revision_id")
            if active not in self.adapter_revision_ids:
                raise ValueError("active_revision_id must be one of adapter_revision_ids.")
            object.__setattr__(self, "active_revision_id", active)
        object.__setattr__(
            self,
            "rollout_record_ids",
            tuple(_non_empty_string(item, name="rollout_record_id") for item in self.rollout_record_ids),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def with_revision(self, revision_id: str, *, active: bool = True) -> "PolicyRecord":
        revision_id = _non_empty_string(revision_id, name="revision_id")
        revision_ids = self.adapter_revision_ids
        if revision_id not in revision_ids:
            revision_ids = (*revision_ids, revision_id)
        return PolicyRecord(
            policy_id=self.policy_id,
            base_deployment_id=self.base_deployment_id,
            rank=self.rank,
            target_modules=self.target_modules,
            adapter_revision_ids=revision_ids,
            active_revision_id=revision_id if active else self.active_revision_id,
            rollout_record_ids=self.rollout_record_ids,
            metadata=self.metadata,
        )

    def with_rollout_record(self, rollout_record_id: str) -> "PolicyRecord":
        rollout_record_id = _non_empty_string(rollout_record_id, name="rollout_record_id")
        if rollout_record_id in self.rollout_record_ids:
            return self
        return PolicyRecord(
            policy_id=self.policy_id,
            base_deployment_id=self.base_deployment_id,
            rank=self.rank,
            target_modules=self.target_modules,
            adapter_revision_ids=self.adapter_revision_ids,
            active_revision_id=self.active_revision_id,
            rollout_record_ids=(*self.rollout_record_ids, rollout_record_id),
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "base_deployment_id": self.base_deployment_id,
            "rank": self.rank,
            "target_modules": list(self.target_modules),
            "adapter_revision_ids": list(self.adapter_revision_ids),
            "active_revision_id": self.active_revision_id,
            "rollout_record_ids": list(self.rollout_record_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PolicySession:
    """Temporary trainer restore session for a policy and adapter revision."""

    session_id: str
    policy_id: str
    base_deployment_id: str
    adapter_revision_id: str | None
    trainer_state_path: str | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _non_empty_string(self.session_id, name="session_id"))
        object.__setattr__(self, "policy_id", _non_empty_string(self.policy_id, name="policy_id"))
        object.__setattr__(
            self, "base_deployment_id", _non_empty_string(self.base_deployment_id, name="base_deployment_id")
        )
        if self.adapter_revision_id is not None:
            object.__setattr__(
                self,
                "adapter_revision_id",
                _non_empty_string(self.adapter_revision_id, name="adapter_revision_id"),
            )
        if self.trainer_state_path is not None:
            object.__setattr__(
                self,
                "trainer_state_path",
                _non_empty_string(self.trainer_state_path, name="trainer_state_path"),
            )
        if not isinstance(self.closed, bool):
            raise TypeError("closed must be a boolean.")

    def close(self) -> "PolicySession":
        return PolicySession(
            session_id=self.session_id,
            policy_id=self.policy_id,
            base_deployment_id=self.base_deployment_id,
            adapter_revision_id=self.adapter_revision_id,
            trainer_state_path=self.trainer_state_path,
            closed=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "policy_id": self.policy_id,
            "base_deployment_id": self.base_deployment_id,
            "adapter_revision_id": self.adapter_revision_id,
            "trainer_state_path": self.trainer_state_path,
            "closed": self.closed,
        }


@dataclass(frozen=True)
class ServingResidency:
    """Serving residency for one adapter revision."""

    revision_id: str
    tier: AdapterResidencyTier = COOL_STORED
    device: str | None = None
    batch_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision_id", _non_empty_string(self.revision_id, name="revision_id"))
        tier = _normalize_residency_tier(self.tier)
        object.__setattr__(self, "tier", tier)
        if self.device is not None:
            object.__setattr__(self, "device", _non_empty_string(self.device, name="device"))
        if self.batch_key is not None:
            object.__setattr__(self, "batch_key", _non_empty_string(self.batch_key, name="batch_key"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "tier": self.tier,
            "device": self.device,
            "batch_key": self.batch_key,
        }


class AdapterPolicyRegistry:
    """Registry for MinT policy identity and adapter revision lifecycle."""

    def __init__(self) -> None:
        self.base_deployments: dict[str, BaseDeployment] = {}
        self.policies: dict[str, PolicyRecord] = {}
        self.revisions: dict[str, AdapterRevision] = {}
        self.sessions: dict[str, PolicySession] = {}

    def register_base_deployment(self, deployment: BaseDeployment) -> BaseDeployment:
        if deployment.base_deployment_id in self.base_deployments:
            raise ValueError(f"base deployment already exists: {deployment.base_deployment_id}")
        self.base_deployments[deployment.base_deployment_id] = deployment
        return deployment

    def create_policy(
        self,
        *,
        policy_id: str,
        base_deployment_id: str,
        rank: int,
        target_modules: Sequence[str] | str,
        metadata: Mapping[str, Any] | None = None,
    ) -> PolicyRecord:
        policy_id = _non_empty_string(policy_id, name="policy_id")
        base_deployment_id = _non_empty_string(base_deployment_id, name="base_deployment_id")
        if policy_id in self.policies:
            raise ValueError(f"policy already exists: {policy_id}")
        if base_deployment_id not in self.base_deployments:
            raise KeyError(f"unknown base deployment: {base_deployment_id}")
        policy = PolicyRecord(
            policy_id=policy_id,
            base_deployment_id=base_deployment_id,
            rank=rank,
            target_modules=_normalize_target_modules(target_modules),
            metadata=_metadata(metadata),
        )
        self.policies[policy_id] = policy
        return policy

    def create_revision(
        self,
        *,
        policy_id: str,
        revision_id: str,
        adapter_path: str,
        parent_revision_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        set_active: bool = True,
    ) -> AdapterRevision:
        policy = self.get_policy(policy_id)
        revision_id = _non_empty_string(revision_id, name="revision_id")
        if revision_id in self.revisions:
            raise ValueError(f"adapter revision already exists: {revision_id}")
        if parent_revision_id is not None and parent_revision_id not in self.revisions:
            raise KeyError(f"unknown parent revision: {parent_revision_id}")
        revision = AdapterRevision(
            revision_id=revision_id,
            policy_id=policy.policy_id,
            base_deployment_id=policy.base_deployment_id,
            adapter_path=adapter_path,
            rank=policy.rank,
            target_modules=policy.target_modules,
            parent_revision_id=parent_revision_id,
            metadata=_metadata(metadata),
        )
        self.revisions[revision_id] = revision
        self.policies[policy.policy_id] = policy.with_revision(revision_id, active=set_active)
        return revision

    def set_active_revision(self, policy_id: str, revision_id: str) -> PolicyRecord:
        policy = self.get_policy(policy_id)
        revision = self.get_revision(revision_id)
        if revision.policy_id != policy.policy_id:
            raise ValueError("adapter revision belongs to a different policy.")
        self.policies[policy.policy_id] = policy.with_revision(revision_id, active=True)
        return self.policies[policy.policy_id]

    def record_rollout(self, policy_id: str, rollout_record_id: str) -> PolicyRecord:
        policy = self.get_policy(policy_id)
        self.policies[policy.policy_id] = policy.with_rollout_record(rollout_record_id)
        return self.policies[policy.policy_id]

    def create_policy_session(
        self,
        *,
        session_id: str,
        policy_id: str,
        trainer_state_path: str | None = None,
    ) -> PolicySession:
        policy = self.get_policy(policy_id)
        session_id = _non_empty_string(session_id, name="session_id")
        if session_id in self.sessions:
            raise ValueError(f"policy session already exists: {session_id}")
        session = PolicySession(
            session_id=session_id,
            policy_id=policy.policy_id,
            base_deployment_id=policy.base_deployment_id,
            adapter_revision_id=policy.active_revision_id,
            trainer_state_path=trainer_state_path,
        )
        self.sessions[session_id] = session
        return session

    def close_policy_session(self, session_id: str) -> PolicySession:
        session_id = _non_empty_string(session_id, name="session_id")
        if session_id not in self.sessions:
            raise KeyError(f"unknown policy session: {session_id}")
        self.sessions[session_id] = self.sessions[session_id].close()
        return self.sessions[session_id]

    def get_policy(self, policy_id: str) -> PolicyRecord:
        policy_id = _non_empty_string(policy_id, name="policy_id")
        if policy_id not in self.policies:
            raise KeyError(f"unknown policy: {policy_id}")
        return self.policies[policy_id]

    def get_revision(self, revision_id: str) -> AdapterRevision:
        revision_id = _non_empty_string(revision_id, name="revision_id")
        if revision_id not in self.revisions:
            raise KeyError(f"unknown adapter revision: {revision_id}")
        return self.revisions[revision_id]

    def active_revision(self, policy_id: str) -> AdapterRevision | None:
        policy = self.get_policy(policy_id)
        if policy.active_revision_id is None:
            return None
        return self.get_revision(policy.active_revision_id)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "base_deployments": [
                deployment.to_dict() for deployment in self.base_deployments.values()
            ],
            "policies": [policy.to_dict() for policy in self.policies.values()],
            "adapter_revisions": [revision.to_dict() for revision in self.revisions.values()],
            "policy_sessions": [session.to_dict() for session in self.sessions.values()],
        }


class AdapterResidencyManager:
    """Warm/hot adapter residency tracker with LRU-style demotion."""

    def __init__(self, *, warm_capacity: int = 128, hot_capacity: int = 32) -> None:
        self.warm_capacity = _non_negative_int(warm_capacity, name="warm_capacity")
        self.hot_capacity = _non_negative_int(hot_capacity, name="hot_capacity")
        self._residency: OrderedDict[str, ServingResidency] = OrderedDict()

    def register_revision(
        self,
        revision: AdapterRevision,
        *,
        tier: AdapterResidencyTier = COOL_STORED,
        device: str | None = None,
        batch_key: str | None = None,
    ) -> ServingResidency:
        if revision.revision_id in self._residency:
            raise ValueError(f"adapter revision already has residency: {revision.revision_id}")
        return self.set_residency(
            revision.revision_id,
            tier=tier,
            device=device,
            batch_key=batch_key,
        )

    def set_residency(
        self,
        revision_id: str,
        *,
        tier: AdapterResidencyTier,
        device: str | None = None,
        batch_key: str | None = None,
    ) -> ServingResidency:
        revision_id = _non_empty_string(revision_id, name="revision_id")
        tier = _normalize_residency_tier(tier)
        if tier == COOL_STORED:
            device = None
            batch_key = None
        elif tier == WARM_CPU and device is None:
            device = "cpu"
        residency = ServingResidency(
            revision_id=revision_id,
            tier=tier,
            device=device,
            batch_key=batch_key,
        )
        self._residency[revision_id] = residency
        self._residency.move_to_end(revision_id)
        self._enforce_capacity(WARM_CPU, self.warm_capacity)
        self._enforce_capacity(HOT_GPU, self.hot_capacity)
        return self._residency[revision_id]

    def promote_to_warm(self, revision_id: str, *, device: str = "cpu") -> ServingResidency:
        return self.set_residency(revision_id, tier=WARM_CPU, device=device)

    def promote_to_hot(
        self,
        revision_id: str,
        *,
        device: str,
        batch_key: str | None = None,
    ) -> ServingResidency:
        return self.set_residency(revision_id, tier=HOT_GPU, device=device, batch_key=batch_key)

    def cool_down(self, revision_id: str) -> ServingResidency:
        return self.set_residency(revision_id, tier=COOL_STORED)

    def get(self, revision_id: str) -> ServingResidency:
        revision_id = _non_empty_string(revision_id, name="revision_id")
        if revision_id not in self._residency:
            raise KeyError(f"unknown residency revision: {revision_id}")
        self._residency.move_to_end(revision_id)
        return self._residency[revision_id]

    def plan_batch(self, revision_ids: Sequence[str]) -> dict[str, list[str]]:
        plan = {COOL_STORED: [], WARM_CPU: [], HOT_GPU: []}
        for revision_id in revision_ids:
            residency = self.get(revision_id)
            plan[residency.tier].append(residency.revision_id)
        return plan

    def summary(self) -> dict[str, Any]:
        counts = {COOL_STORED: 0, WARM_CPU: 0, HOT_GPU: 0}
        for residency in self._residency.values():
            counts[residency.tier] += 1
        return {
            "warm_capacity": self.warm_capacity,
            "hot_capacity": self.hot_capacity,
            "counts": counts,
            "residency": [residency.to_dict() for residency in self._residency.values()],
        }

    def _enforce_capacity(self, tier: AdapterResidencyTier, capacity: int) -> None:
        while sum(1 for item in self._residency.values() if item.tier == tier) > capacity:
            for revision_id, residency in list(self._residency.items()):
                if residency.tier == tier:
                    self._residency[revision_id] = ServingResidency(revision_id=revision_id)
                    break


__all__ = [
    "AdapterPolicyRegistry",
    "AdapterResidencyManager",
    "AdapterResidencyTier",
    "AdapterRevision",
    "BaseDeployment",
    "COOL_STORED",
    "HOT_GPU",
    "PolicyRecord",
    "PolicySession",
    "ServingResidency",
    "WARM_CPU",
]
