# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Diversity majority-vote primitives for Megatron Lite.

The Scaling PEFT paper reports collective gains from many LoRA adapters whose
training data differs by permutation or masking. This module represents the
local, deterministic contract needed before that result can be reproduced:
adapter identities, collaboration versus repetition arms, vote records, majority
aggregation, and paper log-fit anchors as reference-only metadata. It does not train adapters, collect model votes, run AIME24, load data or weights, or claim the paper's empirical accuracy.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

COLLABORATION_DISTINCT_ADAPTERS = "collaboration_distinct_adapters"
REPETITION_SINGLE_ADAPTER = "repetition_single_adapter"
MAJORITY_VOTE = "majority_vote"
PAPER_REFERENCE_LOG_FIT = "accuracy = 0.386 + 0.0172 * ln(k)"
PAPER_REFERENCE_R2 = 0.888
PAPER_REFERENCE_BASELINE_ACCURACY = 0.373
PAPER_REFERENCE_REPETITION_SATURATION_ACCURACY = 0.433
PAPER_REFERENCE_REPETITION_SATURATION_K = 24
PAPER_REFERENCE_MAX_K = 198

ArmType = Literal["collaboration_distinct_adapters", "repetition_single_adapter"]


def _type_name(value: Any) -> str:
    return type(value).__name__


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {_type_name(value)}.")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {_type_name(value)}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return int(value)


def _normalize_arm_type(value: Any, *, name: str = "arm_type") -> ArmType:
    value = _non_empty_string(value, name=name)
    if value not in {COLLABORATION_DISTINCT_ADAPTERS, REPETITION_SINGLE_ADAPTER}:
        raise ValueError(
            "arm_type must be collaboration_distinct_adapters or repetition_single_adapter."
        )
    return value  # type: ignore[return-value]


def _pending_status(value: Any, *, name: str) -> str:
    value = _non_empty_string(value, name=name)
    if not value.startswith("pending_real_"):
        raise ValueError(f"{name} must start with 'pending_real_', got {value!r}.")
    return value


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a boolean or None, got {_type_name(value)}.")


@dataclass(frozen=True)
class AdapterVariantSpec:
    """One LoRA adapter identity in a diversity ensemble."""

    adapter_id: str
    base_model: str = "Qwen3-30B"
    adapter_type: str = "lora"
    diversity_source: str = "data_permutation_or_masking"
    training_status: str = "pending_real_adapter_training"
    vote_status: str = "pending_real_aime24_vote"
    paper_metric_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", _non_empty_string(self.adapter_id, name="adapter_id"))
        object.__setattr__(self, "base_model", _non_empty_string(self.base_model, name="base_model"))
        object.__setattr__(self, "adapter_type", _non_empty_string(self.adapter_type, name="adapter_type"))
        object.__setattr__(
            self,
            "diversity_source",
            _non_empty_string(self.diversity_source, name="diversity_source"),
        )
        if self.adapter_type != "lora":
            raise ValueError("adapter_type must be lora.")
        if self.diversity_source != "data_permutation_or_masking":
            raise ValueError("diversity_source must be data_permutation_or_masking.")
        object.__setattr__(
            self,
            "training_status",
            _pending_status(self.training_status, name="training_status"),
        )
        object.__setattr__(self, "vote_status", _pending_status(self.vote_status, name="vote_status"))
        if self.paper_metric_claimed is not False:
            raise ValueError("AdapterVariantSpec must not claim paper metrics.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "base_model": self.base_model,
            "adapter_type": self.adapter_type,
            "diversity_source": self.diversity_source,
            "training_status": self.training_status,
            "vote_status": self.vote_status,
            "paper_metric_claimed": False,
        }


@dataclass(frozen=True)
class VoteCollectionArm:
    """A pending evaluation arm for distinct-adapter or repetition voting."""

    k: int
    arm_type: ArmType
    benchmark: str = "AIME24"
    aggregation: str = MAJORITY_VOTE
    accuracy_status: str = "pending_real_metric"
    claim_ready: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "k", _positive_int(self.k, name="k"))
        object.__setattr__(self, "arm_type", _normalize_arm_type(self.arm_type))
        object.__setattr__(self, "benchmark", _non_empty_string(self.benchmark, name="benchmark"))
        if self.aggregation != MAJORITY_VOTE:
            raise ValueError("aggregation must be majority_vote.")
        object.__setattr__(
            self,
            "accuracy_status",
            _pending_status(self.accuracy_status, name="accuracy_status"),
        )
        if self.claim_ready is not False:
            raise ValueError("VoteCollectionArm must keep claim_ready=False until real metrics exist.")

    @property
    def arm_id(self) -> str:
        return f"{self.benchmark.lower()}/{self.arm_type}/k{self.k}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "benchmark": self.benchmark,
            "k": self.k,
            "arm_type": self.arm_type,
            "aggregation": self.aggregation,
            "accuracy_status": self.accuracy_status,
            "claim_ready": False,
        }


@dataclass(frozen=True)
class VoteRecord:
    """One model answer before majority aggregation."""

    problem_id: str
    adapter_id: str
    sample_id: str
    answer: str
    normalized_answer: str
    vote_group_k: int
    arm_type: ArmType
    is_correct: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _non_empty_string(self.problem_id, name="problem_id"))
        object.__setattr__(self, "adapter_id", _non_empty_string(self.adapter_id, name="adapter_id"))
        object.__setattr__(self, "sample_id", _non_empty_string(self.sample_id, name="sample_id"))
        object.__setattr__(self, "answer", _non_empty_string(self.answer, name="answer"))
        object.__setattr__(
            self,
            "normalized_answer",
            _non_empty_string(self.normalized_answer, name="normalized_answer"),
        )
        object.__setattr__(self, "vote_group_k", _positive_int(self.vote_group_k, name="vote_group_k"))
        object.__setattr__(self, "arm_type", _normalize_arm_type(self.arm_type))
        object.__setattr__(self, "is_correct", _optional_bool(self.is_correct, name="is_correct"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "adapter_id": self.adapter_id,
            "sample_id": self.sample_id,
            "answer": self.answer,
            "normalized_answer": self.normalized_answer,
            "is_correct": self.is_correct,
            "vote_group_k": self.vote_group_k,
            "arm_type": self.arm_type,
        }


@dataclass(frozen=True)
class MajorityVoteResult:
    """Majority-vote output for one problem and one arm."""

    problem_id: str
    vote_group_k: int
    arm_type: ArmType
    normalized_answer: str
    vote_count: int
    total_votes: int
    tied: bool
    is_correct: bool | None = None
    paper_metric_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _non_empty_string(self.problem_id, name="problem_id"))
        object.__setattr__(self, "vote_group_k", _positive_int(self.vote_group_k, name="vote_group_k"))
        object.__setattr__(self, "arm_type", _normalize_arm_type(self.arm_type))
        object.__setattr__(
            self,
            "normalized_answer",
            _non_empty_string(self.normalized_answer, name="normalized_answer"),
        )
        object.__setattr__(self, "vote_count", _positive_int(self.vote_count, name="vote_count"))
        object.__setattr__(self, "total_votes", _positive_int(self.total_votes, name="total_votes"))
        if self.vote_count > self.total_votes:
            raise ValueError("vote_count must not exceed total_votes.")
        if not isinstance(self.tied, bool):
            raise TypeError("tied must be a boolean.")
        object.__setattr__(self, "is_correct", _optional_bool(self.is_correct, name="is_correct"))
        if self.paper_metric_claimed is not False:
            raise ValueError("MajorityVoteResult must not claim paper metrics.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "vote_group_k": self.vote_group_k,
            "arm_type": self.arm_type,
            "normalized_answer": self.normalized_answer,
            "vote_count": self.vote_count,
            "total_votes": self.total_votes,
            "tied": self.tied,
            "is_correct": self.is_correct,
            "paper_metric_claimed": False,
        }


def build_adapter_variants(
    *,
    count: int = PAPER_REFERENCE_MAX_K,
    prefix: str = "adapter/qwen3-30b-diversity",
) -> tuple[AdapterVariantSpec, ...]:
    count = _positive_int(count, name="count")
    prefix = _non_empty_string(prefix, name="prefix")
    return tuple(
        AdapterVariantSpec(adapter_id=f"{prefix}/{index:03d}")
        for index in range(1, count + 1)
    )


def build_vote_collection_arms(
    *,
    collaboration_k: Sequence[int] = (1, 10, 100, PAPER_REFERENCE_MAX_K),
    repetition_k: Sequence[int] = (1, 10, PAPER_REFERENCE_REPETITION_SATURATION_K, 100),
) -> tuple[VoteCollectionArm, ...]:
    collaboration = tuple(
        VoteCollectionArm(k=_positive_int(k, name="collaboration_k"), arm_type=COLLABORATION_DISTINCT_ADAPTERS)
        for k in collaboration_k
    )
    repetition = tuple(
        VoteCollectionArm(k=_positive_int(k, name="repetition_k"), arm_type=REPETITION_SINGLE_ADAPTER)
        for k in repetition_k
    )
    if not collaboration or not repetition:
        raise ValueError("collaboration_k and repetition_k must be non-empty.")
    return collaboration + repetition


def _validate_vote_group(records: tuple[VoteRecord, ...]) -> tuple[str, int, ArmType]:
    if not records:
        raise ValueError("records must be non-empty.")
    problem_ids = {record.problem_id for record in records}
    vote_group_ks = {record.vote_group_k for record in records}
    arm_types = {record.arm_type for record in records}
    if len(problem_ids) != 1 or len(vote_group_ks) != 1 or len(arm_types) != 1:
        raise ValueError("records must share problem_id, vote_group_k, and arm_type.")
    problem_id = next(iter(problem_ids))
    vote_group_k = next(iter(vote_group_ks))
    arm_type = next(iter(arm_types))
    if len(records) != vote_group_k:
        raise ValueError("record count must match vote_group_k.")
    sample_ids = [record.sample_id for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique within a vote group.")
    adapter_ids = [record.adapter_id for record in records]
    if arm_type == COLLABORATION_DISTINCT_ADAPTERS and len(set(adapter_ids)) != len(adapter_ids):
        raise ValueError("collaboration arms require distinct adapter_id values.")
    if arm_type == REPETITION_SINGLE_ADAPTER and len(set(adapter_ids)) != 1:
        raise ValueError("repetition arms require one repeated adapter_id.")
    return problem_id, vote_group_k, arm_type


def majority_vote(records: Iterable[VoteRecord]) -> MajorityVoteResult:
    records = tuple(records)
    problem_id, vote_group_k, arm_type = _validate_vote_group(records)
    counts = Counter(record.normalized_answer for record in records)
    max_votes = max(counts.values())
    winners = sorted(answer for answer, count in counts.items() if count == max_votes)
    winner = winners[0]
    correctness = {record.is_correct for record in records if record.normalized_answer == winner}
    return MajorityVoteResult(
        problem_id=problem_id,
        vote_group_k=vote_group_k,
        arm_type=arm_type,
        normalized_answer=winner,
        vote_count=max_votes,
        total_votes=len(records),
        tied=len(winners) > 1,
        is_correct=correctness.pop() if len(correctness) == 1 else None,
    )


def accuracy_from_majority_votes(results: Iterable[MajorityVoteResult]) -> float:
    results = tuple(results)
    if not results:
        raise ValueError("results must be non-empty.")
    if any(result.is_correct is None for result in results):
        raise ValueError("all results must have is_correct before accuracy can be computed.")
    return sum(1 for result in results if result.is_correct) / float(len(results))


def paper_reference_accuracy(k: int) -> dict[str, Any]:
    k = _positive_int(k, name="k")
    return {
        "k": k,
        "formula": PAPER_REFERENCE_LOG_FIT,
        "reported_r2": PAPER_REFERENCE_R2,
        "reference_accuracy": 0.386 + 0.0172 * math.log(float(k)),
        "metric_status": "paper_reference_not_local_result",
    }


def paper_reference_vote_contract() -> dict[str, Any]:
    return {
        "max_k": PAPER_REFERENCE_MAX_K,
        "collaboration_k": [1, 10, 100, PAPER_REFERENCE_MAX_K],
        "repetition_k": [1, 10, PAPER_REFERENCE_REPETITION_SATURATION_K, 100],
        "baseline_accuracy": PAPER_REFERENCE_BASELINE_ACCURACY,
        "repetition_saturation_accuracy": PAPER_REFERENCE_REPETITION_SATURATION_ACCURACY,
        "log_fit": PAPER_REFERENCE_LOG_FIT,
        "reported_r2": PAPER_REFERENCE_R2,
        "metric_status": "paper_reference_not_local_result",
    }


__all__ = [
    "AdapterVariantSpec",
    "ArmType",
    "COLLABORATION_DISTINCT_ADAPTERS",
    "MAJORITY_VOTE",
    "MajorityVoteResult",
    "PAPER_REFERENCE_BASELINE_ACCURACY",
    "PAPER_REFERENCE_LOG_FIT",
    "PAPER_REFERENCE_MAX_K",
    "PAPER_REFERENCE_R2",
    "PAPER_REFERENCE_REPETITION_SATURATION_ACCURACY",
    "PAPER_REFERENCE_REPETITION_SATURATION_K",
    "REPETITION_SINGLE_ADAPTER",
    "VoteCollectionArm",
    "VoteRecord",
    "accuracy_from_majority_votes",
    "build_adapter_variants",
    "build_vote_collection_arms",
    "majority_vote",
    "paper_reference_accuracy",
    "paper_reference_vote_contract",
]
