# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron"
    / "lite"
    / "primitive"
    / "modules"
    / "diversity_vote.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("diversity_vote_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_diversity_vote_builds_paper_shaped_adapter_ensemble_without_claims():
    vote = _load_module()

    variants = vote.build_adapter_variants()
    arms = vote.build_vote_collection_arms()
    contract = vote.paper_reference_vote_contract()
    reference_k198 = vote.paper_reference_accuracy(198)

    assert len(variants) == 198
    assert len({variant.adapter_id for variant in variants}) == 198
    assert {variant.base_model for variant in variants} == {"Qwen3-30B"}
    assert {variant.adapter_type for variant in variants} == {"lora"}
    assert {variant.diversity_source for variant in variants} == {"data_permutation_or_masking"}
    assert all(variant.paper_metric_claimed is False for variant in variants)
    assert [(arm.arm_type, arm.k) for arm in arms[:4]] == [
        ("collaboration_distinct_adapters", 1),
        ("collaboration_distinct_adapters", 10),
        ("collaboration_distinct_adapters", 100),
        ("collaboration_distinct_adapters", 198),
    ]
    assert [(arm.arm_type, arm.k) for arm in arms[4:]] == [
        ("repetition_single_adapter", 1),
        ("repetition_single_adapter", 10),
        ("repetition_single_adapter", 24),
        ("repetition_single_adapter", 100),
    ]
    assert all(arm.claim_ready is False for arm in arms)
    assert contract["metric_status"] == "paper_reference_not_local_result"
    assert reference_k198["metric_status"] == "paper_reference_not_local_result"


def test_diversity_vote_majority_vote_collaboration_and_repetition_rules():
    vote = _load_module()

    collaboration_records = (
        vote.VoteRecord("p1", "adapter/001", "s1", "A", "a", 3, "collaboration_distinct_adapters", True),
        vote.VoteRecord("p1", "adapter/002", "s2", "A", "a", 3, "collaboration_distinct_adapters", True),
        vote.VoteRecord("p1", "adapter/003", "s3", "B", "b", 3, "collaboration_distinct_adapters", False),
    )
    collaboration_result = vote.majority_vote(collaboration_records)

    assert collaboration_result.normalized_answer == "a"
    assert collaboration_result.vote_count == 2
    assert collaboration_result.total_votes == 3
    assert collaboration_result.tied is False
    assert collaboration_result.is_correct is True
    assert collaboration_result.paper_metric_claimed is False

    repetition_records = (
        vote.VoteRecord("p2", "adapter/001", "s1", "C", "c", 3, "repetition_single_adapter", False),
        vote.VoteRecord("p2", "adapter/001", "s2", "D", "d", 3, "repetition_single_adapter", True),
        vote.VoteRecord("p2", "adapter/001", "s3", "D", "d", 3, "repetition_single_adapter", True),
    )
    repetition_result = vote.majority_vote(repetition_records)

    assert repetition_result.normalized_answer == "d"
    assert repetition_result.is_correct is True
    assert vote.accuracy_from_majority_votes((collaboration_result, repetition_result)) == 1.0


def test_diversity_vote_rejects_invalid_vote_groups_and_unverified_accuracy():
    vote = _load_module()

    with pytest.raises(ValueError, match="distinct adapter_id"):
        vote.majority_vote(
            (
                vote.VoteRecord("p1", "adapter/001", "s1", "A", "a", 2, "collaboration_distinct_adapters"),
                vote.VoteRecord("p1", "adapter/001", "s2", "B", "b", 2, "collaboration_distinct_adapters"),
            )
        )
    with pytest.raises(ValueError, match="one repeated adapter_id"):
        vote.majority_vote(
            (
                vote.VoteRecord("p1", "adapter/001", "s1", "A", "a", 2, "repetition_single_adapter"),
                vote.VoteRecord("p1", "adapter/002", "s2", "A", "a", 2, "repetition_single_adapter"),
            )
        )
    with pytest.raises(ValueError, match="record count must match vote_group_k"):
        vote.majority_vote(
            (
                vote.VoteRecord("p1", "adapter/001", "s1", "A", "a", 3, "collaboration_distinct_adapters"),
                vote.VoteRecord("p1", "adapter/002", "s2", "A", "a", 3, "collaboration_distinct_adapters"),
            )
        )
    with pytest.raises(ValueError, match="all results must have is_correct"):
        vote.accuracy_from_majority_votes(
            (
                vote.majority_vote(
                    (
                        vote.VoteRecord("p1", "adapter/001", "s1", "A", "a", 1, "collaboration_distinct_adapters"),
                    )
                ),
            )
        )


def test_diversity_vote_ties_are_deterministic_but_not_paper_claims():
    vote = _load_module()
    result = vote.majority_vote(
        (
            vote.VoteRecord("p1", "adapter/001", "s1", "B", "b", 2, "collaboration_distinct_adapters", False),
            vote.VoteRecord("p1", "adapter/002", "s2", "A", "a", 2, "collaboration_distinct_adapters", True),
        )
    )

    assert result.tied is True
    assert result.normalized_answer == "a"
    assert result.is_correct is True
    with pytest.raises(ValueError, match="must not claim paper metrics"):
        vote.MajorityVoteResult(
            problem_id="p1",
            vote_group_k=1,
            arm_type="collaboration_distinct_adapters",
            normalized_answer="a",
            vote_count=1,
            total_votes=1,
            tied=False,
            paper_metric_claimed=True,
        )
