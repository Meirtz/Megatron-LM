# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron"
    / "lite"
    / "primitive"
    / "modules"
    / "adapter_rank.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("adapter_rank_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rslora_alpha_rules_encode_eq2_rank_dependence_without_results():
    rank = _load_module()

    arms = rank.build_rslora_lr_transfer_arms(ranks=(1, 4, 16, 64), reference_rank=16)
    by_rule = {}
    for arm in arms:
        by_rule.setdefault(arm.alpha_rule, []).append(arm)

    sqrt_arms = by_rule["sqrt_rank_rslora"]
    assert all(arm.use_rslora for arm in sqrt_arms)
    assert {round(arm.runtime_scale, 12) for arm in sqrt_arms} == {8.0}
    assert {round(arm.eq2_alpha_squared_over_rank, 12) for arm in sqrt_arms} == {64.0}
    assert all(arm.scaling_convention == "alpha_over_sqrt_rank" for arm in sqrt_arms)

    const_arms = by_rule["const_alpha"]
    assert [arm.eq2_alpha_squared_over_rank for arm in const_arms] == [1024.0, 256.0, 64.0, 16.0]
    assert all(not arm.use_rslora for arm in const_arms)

    fixed_arms = by_rule["fixed_alpha_over_rank"]
    assert all(math.isclose(arm.runtime_scale, 1.0) for arm in fixed_arms)
    assert [arm.eq2_alpha_squared_over_rank for arm in fixed_arms] == [1.0, 4.0, 16.0, 64.0]
    assert all(arm.claims_paper_results is False for arm in arms)


def test_rank_regime_contract_builds_paper_216_shape_as_pending_work():
    rank = _load_module()

    contract = rank.build_rank_sweep_contract()
    invariants = rank.rank_sweep_invariants(contract)

    assert len(contract.rank_regime_arms) == 216
    assert len(contract.rank1_olora_tail_arms) == 48
    assert invariants["rank_regime_run_count"] == 216
    assert invariants["rank_regime_matches_paper_216_shape"] is True
    assert invariants["rank1_olora_tail_run_count"] == 48
    assert invariants["rank1_olora_tail_matches_paper_48_shape"] is True
    assert invariants["sqrt_rank_rslora_eq2_factor_constant"] is True
    assert invariants["sqrt_rank_rslora_runtime_scale_constant"] is True
    assert invariants["all_rank_regime_arms_pending_results"] is True
    assert invariants["all_rank1_olora_tail_arms_pending_results"] is True
    assert invariants["all_rslora_arms_pending_lr_grid"] is True
    assert invariants["paper_results_not_claimed"] is True
    assert contract.paper_exact is False
    assert contract.claims_paper_results is False
    assert contract.launches_training is False
    rank.validate_rank_sweep_contract(contract)

    payload = contract.to_dict()
    assert payload["status"] == "pass_rank_sweep_contract"
    assert payload["paper_exact"] is False
    assert payload["claims_paper_results"] is False
    assert payload["launches_training"] is False
    assert "216 completed" in " ".join(payload["missing_for_paper_exact"])


def test_rank_regime_contract_rejects_bad_shapes_and_metric_claims():
    rank = _load_module()

    small = rank.build_rank_sweep_contract(
        ranks=(1, 16),
        batch_sizes=(16,),
        seeds=(0,),
        reference_rank=16,
    )
    assert small.to_dict()["status"] == "fail_rank_sweep_contract"
    with pytest.raises(ValueError, match="216-arm"):
        rank.validate_rank_sweep_contract(small)

    with pytest.raises(ValueError, match="duplicates"):
        rank.build_rank_sweep_contract(ranks=(1, 1, 16), reference_rank=16)
    with pytest.raises(ValueError, match="reference_rank"):
        rank.build_rank_sweep_contract(ranks=(1, 2, 4), reference_rank=16)
    with pytest.raises(ValueError, match="must not claim paper metrics"):
        rank.RankRegimeArm(rank=1, effective_batch_size=16, seed=0, claims_paper_results=True)
    with pytest.raises(ValueError, match="must not claim paper metrics"):
        rank.RsLoraTransferArm(
            alpha_rule="sqrt_rank_rslora",
            rank=1,
            alpha=8.0,
            use_rslora=True,
            scaling_convention="alpha_over_sqrt_rank",
            runtime_scale=8.0,
            eq2_alpha_squared_over_rank=64.0,
            claims_paper_results=True,
        )
    with pytest.raises(ValueError, match="rank must be 1"):
        rank.Rank1OloraTailArm(
            init_lora_weights="olora_tail",
            rank=2,
            effective_batch_size=16,
            seed=0,
        )
    with pytest.raises(ValueError, match="must not claim paper metrics"):
        rank.Rank1OloraTailArm(
            init_lora_weights="olora_tail",
            effective_batch_size=16,
            seed=0,
            claims_paper_results=True,
        )


def test_rank_sweep_contract_status_rejects_false_invariants():
    rank = _load_module()

    contract = rank.build_rank_sweep_contract()
    bad_arms = (
        rank.RankRegimeArm(
            rank=contract.rank_regime_arms[0].rank,
            effective_batch_size=contract.rank_regime_arms[0].effective_batch_size,
            seed=contract.rank_regime_arms[0].seed,
            result_status="done_without_contract_refresh",
        ),
        *contract.rank_regime_arms[1:],
    )
    bad_contract = rank.RankSweepContract(
        ranks=contract.ranks,
        batch_sizes=contract.batch_sizes,
        seeds=contract.seeds,
        rslora_lr_transfer_arms=contract.rslora_lr_transfer_arms,
        rank_regime_arms=bad_arms,
    )

    assert bad_contract.invariants()["all_rank_regime_arms_pending_results"] is False
    assert bad_contract.to_dict()["status"] == "fail_rank_sweep_contract"
    with pytest.raises(ValueError, match="all_rank_regime_arms_pending_results"):
        rank.validate_rank_sweep_contract(bad_contract)


def _rank_regime_metric(rank_value: int, batch_size: int, seed: int) -> float:
    batch_offset = 0.001 * (batch_size / 16.0)
    if rank_value in (16, 32):
        return 60.0 + (0.2 if rank_value == 32 else 0.0) + (0.01 * seed) + batch_offset
    if rank_value in (1, 2, 4):
        return (60.4 + batch_offset) if seed == 0 else (43.0 + rank_value + 0.01 * seed + batch_offset)
    if rank_value >= 64:
        return 58.8 + (0.02 * seed) + batch_offset
    return 56.0 + (0.02 * seed) + batch_offset


def _rank_regime_results(rank) -> list[dict[str, object]]:
    results = []
    for arm in rank.build_rank_sweep_contract().rank_regime_arms:
        results.append(
            {
                "status": "pass",
                "evidence_type": "paper_scale_rl_run",
                "explicit_user_confirmation": True,
                "gpu_or_training_launched": True,
                "optimizer_step_evidence": True,
                "rank": arm.rank,
                "effective_batch_size": arm.effective_batch_size,
                "seed": arm.seed,
                "steps": arm.steps,
                "optimizer_steps": arm.steps,
                "metrics": {
                    "math_accuracy": _rank_regime_metric(
                        arm.rank,
                        arm.effective_batch_size,
                        arm.seed,
                    )
                },
            }
        )
    return results


_RSLORA_LR_GRID = (1e-6, 3e-6, 1e-5, 3e-5)


def _fixed_alpha_best_lr(rank_value: int) -> float:
    if rank_value <= 4:
        return 3e-5
    if rank_value <= 16:
        return 1e-5
    if rank_value <= 64:
        return 3e-6
    return 1e-6


def _rslora_lr_transfer_metric(alpha_rule: str, rank_value: int, learning_rate: float) -> float:
    lr_index = {lr: index for index, lr in enumerate(_RSLORA_LR_GRID)}
    if alpha_rule == "sqrt_rank_rslora":
        penalties = {1e-6: 2.5, 3e-6: 0.4, 1e-5: 0.0, 3e-5: 3.0}
        return 70.0 - penalties[learning_rate] - (0.001 * math.log2(rank_value))
    if alpha_rule == "const_alpha":
        penalties = {1e-6: 1.2, 3e-6: 0.0, 1e-5: 1.5, 3e-5: 5.0}
        return 69.7 - penalties[learning_rate] - (0.001 * math.log2(rank_value))
    best_lr = _fixed_alpha_best_lr(rank_value)
    return 69.6 - (1.1 * abs(lr_index[learning_rate] - lr_index[best_lr]))


def _rslora_lr_transfer_results(rank) -> list[dict[str, object]]:
    results = []
    for arm in rank.build_rank_sweep_contract().rslora_lr_transfer_arms:
        for learning_rate in _RSLORA_LR_GRID:
            results.append(
                {
                    "status": "pass",
                    "evidence_type": "rslora_lr_transfer_run",
                    "explicit_user_confirmation": True,
                    "gpu_or_training_launched": True,
                    "optimizer_step_evidence": True,
                    "alpha_rule": arm.alpha_rule,
                    "rank": arm.rank,
                    "alpha": arm.alpha,
                    "use_rslora": arm.use_rslora,
                    "scaling_convention": arm.scaling_convention,
                    "learning_rate": learning_rate,
                    "seed": 0,
                    "optimizer_steps": 500,
                    "metrics": {
                        "eval_accuracy": _rslora_lr_transfer_metric(
                            arm.alpha_rule,
                            arm.rank,
                            learning_rate,
                        )
                    },
                }
            )
    return results


def _rank1_metric(init_lora_weights: str, batch_size: int, seed: int) -> float:
    seed_offset = 0.02 * seed
    if init_lora_weights == "olora_tail":
        return {16: 21.0, 32: 20.5, 64: 20.0, 128: 19.5}[batch_size] + seed_offset
    return {16: 15.0, 32: 5.0, 64: -5.0, 128: -18.0}[batch_size] - seed_offset


def _rank1_olora_tail_results(rank) -> list[dict[str, object]]:
    results = []
    for arm in rank.build_rank_sweep_contract().rank1_olora_tail_arms:
        results.append(
            {
                "status": "pass",
                "evidence_type": "rank1_olora_tail_run",
                "explicit_user_confirmation": True,
                "gpu_or_training_launched": True,
                "optimizer_step_evidence": True,
                "rank": arm.rank,
                "alpha": arm.alpha,
                "init_lora_weights": arm.init_lora_weights,
                "effective_batch_size": arm.effective_batch_size,
                "seed": arm.seed,
                "steps": arm.steps,
                "optimizer_steps": arm.steps,
                "metrics": {
                    "gain_percent": _rank1_metric(
                        arm.init_lora_weights,
                        arm.effective_batch_size,
                        arm.seed,
                    )
                },
            }
        )
    return results


def test_rank_regime_result_analyzer_checks_216_run_paper_pattern_without_launching():
    rank = _load_module()

    analysis = rank.analyze_rank_regime_results(
        _rank_regime_results(rank),
        pattern_tolerance=0.5,
    )

    assert analysis["status"] == "pass_rank_regime_result_analysis"
    assert analysis["observed_run_count"] == rank.PAPER_RANK_REGIME_RUN_COUNT
    assert analysis["missing_run_count"] == 0
    assert analysis["paper_exact"] is False
    assert analysis["claims_exact_paper_results"] is False
    assert analysis["paper_pattern_supported"] is True
    assert analysis["paper_pattern_status"] == "pass_paper_rank_regime_pattern"
    assert analysis["pattern_checks"]["deployment_default_has_highest_mean"] is True
    assert analysis["pattern_checks"]["frontier_reliability_is_lower"] is True
    assert analysis["regimes"]["deployment_default_ranks_16_32"]["count"] == 48


def test_rank_regime_result_analyzer_rejects_missing_or_unconfirmed_runs():
    rank = _load_module()
    results = _rank_regime_results(rank)

    missing = rank.analyze_rank_regime_results(results[:-1], pattern_tolerance=0.5)
    assert missing["status"] == "fail_rank_regime_result_analysis"
    assert missing["missing_run_count"] == 1
    assert missing["paper_pattern_supported"] is False

    unconfirmed = _rank_regime_results(rank)
    unconfirmed[0]["gpu_or_training_launched"] = False
    analysis = rank.analyze_rank_regime_results(unconfirmed, pattern_tolerance=0.5)
    assert analysis["status"] == "fail_rank_regime_result_analysis"
    assert any("gpu_or_training_launched" in failure for failure in analysis["failures"])
    assert analysis["paper_pattern_supported"] is False


def test_rslora_lr_transfer_analyzer_checks_reusable_band_pattern_without_launching():
    rank = _load_module()

    analysis = rank.analyze_rslora_lr_transfer_results(
        _rslora_lr_transfer_results(rank),
        expected_learning_rates=_RSLORA_LR_GRID,
        expected_seeds=(0,),
        reusable_band_tolerance=0.75,
        metric_tolerance=0.5,
    )

    assert analysis["status"] == "pass_rslora_lr_transfer_result_analysis"
    assert analysis["expected_result_count"] == 108
    assert analysis["missing_result_count"] == 0
    assert analysis["paper_exact"] is False
    assert analysis["claims_exact_paper_results"] is False
    assert analysis["rslora_transfer_pattern_supported"] is True
    assert analysis["paper_pattern_status"] == "pass_paper_rslora_lr_transfer_pattern"
    assert analysis["pattern_checks"]["sqrt_rank_has_largest_reusable_lr_band"] is True
    assert analysis["pattern_checks"]["fixed_alpha_over_rank_best_lr_moves_down_with_rank"] is True
    assert analysis["rule_summaries"]["sqrt_rank_rslora"]["reusable_lr_band_count"] == 2


def test_rslora_lr_transfer_analyzer_rejects_missing_unconfirmed_or_mismatched_runs():
    rank = _load_module()
    results = _rslora_lr_transfer_results(rank)

    missing = rank.analyze_rslora_lr_transfer_results(
        results[:-1],
        expected_learning_rates=_RSLORA_LR_GRID,
        expected_seeds=(0,),
        reusable_band_tolerance=0.75,
        metric_tolerance=0.5,
    )
    assert missing["status"] == "fail_rslora_lr_transfer_result_analysis"
    assert missing["missing_result_count"] == 1
    assert missing["paper_pattern_supported"] is False

    unconfirmed = _rslora_lr_transfer_results(rank)
    unconfirmed[0]["explicit_user_confirmation"] = False
    unconfirmed[1]["alpha"] = 123.0
    unconfirmed[2]["claims_paper_results"] = True
    analysis = rank.analyze_rslora_lr_transfer_results(
        unconfirmed,
        expected_learning_rates=_RSLORA_LR_GRID,
        expected_seeds=(0,),
        reusable_band_tolerance=0.75,
        metric_tolerance=0.5,
    )
    assert analysis["status"] == "fail_rslora_lr_transfer_result_analysis"
    assert any("explicit_user_confirmation" in failure for failure in analysis["failures"])
    assert any(".alpha must match" in failure for failure in analysis["failures"])
    assert any("must not claim paper-exact results" in failure for failure in analysis["failures"])


def test_rank1_olora_tail_analyzer_checks_fig15_stability_without_launching():
    rank = _load_module()

    analysis = rank.analyze_rank1_olora_tail_results(
        _rank1_olora_tail_results(rank),
        min_olora_tail_mean_gain=15.0,
        min_standard_degradation_drop=20.0,
    )

    assert analysis["status"] == "pass_rank1_olora_tail_result_analysis"
    assert analysis["expected_run_count"] == rank.PAPER_RANK1_OLORA_TAIL_RUN_COUNT
    assert analysis["observed_run_count"] == 48
    assert analysis["paper_exact"] is False
    assert analysis["claims_exact_paper_results"] is False
    assert analysis["rank1_stability_pattern_supported"] is True
    assert analysis["paper_pattern_status"] == "pass_paper_rank1_olora_tail_stability_pattern"
    assert analysis["pattern_checks"]["olora_tail_positive_across_batches"] is True
    assert analysis["pattern_checks"]["standard_lora_degrades_with_batch"] is True
    assert analysis["per_init"]["olora_tail"]["batch_count"] == 4


def test_rank1_olora_tail_analyzer_rejects_missing_unconfirmed_or_bad_claims():
    rank = _load_module()
    results = _rank1_olora_tail_results(rank)

    missing = rank.analyze_rank1_olora_tail_results(results[:-1])
    assert missing["status"] == "fail_rank1_olora_tail_result_analysis"
    assert missing["missing_run_count"] == 1
    assert missing["paper_pattern_supported"] is False

    bad = _rank1_olora_tail_results(rank)
    bad[0]["gpu_or_training_launched"] = False
    bad[1]["alpha"] = 123.0
    bad[2]["claims_paper_results"] = True
    bad[3]["init_lora_weights"] = "pissa"
    analysis = rank.analyze_rank1_olora_tail_results(bad)
    assert analysis["status"] == "fail_rank1_olora_tail_result_analysis"
    assert any("gpu_or_training_launched" in failure for failure in analysis["failures"])
    assert any(".alpha must match" in failure for failure in analysis["failures"])
    assert any("must not claim paper-exact results" in failure for failure in analysis["failures"])
    assert any("init_lora_weights must be standard or olora_tail" in failure for failure in analysis["failures"])


def test_rank_reference_contract_is_not_local_evidence():
    rank = _load_module()

    reference = rank.paper_reference_rank_regime_contract()
    assert reference["paper_reference_not_local_result"] is True
    assert reference["claims_paper_results"] is False
    assert "9 ranks x 4 batch sizes x 6 seeds" in reference["rank_regimes"]
    assert "Fig.15 rank-1" in reference["rank1_olora_tail"]
