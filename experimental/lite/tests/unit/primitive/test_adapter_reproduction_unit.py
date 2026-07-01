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
    / "adapter_reproduction.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("adapter_reproduction_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fig14_arm(repro, name: str, score: float) -> dict[str, object]:
    is_olora = name == "olora_tail"
    return {
        "name": name,
        "status": "pass",
        "evidence_type": "paper_scale_rl_run",
        "explicit_user_confirmation": True,
        "gpu_or_training_launched": True,
        "optimizer_step_evidence": True,
        "paper_exact": False,
        "claims_paper_results": False,
        "model_family": repro.FIG14_MODEL_FAMILY,
        "dataset": repro.FIG14_DATASET,
        "algorithm": repro.FIG14_ALGORITHM,
        "rank": 16,
        "steps": 500,
        "optimizer_steps": 500,
        "effective_batch_size": 32,
        "lora_alpha": 32.0,
        "learning_rate": 1e-5,
        "init_lora_weights": "olora_tail" if is_olora else "standard",
        "olora_tail_applied": is_olora,
        "target_modules": list(repro.FIG14_TARGET_MODULES),
        "metrics": {metric: score for metric in repro.FIG14_METRICS},
    }


def _fig14_results(repro) -> list[dict[str, object]]:
    return [
        _fig14_arm(repro, "lora_baseline", 56.3),
        _fig14_arm(repro, "olora_tail", 58.3),
    ]


def test_fig14_contract_is_local_no_launch_no_claim():
    repro = _load_module()

    contract = repro.build_fig14_olora_tail_contract()
    payload = contract.to_dict()

    assert payload["status"] == "pass_fig14_olora_tail_contract"
    assert payload["paper_exact"] is False
    assert payload["claims_paper_results"] is False
    assert payload["launches_training"] is False
    assert {arm["name"] for arm in payload["arms"]} == {"lora_baseline", "olora_tail"}
    assert all(arm["result_status"] == "pending_gpu_training_and_eval" for arm in payload["arms"])
    assert "GSM8K" in payload["arms"][0]["metrics"]
    assert "user-confirmed LoRA baseline" in " ".join(payload["missing_for_paper_exact"])

    with pytest.raises(ValueError, match="must not claim paper metrics"):
        repro.Fig14ArmContract(
            name="lora_baseline",
            init_lora_weights="standard",
            olora_tail_applied=False,
            claims_paper_results=True,
        )
    with pytest.raises(ValueError, match="no-launch/no-claim"):
        repro.Fig14OloraTailContract(arms=contract.arms, launches_training=True)


def test_fig14_result_analyzer_checks_paper_shape_without_launching():
    repro = _load_module()

    analysis = repro.analyze_fig14_olora_tail_results(_fig14_results(repro))

    assert analysis["status"] == "pass_fig14_olora_tail_result_analysis"
    assert analysis["paper_exact"] is False
    assert analysis["claims_exact_paper_results"] is False
    assert analysis["failures"] == []
    assert round(analysis["observed_averages"]["lora_baseline"], 6) == 56.3
    assert round(analysis["observed_averages"]["olora_tail"], 6) == 58.3
    assert round(analysis["observed_absolute_delta"], 6) == 2.0
    assert analysis["pattern_checks"]["two_arm_coverage"] is True
    assert analysis["pattern_checks"]["all_results_have_real_run_evidence"] is True
    assert analysis["pattern_checks"]["olora_tail_beats_lora_average"] is True
    assert analysis["pattern_checks"]["averages_match_paper_reported_values"] is True
    assert analysis["pattern_checks"]["absolute_delta_matches_paper"] is True
    assert analysis["paper_pattern_supported"] is True
    assert analysis["paper_pattern_status"] == "pass_paper_fig14_olora_tail_pattern"


def test_fig14_result_analyzer_rejects_missing_unconfirmed_or_spoofed_results():
    repro = _load_module()

    missing = repro.analyze_fig14_olora_tail_results(_fig14_results(repro)[:1])
    assert missing["status"] == "fail_fig14_olora_tail_result_analysis"
    assert any("missing Fig.14 arm olora_tail" in failure for failure in missing["failures"])
    assert missing["paper_pattern_supported"] is False

    unconfirmed = _fig14_results(repro)
    unconfirmed[0]["explicit_user_confirmation"] = False
    analysis = repro.analyze_fig14_olora_tail_results(unconfirmed)
    assert analysis["status"] == "fail_fig14_olora_tail_result_analysis"
    assert any("explicit_user_confirmation" in failure for failure in analysis["failures"])

    wrong_target = _fig14_results(repro)
    wrong_target[0]["target_modules"] = ["q_proj"]
    analysis = repro.analyze_fig14_olora_tail_results(wrong_target)
    assert analysis["status"] == "fail_fig14_olora_tail_result_analysis"
    assert any("target_modules" in failure for failure in analysis["failures"])

    wrong_model = _fig14_results(repro)
    wrong_model[0]["model_family"] = "Qwen3-8B"
    analysis = repro.analyze_fig14_olora_tail_results(wrong_model)
    assert analysis["status"] == "fail_fig14_olora_tail_result_analysis"
    assert any("model_family" in failure for failure in analysis["failures"])

    bad_claim = _fig14_results(repro)
    bad_claim[0]["claims_paper_results"] = True
    analysis = repro.analyze_fig14_olora_tail_results(bad_claim)
    assert analysis["status"] == "fail_fig14_olora_tail_result_analysis"
    assert any("must not claim paper-exact results" in failure for failure in analysis["failures"])


def test_fig14_reference_contract_and_lazy_exports_are_not_local_evidence():
    repro = _load_module()
    ref = repro.paper_reference_fig14_contract()

    assert ref["paper_reference_not_local_result"] is True
    assert ref["claims_paper_results"] is False
    assert ref["paper_reported_averages"] == {"lora_baseline": 56.3, "olora_tail": 58.3}

    lite_root = MODULE_PATH.parents[4]
    if str(lite_root) not in sys.path:
        sys.path.insert(0, str(lite_root))
    import megatron.lite.primitive.modules as modules

    contract = modules.build_fig14_olora_tail_contract()
    assert contract.to_dict()["status"] == "pass_fig14_olora_tail_contract"
    assert modules.FIG14_DATASET == repro.FIG14_DATASET
