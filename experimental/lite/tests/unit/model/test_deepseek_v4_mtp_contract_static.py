# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from pathlib import Path


def _lite_model_text() -> str:
    root = Path(__file__).resolve().parents[3]
    return (root / "megatron" / "lite" / "model" / "deepseek_v4" / "lite" / "model.py").read_text()


def _mhc_text() -> str:
    root = Path(__file__).resolve().parents[3]
    return (root / "megatron" / "lite" / "primitive" / "modules" / "attention" / "mhc.py").read_text()


def test_dsv4_mtp_contract_runs_inside_mtp_module_call():
    text = _lite_model_text()
    mtp_loop = text.split("for mtp_layer in self.mtp:", 1)[1].split("return outputs", 1)[0]

    assert "return_contract: bool = False" in text
    assert "return source, self.contract(source)" in text
    assert "return_contract=True" in mtp_loop
    assert "source, contracted = mtp_layer(" in mtp_loop
    assert "outputs.append(contracted)" in mtp_loop
    assert "outputs.append(mtp_layer.contract(source))" not in mtp_loop


def test_dsv4_mhc_head_uses_rms_norm_eps_separately_from_hc_eps():
    text = _mhc_text()

    assert "hc_eps: float, rms_norm_eps: float" in text
    assert "self.eps = hc_eps" in text
    assert "self.rms_norm_eps = rms_norm_eps" in text
    assert "+ self.rms_norm_eps" in text
    assert "F.linear(xf * rsqrt, self.hc_fn.float())" in text
