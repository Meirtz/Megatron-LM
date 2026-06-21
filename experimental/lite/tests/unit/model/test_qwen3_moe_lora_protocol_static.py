# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from pathlib import Path


def test_qwen3_moe_protocol_exposes_runtime_lora_adapter_hooks():
    protocol_py = (
        Path(__file__).resolve().parents[3]
        / "megatron/lite/model/qwen3_moe/lite/protocol.py"
    )
    text = protocol_py.read_text()

    for marker in (
        '"export_lora_adapter_state"',
        '"save_lora_adapter"',
        '"load_lora_adapter_state"',
        '"load_lora_adapter"',
        "def export_lora_adapter_state(",
        "def save_lora_adapter(",
        "def load_lora_adapter_state(",
        "def load_lora_adapter(",
        "megatron.lite.model.qwen3_moe.lite.lora_adapter",
    ):
        assert marker in text
