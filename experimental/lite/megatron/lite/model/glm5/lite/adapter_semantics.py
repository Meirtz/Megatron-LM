# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM5 LoRA adapter semantics for MLA/DSA/MTP-aware PEFT state.

The Scaling PEFT paper calls out GLM5-style stacks as an adapter-semantics
problem: MoE + MLA-style latent projections + Dynamic Sparse Attention (DSA)
+ Multi-Token Prediction (MTP) cannot be treated as a generic LoRA wrapper.
This module is the local, stdlib-only contract for GLM5 adapter keys, target
expansions, metadata fields, DSA exclusions, and MTP layer numbering. It does
not load weights, train adapters, run a GLM5 checkpoint smoke, or claim the
paper's GLM5/GLM5.1 empirical results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

PEFT_PREFIX = "base_model.model.model"
ADAPTER_META_FORMAT = "megatron.lite_glm5_lora_peft_v1"

ATTENTION_ADAPTER_TARGETS: tuple[str, ...] = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
)
MLP_ADAPTER_TARGETS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
ADAPTER_TARGET_MODULES: tuple[str, ...] = ATTENTION_ADAPTER_TARGETS + MLP_ADAPTER_TARGETS
DSA_NON_ADAPTER_TARGETS: tuple[str, ...] = (
    "indexer",
    "eh_proj",
    "eh_norm",
    "indexer_q_proj",
    "indexer_k_proj",
)

TARGET_MODULE_EXPANSIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "all-linear": ADAPTER_TARGET_MODULES,
        "all_linear": ADAPTER_TARGET_MODULES,
        "linear_qkv": ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"),
        "qkv": ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"),
        "q_a": ("q_a_proj",),
        "q_a_proj": ("q_a_proj",),
        "q_b": ("q_b_proj",),
        "q_b_proj": ("q_b_proj",),
        "kv_a": ("kv_a_proj_with_mqa",),
        "kv_a_proj_with_mqa": ("kv_a_proj_with_mqa",),
        "kv_b": ("kv_b_proj",),
        "kv_b_proj": ("kv_b_proj",),
        "linear_proj": ("o_proj",),
        "proj": ("o_proj",),
        "o_proj": ("o_proj",),
        "linear_fc1": ("gate_proj", "up_proj"),
        "fc1": ("gate_proj", "up_proj"),
        "gate_up": ("gate_proj", "up_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
        "gate_proj": ("gate_proj", "up_proj"),
        "up_proj": ("gate_proj", "up_proj"),
        "linear_fc2": ("down_proj",),
        "fc2": ("down_proj",),
        "down": ("down_proj",),
        "down_proj": ("down_proj",),
    }
)

MODEL_METADATA_FIELDS: tuple[str, ...] = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "num_experts",
    "n_shared_experts",
    "moe_intermediate_size",
    "num_nextn_predict_layers",
    "mtp_use_repeated_layer",
)
LORA_METADATA_FIELDS: tuple[str, ...] = (
    "rank",
    "alpha",
    "dropout",
    "use_rslora",
    "scaling_convention",
    "scale",
    "target_modules",
)
PARALLEL_METADATA_FIELDS: tuple[str, ...] = ("tp", "ep", "etp", "pp")

AdapterPathKind = Literal["self_attn", "mlp", "expert_mlp", "shared_experts"]
AdapterSuffix = Literal["lora_A.weight", "lora_B.weight"]


def _type_name(value: Any) -> str:
    return type(value).__name__


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {_type_name(value)}.")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {_type_name(value)}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return int(value)


def _positive_int(value: Any, *, name: str) -> int:
    value = _non_negative_int(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _normalize_suffix(value: Any) -> AdapterSuffix:
    value = _non_empty_string(value, name="suffix")
    if value not in {"lora_A.weight", "lora_B.weight"}:
        raise ValueError("suffix must be lora_A.weight or lora_B.weight.")
    return value  # type: ignore[return-value]


def dedupe_target_modules(targets: str | Sequence[str]) -> tuple[str, ...]:
    """Expand user-facing GLM5 LoRA target aliases into canonical modules."""

    if isinstance(targets, str):
        raw_targets = (targets,)
    elif isinstance(targets, Mapping):
        raise TypeError("target_modules must be a string or sequence of strings.")
    else:
        try:
            raw_targets = tuple(targets)
        except TypeError as exc:
            raise TypeError("target_modules must be a string or sequence of strings.") from exc
    out: list[str] = []
    for target in raw_targets:
        target = _non_empty_string(target, name="target_modules entry")
        modules = TARGET_MODULE_EXPANSIONS.get(target)
        if modules is None:
            supported = ", ".join(sorted(TARGET_MODULE_EXPANSIONS))
            raise ValueError(
                f"Unsupported GLM5 LoRA target module {target!r}. "
                f"Supported adapter targets are attention/MLP projection surfaces only: {supported}."
            )
        for module in modules:
            if module not in out:
                out.append(module)
    return tuple(out)


def mtp_layer_index(num_hidden_layers: int, mtp_idx: int) -> int:
    """Return the PEFT layer index for one MTP transformer layer."""

    return _positive_int(num_hidden_layers, name="num_hidden_layers") + _non_negative_int(
        mtp_idx, name="mtp_idx"
    )


@dataclass(frozen=True)
class Glm5AdapterTensorSpec:
    """Parsed GLM5 LoRA tensor key semantics."""

    layer_idx: int
    path_kind: AdapterPathKind
    module: str
    suffix: AdapterSuffix
    expert_idx: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_idx", _non_negative_int(self.layer_idx, name="layer_idx"))
        if self.path_kind not in {"self_attn", "mlp", "expert_mlp", "shared_experts"}:
            raise ValueError("path_kind must be self_attn, mlp, expert_mlp, or shared_experts.")
        module = _non_empty_string(self.module, name="module")
        valid = ATTENTION_ADAPTER_TARGETS if self.path_kind == "self_attn" else MLP_ADAPTER_TARGETS
        if module not in valid:
            raise ValueError(f"module {module!r} is not valid for {self.path_kind}.")
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "suffix", _normalize_suffix(self.suffix))
        if self.path_kind == "expert_mlp":
            object.__setattr__(self, "expert_idx", _non_negative_int(self.expert_idx, name="expert_idx"))
        elif self.expert_idx is not None:
            raise ValueError("expert_idx is only valid for expert_mlp paths.")

    def to_key(self) -> str:
        stem = f"{PEFT_PREFIX}.layers.{self.layer_idx}"
        if self.path_kind == "self_attn":
            return f"{stem}.self_attn.{self.module}.{self.suffix}"
        if self.path_kind == "mlp":
            return f"{stem}.mlp.{self.module}.{self.suffix}"
        if self.path_kind == "expert_mlp":
            return f"{stem}.mlp.experts.{self.expert_idx}.{self.module}.{self.suffix}"
        return f"{stem}.mlp.shared_experts.{self.module}.{self.suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "path_kind": self.path_kind,
            "module": self.module,
            "suffix": self.suffix,
            "expert_idx": self.expert_idx,
            "key": self.to_key(),
        }


def parse_adapter_tensor_key(key: str) -> Glm5AdapterTensorSpec:
    """Parse one PEFT LoRA tensor key and reject non-GLM5 adapter semantics."""

    key = _non_empty_string(key, name="key")
    suffix = None
    stem = None
    for candidate in ("lora_A.weight", "lora_B.weight"):
        marker = f".{candidate}"
        if key.endswith(marker):
            suffix = candidate
            stem = key[: -len(marker)]
            break
    if suffix is None or stem is None:
        raise ValueError("GLM5 LoRA tensor key must end with .lora_A.weight or .lora_B.weight.")
    prefix = f"{PEFT_PREFIX}.layers."
    if not stem.startswith(prefix):
        raise ValueError(f"GLM5 LoRA tensor key must start with {prefix!r}.")
    parts = stem[len(prefix) :].split(".")
    if not parts or not parts[0].isdigit():
        raise ValueError("GLM5 LoRA tensor key must include an integer layer index.")
    layer_idx = int(parts[0])
    if len(parts) == 3 and parts[1] == "self_attn":
        return Glm5AdapterTensorSpec(layer_idx, "self_attn", parts[2], suffix)
    if len(parts) == 3 and parts[1] == "mlp":
        return Glm5AdapterTensorSpec(layer_idx, "mlp", parts[2], suffix)
    if len(parts) == 5 and parts[1:3] == ["mlp", "experts"] and parts[3].isdigit():
        return Glm5AdapterTensorSpec(layer_idx, "expert_mlp", parts[4], suffix, int(parts[3]))
    if len(parts) == 4 and parts[1:3] == ["mlp", "shared_experts"]:
        return Glm5AdapterTensorSpec(layer_idx, "shared_experts", parts[3], suffix)
    raise ValueError(
        "GLM5 LoRA tensor key must target self_attn, dense mlp, mlp.experts, "
        "or mlp.shared_experts adapter projection surfaces."
    )


def is_supported_adapter_tensor_key(key: str) -> bool:
    try:
        parse_adapter_tensor_key(key)
    except (TypeError, ValueError):
        return False
    return True


def adapter_tensor_keys_for_layer(
    layer_idx: int,
    target_modules: str | Sequence[str],
) -> tuple[str, ...]:
    """Return dense attention/MLP LoRA keys for one main or MTP layer."""

    layer_idx = _non_negative_int(layer_idx, name="layer_idx")
    targets = dedupe_target_modules(target_modules)
    keys: list[str] = []
    for module in targets:
        path_kind: AdapterPathKind = "self_attn" if module in ATTENTION_ADAPTER_TARGETS else "mlp"
        for suffix in ("lora_A.weight", "lora_B.weight"):
            keys.append(Glm5AdapterTensorSpec(layer_idx, path_kind, module, suffix).to_key())
    return tuple(keys)


def adapter_tensor_keys_for_main_and_mtp_layers(
    *,
    main_layer_indices: Sequence[int],
    num_hidden_layers: int,
    num_mtp_layers: int,
    target_modules: str | Sequence[str],
) -> tuple[str, ...]:
    """Return dense GLM5 LoRA keys for main layers and MTP offset layers."""

    main = tuple(_non_negative_int(idx, name="main_layer_indices entry") for idx in main_layer_indices)
    num_hidden_layers = _positive_int(num_hidden_layers, name="num_hidden_layers")
    num_mtp_layers = _non_negative_int(num_mtp_layers, name="num_mtp_layers")
    keys: list[str] = []
    for layer_idx in main:
        keys.extend(adapter_tensor_keys_for_layer(layer_idx, target_modules))
    for mtp_idx in range(num_mtp_layers):
        keys.extend(adapter_tensor_keys_for_layer(mtp_layer_index(num_hidden_layers, mtp_idx), target_modules))
    return tuple(keys)


def glm5_lora_semantics_contract() -> dict[str, Any]:
    """Return the local GLM5 adapter-semantics contract, not paper results."""

    sample_mtp_layer = mtp_layer_index(2, 0)
    sample_keys = adapter_tensor_keys_for_main_and_mtp_layers(
        main_layer_indices=(0,),
        num_hidden_layers=2,
        num_mtp_layers=1,
        target_modules="all-linear",
    )
    invariants = {
        "format_is_glm5_lora_peft_v1": ADAPTER_META_FORMAT == "megatron.lite_glm5_lora_peft_v1",
        "mla_dsa_attention_targets_present": set(ATTENTION_ADAPTER_TARGETS)
        == {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"},
        "mlp_targets_present": set(MLP_ADAPTER_TARGETS) == {"gate_proj", "up_proj", "down_proj"},
        "dsa_indexer_targets_excluded": not any(
            target in ADAPTER_TARGET_MODULES for target in DSA_NON_ADAPTER_TARGETS
        ),
        "all_linear_expands_to_all_targets": dedupe_target_modules("all-linear")
        == ADAPTER_TARGET_MODULES,
        "mtp_layer_index_offsets_after_main_layers": sample_mtp_layer == 2,
        "sample_main_and_mtp_key_count": len(sample_keys) == 32,
        "sample_mtp_key_uses_offset_layer": any(".layers.2.self_attn.q_a_proj." in key for key in sample_keys),
        "model_metadata_tracks_mtp": "num_nextn_predict_layers" in MODEL_METADATA_FIELDS
        and "mtp_use_repeated_layer" in MODEL_METADATA_FIELDS,
        "lora_metadata_tracks_rslora": "use_rslora" in LORA_METADATA_FIELDS
        and "scaling_convention" in LORA_METADATA_FIELDS,
        "paper_results_not_claimed": True,
        "standard_library_only": True,
    }
    return {
        "status": "pass_glm5_lora_semantics_contract"
        if all(value is True for value in invariants.values())
        else "fail_glm5_lora_semantics_contract",
        "paper_exact": False,
        "claims_paper_results": False,
        "claims_glm5_training_smoke": False,
        "claims_weight_materialization": False,
        "adapter_meta_format": ADAPTER_META_FORMAT,
        "adapter_targets": list(ADAPTER_TARGET_MODULES),
        "dsa_non_adapter_targets": list(DSA_NON_ADAPTER_TARGETS),
        "model_metadata_fields": list(MODEL_METADATA_FIELDS),
        "lora_metadata_fields": list(LORA_METADATA_FIELDS),
        "parallel_metadata_fields": list(PARALLEL_METADATA_FIELDS),
        "sample_keys": list(sample_keys),
        "invariants": invariants,
        "missing_for_paper_exact": [
            "materialized GLM5 weights",
            "real adapter import/export over checkpoint tensors",
            "real load/training smoke with MTP enabled",
            "R3 smoke over GLM5 sparse route",
        ],
    }


def validate_glm5_lora_semantics_contract() -> None:
    payload = glm5_lora_semantics_contract()
    if payload["status"] != "pass_glm5_lora_semantics_contract":
        raise ValueError(f"GLM5 LoRA semantics contract failed: {payload['invariants']}")


__all__ = [
    "ADAPTER_META_FORMAT",
    "ADAPTER_TARGET_MODULES",
    "ATTENTION_ADAPTER_TARGETS",
    "DSA_NON_ADAPTER_TARGETS",
    "Glm5AdapterTensorSpec",
    "LORA_METADATA_FIELDS",
    "MLP_ADAPTER_TARGETS",
    "MODEL_METADATA_FIELDS",
    "PARALLEL_METADATA_FIELDS",
    "PEFT_PREFIX",
    "TARGET_MODULE_EXPANSIONS",
    "adapter_tensor_keys_for_layer",
    "adapter_tensor_keys_for_main_and_mtp_layers",
    "dedupe_target_modules",
    "glm5_lora_semantics_contract",
    "is_supported_adapter_tensor_key",
    "mtp_layer_index",
    "parse_adapter_tensor_key",
    "validate_glm5_lora_semantics_contract",
]
