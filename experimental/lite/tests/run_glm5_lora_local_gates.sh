#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_PYTHON_BIN="${TORCH_PYTHON_BIN:-${PYTHON_BIN}}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}:${ROOT}/experimental/lite:${ROOT}/experimental/lite/examples/verl:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m py_compile \
  experimental/lite/megatron/lite/model/glm5/config.py \
  experimental/lite/megatron/lite/model/qwen2/config.py \
  experimental/lite/megatron/lite/model/qwen2/lite/__init__.py \
  experimental/lite/megatron/lite/model/qwen2/lite/checkpoint.py \
  experimental/lite/megatron/lite/model/qwen2/lite/lora_adapter.py \
  experimental/lite/megatron/lite/model/qwen2/lite/model.py \
  experimental/lite/megatron/lite/model/qwen2/lite/protocol.py \
  experimental/lite/megatron/lite/model/glm5/lite/model.py \
  experimental/lite/megatron/lite/model/glm5/lite/protocol.py \
  experimental/lite/megatron/lite/model/glm5/lite/lora_adapter.py \
  experimental/lite/megatron/lite/model/protocol_utils.py \
  experimental/lite/megatron/lite/model/qwen3_moe/lite/model.py \
  experimental/lite/megatron/lite/model/qwen3_moe/lite/protocol.py \
  experimental/lite/megatron/lite/model/qwen3_moe/lite/lora_adapter.py \
  experimental/lite/megatron/lite/primitive/modules/attention/dsa.py \
  experimental/lite/megatron/lite/primitive/modules/router_replay.py \
  experimental/lite/megatron/lite/primitive/modules/router.py \
  experimental/lite/megatron/lite/primitive/utils/moe.py \
  experimental/lite/megatron/lite/primitive/modules/experts.py \
  experimental/lite/megatron/lite/primitive/modules/gqa.py \
  experimental/lite/megatron/lite/primitive/modules/lora.py \
  experimental/lite/megatron/lite/primitive/modules/adapter_rank.py \
  experimental/lite/megatron/lite/primitive/modules/adapter_reproduction.py \
  experimental/lite/examples/verl/verl_mlite/engine/mlite_engine.py \
  experimental/lite/tests/unit/primitive/test_peft_dsa_router_replay_static.py \
  experimental/lite/tests/unit/primitive/test_router_replay_runtime.py \
  experimental/lite/tests/unit/primitive/test_module_primitives_independent_unit.py \
  experimental/lite/tests/unit/primitive/test_adapter_rank_unit.py \
  experimental/lite/tests/unit/primitive/test_adapter_reproduction_unit.py \
  experimental/lite/tests/unit/model/test_qwen2_dense_static.py \
  experimental/lite/tests/unit/model/test_glm5_lite_static.py \
  experimental/lite/tests/unit/model/test_glm5_lora_adapter_runtime.py \
  experimental/lite/tests/unit/runtime/test_runtime_backend_unit.py \
  experimental/lite/tests/unit/verl/test_mlite_engine_checkpoint.py \
  experimental/lite/tests/unit/verl/test_mlite_engine_config.py \
  experimental/lite/tests/unit/verl/test_mlite_engine_r3_static.py

"${PYTHON_BIN}" - <<'PY'
import importlib.util
from pathlib import Path

root = Path.cwd() / "experimental" / "lite"
test_groups = {
    "tests/unit/primitive/test_peft_dsa_router_replay_static.py": [
        "test_lora_config_tracks_rslora_scaling_and_peft_aliases",
        "test_dsa_indexshare_static_contract",
        "test_glm5_lora_static_contract",
        "test_router_replay_static_contract",
    ],
    "tests/unit/model/test_glm5_lite_static.py": [
        "test_glm5_config_reads_hf_architecture_fields",
        "test_glm5_config_honors_mtp_index_share_flag_when_indexer_types_end_at_main_layers",
        "test_glm5_config_tracks_dsa_indexshare_pattern",
        "test_glm5_config_derives_dsa_indexshare_pattern_without_hf_indexer_types",
        "test_glm5_hf_reference_weight_map_matches_dsa_indexshare_and_mtp_layout",
        "test_glm52_reference_config_pins_sparse_arch_low_rank_and_peft_targets",
        "test_glm5_reference_inputs_pin_mindlab_commit_and_peft_boundaries",
    ],
    "tests/unit/model/test_qwen2_dense_static.py": [
        "test_qwen2_config_reads_exact_fig14_target_hf_config",
        "test_qwen2_registry_resolves_exact_target_lite_runtime_name",
    ],
}

for relpath, names in test_groups.items():
    path = root / relpath
    spec = importlib.util.spec_from_file_location(relpath.replace("/", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in names:
        getattr(mod, name)()
        print(f"PASS {relpath}:{name}")

print("static/reference GLM5 LoRA gates passed")
PY

"${TORCH_PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path


class Raises:
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"did not raise {self.exc_type}")
        if not issubclass(exc_type, self.exc_type):
            return False
        if self.match is not None and re.search(self.match, str(exc)) is None:
            raise AssertionError(f"exception {exc!r} did not match {self.match!r}")
        return True


class MonkeyPatch:
    def __init__(self):
        self._undo = []
        self._env_undo = []

    def setattr(self, obj, name, value=None):
        if isinstance(obj, str):
            module_name, attr = obj.rsplit(".", 1)
            module = __import__(module_name, fromlist=[attr])
            old = getattr(module, attr)
            self._undo.append((module, attr, old))
            setattr(module, attr, name)
            return
        old = getattr(obj, name)
        self._undo.append((obj, name, old))
        setattr(obj, name, value)

    def setenv(self, name, value):
        existed = name in os.environ
        old = os.environ.get(name)
        self._env_undo.append((name, existed, old))
        os.environ[name] = value

    def undo(self):
        while self._env_undo:
            name, existed, old = self._env_undo.pop()
            if existed:
                assert old is not None
                os.environ[name] = old
            else:
                os.environ.pop(name, None)
        while self._undo:
            obj, name, old = self._undo.pop()
            setattr(obj, name, old)


def importorskip(name):
    return importlib.import_module(name)


pytest_mark = types.SimpleNamespace(
    mlite=object(),
    parametrize=lambda *args, **kwargs: (lambda fn: fn),
)
pytest = types.SimpleNamespace(
    fixture=lambda *args, **kwargs: (lambda fn: fn),
    importorskip=importorskip,
    raises=lambda exc_type, match=None: Raises(exc_type, match),
    mark=pytest_mark,
)
sys.modules["pytest"] = pytest

root = Path.cwd() / "experimental" / "lite"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

primitive_path = root / "tests/unit/primitive/test_module_primitives_independent_unit.py"
spec = importlib.util.spec_from_file_location(
    "test_module_primitives_independent_unit_shim", primitive_path
)
primitive_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primitive_mod)
primitive_mod.test_lora_config_aliases_and_trainable_param_accounting()
print("PASS primitive:test_lora_config_aliases_and_trainable_param_accounting")
primitive_mod.test_rslora_sqrt_rank_alpha_keeps_scale_constant_across_ranks()
print("PASS primitive:test_rslora_sqrt_rank_alpha_keeps_scale_constant_across_ranks")

adapter_rank_path = root / "tests/unit/primitive/test_adapter_rank_unit.py"
spec = importlib.util.spec_from_file_location("test_adapter_rank_unit_torch_shim", adapter_rank_path)
adapter_rank_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_rank_mod)
for name in (
    "test_rslora_alpha_rules_encode_eq2_rank_dependence_without_results",
    "test_rank_regime_contract_builds_paper_216_shape_as_pending_work",
    "test_rank_regime_contract_rejects_bad_shapes_and_metric_claims",
    "test_rank_sweep_contract_status_rejects_false_invariants",
    "test_rank_regime_result_analyzer_checks_216_run_paper_pattern_without_launching",
    "test_rank_regime_result_analyzer_rejects_missing_or_unconfirmed_runs",
    "test_rslora_lr_transfer_analyzer_checks_reusable_band_pattern_without_launching",
    "test_rslora_lr_transfer_analyzer_rejects_missing_unconfirmed_or_mismatched_runs",
    "test_rank1_olora_tail_analyzer_checks_fig15_stability_without_launching",
    "test_rank1_olora_tail_analyzer_rejects_missing_unconfirmed_or_bad_claims",
    "test_rank_reference_contract_is_not_local_evidence",
):
    getattr(adapter_rank_mod, name)()
    print(f"PASS adapter_rank:{name}")

adapter_reproduction_path = root / "tests/unit/primitive/test_adapter_reproduction_unit.py"
spec = importlib.util.spec_from_file_location(
    "test_adapter_reproduction_unit_torch_shim",
    adapter_reproduction_path,
)
adapter_reproduction_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_reproduction_mod)
for name in (
    "test_fig14_contract_is_local_no_launch_no_claim",
    "test_fig14_result_analyzer_checks_paper_shape_without_launching",
    "test_fig14_result_analyzer_rejects_missing_unconfirmed_or_spoofed_results",
    "test_fig14_reference_contract_and_lazy_exports_are_not_local_evidence",
):
    getattr(adapter_reproduction_mod, name)()
    print(f"PASS adapter_reproduction:{name}")

qwen2_path = root / "tests/unit/model/test_qwen2_dense_static.py"
spec = importlib.util.spec_from_file_location("test_qwen2_dense_static_torch_shim", qwen2_path)
qwen2_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen2_mod)
qwen2_mod.test_qwen2_protocol_builds_tiny_lora_forward_backward()
print("PASS qwen2:test_qwen2_protocol_builds_tiny_lora_forward_backward")
qwen2_mod.test_qwen2_protocol_lora_init_runs_as_post_load_hook()
print("PASS qwen2:test_qwen2_protocol_lora_init_runs_as_post_load_hook")
qwen2_mod.test_qwen2_olora_tail_factors_use_tail_subspace_without_singular_values()
print("PASS qwen2:test_qwen2_olora_tail_factors_use_tail_subspace_without_singular_values")
qwen2_mod.test_qwen2_protocol_lora_init_requires_enabled_lora()
print("PASS qwen2:test_qwen2_protocol_lora_init_requires_enabled_lora")
with tempfile.TemporaryDirectory() as tmpdir:
    qwen2_mod.test_qwen2_lora_adapter_protocol_round_trip(Path(tmpdir))
print("PASS qwen2:test_qwen2_lora_adapter_protocol_round_trip")
with tempfile.TemporaryDirectory() as tmpdir:
    qwen2_mod.test_qwen2_hf_checkpoint_state_dict_round_trip(Path(tmpdir))
print("PASS qwen2:test_qwen2_hf_checkpoint_state_dict_round_trip")

adapter_path = root / "tests/unit/model/test_glm5_lora_adapter_runtime.py"
spec = importlib.util.spec_from_file_location("test_glm5_lora_adapter_runtime_shim", adapter_path)
adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_mod)

static_path = root / "tests/unit/model/test_glm5_lite_static.py"
spec = importlib.util.spec_from_file_location("test_glm5_lite_static_torch_shim", static_path)
static_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_mod)
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_dsa_indexer_topk_torch_backend_masks_invalid_slots(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_dsa_indexer_topk_torch_backend_masks_invalid_slots")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_dsa_build_flat_topk_uses_torch_compact_backend(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_dsa_build_flat_topk_uses_torch_compact_backend")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_dsa_sparse_attn_torch_backend_matches_reference(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_dsa_sparse_attn_torch_backend_matches_reference")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_dsa_training_with_indexer_loss_uses_fused_kernel(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_dsa_training_with_indexer_loss_uses_fused_kernel")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_dsa_training_zero_loss_uses_topk_sparse_attention(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_dsa_training_zero_loss_uses_topk_sparse_attention")
static_mod.test_glm5_dsa_indexshare_loss_fails_explicitly_before_topk_kernel()
print("PASS static_torch:test_glm5_dsa_indexshare_loss_fails_explicitly_before_topk_kernel")
static_mod.test_glm5_dsa_indexshare_training_zero_loss_reuses_full_layer_topk()
print("PASS static_torch:test_glm5_dsa_indexshare_training_zero_loss_reuses_full_layer_topk")
static_mod.test_glm5_dsa_indexshare_shared_layer_requires_source_topk_in_holder()
print("PASS static_torch:test_glm5_dsa_indexshare_shared_layer_requires_source_topk_in_holder")
static_mod.test_glm5_dsa_indexshare_pipeline_split_includes_mtp_layers()
print("PASS static_torch:test_glm5_dsa_indexshare_pipeline_split_includes_mtp_layers")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_protocol_build_model_freezes_base_params_and_reports_lora_stats(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_protocol_build_model_freezes_base_params_and_reports_lora_stats")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_protocol_lora_init_runs_as_post_load_hook(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_protocol_lora_init_runs_as_post_load_hook")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_protocol_lora_init_accepts_mapping_config(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_protocol_lora_init_accepts_mapping_config")
static_mod.test_glm5_protocol_lora_init_requires_enabled_lora()
print("PASS static_torch:test_glm5_protocol_lora_init_requires_enabled_lora")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_qwen3_moe_lora_init_runs_as_post_load_hook(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_qwen3_moe_lora_init_runs_as_post_load_hook")
static_mod.test_qwen3_moe_lora_init_rejects_unsupported_values()
print("PASS static_torch:test_qwen3_moe_lora_init_rejects_unsupported_values")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_lite_tiny_cpu_forward_backward(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_lite_tiny_cpu_forward_backward")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_lora_tiny_cpu_forward_backward_freezes_base_params(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_lora_tiny_cpu_forward_backward_freezes_base_params")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_router_replay_tiny_cpu_record_replay_identity(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_router_replay_tiny_cpu_record_replay_identity")
static_mod.test_glm5_router_replay_preserves_duplicate_topk_slots_for_padding()
print("PASS static_torch:test_glm5_router_replay_preserves_duplicate_topk_slots_for_padding")
static_mod.test_glm5_router_replay_instances_append_mtp_routers_after_main_layers()
print("PASS static_torch:test_glm5_router_replay_instances_append_mtp_routers_after_main_layers")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_glm5_mtp_router_replay_tiny_cpu_record_replay_identity(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_glm5_mtp_router_replay_tiny_cpu_record_replay_identity")
monkeypatch = MonkeyPatch()
try:
    static_mod.test_qwen3_moe_router_replay_tiny_cpu_record_replay_identity(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS static_torch:test_qwen3_moe_router_replay_tiny_cpu_record_replay_identity")
static_mod.test_qwen3_moe_router_replay_instances_append_mtp_routers_after_main_layers()
print("PASS static_torch:test_qwen3_moe_router_replay_instances_append_mtp_routers_after_main_layers")

plain_adapter_tests = [
    "test_lora_adapter_helpers_import_without_transformer_engine",
    "test_lora_adapter_state_guard_rejects_base_weight_keys",
    "test_glm5_lora_adapter_export_materializes_full_tensor_before_local_shard",
    "test_qwen3_lora_adapter_export_materializes_full_tensor_before_local_shard",
    "test_glm5_expert_lora_adapter_export_materializes_full_tensor_before_local_shard",
    "test_qwen3_expert_lora_adapter_export_materializes_full_tensor_before_local_shard",
    "test_glm5_load_lora_adapter_state_copies_into_dtensor_lora_params_without_te",
    "test_qwen3_load_lora_adapter_state_copies_into_dtensor_lora_params_without_te",
    "test_glm5_load_per_expert_lora_state_copies_into_dtensor_grouped_params_without_te",
    "test_qwen3_load_per_expert_lora_state_copies_into_dtensor_grouped_params_without_te",
    "test_olora_tail_factors_use_tail_subspace_without_singular_values",
    "test_qwen3_initialize_lora_olora_tail_supports_attention_and_single_local_expert",
    "test_qwen3_initialize_lora_olora_tail_skips_multi_local_shared_experts",
    "test_initialize_lora_olora_tail_marks_modules_and_blocks_double_init",
    "test_initialize_lora_olora_tail_reaches_mtp_transformer_layers",
    "test_initialize_lora_olora_tail_supports_single_local_routed_expert",
    "test_initialize_lora_olora_tail_supports_per_expert_grouped_lora",
    "test_initialize_lora_olora_tail_copies_into_dtensor_linear_lora",
    "test_initialize_lora_olora_tail_copies_into_dtensor_grouped_lora",
    "test_initialize_lora_olora_tail_skips_multi_local_shared_routed_experts",
    "test_initialize_lora_olora_tail_fsdp_style_wrapper_without_te",
    "test_qwen3_load_lora_adapter_state_reports_missing_and_unexpected_keys_without_te",
    "test_qwen3_lora_adapter_rejects_unsupported_parallel_scopes_without_te",
    "test_glm5_shared_routed_expert_lora_round_trips_without_te",
    "test_glm5_load_lora_adapter_state_reports_missing_and_unexpected_keys_without_te",
    "test_glm5_lora_adapter_rejects_unsupported_parallel_scopes_without_te",
]
tmp_path_adapter_tests = [
    "test_qwen3_lora_adapter_rejects_non_adapter_only_configs_without_te",
    "test_qwen3_lora_adapter_rejects_unexported_target_modules_without_te",
    "test_qwen3_lora_adapter_save_load_validates_metadata_without_te",
    "test_qwen3_lora_adapter_protocol_wrappers_round_trip_without_te",
    "test_qwen3_lora_adapter_fsdp_style_wrappers_round_trip_without_te",
    "test_qwen3_load_lora_adapter_validates_expected_config_without_adapter_config",
    "test_qwen3_per_expert_grouped_lora_round_trips_without_te",
    "test_glm5_protocol_initialize_lora_olora_tail_without_te",
    "test_glm5_shared_routed_expert_lora_save_load_metadata_without_te",
    "test_glm5_lora_adapter_protocol_wrappers_round_trip_without_te",
    "test_glm5_lora_adapter_fsdp_style_wrappers_round_trip_without_te",
    "test_glm5_shared_routed_expert_rslora_round_trips_metadata_without_te",
    "test_glm5_load_lora_adapter_validates_metadata_without_te",
    "test_glm5_load_lora_adapter_rejects_config_mismatches_without_te",
    "test_glm5_load_lora_adapter_rejects_config_metadata_disagreement_without_te",
    "test_glm5_load_lora_adapter_rejects_non_object_json_sidecars_without_te",
    "test_glm5_load_lora_adapter_validates_expected_config_without_adapter_config",
    "test_glm5_load_lora_adapter_rejects_non_adapter_artifact_keys_without_te",
    "test_glm5_lora_adapter_rejects_unsupported_target_modules_without_te",
    "test_glm5_lora_adapter_rejects_unexported_target_modules_without_te",
    "test_glm5_lora_adapter_round_trips_all_linear_with_mtp_without_te",
]

for name in plain_adapter_tests:
    getattr(adapter_mod, name)()
    print(f"PASS adapter:{name}")
monkeypatch = MonkeyPatch()
try:
    adapter_mod.test_olora_tail_factors_reject_non_finite_svd_factors(monkeypatch)
finally:
    monkeypatch.undo()
print("PASS adapter:test_olora_tail_factors_reject_non_finite_svd_factors")
for name in tmp_path_adapter_tests:
    with tempfile.TemporaryDirectory() as tmpdir:
        getattr(adapter_mod, name)(Path(tmpdir))
    print(f"PASS adapter:{name}")

router_path = root / "tests/unit/primitive/test_router_replay_runtime.py"
spec = importlib.util.spec_from_file_location("test_router_replay_runtime_shim", router_path)
router_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router_mod)

for name in [
    "test_router_replay_compact_dtype_matches_roll_reference_bounds",
    "test_padded_routed_experts_to_list_compacts_roll_style_batch",
    "test_router_replay_trace_contract_builds_audit_safe_extras",
    "test_router_replay_trace_contract_rejects_unsafe_or_inconsistent_metadata",
    "test_router_replay_records_and_replays_forward_and_backward_indices",
    "test_router_replay_accepts_string_actions_and_rejects_bad_actions",
    "test_router_replay_record_rejects_bad_recorded_indices",
    "test_router_replay_snapshots_record_and_replay_tensors",
    "test_router_replay_global_helpers_validate_and_snapshot_data",
    "test_router_replay_rejects_missing_bad_shape_and_out_of_range_replay_data",
    "test_router_replay_context_sets_data_actions_and_cleans_up",
    "test_router_replay_context_rejects_bad_actions_and_conflicting_data",
    "test_router_replay_context_handles_multi_router_list_and_tensor_layouts",
    "test_router_replay_context_rejects_bad_per_router_entry_shape_before_pack",
    "test_router_replay_rejects_mismatched_per_router_token_counts_before_pack",
    "test_router_replay_context_accepts_padded_batch_routed_experts",
    "test_router_replay_context_explicit_backward_replay_consumes_saved_indices",
    "test_router_replay_context_pads_true_token_replay_data_to_thd_layout",
    "test_router_replay_context_splits_padded_replay_data_to_cp_local_layout",
    "test_router_replay_context_respects_explicit_thd_replay_layout",
    "test_router_replay_context_uses_unwrapped_model_parallel_state_for_thd_layout",
    "test_router_replay_checkpoint_recompute_switches_to_backward_replay",
]:
    getattr(router_mod, name)()
    print(f"PASS router:{name}")

runtime_path = root / "tests/unit/runtime/test_runtime_backend_unit.py"
spec = importlib.util.spec_from_file_location("test_runtime_backend_unit_shim", runtime_path)
runtime_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime_mod
spec.loader.exec_module(runtime_mod)

with tempfile.TemporaryDirectory() as tmpdir:
    runtime_mod.test_mlite_runtime_forwards_lora_adapter_helpers_to_protocol(Path(tmpdir))
print("PASS runtime:test_mlite_runtime_forwards_lora_adapter_helpers_to_protocol")
runtime_mod.test_mlite_runtime_lora_adapter_helpers_require_protocol_context()
print("PASS runtime:test_mlite_runtime_lora_adapter_helpers_require_protocol_context")
for name in [
    "test_verl_sft_script_maps_lora_env_to_impl_cfg",
    "test_verl_sft_script_maps_lora_adapter_checkpoint_env",
    "test_verl_grpo_script_maps_lora_env_to_actor_impl_cfg",
    "test_verl_grpo_script_maps_lora_adapter_checkpoint_env",
    "test_verl_grpo_script_maps_r3_router_replay_env_to_rollout_and_mlite",
    "test_verl_grpo_script_rejects_r3_without_sglang",
    "test_verl_grpo_script_rejects_r2_router_replay_mode",
    "test_verl_worker_reads_mlite_engine_router_replay_mode_static",
    "test_verl_worker_router_replay_mode_helper_resolves_mlite_and_legacy_configs",
    "test_verl_worker_routing_replay_decorator_assigns_flag_only_when_enabled",
    "test_verl_mlite_engine_honors_disabled_routing_replay_flag_static",
    "test_verl_mlite_engine_routing_replay_helpers_honor_disabled_flag",
]:
    with tempfile.TemporaryDirectory() as tmpdir:
        getattr(runtime_mod, name)(Path(tmpdir))
    print(f"PASS runtime:{name}")

r3_static_path = root / "tests/unit/verl/test_mlite_engine_r3_static.py"
spec = importlib.util.spec_from_file_location("test_mlite_engine_r3_static_shim", r3_static_path)
r3_static_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r3_static_mod
spec.loader.exec_module(r3_static_mod)
for name in [
    "test_mlite_engine_r3_replay_requires_routed_experts_unless_recording",
    "test_mlite_engine_r3_config_guard_rejects_unsupported_layouts",
    "test_mlite_engine_r3_replay_flags_use_config_boolean_parsing",
    "test_mlite_engine_r3_trace_metadata_reaches_packed_batch_extras",
    "test_mlite_engine_r3_trace_metadata_conflicts_fail_before_consuming_payload",
    "test_verl_worker_r3_symmetry_helpers_execute_without_torch_import",
    "test_verl_worker_r3_symmetry_rejects_sequence_packing_without_torch_import",
    "test_verl_worker_rejects_r2_and_unknown_router_replay_modes_without_torch_import",
    "test_verl_worker_routing_replay_decorator_assigns_flag_without_torch_import",
    "test_verl_worker_r3_disabled_roles_clear_trace_conflict_side_channel_static",
    "test_verl_worker_r3_role_endpoints_bind_expected_replay_flags_static",
    "test_verl_agent_loop_r3_postprocess_writes_trace_metadata_static",
    "test_verl_fully_async_r3_partial_rollout_preserves_trace_extra_fields_static",
    "test_roll_r3_real_smoke_events_rejects_trace_metadata_conflicts_static",
    "test_roll_r3_real_smoke_extractor_mtp_router_count_static",
    "test_roll_r3_real_smoke_extractor_expected_router_count_source_of_truth_static",
    "test_roll_r3_real_smoke_events_bundle_router_count_source_of_truth_static",
    "test_roll_r3_real_smoke_training_router_count_matches_rollout_static",
    "test_roll_r3_real_smoke_bundle_collects_with_bundle_router_count_source_of_truth_static",
    "test_verl_ppo_trainer_r3_standard_rollout_writes_missing_trace_metadata_static",
    "test_verl_ppo_trainer_r3_segment_trace_metadata_static",
    "test_verl_ppo_trainer_r3_metadata_without_payload_fails_static",
    "test_verl_ppo_trainer_r3_logprob_outputs_do_not_override_rollout_trace_static",
    "test_verl_ppo_trainer_r3_auxiliary_outputs_do_not_override_rollout_trace_static",
    "test_verl_ppo_trainer_r3_post_reward_transforms_preserve_rollout_trace_static",
    "test_verl_ppo_trainer_r3_reward_extra_infos_do_not_override_rollout_trace_static",
    "test_verl_separation_trainer_r3_preserves_rollout_trace_static",
    "test_verl_rollout_correction_bypass_mode_preserves_r3_trace_static",
    "test_verl_sglang_rollout_r3_capability_contract_static",
    "test_verl_sglang_rollout_r3_shape_helper_executes_without_sglang_import",
    "test_verl_padding_r3_marks_unpadded_routed_experts_as_true_tokens_static",
    "test_verl_vllm_rollout_rejects_r3_router_replay_static",
]:
    getattr(r3_static_mod, name)()
    print(f"PASS r3_static:{name}")


def _stub_module(name: str, *, package: bool = False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    sys.modules[name] = module
    return module


def _install_verl_checkpoint_stubs():
    tensordict_module = _stub_module("tensordict")

    class TensorDict(dict):
        pass

    tensordict_module.TensorDict = TensorDict

    _stub_module("verl", package=True)
    _stub_module("verl.trainer", package=True)
    trainer_config = _stub_module("verl.trainer.config")
    trainer_config.CheckpointConfig = dict
    _stub_module("verl.utils", package=True)
    _stub_module("verl.utils.tensordict_utils")
    device_module = _stub_module("verl.utils.device")
    device_module.get_device_id = lambda: "cpu"
    device_module.get_device_name = lambda: "cpu"
    _stub_module("verl.workers", package=True)
    workers_config = _stub_module("verl.workers.config")

    class HFModelConfig:
        pass

    class OptimizerConfig:
        pass

    workers_config.HFModelConfig = HFModelConfig
    workers_config.OptimizerConfig = OptimizerConfig
    workers_config_engine = _stub_module("verl.workers.config.engine")

    @dataclass
    class EngineConfig:
        param_offload: bool = False
        optimizer_offload: bool = False
        grad_offload: bool = False
        use_fused_kernels: bool = False
        full_determinism: bool = False
        forward_only: bool = False

        def __post_init__(self):
            pass

    workers_config_engine.EngineConfig = EngineConfig
    _stub_module("verl.workers.engine", package=True)
    engine_base = _stub_module("verl.workers.engine.base")

    class BaseEngine:
        def __init__(self):
            pass

        def to(self, **kwargs):
            pass

    class BaseEngineCtx:
        def __init__(self, **kwargs):
            self.engine = kwargs.get("engine")
            self.mode = kwargs.get("mode")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class EngineRegistry:
        @staticmethod
        def register(**kwargs):
            return lambda cls: cls

    engine_base.BaseEngine = BaseEngine
    engine_base.BaseEngineCtx = BaseEngineCtx
    engine_base.EngineRegistry = EngineRegistry
    engine_utils = _stub_module("verl.workers.engine.utils")
    engine_utils.postprocess_batch_func = lambda value: value
    engine_utils.prepare_micro_batches = lambda *args, **kwargs: []


_install_verl_checkpoint_stubs()
checkpoint_path = root / "tests/unit/verl/test_mlite_engine_checkpoint.py"
spec = importlib.util.spec_from_file_location("test_mlite_engine_checkpoint_shim", checkpoint_path)
checkpoint_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checkpoint_mod
spec.loader.exec_module(checkpoint_mod)

checkpoint_mod.test_checkpoint_content_set_parses_exact_keys_and_rejects_bad_values()
print("PASS verl_checkpoint:test_checkpoint_content_set_parses_exact_keys_and_rejects_bad_values")
checkpoint_mod.test_checkpoint_bool_parses_strings_and_rejects_bad_values()
print("PASS verl_checkpoint:test_checkpoint_bool_parses_strings_and_rejects_bad_values")

for name in [
    "test_save_checkpoint_uses_exact_content_keys_not_substrings",
    "test_save_checkpoint_string_false_does_not_enable_lora_adapter",
    "test_save_checkpoint_accepts_bracketed_lora_adapter_contents_string",
    "test_save_checkpoint_string_true_enables_lora_adapter",
    "test_lora_adapter_checkpoint_dir_name_stays_inside_checkpoint",
    "test_save_checkpoint_can_write_lora_adapter_sidecar_without_full_checkpoint",
    "test_save_checkpoint_lora_adapter_passes_init_and_user_kwargs",
    "test_save_checkpoint_lora_adapter_accepts_mapping_kwargs",
    "test_save_checkpoint_lora_adapter_rejects_non_mapping_kwargs_and_metadata",
    "test_save_checkpoint_lora_adapter_validation_runs_before_full_checkpoint_side_effects",
    "test_save_checkpoint_lora_adapter_only_failure_cleans_new_checkpoint_dir",
    "test_save_checkpoint_lora_adapter_only_failure_preserves_existing_checkpoint_dir",
    "test_load_checkpoint_skips_when_contents_exclude_all_mlite_state",
    "test_load_checkpoint_can_restore_lora_adapter_without_full_checkpoint",
    "test_load_checkpoint_string_false_does_not_enable_lora_adapter",
    "test_load_checkpoint_string_true_enables_lora_adapter",
    "test_load_checkpoint_lora_adapter_rejects_non_mapping_kwargs_and_metadata",
    "test_load_checkpoint_lora_adapter_validation_runs_before_full_checkpoint_side_effects",
]:
    monkeypatch = MonkeyPatch()
    try:
        import verl_mlite.engine.mlite_engine as mlite_engine_module

        monkeypatch.setattr(mlite_engine_module.dist, "is_initialized", lambda: False)
        with tempfile.TemporaryDirectory() as tmpdir:
            getattr(checkpoint_mod, name)(Path(tmpdir), monkeypatch)
    finally:
        monkeypatch.undo()
    print(f"PASS verl_checkpoint:{name}")

print("CPU torch GLM5 LoRA gates passed")
PY
