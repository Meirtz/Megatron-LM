"""Static adaptation checks for PEFT, GLM5 DSA IndexShare, and R3 Router Replay."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "megatron" / "lite"


def _read(*parts: str) -> str:
    return (ROOT / Path(*parts)).read_text()


def test_lora_config_tracks_rslora_scaling_and_peft_aliases():
    lora_text = _read("primitive", "modules", "lora.py")
    qwen_adapter_text = _read("model", "qwen3_moe", "lite", "lora_adapter.py")
    qwen_protocol_text = _read("model", "qwen3_moe", "lite", "protocol.py")
    gqa_text = _read("primitive", "modules", "gqa.py")
    experts_text = _read("primitive", "modules", "experts.py")

    assert "use_rslora: bool = False" in lora_text
    assert "def lora_scale" in lora_text
    assert "math.sqrt(float(rank)) if use_rslora else float(rank)" in lora_text
    assert '"all_linear": "all-linear"' in lora_text
    assert '"all-linear": _DEFAULT_TARGET_MODULES' in lora_text
    assert "out.update(_TARGET_EXPANSIONS.get(canonical, (canonical,)))" in lora_text
    assert '"lora_alpha" in values and "alpha" not in values' in lora_text
    assert '"lora_dropout" in values and "dropout" not in values' in lora_text
    assert '"r" in values and "rank" not in values' in lora_text
    assert "def _normalize_target_modules" in lora_text
    assert 'values["target_modules"] = _normalize_target_modules' in lora_text
    assert "LoRA config target_modules must be a string or sequence of strings" in lora_text
    assert "LoRA config target_modules entries must be strings" in lora_text
    assert "LoRA config target_modules entries must be non-empty strings" in lora_text
    assert "LoRA config target_modules must be non-empty when LoRA is enabled" in lora_text
    assert "def validate_lora_dropout_value" in lora_text
    assert "must be between 0 and 1 inclusive" in lora_text
    assert '"q_a": "q_a_proj"' in lora_text
    assert '"o_proj": "linear_proj"' in lora_text
    assert '"gate_proj": "linear_fc1"' in lora_text
    assert '"up_proj": "linear_fc1"' in lora_text
    assert "def materialized_delta_weight" in lora_text

    assert "use_rslora=lora.use_rslora" in gqa_text
    assert "use_rslora=lora.use_rslora" in experts_text
    assert '"use_rslora": lora.use_rslora' in qwen_adapter_text
    assert "_infer_native_use_rslora" in qwen_adapter_text
    assert "def _infer_native_dropout" in qwen_adapter_text
    assert "Adapter config lora_dropout=" in qwen_adapter_text
    assert "def _validate_init_lora_weights" in qwen_adapter_text
    assert "Adapter config init_lora_weights must be" in qwen_adapter_text
    assert "def _validate_adapter_only_config" in qwen_adapter_text
    assert "Adapter config bias=" in qwen_adapter_text
    assert "Adapter config fan_in_fan_out=True is not supported." in qwen_adapter_text
    assert "Adapter config modules_to_save is not supported" in qwen_adapter_text
    assert "caller-provided lora_config" in qwen_adapter_text
    assert "_WRAPPED_MODULE_ATTRS" in qwen_adapter_text
    assert '"_fsdp_wrapped_module"' in qwen_adapter_text
    assert "def export_lora_adapter_state" in qwen_protocol_text
    assert "def initialize_lora_olora_tail" in qwen_protocol_text
    assert "def save_lora_adapter" in qwen_protocol_text
    assert "def load_lora_adapter_state" in qwen_protocol_text
    assert "def load_lora_adapter" in qwen_protocol_text
    assert '"export_lora_adapter_state"' in qwen_protocol_text
    assert '"initialize_lora_olora_tail"' in qwen_protocol_text
    assert '"save_lora_adapter"' in qwen_protocol_text
    assert '"load_lora_adapter_state"' in qwen_protocol_text
    assert '"load_lora_adapter"' in qwen_protocol_text
    assert "consumed_keys: set[str] = set()" in qwen_adapter_text
    assert "return _require_tensor(state, key, consumed_keys)" in qwen_adapter_text
    assert "Unexpected adapter tensor keys" in qwen_adapter_text
    assert "def _dedupe_target_modules" in qwen_adapter_text
    assert "def _validate_adapter_state_is_lora_only" in qwen_adapter_text
    assert "def initialize_lora_olora_tail" in qwen_adapter_text
    assert '"olora_tail"' in qwen_adapter_text
    assert "Qwen3-MoE OLoRA-tail currently supports tp=1" in qwen_adapter_text
    assert "LoRA adapter state is empty" in qwen_adapter_text
    assert 'key.endswith(".lora_A.weight")' in qwen_adapter_text
    assert 'key.endswith(".lora_B.weight")' in qwen_adapter_text
    assert "LoRA adapter export produced non-adapter tensor keys" in qwen_adapter_text
    assert "_validate_adapter_state_is_lora_only(state)" in qwen_adapter_text
    assert '"all-linear": _ADAPTER_TARGET_MODULES' in qwen_adapter_text
    assert '"gate_proj": ("gate_proj", "up_proj")' in qwen_adapter_text
    assert '"up_proj": ("gate_proj", "up_proj")' in qwen_adapter_text
    assert "config_use_rslora = _adapter_bool(" in qwen_adapter_text
    assert "does not match expected use_rslora" in qwen_adapter_text
    assert "test_qwen3_lora_adapter_rejects_non_adapter_only_configs_without_te" in (
        ROOT.parents[1] / "tests" / "unit" / "model" / "test_glm5_lora_adapter_runtime.py"
    ).read_text()
    assert "test_qwen3_lora_adapter_protocol_wrappers_round_trip_without_te" in (
        ROOT.parents[1] / "tests" / "unit" / "model" / "test_glm5_lora_adapter_runtime.py"
    ).read_text()
    assert "test_qwen3_lora_adapter_fsdp_style_wrappers_round_trip_without_te" in (
        ROOT.parents[1] / "tests" / "unit" / "model" / "test_glm5_lora_adapter_runtime.py"
    ).read_text()
    assert "test_qwen3_per_expert_grouped_lora_round_trips_without_te" in (
        ROOT.parents[1] / "tests" / "unit" / "model" / "test_glm5_lora_adapter_runtime.py"
    ).read_text()
    assert "test_qwen3_initialize_lora_olora_tail_supports_attention_and_single_local_expert" in (
        ROOT.parents[1] / "tests" / "unit" / "model" / "test_glm5_lora_adapter_runtime.py"
    ).read_text()
    assert "test_qwen3_initialize_lora_olora_tail_skips_multi_local_shared_experts" in (
        ROOT.parents[1] / "tests" / "unit" / "model" / "test_glm5_lora_adapter_runtime.py"
    ).read_text()
    qwen_adapter_runtime_test = (
        ROOT.parents[1] / "tests" / "unit" / "model" / "test_glm5_lora_adapter_runtime.py"
    ).read_text()
    assert '"/definitely/missing/qwen_adapter"' in qwen_adapter_runtime_test
    assert '"qwen_unsupported_save"' in qwen_adapter_runtime_test


def test_delta_mem_gqa_attention_wiring_static_contract():
    delta_mem_text = _read("primitive", "modules", "delta_mem.py")
    gqa_text = _read("primitive", "modules", "gqa.py")
    qwen_model_text = _read("model", "qwen3_moe", "lite", "model.py")
    qwen_protocol_text = _read("model", "qwen3_moe", "lite", "protocol.py")
    runtime_text = _read("runtime", "backends", "mlite", "runtime.py")
    runtime_delta_mem_text = _read("runtime", "backends", "mlite", "delta_mem_state.py")
    data_contract_text = _read("runtime", "contracts", "data.py")
    primitive_test_text = (
        ROOT.parents[1] / "tests" / "unit" / "primitive" / "test_module_primitives_independent_unit.py"
    ).read_text()
    runtime_test_text = (
        ROOT.parents[1] / "tests" / "unit" / "runtime" / "test_runtime_backend_unit.py"
    ).read_text()

    assert "class DeltaMemConfig" in delta_mem_text
    assert "def normalize_delta_mem_config" in delta_mem_text
    assert '"r" in values and "rank" not in values' in delta_mem_text
    assert "DeltaMem config enabled=True requires a positive rank" in delta_mem_text
    assert "class DeltaMemAttentionCorrection" in delta_mem_text
    assert "query_delta=self.query_delta_proj(policy_output.readout)" in delta_mem_text
    assert "output_delta=self.output_delta_proj(policy_output.readout)" in delta_mem_text
    assert "class DeltaMemStatePolicy" in delta_mem_text
    assert "class DeltaMemStateManager" in delta_mem_text
    assert "def normalize_delta_mem_state_policy" in delta_mem_text
    assert "state_policy: DeltaMemStatePolicy | Mapping[str, Any] | str | None = None" in delta_mem_text
    assert "self.state_manager = DeltaMemStateManager(self.rank, policy=state_policy)" in delta_mem_text
    assert "def initial_state(self, hidden: torch.Tensor) -> torch.Tensor" in delta_mem_text
    assert "state_indices: torch.Tensor | None = None" in delta_mem_text
    assert '"token-state": "token"' in delta_mem_text
    assert '"sequence-state": "sequence"' in delta_mem_text
    assert '"multi-state": "multi"' in delta_mem_text
    assert "DeltaMem sequence/multi policies require a leading time dimension" in delta_mem_text
    assert "DeltaMem multi-state state_indices are out of range" in delta_mem_text

    assert "delta_mem_config: DeltaMemConfig | Mapping[str, Any] | None = None" in gqa_text
    assert "delta_mem = normalize_delta_mem_config(delta_mem_config)" in gqa_text
    assert "self.delta_mem: DeltaMemAttentionCorrection | None = None" in gqa_text
    assert "GQAttention DeltaMem wiring currently supports tensor parallel size 1" in gqa_text
    assert "GQAttention DeltaMem wiring currently supports context parallel size 1" in gqa_text
    assert "GQAttention DeltaMem wiring currently does not support THD packed sequences" in gqa_text
    assert "delta_mem_state: torch.Tensor | None = None" in gqa_text
    assert "delta_mem_state_indices: torch.Tensor | None = None" in gqa_text
    assert "return_delta_mem_state: bool = False" in gqa_text
    assert "state_policy=delta_mem.state_policy" in gqa_text
    assert "delta_mem_hidden = x.to(dtype=self.delta_mem.query_delta_proj.weight.dtype)" in gqa_text
    assert "state_indices=delta_mem_state_indices" in gqa_text
    assert "delta_mem_result = self.delta_mem(" in gqa_text
    assert "q = q + delta_mem_result.query_delta.to(dtype=q.dtype).reshape_as(q)" in gqa_text
    assert "output = output + delta_mem_result.output_delta.to(dtype=output.dtype)" in gqa_text
    assert "return output, delta_mem_result.next_state" in gqa_text
    assert "def _initial_delta_mem_state" in gqa_text
    assert "return self.delta_mem.initial_state(x)" in gqa_text

    assert "delta_mem_config: DeltaMemConfig | Mapping[str, Any] | None = None" in qwen_model_text
    assert "delta_mem_config=delta_mem_config" in qwen_model_text
    assert "def delta_mem_layer_count" in qwen_model_text
    assert "def delta_mem_mtp_layer_count" in qwen_model_text
    assert "def _normalize_delta_mem_states" in qwen_model_text
    assert "def _normalize_delta_mem_mtp_states" in qwen_model_text
    assert "def _normalize_delta_mem_state_indices" in qwen_model_text
    assert "def _normalize_delta_mem_mtp_state_indices" in qwen_model_text
    assert "def _split_delta_mem_states_payload" in qwen_model_text
    assert "def _split_delta_mem_state_indices_payload" in qwen_model_text
    assert "delta_mem_states: Sequence[torch.Tensor | None] | Mapping[str, Any] | None = None" in qwen_model_text
    assert "delta_mem_mtp_states: Sequence[torch.Tensor | None] | None = None" in qwen_model_text
    assert "delta_mem_state_indices: Sequence[torch.Tensor | None] | torch.Tensor | Mapping[str, Any] | None = None" in qwen_model_text
    assert "delta_mem_mtp_state_indices: Sequence[torch.Tensor | None] | torch.Tensor | None = None" in qwen_model_text
    assert "return_delta_mem_states: bool = False" in qwen_model_text
    assert "use_delta_mem_state_handoff = (" in qwen_model_text
    assert "return_delta_mem_state=True" in qwen_model_text
    assert 'output["delta_mem_states"] = next_delta_mem_states' in qwen_model_text
    assert 'output["delta_mem_mtp_states"] = next_delta_mem_mtp_states' in qwen_model_text
    assert "return outputs, next_delta_mem_states" in qwen_model_text
    assert "Qwen3MoE MTP delta_mem_states length must match MTP prediction depths" in qwen_model_text
    assert "Qwen3MoEModel delta_mem_mtp_states require enabled MTP." in qwen_model_text
    assert "Qwen3MoEModel delta_mem_mtp_state_indices require enabled MTP." in qwen_model_text
    assert (
        "self._normalize_delta_mem_mtp_states(delta_mem_mtp_states)\n"
        "                if use_delta_mem_state_handoff\n"
        "                else []"
    ) in qwen_model_text
    assert (
        "self._normalize_delta_mem_mtp_state_indices(delta_mem_mtp_state_indices)\n"
        "                if use_delta_mem_state_handoff\n"
        "                else []"
    ) in qwen_model_text
    assert "delta_mem: DeltaMemConfig | Mapping[str, Any] | None = None" in qwen_protocol_text
    assert "delta_mem_config = normalize_delta_mem_config(impl_cfg.delta_mem)" in qwen_protocol_text
    assert "delta_mem_config=delta_mem_config" in qwen_protocol_text
    assert '"delta_mem_config": delta_mem_config' in qwen_protocol_text
    assert '"delta_mem_mtp_states"' in qwen_protocol_text
    assert '"delta_mem_mtp_state_indices"' in qwen_protocol_text
    assert "kwargs[key] = extras[key]" in qwen_protocol_text

    assert "class DeltaMemRuntimeState" in runtime_delta_mem_text
    assert 'DELTA_MEM_STATE_SCOPE_KEY = "delta_mem_state_scope"' in runtime_delta_mem_text
    assert 'DELTA_MEM_RUNTIME_STATES_KEY = "_delta_mem_runtime_states"' in runtime_delta_mem_text
    assert 'DELTA_MEM_RUNTIME_STATE_FORMAT = "megatron_lite.delta_mem_runtime_state.v1"' in runtime_delta_mem_text
    assert 'DELTA_MEM_RUNTIME_STATE_FILE = "delta_mem_runtime_state.pt"' in runtime_delta_mem_text
    assert 'return "step"' in runtime_delta_mem_text
    assert 'return "handle"' in runtime_delta_mem_text
    assert "forward_extras[\"return_delta_mem_states\"] = True" in runtime_delta_mem_text
    assert '"delta_mem_states" not in forward_extras' in runtime_delta_mem_text
    assert '"delta_mem_mtp_states" not in forward_extras' in runtime_delta_mem_text
    assert 'return {"main": main_states, "mtp": output.get("delta_mem_mtp_states")}' in runtime_delta_mem_text
    assert "value.detach()" in runtime_delta_mem_text
    assert "def export_delta_mem_runtime_state" in runtime_delta_mem_text
    assert "def import_delta_mem_runtime_state" in runtime_delta_mem_text
    assert "def save_delta_mem_runtime_state" in runtime_delta_mem_text
    assert "def load_delta_mem_runtime_state" in runtime_delta_mem_text
    assert "def reset_delta_mem_runtime_state" in runtime_delta_mem_text
    assert '"adapter_export_compatible": True' in runtime_delta_mem_text
    assert "def wrap_delta_mem_forward_step" in runtime_delta_mem_text
    assert "delta-mem runtime lifecycle requires enabled DeltaMem config" in runtime_delta_mem_text
    assert "delta_mem_runtime_lifecycle_requested" in runtime_text
    assert "wrap_delta_mem_forward_step" in runtime_text
    assert "save_delta_mem_state = bool(kwargs.pop(\"save_delta_mem_state\", True))" in runtime_text
    assert "load_delta_mem_state = bool(kwargs.pop(\"load_delta_mem_state\", True))" in runtime_text
    assert "save_delta_mem_runtime_state(handle, delta_mem_path, require_state=False)" in runtime_text
    assert "load_delta_mem_runtime_state(handle, path, require_exists=False)" in runtime_text
    assert "def export_delta_mem_runtime_state(self, handle: ModelHandle, **kwargs)" in runtime_text
    assert "def save_delta_mem_runtime_state" in runtime_text
    assert "def load_delta_mem_runtime_state" in runtime_text
    assert "def reset_delta_mem_runtime_state" in runtime_text
    assert "MLite delta-mem runtime lifecycle currently supports pipeline parallel size 1" in runtime_text
    assert "delta_mem_manager.finish(handle)" in runtime_text
    assert "metrics.update(delta_mem_manager.metrics())" in runtime_text
    assert "delta_mem_states=out.get(\"delta_mem_states\") if out else None" in runtime_text
    assert "delta_mem_mtp_states=out.get(\"delta_mem_mtp_states\") if out else None" in runtime_text
    assert "delta_mem_states: Any | None = None" in data_contract_text
    assert "delta_mem_mtp_states: Any | None = None" in data_contract_text

    assert "def test_delta_mem_config_normalizes_mapping_and_aliases" in primitive_test_text
    assert "def test_delta_mem_state_policy_normalizes_aliases_and_rejects_bad_values" in primitive_test_text
    assert "def test_delta_mem_token_state_manager_matches_vectorized_write_and_detach" in primitive_test_text
    assert "def test_delta_mem_sequence_state_manager_recurrently_updates_one_state" in primitive_test_text
    assert "def test_delta_mem_multi_state_manager_routes_writes_to_state_banks" in primitive_test_text
    assert "def test_delta_mem_attention_correction_emits_query_and_output_deltas" in primitive_test_text
    assert "def test_delta_mem_attention_correction_uses_sequence_state_policy" in primitive_test_text
    assert "def test_delta_mem_attention_correction_uses_multi_state_policy_indices" in primitive_test_text
    assert (
        "def test_mlite_runtime_delta_mem_step_scope_carries_detached_states_across_microbatches"
        in runtime_test_text
    )
    assert "def test_mlite_runtime_delta_mem_handle_scope_persists_detached_state" in runtime_test_text
    assert (
        "def test_mlite_runtime_delta_mem_carries_main_and_mtp_states_across_microbatches"
        in runtime_test_text
    )
    assert "def test_mlite_runtime_delta_mem_state_explicit_save_load_reset_round_trip" in runtime_test_text
    assert "def test_mlite_runtime_checkpoint_saves_and_loads_delta_mem_sidecar" in runtime_test_text
    assert '"_delta_mem_runtime_states": [torch.tensor([7.0])]' in runtime_test_text


def test_mint_adapter_lifecycle_static_contract():
    lifecycle_text = _read("primitive", "modules", "adapter_lifecycle.py")
    modules_init_text = _read("primitive", "modules", "__init__.py")
    lifecycle_test_text = (
        ROOT.parents[1] / "tests" / "unit" / "primitive" / "test_adapter_lifecycle_unit.py"
    ).read_text()

    assert "policy identity is separate from compute residency" in lifecycle_text
    assert 'COOL_STORED = "cool"' in lifecycle_text
    assert 'WARM_CPU = "warm"' in lifecycle_text
    assert 'HOT_GPU = "hot"' in lifecycle_text
    assert "class BaseDeployment" in lifecycle_text
    assert "class PolicyRecord" in lifecycle_text
    assert "class PolicySession" in lifecycle_text
    assert "class AdapterRevision" in lifecycle_text
    assert "class ServingResidency" in lifecycle_text
    assert "class AdapterPolicyRegistry" in lifecycle_text
    assert "class AdapterResidencyManager" in lifecycle_text
    assert "def create_policy" in lifecycle_text
    assert "def create_revision" in lifecycle_text
    assert "def set_active_revision" in lifecycle_text
    assert "def create_policy_session" in lifecycle_text
    assert "def close_policy_session" in lifecycle_text
    assert "def record_rollout" in lifecycle_text
    assert "def set_residency" in lifecycle_text
    assert "def promote_to_warm" in lifecycle_text
    assert "def promote_to_hot" in lifecycle_text
    assert "def cool_down" in lifecycle_text
    assert "def plan_batch" in lifecycle_text
    assert "def _enforce_capacity" in lifecycle_text
    assert "does not load weights" in lifecycle_text
    assert "claim any paper evaluation result" in lifecycle_text
    assert '"AdapterPolicyRegistry"' in modules_init_text
    assert '"AdapterResidencyManager"' in modules_init_text
    assert '"AdapterRevision"' in modules_init_text
    assert '"ServingResidency"' in modules_init_text
    assert "test_adapter_policy_registry_tracks_policy_revision_and_session_lifecycle" in lifecycle_test_text
    assert "test_adapter_residency_manager_separates_identity_from_compute_residency" in lifecycle_test_text


def test_lora_memory_capacity_static_contract():
    memory_text = _read("primitive", "modules", "adapter_memory.py")
    modules_init_text = _read("primitive", "modules", "__init__.py")
    memory_test_text = (
        ROOT.parents[1] / "tests" / "unit" / "primitive" / "test_adapter_memory_unit.py"
    ).read_text()

    assert "capacity efficiency as memory tokens per" in memory_text
    assert "does not load datasets, train adapters, evaluate accuracy" in memory_text
    assert "CAPACITY_HIGH_ACCURACY_MAX = 1e-3" in memory_text
    assert "CAPACITY_COLLAPSE_MIN = 1e-2" in memory_text
    assert "class LinearModuleSpec" in memory_text
    assert "class LoraMemoryPlan" in memory_text
    assert "def lora_trainable_params" in memory_text
    assert "return rank * (in_features + out_features)" in memory_text
    assert "def capacity_efficiency" in memory_text
    assert "return float(memory_tokens) / float(trainable_params)" in memory_text
    assert "def classify_capacity_efficiency" in memory_text
    assert '"below_transition_expected_high_accuracy"' in memory_text
    assert '"transition_band_expected_degradation"' in memory_text
    assert '"above_collapse_expected_failure"' in memory_text
    assert "def build_lora_memory_plan" in memory_text
    assert "def build_target_module_ablation_plans" in memory_text
    assert "def build_rank_shift_plans" in memory_text
    assert "def paper_reference_target_order" in memory_text
    assert "paper_accuracy_claimed: bool = False" in memory_text
    assert "paper_module_ranking_claimed: bool = False" in memory_text
    assert '"mlp", "attention", "all", "unembed"' in memory_text
    assert '"LinearModuleSpec"' in modules_init_text
    assert '"LoraMemoryPlan"' in modules_init_text
    assert '"build_target_module_ablation_plans"' in modules_init_text
    assert "test_lora_memory_capacity_classifies_paper_ratio_regimes" in memory_test_text
    assert "test_lora_memory_target_module_ablation_plan_keeps_paper_ranking_unclaimed" in memory_test_text
    assert "test_lora_memory_rank_shift_moves_fixed_memory_across_regimes" in memory_test_text


def test_context_learning_write_policy_static_contract():
    context_text = _read("primitive", "modules", "context_learning.py")
    modules_init_text = _read("primitive", "modules", "__init__.py")
    context_test_text = (
        ROOT.parents[1] / "tests" / "unit" / "primitive" / "test_context_learning_unit.py"
    ).read_text()

    assert "Context Learning write-policy primitives" in context_text
    assert "repeated Context Distillation" in context_text
    assert "does not call a teacher model" in context_text
    assert "load datasets or weights, launch" in context_text
    assert "claim the paper's empirical result" in context_text
    assert 'QUERY_ONLY_ROLLOUT = "query_only_rollout"' in context_text
    assert 'TEACHER_CONTEXT_SCORING = "teacher_context_scoring"' in context_text
    assert 'QUERY_ONLY_POLICY_UPDATE = "query_only_policy_update"' in context_text
    assert 'QUERY_ONLY_INFERENCE_CONTRACT = "query_only_inference_contract"' in context_text
    assert 'LORA_ADAPTER_TARGET = "lora_adapter"' in context_text
    assert "class QueryOnlyPayload" in context_text
    assert "class TeacherContextScoringPayload" in context_text
    assert "class QueryOnlyRolloutRecord" in context_text
    assert "class TeacherContextScoringRequest" in context_text
    assert "class QueryOnlyAdapterUpdatePlan" in context_text
    assert "class QueryOnlyInferenceContract" in context_text
    assert "class ContextLearningPlan" in context_text
    assert "def build_context_learning_plan" in context_text
    assert "def context_learning_invariants" in context_text
    assert "def assert_context_learning_plan" in context_text
    assert "query-only payload must not carry context" in context_text
    assert "reward_source must be teacher_context_scoring" in context_text
    assert "update_target must be lora_adapter" in context_text
    assert "calls_teacher_model: bool = False" in context_text
    assert "paper_results_claimed: bool = False" in context_text
    assert '"context_not_copied_into_rollout_or_update_payload"' in context_text
    assert '"ContextLearningPlan"' in modules_init_text
    assert '"QueryOnlyPayload"' in modules_init_text
    assert '"TeacherContextScoringRequest"' in modules_init_text
    assert '"build_context_learning_plan"' in modules_init_text
    assert '"context_learning_invariants"' in modules_init_text
    assert "test_context_learning_plan_keeps_context_only_in_teacher_scoring" in context_test_text
    assert "test_context_learning_plan_rejects_context_leaks_and_wrong_reward_source" in context_test_text
    assert "test_context_learning_plan_keeps_paper_results_unclaimed" in context_test_text


def test_diversity_majority_vote_static_contract():
    vote_text = _read("primitive", "modules", "diversity_vote.py")
    modules_init_text = _read("primitive", "modules", "__init__.py")
    vote_test_text = (
        ROOT.parents[1] / "tests" / "unit" / "primitive" / "test_diversity_vote_unit.py"
    ).read_text()

    assert "Diversity majority-vote primitives" in vote_text
    assert "collaboration versus repetition arms" in vote_text
    assert "does not train adapters, collect model votes, run AIME24" in vote_text
    assert "claim the paper's empirical accuracy" in vote_text
    assert 'COLLABORATION_DISTINCT_ADAPTERS = "collaboration_distinct_adapters"' in vote_text
    assert 'REPETITION_SINGLE_ADAPTER = "repetition_single_adapter"' in vote_text
    assert 'MAJORITY_VOTE = "majority_vote"' in vote_text
    assert 'PAPER_REFERENCE_LOG_FIT = "accuracy = 0.386 + 0.0172 * ln(k)"' in vote_text
    assert "PAPER_REFERENCE_MAX_K = 198" in vote_text
    assert "class AdapterVariantSpec" in vote_text
    assert "class VoteCollectionArm" in vote_text
    assert "class VoteRecord" in vote_text
    assert "class MajorityVoteResult" in vote_text
    assert "def build_adapter_variants" in vote_text
    assert "def build_vote_collection_arms" in vote_text
    assert "def majority_vote" in vote_text
    assert "def accuracy_from_majority_votes" in vote_text
    assert "def paper_reference_accuracy" in vote_text
    assert "def paper_reference_vote_contract" in vote_text
    assert "AdapterVariantSpec must not claim paper metrics" in vote_text
    assert "MajorityVoteResult must not claim paper metrics" in vote_text
    assert "collaboration arms require distinct adapter_id values" in vote_text
    assert "repetition arms require one repeated adapter_id" in vote_text
    assert '"paper_reference_not_local_result"' in vote_text
    assert '"AdapterVariantSpec"' in modules_init_text
    assert '"VoteRecord"' in modules_init_text
    assert '"MajorityVoteResult"' in modules_init_text
    assert '"build_adapter_variants"' in modules_init_text
    assert '"majority_vote"' in modules_init_text
    assert '"paper_reference_vote_contract"' in modules_init_text
    assert "test_diversity_vote_builds_paper_shaped_adapter_ensemble_without_claims" in vote_test_text
    assert "test_diversity_vote_majority_vote_collaboration_and_repetition_rules" in vote_test_text
    assert "test_diversity_vote_rejects_invalid_vote_groups_and_unverified_accuracy" in vote_test_text
    assert "test_diversity_vote_ties_are_deterministic_but_not_paper_claims" in vote_test_text


def test_adapter_population_user_simulator_static_contract():
    population_text = _read("primitive", "modules", "adapter_population.py")
    modules_init_text = _read("primitive", "modules", "__init__.py")
    population_test_text = (
        ROOT.parents[1] / "tests" / "unit" / "primitive" / "test_adapter_population_unit.py"
    ).read_text()

    assert "Per-user adapter-population primitives" in population_text
    assert "OASIS user-simulator results" in population_text
    assert "per-user LoRA policies preserve heterogeneity" in population_text
    assert "It does not create OASIS data" in population_text
    assert "simulators, train adapters" in population_text
    assert "claim paper metrics" in population_text
    assert 'PER_USER_LORA = "per_user_lora"' in population_text
    assert 'SHARED_BASE = "shared_base"' in population_text
    assert 'LORA_ADAPTER = "lora"' in population_text
    assert 'NO_ADAPTER = "none"' in population_text
    assert "HETEROGENEITY_METRICS" in population_text
    assert "ACTION_ECOLOGY_METRICS" in population_text
    assert "class BaseDeploymentSpec" in population_text
    assert "class UserPolicyRecord" in population_text
    assert "class SharedBaselinePolicy" in population_text
    assert "class PopulationEvaluationArm" in population_text
    assert "class AdapterPopulationMetricSchema" in population_text
    assert "class StructuralScalingArm" in population_text
    assert "class AdapterPopulationPlan" in population_text
    assert "def build_adapter_population_plan" in population_text
    assert "def build_population_evaluation_arms" in population_text
    assert "def adapter_population_invariants" in population_text
    assert "def validate_adapter_population_plan" in population_text
    assert "def paper_reference_population_contract" in population_text
    assert "shared baseline must not have an adapter_revision_id" in population_text
    assert "per-user policy adapter_type must be lora" in population_text
    assert "AdapterPopulationPlan does not run user simulators" in population_text
    assert "AdapterPopulationPlan must not claim paper results" in population_text
    assert '"paper_reference_not_local_result"' in population_text
    assert '"AdapterPopulationPlan"' in modules_init_text
    assert '"UserPolicyRecord"' in modules_init_text
    assert '"SharedBaselinePolicy"' in modules_init_text
    assert '"PopulationEvaluationArm"' in modules_init_text
    assert '"build_adapter_population_plan"' in modules_init_text
    assert '"adapter_population_invariants"' in modules_init_text
    assert '"paper_reference_population_contract"' in modules_init_text
    assert "test_adapter_population_plan_builds_per_user_and_shared_baseline_arms" in population_test_text
    assert "test_adapter_population_plan_rejects_duplicate_users_and_metric_claims" in population_test_text
    assert "test_adapter_population_shared_baseline_and_arm_guards" in population_test_text
    assert "test_adapter_population_reference_contract_is_not_local_metric" in population_test_text


def test_dsa_indexshare_static_contract():
    dsa_text = _read("primitive", "modules", "attention", "dsa.py")
    glm5_text = _read("model", "glm5", "lite", "model.py")
    glm5_cfg_text = _read("model", "glm5", "config.py")
    glm5_static_test = (
        ROOT.parents[1] / "tests" / "unit" / "model" / "test_glm5_lite_static.py"
    ).read_text()

    assert 'DSA_INDEX_SHARE_TOPK_HOLDER_ATTR = "_dsa_index_share_topk_holder"' in dsa_text
    assert "self.skip_topk = dsa_indexer_type == \"shared\"" in dsa_text
    assert "self.indexer: DSAIndexer | None = None" in dsa_text
    assert "if not self.skip_topk:" in dsa_text
    assert "topk_indices = self._load_index_share_topk" in dsa_text
    assert "self._store_index_share_topk(" in dsa_text
    assert "index_share_segment_idx=idx" in dsa_text
    assert "DSA IndexShare training with dsa_indexer_loss_coeff > 0" in dsa_text

    assert "_validate_dsa_index_share_pipeline_split" in glm5_text
    assert "Glm5DSAAttention(config, ps, layer_idx, lora_config=lora_config)" in glm5_text
    assert "dsa_indexer_type=config.dsa_indexer_type(layer_idx)" in glm5_text
    assert "dsa_source_layer_idx=config.dsa_source_compute_layer(layer_idx)" in glm5_text

    assert '"indexer_types"' in glm5_cfg_text
    assert "def dsa_indexer_type" in glm5_cfg_text
    assert "def dsa_source_compute_layer" in glm5_cfg_text

    assert "def test_glm5_hf_reference_weight_map_matches_dsa_indexshare_and_mtp_layout" in glm5_static_test
    assert "def test_glm5_dsa_indexshare_loss_fails_explicitly_before_topk_kernel" in glm5_static_test
    assert (
        "def test_glm5_dsa_indexshare_training_zero_loss_reuses_full_layer_topk"
        in glm5_static_test
    )
    assert (
        "def test_glm5_dsa_indexshare_shared_layer_requires_source_topk_in_holder"
        in glm5_static_test
    )
    assert "def test_glm5_lite_tiny_cpu_forward_backward" in glm5_static_test
    assert (
        "def test_glm5_protocol_build_model_freezes_base_params_and_reports_lora_stats"
        in glm5_static_test
    )
    assert "def test_glm5_lora_tiny_cpu_forward_backward_freezes_base_params" in glm5_static_test
    assert "def test_glm5_router_replay_tiny_cpu_record_replay_identity" in glm5_static_test
    assert "def test_qwen3_moe_router_replay_tiny_cpu_record_replay_identity" in glm5_static_test
    assert (
        "def test_qwen3_moe_router_replay_instances_append_mtp_routers_after_main_layers"
        in glm5_static_test
    )
    assert "model.safetensors.index.json" in glm5_static_test
    assert "has_indexer_weights is (indexer_type == \"full\")" in glm5_static_test
    assert "model.layers.{mtp_layer_idx}.eh_proj.weight" in glm5_static_test
    assert "assert \"eh_proj\" not in adapter_targets" in glm5_static_test


def test_glm5_lora_static_contract():
    dsa_text = _read("primitive", "modules", "attention", "dsa.py")
    glm5_text = _read("model", "glm5", "lite", "model.py")
    glm5_protocol_text = _read("model", "glm5", "lite", "protocol.py")
    glm5_lite_init_text = _read("model", "glm5", "lite", "__init__.py")
    glm5_adapter_text = _read("model", "glm5", "lite", "lora_adapter.py")
    glm5_adapter_semantics_text = _read("model", "glm5", "lite", "adapter_semantics.py")
    glm5_checkpoint_text = _read("model", "glm5", "lite", "checkpoint.py")
    glm5_adapter_runtime_test = (
        ROOT.parents[1]
        / "tests"
        / "unit"
        / "model"
        / "test_glm5_lora_adapter_runtime.py"
    ).read_text()

    assert "lora_config: LoraConfig | Mapping[str, Any] | None = None" in dsa_text
    assert "input_projection_lora = lora.enabled and lora.targets_module(\"linear_qkv\")" in dsa_text
    for name in ("q_a_lora", "q_b_lora", "kv_a_lora", "kv_b_lora", "o_lora"):
        assert name in dsa_text
    assert "self.kv_b_lora.materialized_delta_weight()" in dsa_text

    assert "class DenseMLP" in glm5_text
    assert "self.gate_up_lora: LinearLoRA | None = None" in glm5_text
    assert "self.down_lora: LinearLoRA | None = None" in glm5_text
    assert "self._gate_up_lora_input(x)" in glm5_text
    assert "class SharedExpert" in glm5_text
    assert "self.shared_expert = SharedExpert(config, ps, lora_config=lora_config)" in glm5_text
    assert "lora_config=lora_config" in glm5_text
    assert "class Glm5ForCausalLM" in glm5_text
    assert "self.model = Glm5Model(" in glm5_text
    assert "def _build_glm5_pipeline_layers" in glm5_text
    assert "def state_dict(self, *args, **kwargs)" in glm5_text
    assert "export_hf_weights(self.model, self.config, self.ps)" in glm5_text
    assert 'kwargs["hidden_states"] = hidden_states.transpose(0, 1).contiguous()' in glm5_text
    assert 'output["hidden_states"] = output["hidden_states"].transpose(0, 1).contiguous()' in glm5_text
    assert "Glm5Router = Glm5SigmoidTopKRouter" in glm5_text

    assert "lora: LoraConfig | Mapping[str, Any] | None = None" in glm5_protocol_text
    assert "lora_config = normalize_lora_config(impl_cfg.lora)" in glm5_protocol_text
    assert "freeze_non_lora_params(chunk)" in glm5_protocol_text
    assert '"lora_stats": lora_stats' in glm5_protocol_text
    assert "PP / VPP / EP / CP are inherited" in glm5_protocol_text
    assert "TP>1 / ETP>1 are unsupported" in glm5_protocol_text
    assert "if p.tp > 1:" in glm5_protocol_text
    assert "if etp > 1:" in glm5_protocol_text
    assert "def __getattr__(name: str)" in glm5_lite_init_text
    assert "if name == \"Glm5Model\":" in glm5_lite_init_text
    assert "from megatron.lite.model.glm5.lite.model import Glm5Model" in glm5_lite_init_text

    assert "def export_lora_adapter_state" in glm5_adapter_text
    assert "def initialize_lora_olora_tail" in glm5_adapter_text
    assert 'B0 = U_tail' in glm5_adapter_text
    assert 'A0 = V_tail.T' in glm5_adapter_text
    assert "torch.linalg.svd" in glm5_adapter_text
    assert '\"olora_tail\"' in glm5_adapter_text
    assert "def _grouped_base_weight" in glm5_adapter_text
    assert "def _init_grouped_lora_olora_tail" in glm5_adapter_text
    assert "OLoRA-tail initialization is undefined" in glm5_adapter_text
    assert "_olora_tail_initialized" in glm5_adapter_text
    assert "Singular values are intentionally" in glm5_adapter_text
    assert "def save_lora_adapter" in glm5_adapter_text
    assert "def load_lora_adapter_state" in glm5_adapter_text
    assert "def load_lora_adapter" in glm5_adapter_text
    assert "_WRAPPED_MODULE_ATTRS" in glm5_adapter_text
    assert '"_fsdp_wrapped_module"' in glm5_adapter_text
    assert "inner_model = getattr(current, \"model\", None)" in glm5_adapter_text
    assert "hasattr(inner_model, \"layer_indices\")" in glm5_adapter_text
    assert "def _unwrap_glm5_model" in glm5_checkpoint_text
    assert "inner = getattr(base_model, \"model\", None)" in glm5_checkpoint_text
    assert "export_model = _unwrap_glm5_model(model)" in glm5_checkpoint_text
    assert "def _resolve_hf_tensor" in glm5_checkpoint_text
    assert "gate_and_up_projs" in glm5_checkpoint_text
    assert "def _resolve_named_parameter_tensor" in glm5_checkpoint_text
    assert "def _slice_to_target_shape" in glm5_checkpoint_text
    assert "def save_weights" in glm5_checkpoint_text
    assert "def _iter_adapter_layers" in glm5_adapter_text
    assert "yield model_cfg.num_hidden_layers + mtp_idx, mtp_layer.transformer_layer" in glm5_adapter_text
    assert glm5_adapter_text.count("for layer_idx, layer in _iter_adapter_layers(chunks, model_cfg):") >= 3
    assert '"num_nextn_predict_layers": model_cfg.num_nextn_predict_layers' in glm5_adapter_text
    assert '"mtp_use_repeated_layer": model_cfg.mtp_use_repeated_layer' in glm5_adapter_text
    assert "def _infer_native_dropout" in glm5_adapter_text
    assert "Adapter config lora_dropout=" in glm5_adapter_text
    assert "def _dedupe_target_modules" in glm5_adapter_text
    assert "Unsupported GLM5 LoRA target module" in glm5_adapter_text
    assert "def _validate_parallel_scope" in glm5_adapter_text
    assert "GLM5 LoRA adapter import/export currently supports tp=1" in glm5_adapter_text
    assert "GLM5 LoRA adapter import/export currently supports pp=1" in glm5_adapter_text
    assert "GLM5 LoRA adapter import/export currently supports etp=1" in glm5_adapter_text
    assert "ADAPTER_TARGET_MODULES as _ADAPTER_TARGET_MODULES" in glm5_adapter_text
    assert "TARGET_MODULE_EXPANSIONS as _TARGET_MODULE_EXPANSIONS" in glm5_adapter_text
    assert "modules = _TARGET_MODULE_EXPANSIONS.get(target)" in glm5_adapter_text
    assert '"all-linear": ADAPTER_TARGET_MODULES' in glm5_adapter_semantics_text
    assert '"linear_qkv": ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj")' in glm5_adapter_semantics_text
    assert '"linear_fc1": ("gate_proj", "up_proj")' in glm5_adapter_semantics_text
    assert '"gate_proj": ("gate_proj", "up_proj")' in glm5_adapter_semantics_text
    assert '"up_proj": ("gate_proj", "up_proj")' in glm5_adapter_semantics_text
    assert "return _dedupe_target_modules(lora_config.target_modules)" in glm5_adapter_text
    assert "def _adapter_bool" in glm5_adapter_text
    assert "Adapter config {key} must be a boolean" in glm5_adapter_text
    assert "def _validate_init_lora_weights" in glm5_adapter_text
    assert "Adapter config init_lora_weights must be" in glm5_adapter_text
    assert "def _validate_adapter_only_config" in glm5_adapter_text
    assert "Adapter config bias={bias!r} is not supported" in glm5_adapter_text
    assert "Adapter config fan_in_fan_out=True is not supported" in glm5_adapter_text
    assert "Adapter config modules_to_save is not supported" in glm5_adapter_text
    assert "def _validate_adapter_metadata" in glm5_adapter_text
    assert "def _validate_lora_metadata" in glm5_adapter_text
    assert "def _validate_adapter_config_metadata_consistency" in glm5_adapter_text
    assert "def _validate_expected_lora_config" in glm5_adapter_text
    assert "Adapter import requires an enabled expected LoRA config" in glm5_adapter_text
    assert "caller-provided lora_config" in glm5_adapter_text
    assert "Expected Megatron Lite GLM5 adapter metadata format" in glm5_adapter_text
    assert "Adapter metadata num_tensors" in glm5_adapter_text
    assert "Adapter metadata num_parameters" in glm5_adapter_text
    assert "Adapter metadata model.{key}" in glm5_adapter_text
    assert "Adapter metadata lora.target_modules do not match adapter tensors" in glm5_adapter_text
    assert "Adapter metadata lora.scaling_convention" in glm5_adapter_text
    assert "Adapter metadata lora.scale" in glm5_adapter_text
    assert "Adapter config init_lora_weights" in glm5_adapter_text
    assert "def _validate_adapter_state_is_lora_only" in glm5_adapter_text
    assert "LoRA adapter state is empty" in glm5_adapter_text
    assert "def _validate_exported_target_modules" in glm5_adapter_text
    assert "LoRA config target_modules do not match exported adapter tensors" in glm5_adapter_text
    assert 'key.endswith(".lora_A.weight")' in glm5_adapter_text
    assert 'key.endswith(".lora_B.weight")' in glm5_adapter_text
    assert "LoRA adapter export produced non-adapter tensor keys" in glm5_adapter_text
    assert glm5_adapter_text.count("_validate_adapter_state_is_lora_only(state)") >= 2
    assert '"peft_type": "LORA"' in glm5_adapter_text
    assert '"task_type": "CAUSAL_LM"' in glm5_adapter_text
    assert '"base_model_name_or_path": base_model_name_or_path' in glm5_adapter_text
    assert '"r": lora.rank' in glm5_adapter_text
    assert '"lora_alpha": _json_number(_effective_lora_alpha(lora))' in glm5_adapter_text
    assert '"lora_dropout": lora.dropout' in glm5_adapter_text
    assert '"use_rslora": lora.use_rslora' in glm5_adapter_text
    assert '"target_modules": _target_modules_from_lora_config(lora)' in glm5_adapter_text
    assert '"init_lora_weights": init_lora_weights_value' in glm5_adapter_text
    assert '"modules_to_save": None' in glm5_adapter_text
    assert '"num_parameters": int(sum(t.numel() for t in state.values()))' in glm5_adapter_text
    assert '"scaling_convention"' in glm5_adapter_text
    assert '"alpha_over_sqrt_rank"' in glm5_adapter_text
    assert '"alpha_over_rank"' in glm5_adapter_text
    assert '"parallel": {' in glm5_adapter_text
    assert "def _require_tensor_shape" in glm5_adapter_text
    assert "Adapter tensor {key!r} has shape" in glm5_adapter_text
    assert "def _to_lora_param" in glm5_adapter_text
    assert "consumed_keys" in glm5_adapter_text
    assert "def _validate_all_expert_adapter_keys" in glm5_adapter_text
    assert "for expert_idx in range(model_cfg.num_experts)" in glm5_adapter_text
    assert "def _expert_lora_is_shared" in glm5_adapter_text
    assert "def _expand_shared_expert_lora" in glm5_adapter_text
    assert "def _expert_lora_representation" in glm5_adapter_text
    assert '"expert_lora_representation": _expert_lora_representation' in glm5_adapter_text
    assert '"shared_local_expert_group"' in glm5_adapter_text
    assert "Adapter metadata expert_lora_representation" in glm5_adapter_text
    assert "can only import PEFT adapters" in glm5_adapter_text
    assert "Unexpected adapter tensor keys" in glm5_adapter_text
    assert '"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"' in glm5_adapter_text
    assert "self_attn.{module}.{suffix}.weight" in glm5_adapter_text
    assert "mlp.experts.{expert_idx}.{module}.{suffix}.weight" in glm5_adapter_text
    assert "mlp.shared_experts.{module}.{suffix}.weight" in glm5_adapter_text
    assert "ADAPTER_META_FORMAT as _ADAPTER_META_FORMAT" in glm5_adapter_text
    assert "megatron.lite_glm5_lora_peft_v1" in glm5_adapter_semantics_text
    assert "export_lora_adapter_state" in glm5_protocol_text
    assert "initialize_lora_olora_tail" in glm5_protocol_text
    assert "save_lora_adapter" in glm5_protocol_text
    assert "load_lora_adapter_state" in glm5_protocol_text
    assert "load_lora_adapter" in glm5_protocol_text

    assert "test_olora_tail_factors_use_tail_subspace_without_singular_values" in glm5_adapter_runtime_test
    assert "test_initialize_lora_olora_tail_marks_modules_and_blocks_double_init" in glm5_adapter_runtime_test
    assert "test_glm5_protocol_initialize_lora_olora_tail_without_te" in glm5_adapter_runtime_test
    assert (
        "test_initialize_lora_olora_tail_reaches_mtp_transformer_layers"
        in glm5_adapter_runtime_test
    )
    assert "test_initialize_lora_olora_tail_supports_single_local_routed_expert" in glm5_adapter_runtime_test
    assert "test_initialize_lora_olora_tail_supports_per_expert_grouped_lora" in glm5_adapter_runtime_test
    assert (
        "test_initialize_lora_olora_tail_skips_multi_local_shared_routed_experts"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_initialize_lora_olora_tail_fsdp_style_wrapper_without_te"
        in glm5_adapter_runtime_test
    )
    assert "test_lora_adapter_helpers_import_without_transformer_engine" in glm5_adapter_runtime_test
    assert "test_lora_adapter_state_guard_rejects_base_weight_keys" in glm5_adapter_runtime_test
    assert "test_glm5_shared_routed_expert_lora_round_trips_without_te" in glm5_adapter_runtime_test
    assert (
        "test_glm5_load_lora_adapter_state_reports_missing_and_unexpected_keys_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_shared_routed_expert_lora_save_load_metadata_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_lora_adapter_protocol_wrappers_round_trip_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_lora_adapter_fsdp_style_wrappers_round_trip_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_shared_routed_expert_rslora_round_trips_metadata_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_load_lora_adapter_validates_metadata_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_load_lora_adapter_rejects_config_mismatches_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_load_lora_adapter_rejects_config_metadata_disagreement_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_load_lora_adapter_validates_expected_config_without_adapter_config"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_load_lora_adapter_rejects_non_adapter_artifact_keys_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_lora_adapter_rejects_unsupported_target_modules_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_lora_adapter_rejects_unexported_target_modules_without_te"
        in glm5_adapter_runtime_test
    )
    assert (
        "test_glm5_lora_adapter_rejects_unsupported_parallel_scopes_without_te"
        in glm5_adapter_runtime_test
    )
    assert '"/definitely/missing/glm5_adapter"' in glm5_adapter_runtime_test
    assert '"glm5_unsupported_save"' in glm5_adapter_runtime_test
    assert "assert not save_dir.exists()" in glm5_adapter_runtime_test
    assert (
        "test_glm5_lora_adapter_round_trips_all_linear_with_mtp_without_te"
        in glm5_adapter_runtime_test
    )
    assert "test_glm5_lora_adapter_round_trips_all_linear_with_mtp" in glm5_adapter_runtime_test
    assert 'pytest.importorskip("torch")' in glm5_adapter_runtime_test
    assert 'pytest.importorskip("safetensors.torch")' in glm5_adapter_runtime_test
    assert 'pytest.importorskip("transformer_engine.pytorch")' in glm5_adapter_runtime_test
    assert '"target_modules": "all-linear"' in glm5_adapter_runtime_test
    assert "base_model.model.model.layers.2.self_attn.q_a_proj.lora_A.weight" in glm5_adapter_runtime_test
    assert "base_model.model.model.layers.2.mlp.gate_proj.lora_B.weight" in glm5_adapter_runtime_test
    assert "base_model.model.model.layers.2.mlp.up_proj.lora_B.weight" in glm5_adapter_runtime_test
    assert 'init_lora_weights="olora_tail"' in glm5_adapter_runtime_test
    assert '"megatron.lite_glm5_lora_peft_v1"' in glm5_adapter_runtime_test


def test_router_replay_static_contract():
    router_replay_text = _read("primitive", "modules", "router_replay.py")
    router_text = _read("primitive", "modules", "router.py")
    moe_text = _read("primitive", "utils", "moe.py")
    recompute_text = _read("primitive", "recompute.py")
    modules_init_text = _read("primitive", "modules", "__init__.py")
    protocol_utils_text = _read("model", "protocol_utils.py")
    data_contract_text = _read("runtime", "contracts", "data.py")
    mlite_engine_text = (
        ROOT.parents[1] / "examples" / "verl" / "verl_mlite" / "engine" / "mlite_engine.py"
    ).read_text()
    glm5_text = _read("model", "glm5", "lite", "model.py")
    glm5_protocol_text = _read("model", "glm5", "lite", "protocol.py")
    qwen_text = _read("model", "qwen3_moe", "lite", "model.py")
    qwen_protocol_text = _read("model", "qwen3_moe", "lite", "protocol.py")
    router_replay_runtime_test = (
        ROOT.parents[1] / "tests" / "unit" / "primitive" / "test_router_replay_runtime.py"
    ).read_text()

    tree = ast.parse(router_replay_text)
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert {"RouterReplay", "RouterReplayAction"}.issubset(class_names)
    assert "RECORD" in router_replay_text
    assert "REPLAY_FORWARD" in router_replay_text
    assert "REPLAY_BACKWARD" in router_replay_text
    assert "global_router_replay_instances" in router_replay_text
    assert "weakref.ref(self)" in router_replay_text
    assert "def _live_global_instances" in router_replay_text
    assert "def get_routed_experts_dtype" in router_replay_text
    assert "max_expert_idx <= 255" in router_replay_text
    assert "max_expert_idx <= 32767" in router_replay_text
    assert "def get_replay_topk" in router_replay_text
    assert "def _validated_replay_indices" in router_replay_text
    assert "RouterReplay {description} shape mismatch" in router_replay_text
    assert "RouterReplay {description} are out of range" in router_replay_text
    assert "class RouterReplayTraceContract" in router_replay_text
    assert "def build_router_replay_trace_contract" in router_replay_text
    assert "def normalize_routed_experts_digest" in router_replay_text
    assert "ROUTER_REPLAY_TRACE_COLLECTION_PATHS" in router_replay_text
    assert "ROUTER_REPLAY_TRACE_SOURCES" in router_replay_text
    assert "ROUTER_REPLAY_LAYOUTS" in router_replay_text
    assert '"pass_router_replay_trace_contract"' in router_replay_text
    assert '"paper_exact": False' in router_replay_text
    assert '"claims_paper_results": self.claims_paper_results' in router_replay_text
    assert "RouterReplayTraceContract is a local no-launch contract" in router_replay_text
    assert "routed_experts_segments traces must use router_replay_layout='true_tokens'" in router_replay_text
    assert "def action_context" in router_replay_text
    assert "self.replay_backward_enabled = False" in router_replay_text
    assert "def enable_replay_backward_queue" in router_replay_text
    assert "save_for_backward: bool | None = None" in router_replay_text
    assert "save_for_backward = self.replay_backward_enabled" in router_replay_text
    assert "def _snapshot_index_tensor" in router_replay_text
    assert "self.replay_backward_list.append(self._snapshot_index_tensor(topk_indices))" in router_replay_text
    assert "self.replay_backward_list.pop(0)" in router_replay_text
    assert "scores.gather(1, indices)" in router_replay_text

    assert "router_replay: Any | None = None" in moe_text
    assert "router_replay.get_replay_topk" in moe_text
    assert "if fused and replay_active:" in moe_text
    assert "fused = False" in moe_text

    assert "routed_experts: torch.Tensor | list[torch.Tensor] | None" in data_contract_text
    assert "def router_replay_context" in protocol_utils_text
    assert 'extras.get("router_replay_action")' in protocol_utils_text
    assert "RouterReplayAction.REPLAY_FORWARD" in protocol_utils_text
    assert "record_routed_experts" in protocol_utils_text
    assert "def padded_routed_experts_to_list" in protocol_utils_text
    assert "def concat_routed_experts_segments" in protocol_utils_text
    assert "multi-turn/agentic R3 traces" in protocol_utils_text
    assert "[batch, seq, routers, topk]" in protocol_utils_text
    assert "get_routed_experts_dtype(max_expert_idx)" in protocol_utils_text
    assert "import hashlib" in protocol_utils_text
    assert "def routed_experts_digest" in protocol_utils_text
    assert "def routed_experts_segments_digest" in protocol_utils_text
    assert "mlite-router-replay-routed-experts-v1" in protocol_utils_text
    assert "hashlib.sha256()" in protocol_utils_text
    assert 'to(device="cpu", dtype=torch.int64)' in protocol_utils_text
    assert "sha256:{hasher.hexdigest()}" in protocol_utils_text
    assert '"routed_experts_digest"' in protocol_utils_text
    assert '"routed_experts_segments_digest"' in protocol_utils_text
    assert "_routed_experts_to_list" in protocol_utils_text
    assert "RouterReplayAction.REPLAY_BACKWARD" in protocol_utils_text
    assert "save_for_backward=(" in protocol_utils_text
    assert "ROLL/SGLang-style" in data_contract_text
    assert "[batch, seq, routers, topk]" in data_contract_text
    assert "routed_experts = self._routed_experts_for_packing(micro_batch)" in mlite_engine_text
    assert "routed_experts=routed_experts" in mlite_engine_text
    assert "extras=self._router_replay_extras(" in mlite_engine_text
    assert "seq_lens=seq_lens" in mlite_engine_text
    assert "def _routed_experts_for_packing" in mlite_engine_text
    assert "def _compact_routed_experts_for_packing" in mlite_engine_text
    assert "routed_experts.values(), context=\"nested routed_experts values\"" in mlite_engine_text
    assert "get_routed_experts_dtype(max_expert_id)" in mlite_engine_text
    assert "must contain non-negative expert ids" in mlite_engine_text
    assert "_ROUTER_REPLAY_LAYOUT_ALIASES" in mlite_engine_text
    assert "def _normalize_router_replay_layout" in mlite_engine_text
    assert "requires router_replay_layout" in mlite_engine_text
    assert "true_tokens, full_padded, or cp_local" in mlite_engine_text
    assert "_ROUTER_REPLAY_TRACE_COLLECTION_PATHS" in mlite_engine_text
    assert "_ROUTER_REPLAY_TRACE_SOURCES" in mlite_engine_text
    assert "def _required_router_replay_trace_metadata" in mlite_engine_text
    assert "router_replay_trace_collection_path when replay payload is provided" in mlite_engine_text
    assert "router_replay_trace_source={expected_source!r}" in mlite_engine_text
    assert "{key}=True when replay payload is provided" in mlite_engine_text
    assert "def _router_replay_extras" in mlite_engine_text
    assert '"router_replay_action",' in mlite_engine_text

    assert "_replay_routers" in recompute_text
    assert "router.enable_replay_backward_queue()" in recompute_text
    assert "RouterReplayAction.REPLAY_BACKWARD" in recompute_text
    assert "previous_actions" in recompute_text

    assert "router_replay: RouterReplay | None = None" in router_text
    assert "self.router_replay = router_replay" in router_text
    assert "router_replay=self.router_replay" in router_text

    assert '"RouterReplay"' in modules_init_text
    assert '"RouterReplayAction"' in modules_init_text
    assert '"RouterReplayTraceContract"' in modules_init_text
    assert '"build_router_replay_trace_contract"' in modules_init_text
    assert '"normalize_routed_experts_digest"' in modules_init_text
    assert '"get_routed_experts_dtype"' in modules_init_text
    assert "router_replay: RouterReplay | None = None" in glm5_text
    assert "router_replay=self.router_replay" in glm5_text
    assert "def set_router_replay_data" in glm5_text
    assert "def set_router_replay_action" in glm5_text
    assert "def clear_router_replay_action" in glm5_text
    assert "def clear_router_replay_indices" in glm5_text
    assert "def recorded_routed_experts" in glm5_text
    assert "RouterReplayAction.RECORD" in glm5_text
    assert 'output["routed_experts"] = routed_experts' in glm5_text
    assert "router_replay_context" in glm5_protocol_text
    assert "with router_replay_context(model, batch):" in glm5_protocol_text
    assert "router_replay: bool = False" in glm5_protocol_text
    assert "router_replay=impl_cfg.router_replay" in glm5_protocol_text

    assert "RouterReplay(layer_idx=layer_idx)" in qwen_text
    assert "def set_router_replay_data" in qwen_text
    assert "def set_router_replay_action" in qwen_text
    assert "def clear_router_replay_action" in qwen_text
    assert "def clear_router_replay_indices" in qwen_text
    assert "def recorded_routed_experts" in qwen_text
    assert "RouterReplayAction.RECORD" in qwen_text
    assert 'output["routed_experts"] = routed_experts' in qwen_text
    assert "router_replay_context" in qwen_protocol_text
    assert "with router_replay_context(model, batch):" in qwen_protocol_text
    assert "router_replay: bool = False" in qwen_protocol_text
    assert "router_replay=impl_cfg.router_replay" in qwen_protocol_text

    assert "test_router_replay_records_and_replays_forward_and_backward_indices" in router_replay_runtime_test
    assert "test_router_replay_compact_dtype_matches_roll_reference_bounds" in router_replay_runtime_test
    assert "test_padded_routed_experts_to_list_compacts_roll_style_batch" in router_replay_runtime_test
    assert (
        "test_concat_routed_experts_segments_matches_roll_multi_turn_contract"
        in router_replay_runtime_test
    )
    assert "test_routed_experts_digest_canonicalizes_supported_layouts" in (
        router_replay_runtime_test
    )
    assert "test_routed_experts_segments_digest_matches_concatenated_contract" in (
        router_replay_runtime_test
    )
    assert "test_router_replay_trace_contract_builds_audit_safe_extras" in (
        router_replay_runtime_test
    )
    assert "test_router_replay_trace_contract_rejects_unsafe_or_inconsistent_metadata" in (
        router_replay_runtime_test
    )
    assert "test_router_replay_rejects_missing_bad_shape_and_out_of_range_replay_data" in router_replay_runtime_test
    assert "test_router_replay_context_sets_data_actions_and_cleans_up" in router_replay_runtime_test
    assert "test_router_replay_context_rejects_bad_actions_and_conflicting_data" in router_replay_runtime_test
    assert 'pytest.importorskip("torch")' in router_replay_runtime_test
    assert "RouterReplayAction.RECORD" in router_replay_runtime_test
    assert "RouterReplayAction.REPLAY_FORWARD" in router_replay_runtime_test
    assert "RouterReplayAction.REPLAY_BACKWARD" in router_replay_runtime_test
    assert (
        "test_router_replay_context_handles_multi_router_list_and_tensor_layouts"
        in router_replay_runtime_test
    )
    assert (
        "test_router_replay_context_accepts_padded_batch_routed_experts"
        in router_replay_runtime_test
    )
    assert (
        "test_router_replay_context_explicit_backward_replay_consumes_saved_indices"
        in router_replay_runtime_test
    )
    assert (
        "test_router_replay_checkpoint_recompute_switches_to_backward_replay"
        in router_replay_runtime_test
    )
    assert "save_for_backward=True" in router_replay_runtime_test
    assert "assert router.replay_backward_list == []" in router_replay_runtime_test
    assert "scores.gather(1, replay_indices)" in router_replay_runtime_test
    assert "router_replay_context(model, batch)" in router_replay_runtime_test
