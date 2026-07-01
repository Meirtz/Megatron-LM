# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared model modules owned by Megatron Lite."""

from __future__ import annotations

_EXPORTS = {
    "ALPHA_SCALING_RULES": (
        "megatron.lite.primitive.modules.adapter_rank",
        "ALPHA_SCALING_RULES",
    ),
    "AdapterPolicyRegistry": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "AdapterPolicyRegistry",
    ),
    "AdapterResidencyManager": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "AdapterResidencyManager",
    ),
    "AdapterRevision": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "AdapterRevision",
    ),
    "AdapterPopulationMetricSchema": (
        "megatron.lite.primitive.modules.adapter_population",
        "AdapterPopulationMetricSchema",
    ),
    "AdapterPopulationPlan": (
        "megatron.lite.primitive.modules.adapter_population",
        "AdapterPopulationPlan",
    ),
    "AdapterVariantSpec": (
        "megatron.lite.primitive.modules.diversity_vote",
        "AdapterVariantSpec",
    ),
    "CAPACITY_COLLAPSE_MIN": (
        "megatron.lite.primitive.modules.adapter_memory",
        "CAPACITY_COLLAPSE_MIN",
    ),
    "CAPACITY_HIGH_ACCURACY_MAX": (
        "megatron.lite.primitive.modules.adapter_memory",
        "CAPACITY_HIGH_ACCURACY_MAX",
    ),
    "BaseDeployment": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "BaseDeployment",
    ),
    "BaseDeploymentSpec": (
        "megatron.lite.primitive.modules.adapter_population",
        "BaseDeploymentSpec",
    ),
    "ContextLearningPlan": (
        "megatron.lite.primitive.modules.context_learning",
        "ContextLearningPlan",
    ),
    "COLLABORATION_DISTINCT_ADAPTERS": (
        "megatron.lite.primitive.modules.diversity_vote",
        "COLLABORATION_DISTINCT_ADAPTERS",
    ),
    "COOL_STORED": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "COOL_STORED",
    ),
    "DEFAULT_BATCH_SIZES": (
        "megatron.lite.primitive.modules.adapter_rank",
        "DEFAULT_BATCH_SIZES",
    ),
    "DEFAULT_RANKS": (
        "megatron.lite.primitive.modules.adapter_rank",
        "DEFAULT_RANKS",
    ),
    "DEFAULT_REFERENCE_ALPHA": (
        "megatron.lite.primitive.modules.adapter_rank",
        "DEFAULT_REFERENCE_ALPHA",
    ),
    "DEFAULT_REFERENCE_RANK": (
        "megatron.lite.primitive.modules.adapter_rank",
        "DEFAULT_REFERENCE_RANK",
    ),
    "DEFAULT_SEEDS": (
        "megatron.lite.primitive.modules.adapter_rank",
        "DEFAULT_SEEDS",
    ),
    "DEFAULT_STEPS": (
        "megatron.lite.primitive.modules.adapter_rank",
        "DEFAULT_STEPS",
    ),
    "FIG14_ALGORITHM": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "FIG14_ALGORITHM",
    ),
    "FIG14_DATASET": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "FIG14_DATASET",
    ),
    "FIG14_EXPECTED_ABSOLUTE_DELTA": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "FIG14_EXPECTED_ABSOLUTE_DELTA",
    ),
    "FIG14_METRICS": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "FIG14_METRICS",
    ),
    "FIG14_MODEL_FAMILY": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "FIG14_MODEL_FAMILY",
    ),
    "FIG14_PAPER_AVERAGES": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "FIG14_PAPER_AVERAGES",
    ),
    "FIG14_PAPER_AVERAGE_ATOL": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "FIG14_PAPER_AVERAGE_ATOL",
    ),
    "FIG14_TARGET_MODULES": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "FIG14_TARGET_MODULES",
    ),
    "Fig14ArmContract": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "Fig14ArmContract",
    ),
    "Fig14OloraTailContract": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "Fig14OloraTailContract",
    ),
    "PolicyRecord": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "PolicyRecord",
    ),
    "PolicySession": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "PolicySession",
    ),
    "PAPER_RANK_REGIME_RUN_COUNT": (
        "megatron.lite.primitive.modules.adapter_rank",
        "PAPER_RANK_REGIME_RUN_COUNT",
    ),
    "PAPER_RANK1_OLORA_TAIL_RUN_COUNT": (
        "megatron.lite.primitive.modules.adapter_rank",
        "PAPER_RANK1_OLORA_TAIL_RUN_COUNT",
    ),
    "RANK1_INIT_VALUES": (
        "megatron.lite.primitive.modules.adapter_rank",
        "RANK1_INIT_VALUES",
    ),
    "Rank1OloraTailArm": (
        "megatron.lite.primitive.modules.adapter_rank",
        "Rank1OloraTailArm",
    ),
    "RankRegimeArm": (
        "megatron.lite.primitive.modules.adapter_rank",
        "RankRegimeArm",
    ),
    "RankSweepContract": (
        "megatron.lite.primitive.modules.adapter_rank",
        "RankSweepContract",
    ),
    "RsLoraTransferArm": (
        "megatron.lite.primitive.modules.adapter_rank",
        "RsLoraTransferArm",
    ),
    "ServingResidency": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "ServingResidency",
    ),
    "WARM_CPU": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "WARM_CPU",
    ),
    "DeltaMemConfig": (
        "megatron.lite.primitive.modules.delta_mem",
        "DeltaMemConfig",
    ),
    "DeltaMemAttentionCorrection": (
        "megatron.lite.primitive.modules.delta_mem",
        "DeltaMemAttentionCorrection",
    ),
    "DeltaMemAttentionCorrectionOutput": (
        "megatron.lite.primitive.modules.delta_mem",
        "DeltaMemAttentionCorrectionOutput",
    ),
    "DeltaMemGranularity": ("megatron.lite.primitive.modules.delta_mem", "DeltaMemGranularity"),
    "DeltaMemState": ("megatron.lite.primitive.modules.delta_mem", "DeltaMemState"),
    "DeltaMemStateManager": ("megatron.lite.primitive.modules.delta_mem", "DeltaMemStateManager"),
    "DeltaMemStatePolicy": ("megatron.lite.primitive.modules.delta_mem", "DeltaMemStatePolicy"),
    "DeltaMemStatePolicyOutput": (
        "megatron.lite.primitive.modules.delta_mem",
        "DeltaMemStatePolicyOutput",
    ),
    "DeltaMemWriteProjection": (
        "megatron.lite.primitive.modules.delta_mem",
        "DeltaMemWriteProjection",
    ),
    "DeltaMemWriteTensors": (
        "megatron.lite.primitive.modules.delta_mem",
        "DeltaMemWriteTensors",
    ),
    "EXPECTED_STAGE_ORDER": (
        "megatron.lite.primitive.modules.context_learning",
        "EXPECTED_STAGE_ORDER",
    ),
    "Experts": ("megatron.lite.primitive.modules.experts", "Experts"),
    "GatedDeltaNet": ("megatron.lite.primitive.modules.gated_delta_net", "GatedDeltaNet"),
    "GQAttention": ("megatron.lite.primitive.modules.gqa", "GQAttention"),
    "HOT_GPU": (
        "megatron.lite.primitive.modules.adapter_lifecycle",
        "HOT_GPU",
    ),
    "LinearModuleSpec": (
        "megatron.lite.primitive.modules.adapter_memory",
        "LinearModuleSpec",
    ),
    "LORA_ADAPTER_TARGET": (
        "megatron.lite.primitive.modules.context_learning",
        "LORA_ADAPTER_TARGET",
    ),
    "LORA_ADAPTER": (
        "megatron.lite.primitive.modules.adapter_population",
        "LORA_ADAPTER",
    ),
    "LoraMemoryPlan": (
        "megatron.lite.primitive.modules.adapter_memory",
        "LoraMemoryPlan",
    ),
    "MAJORITY_VOTE": (
        "megatron.lite.primitive.modules.diversity_vote",
        "MAJORITY_VOTE",
    ),
    "MajorityVoteResult": (
        "megatron.lite.primitive.modules.diversity_vote",
        "MajorityVoteResult",
    ),
    "MTPBlock": ("megatron.lite.primitive.modules.mtp", "MTPBlock"),
    "MTPDecoderLayer": ("megatron.lite.primitive.modules.mtp", "MTPDecoderLayer"),
    "MTPLossAutoScaler": ("megatron.lite.primitive.modules.mtp", "MTPLossAutoScaler"),
    "MoEAuxLossAutoScaler": ("megatron.lite.primitive.modules.moe", "MoEAuxLossAutoScaler"),
    "MultimodalRotaryEmbedding": (
        "megatron.lite.primitive.modules.mrope",
        "MultimodalRotaryEmbedding",
    ),
    "RouterReplay": ("megatron.lite.primitive.modules.router_replay", "RouterReplay"),
    "RouterReplayAction": (
        "megatron.lite.primitive.modules.router_replay",
        "RouterReplayAction",
    ),
    "RouterReplayTraceContract": (
        "megatron.lite.primitive.modules.router_replay",
        "RouterReplayTraceContract",
    ),
    "QueryOnlyAdapterUpdatePlan": (
        "megatron.lite.primitive.modules.context_learning",
        "QueryOnlyAdapterUpdatePlan",
    ),
    "QueryOnlyInferenceContract": (
        "megatron.lite.primitive.modules.context_learning",
        "QueryOnlyInferenceContract",
    ),
    "QueryOnlyPayload": (
        "megatron.lite.primitive.modules.context_learning",
        "QueryOnlyPayload",
    ),
    "QueryOnlyRolloutRecord": (
        "megatron.lite.primitive.modules.context_learning",
        "QueryOnlyRolloutRecord",
    ),
    "RL_STYLE_POLICY_UPDATE": (
        "megatron.lite.primitive.modules.context_learning",
        "RL_STYLE_POLICY_UPDATE",
    ),
    "REPETITION_SINGLE_ADAPTER": (
        "megatron.lite.primitive.modules.diversity_vote",
        "REPETITION_SINGLE_ADAPTER",
    ),
    "PopulationEvaluationArm": (
        "megatron.lite.primitive.modules.adapter_population",
        "PopulationEvaluationArm",
    ),
    "SHARED_BASE": (
        "megatron.lite.primitive.modules.adapter_population",
        "SHARED_BASE",
    ),
    "SharedBaselinePolicy": (
        "megatron.lite.primitive.modules.adapter_population",
        "SharedBaselinePolicy",
    ),
    "StructuralScalingArm": (
        "megatron.lite.primitive.modules.adapter_population",
        "StructuralScalingArm",
    ),
    "TeacherContextScoringPayload": (
        "megatron.lite.primitive.modules.context_learning",
        "TeacherContextScoringPayload",
    ),
    "TeacherContextScoringRequest": (
        "megatron.lite.primitive.modules.context_learning",
        "TeacherContextScoringRequest",
    ),
    "VoteCollectionArm": (
        "megatron.lite.primitive.modules.diversity_vote",
        "VoteCollectionArm",
    ),
    "VoteRecord": (
        "megatron.lite.primitive.modules.diversity_vote",
        "VoteRecord",
    ),
    "UserPolicyRecord": (
        "megatron.lite.primitive.modules.adapter_population",
        "UserPolicyRecord",
    ),
    "assert_context_learning_plan": (
        "megatron.lite.primitive.modules.context_learning",
        "assert_context_learning_plan",
    ),
    "adapter_population_invariants": (
        "megatron.lite.primitive.modules.adapter_population",
        "adapter_population_invariants",
    ),
    "alpha_for_scaling_rule": (
        "megatron.lite.primitive.modules.adapter_rank",
        "alpha_for_scaling_rule",
    ),
    "analyze_fig14_olora_tail_results": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "analyze_fig14_olora_tail_results",
    ),
    "analyze_rank1_olora_tail_results": (
        "megatron.lite.primitive.modules.adapter_rank",
        "analyze_rank1_olora_tail_results",
    ),
    "analyze_rank_regime_results": (
        "megatron.lite.primitive.modules.adapter_rank",
        "analyze_rank_regime_results",
    ),
    "analyze_rslora_lr_transfer_results": (
        "megatron.lite.primitive.modules.adapter_rank",
        "analyze_rslora_lr_transfer_results",
    ),
    "build_context_learning_plan": (
        "megatron.lite.primitive.modules.context_learning",
        "build_context_learning_plan",
    ),
    "build_fig14_olora_tail_contract": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "build_fig14_olora_tail_contract",
    ),
    "build_adapter_variants": (
        "megatron.lite.primitive.modules.diversity_vote",
        "build_adapter_variants",
    ),
    "build_adapter_population_plan": (
        "megatron.lite.primitive.modules.adapter_population",
        "build_adapter_population_plan",
    ),
    "build_population_evaluation_arms": (
        "megatron.lite.primitive.modules.adapter_population",
        "build_population_evaluation_arms",
    ),
    "build_vote_collection_arms": (
        "megatron.lite.primitive.modules.diversity_vote",
        "build_vote_collection_arms",
    ),
    "build_router_replay_trace_contract": (
        "megatron.lite.primitive.modules.router_replay",
        "build_router_replay_trace_contract",
    ),
    "get_routed_experts_dtype": (
        "megatron.lite.primitive.modules.router_replay",
        "get_routed_experts_dtype",
    ),
    "normalize_routed_experts_digest": (
        "megatron.lite.primitive.modules.router_replay",
        "normalize_routed_experts_digest",
    ),
    "normalize_router_replay_layout": (
        "megatron.lite.primitive.modules.router_replay",
        "normalize_router_replay_layout",
    ),
    "normalize_router_replay_trace_collection_path": (
        "megatron.lite.primitive.modules.router_replay",
        "normalize_router_replay_trace_collection_path",
    ),
    "normalize_router_replay_trace_source": (
        "megatron.lite.primitive.modules.router_replay",
        "normalize_router_replay_trace_source",
    ),
    "SigmoidTopKRouter": ("megatron.lite.primitive.modules.router", "SigmoidTopKRouter"),
    "SwiGLUMLP": ("megatron.lite.primitive.modules.mlp", "SwiGLUMLP"),
    "TokenDispatcher": ("megatron.lite.primitive.modules.dispatcher", "TokenDispatcher"),
    "TopKRouter": ("megatron.lite.primitive.modules.router", "TopKRouter"),
    "_AllToAll": ("megatron.lite.primitive.modules.moe", "_AllToAll"),
    "delta_mem_read": ("megatron.lite.primitive.modules.delta_mem", "delta_mem_read"),
    "delta_mem_write": ("megatron.lite.primitive.modules.delta_mem", "delta_mem_write"),
    "eq2_alpha_squared_over_rank": (
        "megatron.lite.primitive.modules.adapter_rank",
        "eq2_alpha_squared_over_rank",
    ),
    "build_lora_memory_plan": (
        "megatron.lite.primitive.modules.adapter_memory",
        "build_lora_memory_plan",
    ),
    "build_rank1_olora_tail_arms": (
        "megatron.lite.primitive.modules.adapter_rank",
        "build_rank1_olora_tail_arms",
    ),
    "build_rank_regime_arms": (
        "megatron.lite.primitive.modules.adapter_rank",
        "build_rank_regime_arms",
    ),
    "build_rank_shift_plans": (
        "megatron.lite.primitive.modules.adapter_memory",
        "build_rank_shift_plans",
    ),
    "build_rank_sweep_contract": (
        "megatron.lite.primitive.modules.adapter_rank",
        "build_rank_sweep_contract",
    ),
    "build_rslora_lr_transfer_arms": (
        "megatron.lite.primitive.modules.adapter_rank",
        "build_rslora_lr_transfer_arms",
    ),
    "build_target_module_ablation_plans": (
        "megatron.lite.primitive.modules.adapter_memory",
        "build_target_module_ablation_plans",
    ),
    "capacity_efficiency": (
        "megatron.lite.primitive.modules.adapter_memory",
        "capacity_efficiency",
    ),
    "classify_capacity_efficiency": (
        "megatron.lite.primitive.modules.adapter_memory",
        "classify_capacity_efficiency",
    ),
    "context_learning_invariants": (
        "megatron.lite.primitive.modules.context_learning",
        "context_learning_invariants",
    ),
    "accuracy_from_majority_votes": (
        "megatron.lite.primitive.modules.diversity_vote",
        "accuracy_from_majority_votes",
    ),
    "lora_trainable_params": (
        "megatron.lite.primitive.modules.adapter_memory",
        "lora_trainable_params",
    ),
    "lora_runtime_scale": (
        "megatron.lite.primitive.modules.adapter_rank",
        "lora_runtime_scale",
    ),
    "mlite_lora_overrides": (
        "megatron.lite.primitive.modules.adapter_rank",
        "mlite_lora_overrides",
    ),
    "normalize_delta_mem_config": (
        "megatron.lite.primitive.modules.delta_mem",
        "normalize_delta_mem_config",
    ),
    "normalize_delta_mem_state_policy": (
        "megatron.lite.primitive.modules.delta_mem",
        "normalize_delta_mem_state_policy",
    ),
    "paper_reference_target_order": (
        "megatron.lite.primitive.modules.adapter_memory",
        "paper_reference_target_order",
    ),
    "paper_reference_population_contract": (
        "megatron.lite.primitive.modules.adapter_population",
        "paper_reference_population_contract",
    ),
    "paper_reference_fig14_contract": (
        "megatron.lite.primitive.modules.adapter_reproduction",
        "paper_reference_fig14_contract",
    ),
    "paper_reference_rank_regime_contract": (
        "megatron.lite.primitive.modules.adapter_rank",
        "paper_reference_rank_regime_contract",
    ),
    "majority_vote": (
        "megatron.lite.primitive.modules.diversity_vote",
        "majority_vote",
    ),
    "paper_reference_accuracy": (
        "megatron.lite.primitive.modules.diversity_vote",
        "paper_reference_accuracy",
    ),
    "paper_reference_vote_contract": (
        "megatron.lite.primitive.modules.diversity_vote",
        "paper_reference_vote_contract",
    ),
    "rank_sweep_invariants": (
        "megatron.lite.primitive.modules.adapter_rank",
        "rank_sweep_invariants",
    ),
    "validate_adapter_population_plan": (
        "megatron.lite.primitive.modules.adapter_population",
        "validate_adapter_population_plan",
    ),
    "validate_rank_sweep_contract": (
        "megatron.lite.primitive.modules.adapter_rank",
        "validate_rank_sweep_contract",
    ),
    "select_target_group_modules": (
        "megatron.lite.primitive.modules.adapter_memory",
        "select_target_group_modules",
    ),
    "split_grouped_qkvg": ("megatron.lite.primitive.modules.gqa_utils", "split_grouped_qkvg"),
    "trainable_params_for_modules": (
        "megatron.lite.primitive.modules.adapter_memory",
        "trainable_params_for_modules",
    ),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "ALPHA_SCALING_RULES",
    "AdapterPolicyRegistry",
    "AdapterPopulationMetricSchema",
    "AdapterPopulationPlan",
    "AdapterResidencyManager",
    "AdapterRevision",
    "AdapterVariantSpec",
    "BaseDeployment",
    "BaseDeploymentSpec",
    "CAPACITY_COLLAPSE_MIN",
    "CAPACITY_HIGH_ACCURACY_MAX",
    "ContextLearningPlan",
    "COLLABORATION_DISTINCT_ADAPTERS",
    "COOL_STORED",
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_RANKS",
    "DEFAULT_REFERENCE_ALPHA",
    "DEFAULT_REFERENCE_RANK",
    "DEFAULT_SEEDS",
    "DEFAULT_STEPS",
    "DeltaMemConfig",
    "DeltaMemAttentionCorrection",
    "DeltaMemAttentionCorrectionOutput",
    "DeltaMemGranularity",
    "DeltaMemState",
    "DeltaMemStateManager",
    "DeltaMemStatePolicy",
    "DeltaMemStatePolicyOutput",
    "DeltaMemWriteProjection",
    "DeltaMemWriteTensors",
    "EXPECTED_STAGE_ORDER",
    "Experts",
    "FIG14_ALGORITHM",
    "FIG14_DATASET",
    "FIG14_EXPECTED_ABSOLUTE_DELTA",
    "FIG14_METRICS",
    "FIG14_MODEL_FAMILY",
    "FIG14_PAPER_AVERAGES",
    "FIG14_PAPER_AVERAGE_ATOL",
    "FIG14_TARGET_MODULES",
    "Fig14ArmContract",
    "Fig14OloraTailContract",
    "GatedDeltaNet",
    "GQAttention",
    "HOT_GPU",
    "LinearModuleSpec",
    "LORA_ADAPTER",
    "LORA_ADAPTER_TARGET",
    "LoraMemoryPlan",
    "MAJORITY_VOTE",
    "MajorityVoteResult",
    "MTPBlock",
    "MTPDecoderLayer",
    "MTPLossAutoScaler",
    "MoEAuxLossAutoScaler",
    "MultimodalRotaryEmbedding",
    "PAPER_RANK1_OLORA_TAIL_RUN_COUNT",
    "PAPER_RANK_REGIME_RUN_COUNT",
    "PolicyRecord",
    "PolicySession",
    "PopulationEvaluationArm",
    "QueryOnlyAdapterUpdatePlan",
    "QueryOnlyInferenceContract",
    "QueryOnlyPayload",
    "QueryOnlyRolloutRecord",
    "RL_STYLE_POLICY_UPDATE",
    "REPETITION_SINGLE_ADAPTER",
    "RANK1_INIT_VALUES",
    "Rank1OloraTailArm",
    "RankRegimeArm",
    "RankSweepContract",
    "RouterReplay",
    "RouterReplayAction",
    "RouterReplayTraceContract",
    "RsLoraTransferArm",
    "SHARED_BASE",
    "SigmoidTopKRouter",
    "ServingResidency",
    "SharedBaselinePolicy",
    "StructuralScalingArm",
    "SwiGLUMLP",
    "TeacherContextScoringPayload",
    "TeacherContextScoringRequest",
    "VoteCollectionArm",
    "VoteRecord",
    "UserPolicyRecord",
    "WARM_CPU",
    "accuracy_from_majority_votes",
    "adapter_population_invariants",
    "alpha_for_scaling_rule",
    "analyze_fig14_olora_tail_results",
    "analyze_rank1_olora_tail_results",
    "analyze_rank_regime_results",
    "analyze_rslora_lr_transfer_results",
    "assert_context_learning_plan",
    "build_adapter_population_plan",
    "build_adapter_variants",
    "build_context_learning_plan",
    "build_fig14_olora_tail_contract",
    "build_population_evaluation_arms",
    "build_router_replay_trace_contract",
    "build_vote_collection_arms",
    "get_routed_experts_dtype",
    "normalize_routed_experts_digest",
    "normalize_router_replay_layout",
    "normalize_router_replay_trace_collection_path",
    "normalize_router_replay_trace_source",
    "split_grouped_qkvg",
    "TokenDispatcher",
    "TopKRouter",
    "_AllToAll",
    "build_lora_memory_plan",
    "build_rank1_olora_tail_arms",
    "build_rank_regime_arms",
    "build_rank_shift_plans",
    "build_rank_sweep_contract",
    "build_rslora_lr_transfer_arms",
    "build_target_module_ablation_plans",
    "capacity_efficiency",
    "classify_capacity_efficiency",
    "context_learning_invariants",
    "delta_mem_read",
    "delta_mem_write",
    "eq2_alpha_squared_over_rank",
    "lora_trainable_params",
    "lora_runtime_scale",
    "majority_vote",
    "mlite_lora_overrides",
    "normalize_delta_mem_config",
    "normalize_delta_mem_state_policy",
    "paper_reference_accuracy",
    "paper_reference_fig14_contract",
    "paper_reference_population_contract",
    "paper_reference_rank_regime_contract",
    "paper_reference_target_order",
    "paper_reference_vote_contract",
    "rank_sweep_invariants",
    "select_target_group_modules",
    "trainable_params_for_modules",
    "validate_adapter_population_plan",
    "validate_rank_sweep_contract",
]
