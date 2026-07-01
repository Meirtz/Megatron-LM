# MLite LoRA / OLoRA-tail 适配执行计划

状态日期：2026-06-21。

本文是 `experimental/lite/docs/glm5_lora_adaptation_plan_zh.md` 的执行版摘要：
它把 MLite 当前 LoRA/OLoRA-tail 状态拆成三条线：

1. 基本复现：先证明本地代码、checkpoint 映射、adapter lifecycle 和小规模训练步是可信的。
2. 认知对齐：把论文、MindLab GLM5.2 参考实现和 MLite 设计里的概念边界统一。
3. 完整功能适配：补齐 runtime、VERL、FSDP2、分布式和 paper-facing reproduction 所需证据。

本文不声明论文复现已经完成。当前已经有不少工程证据，但精确 Fig.14 仍缺真实 DAPO
RL 训练、500-step 双臂结果、baseline、评测输出和用户确认的 GPU/SLURM 运行证据。

## 0. 当前状态快照

### 0.1 已固定参考

- MindLab GLM5.2 Megatron fork：
  `references/external/Megatron-GLM5.2`，commit
  `e08610def780d9deba2778f1dbe268c0bca4d791`。这个 commit 是 GLM5
  base architecture 参考，覆盖 DSA、MLA、MTP、CP/THD、MoE dispatcher 和
  pipeline schedule；它不是直接的 PEFT LoRA adapter 实现。
- HF GLM-5.2 snapshot：
  `references/external/hf-zai-org-GLM-5.2/config.json`。
- 精确 Fig.14 目标模型 snapshot：
  `references/external/hf-deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B`。
  该模型是 dense Qwen2：`model_type="qwen2"`、
  `architectures=["Qwen2ForCausalLM"]`。
- 论文笔记：
  `peft-mint-repro/references/main/insights.md`。
- 已准备的相关 repo / reference：
  `references/external/DAPO`、`references/external/verl-recipe`、
  `peft-mint-repro`、`references/external/Megatron-GLM5.2`。
- ROLL R3 reference：
  `references/external/ROLL`，commit
  `7f9d4d3873a87679c8d070085a4bd3bdeea7e589`。该 repo 当前提供
  SGLang rollout + Megatron train 的 Router Replay (R3) 端到端参考；
  R2 尚未实现，R3 + sequence packing 在 ROLL 中明确不支持。当前分析记录在
  `runs/20260620-mlite-olora-tail-rl-repro/roll_r3_reference_analysis.md`。

### 0.2 当前已具备的工程证据

- GLM5 LoRA 单 GPU B300A TE/CUDA suite 已通过：
  `runs/20260619-mlite-glm5-lora-te-gpu-smoke` 覆盖 preflight、TE adapter
  round trip、DSA dependency、LoRA forward/backward/optimizer/save-load、
  Router Replay record/replay 和 OLoRA-tail init。后续 fused-indexer rerun 也证明
  GLM5-like 32-index-head shape 可走默认 fused indexer。
- Dense Qwen2 精确模型族已经进入 MLite 最小路径：
  `qwen2` registry identity、TP=1 runtime slice、LoRA freeze/forward/backward、
  adapter save/load/import/export、OLoRA-tail post-load init、HF checkpoint
  pack/unpack 映射都已具备本地门禁。
- DeepSeek-R1-Distill-Qwen-1.5B 真实 checkpoint 已验证：
  339/339 BF16 tensors 匹配，q/k/v bias 到 native fused qkv bias 的
  pack/unpack 被检查，full BF16 model load smoke 已能通过 Qwen2 protocol 加载真实权重。
- Qwen2 tiny local contract 已能产生标准 LoRA 与 OLoRA-tail 两条本地训练步证据：
  有 finite loss、非零 LoRA gradient、base grad absent、optimizer step、非零 adapter
  delta 和 adapter sidecar。
- Fig.14 exact-route run 包现在有受保护的数据准备契约：
  `prepare_fig14_data.sh` dry-run 会记录 DAPO-Math-17k 与 AIME-2024 parquet
  计划路径，`validate_fig14_data.py` 可在训练文件系统里把 parquet 存在性、
  指定 HF source URL/provenance、manifest sha256 和 schema/sample 检查升级为硬门禁。
- Fig.14 result package 现在也有受保护的数据校验证据链：
  `fig14-results.template.json`、collection manifest、collector、analyzer、finalizer、
  run-bundle preparer/validator 和 repro spec 都必须保留
  `validate_fig14_data.py` 的 `data_validation` 证据；`readiness_check.py` 和
  `adaptation_phase_matrix.py` 都会拒绝 status-only 的伪通过。
- Fig.14 exact-route entrypoint 现在会显式选择 VERL 的 MLite actor：
  `hydra.searchpath=[pkg://verl_mlite.config]`、
  `actor@actor_rollout_ref.actor=mlite_actor`、
  `actor_rollout_ref.actor.engine.model_name=qwen2`、Qwen2 dist-opt、LoRA
  sidecar checkpoint override，以及 `data.seed`、`trainer.seed`、
  `actor_rollout_ref.actor.data_loader_seed`、`actor_rollout_ref.actor.engine.seed`
  已被 `audit_fig14_verl_mlite_config.py` dry-run 审计。
- 同一个 exact-route entrypoint 的真实 launch 分支现在会在 `ray job submit`
  之前强制检查 `DATASET_PATH`、`TEST_FILE`、HF config/checkpoint、`PYTHON_BIN`
  和 `DATA_VALIDATOR`，写出 prelaunch data manifest，并执行
  `validate_fig14_data.py --require-ready-files --require-source-urls
  --require-manifest-sha256`。这把 AIME validation parquet、指定 HF source
  provenance、manifest sha256 和 DAPO schema/sample 检查放到了真正启动前的硬门禁里。
- `test_fig14_entrypoint_prelaunch_guards.py` 已把这些 real-branch 守卫变成
  负向本地 gate：缺确认、缺 Ray 地址、缺 train/test parquet、缺 validator、
  以及坏 parquet 都会在 `ray job submit` 前失败，并用 fake `ray` marker 证明
  没有触发 Ray。
- Fig.14 数据现在已在本地文件系统通过 strict ready 校验：
  `fig14-data-validation.require-ready.json` 为 `status=pass_ready_data`，
  train parquet 为 `/Users/lmei/verl/data/dapo-math-17k.parquet`
  (`sha256=534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941`)，
  test parquet 为 `/Users/lmei/verl/data/aime-2024.parquet`
  (`sha256=12154e38a716d12db5731f9a022ae69a610c4f7d0e0dcc04e902887a686877e7`)。
  这只证明本地数据 artifact ready；如果真实训练迁移到 cluster / 训练文件系统，
  仍需在目标 namespace 重新执行同一套 `validate_fig14_data.py
  --require-ready-files --require-source-urls --require-manifest-sha256`。
- GLM5/Qwen3-MoE adapter lifecycle 已有大量 CPU gate：
  LoRA-only safetensors、strict/non-strict load、metadata JSON 边界、rsLoRA
  scaling convention、FSDP-style unwrap、DTensor materialize/load、per-expert
  grouped LoRA、shared local expert fold/unfold 等。
- FSDP2 adapter roundtrip run 包已准备：
  `runs/20260620-mlite-glm5-fsdp2-adapter-roundtrip`，但尚未启动真实 GPU。
- ROLL/R3 已完成 no-GPU 适配闭环：
  MLite 本地 Router Replay primitive、ROLL/SGLang padded trace
  `[batch, seq, routers, topk]` 到 MLite true-token/list layout 的转换、
  multi-turn/agentic routed-experts segment concat、VERL/MLite worker 和 engine
  helper、SGLang-only R3 launcher guard、stale trace suppression、R2/vLLM+R3
  early failure、standard PPO balance 后重新校验 R3 trace metadata、standard PPO
  同时校验 direct `routed_experts` 与 `routed_experts_segments` trace metadata、
  标准 PPO 对同一 batch 同时携带 direct `routed_experts` 与
  `routed_experts_segments` 的混合 source-of-truth 早失败、
  标准 PPO 对 `routed_experts_digest` 做 batch 内 uniform 与 sha256 格式校验、
  标准 PPO 对只剩 `router_replay_trace_*`/digest 但缺少 trace payload 的
  metadata-only batch 早失败、
  logprob/ref、reward-model、critic-value 和 reward-extra infos 都不覆盖 rollout
  trace、KL/correction/advantage 等 post-reward transforms 不丢弃 rollout trace、
  rollout-correction bypass mode 不改写 R3 trace source、fully-async/separation
  trainer 在 REMAX baseline、generation union、reward/logprob/ref/critic union、
  reward extras 和 KL/correction/advantage 路径上保留 rollout trace、static audit
  和 dry-run preflight 都已进入 `runs/20260620-mlite-olora-tail-rl-repro`。
- R3 checkpoint/run handoff 工具已准备：
  synthetic GSM8K parquet、MLite LoRA/R3 local-gate recorder、bridge-smoke
  analyzer、structured event contract validator、real-smoke bundle
  collector/validator、model-path validator、GLM5 checkpoint materialization
  planner、local MoE checkpoint discovery、materialized checkpoint intake、
  launch readiness checker 和 phase matrix 聚合门禁都已落盘；launch readiness
  现在也要求 `mlite-lora-local-gates.json=pass_local_gates`，phase matrix
  和 deliverable audit 都显式校验 R3/SGLang、sequence-packing rejection、
  padding bridge、trace-conflict early-fail 和 vLLM+R3 hard-fail 等关键
  local-gate markers。
- 当前三阶段 phase matrix：
  `mlite-lora-r3-adaptation-phase-matrix.json` 为
  `status=in_progress_blocked_by_external_artifacts`，
  `basic_reproduction=pass`、`cognition_alignment=pass`、
  `full_functional_adaptation=blocked`；`basic_reproduction` 现在显式包含
  `fig14_result_data_validation_contract=pass`，不再只通过
  `exact_fig14_full_code_path_supported` 间接吸收这条证据。R3 real-smoke
  完成项现在必须同时满足 analyzer `status=pass` 和 launch readiness
  `status=pass_r3_real_smoke`。

### 0.3 当前缺口

- 没有真实 DeepSeek-R1-Distill-Qwen-1.5B GPU train-step 证据。
- Fig.14 数据 artifact 已在本地 ready，但这不是论文复现证据：
  `fig14-data-validation.require-ready.json` 已通过 parquet 存在性、指定
  HF source URL/provenance、manifest sha256、正 size、schema/sample 检查。
  如果真实训练在集群或不同挂载路径运行，仍要在目标训练 namespace 重新校验
  DAPO-Math-17k/AIME parquet，并把对应 validation JSON 写入 run bundle。
- Dense Qwen2 MLite protocol 已补上 VERL/RL dist-opt 协议路径：
  `audit_fig14_verl_mlite_config.py` 现在记录
  `qwen2_optimizer.ready_for_verl_rl_training=true`，
  exact-route command contract 会显式传入
  `+actor_rollout_ref.actor.engine.impl_cfg.optimizer=dist_opt`。
  该 audit 还会对 standard LoRA 与 OLoRA-tail 两个 dry-run arm 同时硬检查
  Fig.14 paper recipe knobs：DeepSeek-R1-Distill-Qwen-1.5B、Qwen2、
  DAPO-Math-17k、500 steps、LR 1e-5、constant schedule、effective batch 32、
  rank 16、alpha 32、dropout 0、`use_rslora=false`、以及
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` 全 attention+MLP
  projection target。两个 arm 只允许在 `lora_init=standard/olora_tail` 上分叉。
  这只表示代码路径和 dry-run 配置已对齐，仍不是 GPU/RL 或论文复现证据。
- 没有 standard LoRA vs OLoRA-tail 的 500-step 双臂 paper-scale 结果。
- 没有 GSM8K、MATH500、AIME22/23/24/25 的评测产物。
- 没有 dense Qwen2 分布式 runtime / RL evidence。
- rsLoRA 目前只有 primitive formula/local gate 和 no-GPU sweep contract：
  `rslora-rank-sweep-matrix.json` 已固化三种 alpha 规则、27 个
  LR-transfer arms、Qwen3-8B 9 ranks × 4 batch × 6 seeds = 216 个
  rank-regime arms、以及 rank-1 standard/OLoRA-tail 对照 arms；但它仍
  `paper_exact=false`，缺 Figs.16-18 精确 LR grid、真实 Qwen3 GPU 训练、
  adapter deltas、RL metrics 和 eval outputs。
- GLM5 完整路径仍缺真实 FSDP2、EP、PP、CP/THD default backend 的最终强证据。
- 没有本地 launch-ready 的 MoE R3 checkpoint：
  `references/external/hf-zai-org-GLM-5.2` config-ready，但 282 个
  safetensors shard 仍是 Git LFS pointer；本地 dense
  DeepSeek-R1-Distill-Qwen-1.5B 权重完整，但 dense Qwen2 不是 R3 路线。
- GLM5 weight materialization 当前已具备 `git-lfs/3.7.1` 工具，但仍缺显式用户确认和足够
  存储空间；planner 估算需要约 1.51TB 原始权重，含 buffer 约 1.58TB，而当前目标路径可用空间
  约 0.46TB。
- 已新增 no-download storage probe：`roll-r3-glm5-storage-targets.json` 当前显示已探测 7 个
  本机/Volumes 候选，`ready_candidate_count=0`，因此 GLM5 权重物化前必须先选择新的充足存储路径。
- GLM5 weight materialization 已有 no-download guarded plan：真实物化只能在
  `CONFIRM_WEIGHT_DOWNLOAD=yes`、`git lfs version`、磁盘空间检查都通过后执行；
  物化脚本结束后会重新运行
  `validate_roll_r3_model_path.py --model-path <materialized-checkpoint> --require-weight-assets`，
  并把 `post_materialization_validation` 作为进入 R3 launch readiness 前的 source of truth。
- 对新存储目标，materialization planner 已支持
  `--model-path /target/hf-zai-org-GLM-5.2 --metadata-model-path references/external/hf-zai-org-GLM-5.2`：
  当前 pointer snapshot 只提供 config/index/size metadata，生成的脚本会在目标路径做 guarded
  `git clone --filter=blob:none` + `git lfs pull`，仍必须等显式确认后才可执行。
- External readiness probe 已接入 storage-target scan：没有 ready candidate 时
  `glm5_weight_materialization` 保持 blocked；如果后续插入/选择了充足存储并重新 probe，它只会升级到
  `ready_for_user_confirmation`，真实下载仍由 `CONFIRM_WEIGHT_DOWNLOAD=yes` 单独授权。
- Start request 也区分这两个状态：当前 `ready_candidate_count=0` 时推荐
  `select_storage_then_probe`；只有 storage probe 找到可用候选后，才会建议
  `confirm_weight_download_if_desired`。
- R3 model-path gate 现在对 `.safetensors` 做轻量 header 合法性检查，而不是只看
  文件非空；prelaunch bundle validator 也要求 `pending_weight_assets` /
  `pass_ready_model_path` 产物保留 LFS pointer、missing shard、
  `invalid_safetensors_*`、本地大小和 index-too-small 诊断字段，避免坏权重资产在
  bundle 层被折叠成模糊的 pending 状态。
- 这些权重资产诊断现在也被提升为显式 prelaunch contract marker：
  `model_path_weight_asset_diagnostics_checked`，默认 R3 bundle 和 GLM5 proxy
  bundle 都必须验证该 marker 后才可进入 readiness 聚合。
- R3 model-path gate 现在同时报告 `expected_router_replay_instances_without_mtp`
  与 `expected_router_replay_instances_with_mtp_if_enabled`。当前 GLM-5.2
  config 为 main MoE routers `75`，若 `mtp_enable` 在真实 bundle / train-spec /
  evidence summary 中显式开启，则期望 router replay instances 为 `76`；否则
  real-smoke evidence extractor 仍按 without-MTP 口径检查。
- 没有用户确认后的 MLite + SGLang R3 real-smoke result package。
- 完整功能适配仍被 phase matrix 阻塞在外部 artifact / 真实运行证据：
  real-checkpoint GPU/RL train step、materialized MoE checkpoint、
  GLM5 weight materialization、R3 real smoke 和 paper-scale 双臂结果。

## 1. 基本复现阶段

目标：用最小成本证明每个关键机制可独立成立；任何通过项都只按其证据等级命名。

### 1.1 No-GPU reference audit

动作：

- 固定 MindLab commit 与 HF snapshots，记录 changed-file audit。
- 对 GLM5 HF config 审计这些字段：
  `index_topk_freq`、`index_skip_topk_offset`、`indexer_types`、
  `index_share_for_mtp_iteration`、`q_lora_rank`、`kv_lora_rank`、
  `num_nextn_predict_layers`、`scoring_func`、`topk_method`。
- 对 HF weight map 审计：
  full DSA layer 是否有 `self_attn.indexer.*`，shared layer 是否没有；
  MTP appended layer 是否在 `model.layers.78.*`；`eh_proj` 是否只作为 checkpoint state。
- 对论文 Fig.14 knobs 审计：
  DeepSeek-R1-Distill-Qwen-1.5B、DAPO-Math-17k、500 steps、LR 1e-5、
  effective batch 32、rank 16、alpha 32、dropout 0、`use_rslora=false`、
  all attention + MLP projections；standard LoRA 与 OLoRA-tail dry-run arm 的
  共同 knobs 必须一致，只能在 `lora_init` 上分叉。

验收：

- 静态 gate 明确区分：
  `q_lora_rank/kv_lora_rank` 是 MLA base architecture，不是 PEFT target。
- MindLab commit 被标注为架构参考，不被当成 LoRA adapter source。
- Fig.14 精确目标被标注为 dense Qwen2 路线，不用 `qwen2_moe` 或 GLM5 proxy 替代。

推荐命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/readiness_check.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/discover_fig14_inputs.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/adaptation_phase_matrix.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/audit_roll_r3_mlite.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/audit_adaptation_deliverables.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/roll_r3_sglang_smoke_preflight.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/discover_roll_r3_moe_checkpoints.py \
  --root references/external
```

当前 `adaptation_phase_matrix.py` 预期会以非零退出码结束，因为
`full_functional_adaptation=blocked` 是真实状态；判断时以写出的 JSON 和 stdout 为准，
不要把这个非零退出码解读为脚本崩溃。

### 1.2 Local static / CPU gates

动作：

- 跑 `tests/run_glm5_lora_local_gates.sh`，覆盖 GLM5、Qwen3-MoE、Qwen2、
  adapter lifecycle、OLoRA-tail、Router Replay、VERL sidecar dry-run。
- 跑 Qwen2 exact local contracts，生成 standard LoRA 与 OLoRA-tail tiny train-step 证据。
- 跑 Fig.14 collector/finalizer self-test，确认后续 paper result package 的 schema
  不会空跑通过。
- 跑 `prepare_rslora_rank_sweep_matrix.py`，确认 rsLoRA/rank-regime 复现矩阵
  不会把 alpha scaling convention、rank grid、batch/seed grid 或 no-claim
  safety boundary 混掉。
- 跑 readiness/phase-matrix self-test，确认 Fig.14 result data-validation chain
  不能被顶层 status 或缺失 child check 伪造通过。

验收：

- local gate 通过。
- `record_mlite_lora_local_gates.py` 生成 `mlite-lora-local-gates.json`，
  固化 `run_glm5_lora_local_gates.sh` 的最新本地执行证据；当前为
  `status=pass_local_gates`，`pass_line_count=178`。新增 gate 显式覆盖
  dense Qwen2 OLoRA-tail 的 smallest-singular-vector/no-singular-scaling
  语义，以及 `alpha=c*sqrt(rank)` 下 rsLoRA scale 跨 rank 保持常数的语义。
- Qwen2 两条 tiny contract 都是 `pass_local_contract`。
- adapter delta、LoRA grad、base grad absent、sidecar files 都存在。
- `rslora-rank-sweep-matrix.json` 为 `status=pass_rank_sweep_contract`，
  其中 `sqrt_rank_rslora_eq2_factor_constant=true`、
  `sqrt_rank_rslora_runtime_scale_constant=true`、
  `rank_regime_matches_paper_216_shape=true`，同时保持
  `claims_paper_results=false`。
- `claims_exact_fig14_reproduction=false` 保持为 false。

推荐命令：

```bash
PYTHONDONTWRITEBYTECODE=1 TORCH_PYTHON_BIN=/Users/lmei/anaconda3/envs/cua/bin/python \
  bash experimental/lite/tests/run_glm5_lora_local_gates.sh

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mlite/experimental/lite \
  bash runs/20260620-mlite-olora-tail-rl-repro/run_qwen2_exact_local_contracts.sh

PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/collect_fig14_results.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/finalize_fig14_claim.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/validate_fig14_data.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/readiness_check.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/adaptation_phase_matrix.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/audit_fig14_verl_mlite_config.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/test_fig14_entrypoint_prelaunch_guards.py
FIG14_DATA_DRY_RUN=1 VERL_HOME=/home/lmei/verl \
  bash runs/20260620-mlite-olora-tail-rl-repro/prepare_fig14_data.sh
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/validate_fig14_data.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/audit_fig14_verl_mlite_config.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/prepare_roll_r3_synthetic_data.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/validate_roll_r3_synthetic_data.py
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/prepare_roll_r3_checkpoint_materialization.py \
  --model-path references/external/hf-zai-org-GLM-5.2 \
  --output-dir runs/20260620-mlite-olora-tail-rl-repro/roll-r3-glm5-checkpoint-materialization
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/validate_roll_r3_checkpoint_materialization.py \
  --plan runs/20260620-mlite-olora-tail-rl-repro/roll-r3-glm5-checkpoint-materialization/checkpoint-materialization-plan.json
```

### 1.3 单 GPU smoke

动作：

- GLM5 路线：
  用 B300A/GB300 跑 preflight、TE adapter、DSA deps、gpu_forward、router_replay、
  olora_tail；优先默认 fused indexer，fallback 只能作为诊断或 layout 证据。
- Qwen2 精确路线：
  在真实 GPU 上加载 DeepSeek-R1-Distill-Qwen-1.5B，执行短 train step，
  记录 finite loss、LoRA grad、adapter delta、checkpoint_loaded=true、
  adapter sidecar 和 analyzer 结果。

验收：

- GLM5 smoke analyzer `status=pass`，且 backend/fallback 被记录。
- Qwen2 real-checkpoint train-step smoke 通过，但只称为 exact-route smoke；
  仍不能称为 Fig.14 reproduction。
- 所有运行都落在 `runs/<run-id>/`，有 `plan.md`、`train-spec.json`、
  launcher、monitor、status 和 analyzer。

边界：

- 任何 GPU allocation、`srun`、`sbatch` 或长时间运行都必须先得到用户明确确认。

### 1.4 分布式基础 smoke

动作：

- FSDP2 TP=1 adapter roundtrip：
  preflight -> adapter roundtrip -> OLoRA-tail。
- CP/THD Router Replay：
  true-token replay data 到 padded/CP-local token grain 的转换。
- PP boundary：
  合法 DSA IndexShare split 通过，跨 PP source layer 早失败。
- EP Router Replay：
  record/replay 对 local expert dispatch 的 identity smoke。

验收：

- FSDP2 证明 LoRA 参数是 DTensor-like，保存/加载后 re-export drift 为 0。
- CP/THD analyzer 记录 backend、fallback call-count、target shape、replay match。
- EP/PP 不仅有构建通过，还要有明确成功路径或失败边界。

## 2. 认知对齐阶段

目标：避免“能跑但语义错”。这一阶段决定实现和报告的语言。

### 2.1 概念边界

必须保持以下边界：

- PEFT LoRA 是 frozen base 上的持久 adapter state；只能保存 LoRA A/B 和 metadata。
- GLM5 `q_lora_rank`、`kv_lora_rank` 是 MLA 架构低秩投影，不是 PEFT adapter rank。
- OLoRA-tail 是 post-load initialization：
  `B0=U_tail`、`A0=V_tail.T`，使用最小 singular-vector subspace，不乘 singular value。
- rsLoRA 是 scaling convention：
  legacy 为 `alpha/rank`，rsLoRA 为 `alpha/sqrt(rank)`。
- R3 Router Replay 解决 MoE sparse-path TIM：record rollout expert ids，
  training forward/recompute verbatim replay。
- DSA IndexShare 解决 attention top-k 复用，不是 Router Replay，也不是 LoRA。
- MinT/lifecycle 语义里，adapter revision 是 policy identity；router trace 是 rollout
  artifact；DSA/MLA/MTP config 是 base model config。

### 2.2 Adapter lifecycle 不变量

实现必须满足：

- `adapter_model.safetensors` 只能包含 `lora_A.weight` / `lora_B.weight`。
- `adapter_config.json` 与 `megatron.lite_adapter_meta.json` 都必须是标准 JSON object。
- rank、alpha、dropout、target_modules、use_rslora、scale、scaling convention、
  init history、base identity 都要能重建实验配置。
- `strict=False` 只能忽略支持 surface 上的额外 LoRA tensor，不能吞掉 base weight、
  unknown projection、DSA indexer、lm_head 或跨模型 key。
- target_modules 支持 PEFT aliases 和 `all-linear`，但不能把 `eh_proj`、DSA indexer、
  embedding、LM head 放进默认 target。
- 新 adapter 目录写入必须 atomic；失败不能留下半写目录。
- 真实 FSDP2/DTensor 情况下，export 必须 materialize full tensor，load 必须按目标
  mesh/placements distribute 后整参复制。

### 2.3 模型语义对齐矩阵

| 机制 | GLM5/GLM5.2 路线 | Dense Qwen2 Fig.14 路线 | 验收口径 |
|---|---|---|---|
| LoRA target | DSA attention、MLP、routed/shared expert、MTP transformer | qkv/o/gate-up/down dense projections | `all-linear` 展开后只含支持 projection |
| 架构低秩 | `q_lora_rank/kv_lora_rank` 属于 MLA | 无 GLM5 MLA 低秩字段 | 不进入 adapter target |
| OLoRA-tail | post-load，支持 unsharded/DTensor-aware paths，sharded 路径显式边界 | TP=1 支持 fused qkv/o/gate-up/down | 记录 init metadata 和 changed tensors |
| R3 | MoE router record/replay/recompute | Dense Qwen2 不需要 R3 | GLM5/Qwen3 MoE 才做 TIM 证据 |
| DSA IndexShare | full/shared source layer，PP split 校验 | 不适用 | zero-loss path 可训练 |
| Paper Fig.14 | proxy engineering route | 精确 reproduction route | 不能混报 |

### 2.4 报告语言

- Local static / CPU gate：称为 local contract evidence。
- 单 GPU GLM5 smoke：称为 MLite engineering smoke。
- Qwen2 real-checkpoint short train step：称为 exact-route smoke。
- GLM5/Qwen3-MoE OLoRA/R3 跑通：称为 proxy engineering evidence。
- 只有 Fig.14 双臂 500-step DAPO + eval + finalizer 通过，才称为 paper-facing reproduction。

## 3. 完整功能适配阶段

目标：从“局部能跑”变成“训练、保存、加载、分布式、后训练集成都可信”。

### 3.1 GLM5 LoRA injection 完整化

任务：

- 覆盖 DSA attention：
  `q_a_proj`、`q_b_proj`、`kv_a_proj_with_mqa`、`kv_b_proj`、`o_proj`。
- 覆盖 dense MLP：
  fused gate/up 和 down。
- 覆盖 MoE：
  routed experts、shared experts、per-expert grouped LoRA、shared-local expert fold/unfold。
- 覆盖 MTP transformer layers：
  使用 appended HF-style layer index。
- 保持 unsupported：
  DSA indexer、MTP `eh_proj`、embedding、LM head、未设计 gather/scatter 的 TP/ETP
  adapter save/load。

验收：

- `lora_config=None` 时行为完全不变。
- `freeze_non_lora_params` 后只有 adapter trainable。
- protocol `ModelBundle` 记录 trainable/frozen LoRA stats。
- CPU + TE/CUDA + FSDP2 smoke 都覆盖至少一个真实 target。

### 3.2 Adapter import/export 与 runtime API 完整化

任务：

- GLM5、Qwen3-MoE、Qwen2 各自保留模型原生 key map，不互相硬套。
- protocol 暴露：
  `export_lora_adapter_state`、`save_lora_adapter`、
  `load_lora_adapter_state`、`load_lora_adapter`。
- runtime `ModelHandle` 透传 adapter lifecycle helper，先校验 protocol、model_cfg、
  chunks、parallel_state。
- VERL checkpoint 支持 explicit LoRA adapter sidecar：
  `save_contents/load_contents=lora_adapter` 或 `save_lora_adapter=True`。
- SFT/GRPO launcher 暴露：
  `LORA_RANK`、`LORA_ALPHA`、`LORA_DROPOUT`、`LORA_TARGET_MODULES`、
  `LORA_USE_RSLORA`、`LORA_INIT`、`CHECKPOINT_SAVE_CONTENTS`、
  `CHECKPOINT_LOAD_CONTENTS`、`CHECKPOINT_SAVE_LORA_ADAPTER`、
  `LORA_ADAPTER_DIR_NAME`。

验收：

- adapter-only save/load 不改变普通 full checkpoint 默认行为。
- sidecar 路径不能绝对路径或 `..` traversal。
- sidecar kwargs 和 metadata 必须是 object/mapping。
- combined full checkpoint + adapter sidecar 要先预校验，避免 full checkpoint 副作用后才失败。

### 3.3 OLoRA-tail 完整化

任务：

- 继续保持 post-load API：
  不能在空 module construction 时初始化。
- 对 unsharded attention/MLP/shared expert/MTP transformer 和 per-expert grouped LoRA
  保持精确 SVD-tail。
- 对 DTensor/FSDP2 materialize full base weight，写回时用 whole-parameter copy。
- 对 TP/ETP-sharded weights 设计二选一：
  gather/init/scatter，或显式 documented local approximation。没有设计前保持 unsupported。
- 对大矩阵加入 memory guard 或 randomized SVD 设计，避免直接 full SVD 变成长尾炸点。

验收：

- 小矩阵数值 gate 证明不乘 singular value。
- real TE/CUDA 和 real FSDP2 smoke 证明 post-load pass 可用于真实 runtime。
- adapter metadata 记录 `init_lora_weights="olora_tail"`，load 不默认重跑 init。

### 3.4 R3 Runtime 完整化

当前状态：

- 已把 ROLL 的核心 R3 contract 落到 MLite/VERL 的 no-GPU 门禁：
  仅 `INFER_BACKEND=sglang` 可开启 R3，R2 和 vLLM+R3 在 launch 前失败。
- 重新检查 ROLL `SgLangStrategy` 后确认：ROLL 只可作为 R3 routed-experts
  采集/传输/重放的参考；其 SGLang 参数更新接口在 `is_lora=True` 时会触发
  `lora training is not supported with sglang` 断言。因此 MLite LoRA/OLoRA-tail
  的 adapter sidecar、导出/加载和真实 optimizer-step 证据不能从 ROLL SGLang
  LoRA 权重同步路径继承，必须继续由 MLite runtime/checkpoint/evidence 合同证明。
- 已支持 ROLL/SGLang padded routed-experts trace
  `[batch, seq, routers, topk]` 输入，转换为 MLite true-token/list layout 后喂给
  THD/CP per-router replay path。
- 已支持把 multi-turn/agentic R3 segments 沿 token 维合并：
  segment 可以是 padded trace、`[tokens, routers, topk]`、
  `[routers, tokens, topk]` 或 per-router list；3D 歧义 layout 会早失败。
- VERL/MLite engine bridge 已支持通过 non-tensor
  `routed_experts_segments` / `routed_experts_segment_seq_lens` /
  `routed_experts_num_routers` side-channel 接入 multi-turn trace，并仍然尊重
  `enable_routing_replay=False` 的 stale-trace 抑制。
- `enable_routing_replay=False` 会压制 stale trace，避免 reference/logprob 等路径误消费旧
  `routed_experts`。
- `roll_r3_sglang_smoke_preflight.py` 已覆盖 dry-run command、R2/vLLM+R3
  early failure、static R3 audit，以及 ROLL 参考中的 SGLang routed-experts
  capability contract：`sglang>=0.5.6.post3`、
  `enable_return_routed_experts=True`、`return_routed_experts=True`、rollout/train
  双侧 R3 对称开启、R3 时未启用 sequence packing。该 gate 仍不启动
  SGLang/Ray/GPU，也不代表真实 R3 smoke。
- VERL SGLang rollout server 现在也有对齐 ROLL 的 runtime guard：开启
  `enable_rollout_routing_replay` 时要求 `sglang>=0.5.6.post3`，并由 no-GPU
  静态 gate 检查 server args 里的 `enable_return_routed_experts=True`、每个
  request 的 `return_routed_experts=True`、`extract_routed_experts_from_meta_info`
  metadata extraction 以及 `TokenOutput.routed_experts` 返回路径。R3 开启后若
  SGLang response 没有带回 `routed_experts`，rollout 侧会直接 hard fail，不再把
  `None` trace 延迟到 train batch 才暴露。同时，`skip_tokenizer_init` 下的 raw
  `meta_info["routed_experts"]` 和 capturer 输出都会经过统一 helper 转为
  `[tokens, num_hidden_layers, num_experts_per_tok]`，并检查 integer dtype、非空、
  expert id 非负、expert id 小于 config 暴露的专家总数、topk/layer 维度和元素数自洽。
- VERL 标准 PPO rollout path 现在会在 `routed_experts` 进入训练 batch 后立即补齐并
  校验 R3 trace metadata：缺失字段按 `router_generate_request` / `full_padded`
  填充；已有字段必须在整个 batch 内 uniform，`router_replay_trace_source` 必须为
  `routed_experts`，trace collection path 必须属于已知保留 metadata 的路径，且
  `router_replay_trace_metadata_conflicts` 必须为空；如果携带
  `routed_experts_digest`，还必须 batch 内 uniform 且为 sha256 digest。这样普通 rollout path 也不会把
  ROLL 提醒的“先坏后好”或跨 chunk metadata 冲突延迟到 MLite engine 才暴露。
- `smoke_roll_r3_mlite_bridge.py` 与
  `analyze_roll_r3_mlite_bridge_smoke.py` 共同构成 no-GPU bridge contract：前者执行
  tensor/list/multi-turn side-channel helper，并显式绑定 MLite engine 的
  `_compact_routed_experts_for_packing()`、layout normalization 和 required trace
  metadata helper；后者强校验九个子检查和 no-launch / no-download 不变量，包括
  disabled stale segments 抑制、缺失 padded seq_lens 失败、negative expert-id
  segment 失败，以及 segment trace 经 `concat_routed_experts_segments` 转成 replay
  payload 后必须写出 `router_replay_layout=true_tokens` 且保留
  `router_replay_trace_source=routed_experts_segments`。
- `extract_roll_r3_real_smoke_evidence.py` 已作为未来真实 smoke 的离线 evidence
  extractor：从 bundle、train log、JSONL 和 adapter 目录生成 draft result；证据不完整时
  保持 `draft_incomplete`，完整时再交给 `analyze_roll_r3_real_smoke.py` 判定。
  extractor 也会记录所有已映射 evidence path 的多次观测冲突到
  `conflicting_observed_paths`，防止真实日志里先坏后好的值被静默覆盖。
- `roll-r3-real-smoke-result.template.json` 已显式预留
  `rollout.routed_experts_digest`、`batch_bridge.routed_experts_digest`、
  `training.routed_experts_digest` 和 `training.recompute_routed_experts_digest`。
  `validate_roll_r3_real_smoke_bundle.py --output-dir <bundle-dir>` 会检查这些模板路径都是 sha256 填写项，避免
  真实 smoke 采集前的 handoff 模板漏掉 collector/analyzer 已经强制要求的 digest 证据。
  同一个 bundle validator 还会对
  `roll-r3-real-smoke-events.template.jsonl` 与 result template 做 trace-contract
  parity 检查：两边都必须保留 `rollout.trace_collection_path`、
  `batch_bridge.trace_source`、`batch_bridge.router_replay_layout`、
  `packed_batch_extras_trace_keys` 和四个 routed-experts digest carrier，避免
  结构化 JSONL 与最终 result handoff 在 trace source/layout 字段上漂移。
  bundle validator 的 self-test 现在还覆盖负例：result template 缺
  `batch_bridge.trace_source`、events template 缺 `r3.training`、events template
  使用非 64-hex sha256 digest、或 events template 缺
  `batch_bridge.router_replay_layout` 都必须失败。
- R3 real-smoke analyzer 现在要求 adapter artifact 不只是 JSON 里列出文件名：
  `adapter_artifact.path` 必须是当前证据文件系统里的目录，三个 sidecar 文件必须存在且
  非空，`adapter_config.json` 和 `megatron.lite_adapter_meta.json` 必须能解析成 JSON object。
  collector 自测也覆盖空 `adapter_model.safetensors`、非法 JSON sidecar 和 JSON list
  sidecar 这三类负例，防止坏 adapter artifact 进入 extractor/analyzer。
- 两个 real-smoke draft result：
  `roll-r3-real-smoke-result.draft.json` 与
  `roll-r3-glm5-real-smoke-result.draft.json` 即使处于 `draft_incomplete`，也必须保留
  `lora_smoke.adapter_artifact.file_digests` schema；真实 pass 时该字段必须覆盖
  `adapter_model.safetensors`、`adapter_config.json` 和
  `megatron.lite_adapter_meta.json`。
- direct extractor 现在也有 adapter path fallback：如果未传 `--adapter-path`，但
  结构化日志提供了 `adapter_artifact_path` 或
  `lora_smoke.adapter_artifact_path`，`extract_roll_r3_real_smoke_evidence.py`
  会从该目录回填 `files`、`file_digests` 和 `num_tensors`。这不放宽正式
  collector 的 `ADAPTER_PATH` 要求，只是避免已有结构化证据中的 adapter path
  被丢弃。
- 正式 collector 现在先运行 `enrich_roll_r3_real_smoke_events.py`：它以真实
  JSONL 和 `ADAPTER_PATH` 为输入，生成 derived enriched JSONL，并把当前文件系统里
  三件 LoRA sidecar 的 `files`、`file_digests` 和 `num_tensors` 写入 canonical
  `lora_smoke.adapter_artifact`。如果原始 JSONL 已经记录 sidecar 字段但和磁盘
  artifact 不一致，enricher 会失败；因此这一步是 evidence normalization，
  不是放宽 analyzer 的 `adapter_artifact_recorded_file_digests_match=true` 要求。
- `prepare_roll_r3_real_smoke_bundle.py` 生成的 bundle 现在包含
  `collect-evidence.sh`：真实 run 结束后设置 `CONFIRM_REAL_SMOKE_EVIDENCE=yes`，
  并提供 `ADAPTER_PATH`，即可按 adapter sidecar precheck -> validator -> extractor ->
  analyzer 顺序收集结果；该 collector 默认拒绝 `template_only=true` 的模板 JSONL 事件，
  并在 analyzer 通过后刷新 launch readiness；只有 readiness 聚合到
  `pass_r3_real_smoke` 时 collector 才返回成功。collector 本身仍然不启动任何
  Ray/SGLang/GPU/SLURM/训练。
- `roll-r3-real-smoke-events.template.jsonl` 与
  `validate_roll_r3_real_smoke_events.py` 已定义未来真实 run 的结构化事件契约：
  rollout、batch bridge、training、LoRA smoke 和 metrics 五类事件必须覆盖 analyzer
  所需的 R3/LoRA 证据，包括 `enable_rollout_routing_replay=True`、train 侧
  `actor_engine_router_replay_mode="R3"`、`impl_router_replay_enabled=True` 和
  `sequence_packing_enabled=False`。
  模板合同本身要用 `--allow-template-events` 显式校验；真实 smoke JSONL 必须移除或设置
  `template_only=false`。
  launch-readiness gate 会把该事件合同标注为 `schema_template` 或
  `structured_real_events`；`schema_template` 只允许支持 prelaunch/launch-request
  readiness，不能支持 `pass_r3_real_smoke`。真实 smoke pass 必须同时有
  analyzer `status=pass` 和 event contract `structured_real_events`。phase matrix
  也不再接受 status-only 的事件合同伪通过。
  同时，五类 required events 必须恰好各出现一次；缺失或重复都会通过
  `missing_required_events`、`duplicate_required_events` 或 readiness 中的
  `missing_required_event_count`、`duplicate_required_event_count` 阻断 pass。
  readiness 现在还要求真实 smoke analyzer pass 带回
  `gpu_or_training_launched=true`、`starts_sglang=true` 和 boolean `submits_slurm`，
  避免把 no-launch/prelaunch JSON 误升级成 real-smoke pass。
  validator/analyzer/extractor 的 self-test 现在同时覆盖语义负向 case：字段存在但
  train-side R3 未开启、impl replay 为 false 或 sequence packing 为 true 时，必须判定失败；
  extractor self-test 还覆盖 structured JSONL 中的 adapter path fallback，确保
  `adapter_artifact_path` 能回填 sidecar `files`、`file_digests` 和 `num_tensors`。
  如果 `batch_bridge.trace_source=routed_experts_segments`，real-smoke analyzer
  和 structured event validator 还会强制
  `batch_bridge.router_replay_layout=true_tokens`，防止 concat 后的 true-token
  replay payload 被错误标成 padded layout；launch readiness、unblock runbook 和
  external evidence handoff 也会把这条作为 real-smoke ready 条件。
  生成的 `collect-evidence.sh` 和 bundle validator 同样会把
  `collect_evidence_requires_segment_layout_true_tokens` 作为 prelaunch contract，
  因而真实 evidence collection 不能绕过这条 segment/layout 约束。
  同一个 prelaunch contract 现在也显式要求
  `collect_evidence_requires_missing_trace_hard_fail_guard`：真实 evidence collection
  必须证明 R3 开启后缺失 rollout `routed_experts` 会在 rollout/training contract
  层硬失败，而不是静默 fallback 到普通 router。
  同一个 prelaunch contract 也要求
  `collect_evidence_requires_lora_adapter_contract`：真实 evidence collection
  必须让 analyzer 产出 `analysis.lora_adapter_contract`，并证明
  OLoRA-tail init、LoRA optimizer step、非零 adapter delta、三件 LoRA sidecar
  文件和 sidecar sha256 digest 都来自同一份 real-smoke 证据。
  同一个 prelaunch contract 也要求
  `collect_evidence_enriches_adapter_sidecar_event`：真实 evidence collection
  必须先从真实 JSONL 与 `ADAPTER_PATH` 生成带 canonical
  `lora_smoke.adapter_artifact` 的 derived enriched JSONL，再让 event validator、
  extractor 和 analyzer 消费该 evidence 文件。
  同一个 prelaunch contract 也要求
  `collect_evidence_supports_r3_carrier_aliases`：真实 evidence collection 必须能从
  `extra_fields`、`non_tensor_batch`、`PackedBatch.extras` 和 runtime batch extras
  这些真实 carrier 路径读取 `routed_experts_digest` 与 trace metadata，
  避免只支持理想化顶层字段。
  同一个 prelaunch contract 也要求
  `collect_evidence_rejects_trace_metadata_conflicts`：真实 evidence collection
  必须拒绝 `router_replay_trace_metadata_conflicts` 非空的 fully-async/partial-rollout
  trace，避免把跨 chunk 不一致的 rollout metadata 当成可 replay 证据。
  同时覆盖 rollout shape 负向 case：3D trace 必须是 `[tokens, routers, topk]`，
  4D trace 必须是 `[batch, seq, routers, topk]`，且报告的 `routers`、`topk`、
	  `batch_size` 和 `valid_tokens` 必须与 shape 自洽。
  `model.num_experts` 现在也是 real-smoke evidence contract 的一部分；
  `max_expert_id` 必须小于 `model.num_experts`，`topk` 必须不超过
  `model.num_experts`。
  真实 runtime 的 `rollout.sglang_version` 也必须记录并满足 `>=0.5.6.post3`，
  不能只依赖 ROLL 参考源码的静态版本检查。
  JSONL validator 还会检查必需 evidence path 的多次观测是否前后一致；
  如果坏值先出现、好值后来覆盖，仍必须失败；extractor 进一步对所有已映射证据路径
  记录 `conflicting_observed_paths`，并把冲突 draft 保持为 incomplete。

任务：

- GLM5/Qwen3 protocol `_forward_step` 贯通 `PackedBatch.routed_experts`。
- record/replay tensor 必须是 integer expert ids，形状/range/dtype 早校验。
- activation recompute wrapper 切换到 `REPLAY_BACKWARD` 并消费 forward 保存队列。
- true-token replay data 转换到 padded/THD/CP-local grain；已是 padded 或 CP-local 的调试数据
  必须通过 `batch.extras["router_replay_layout"]` 显式声明。
- MTP routers 顺序追加在 main routers 后。
- rollout engine capability check：
  SGLang 必须明确支持并返回 routed experts，否则 R3 hard fail；no-GPU preflight
  当前只验证 ROLL 参考契约和 MLite dry-run flags，真实支持仍需用户确认运行时证明。
- train step R3 enabled 时必须要求 `routed_experts` 存在，不能 silently fallback 到普通 router。
- materialized MoE checkpoint intake 必须先通过 model-path validator、
  real-smoke bundle validator 和 launch readiness checker，再请求用户确认真实运行。

验收：

- CPU primitive/context/model gates。
- `mlite-lora-local-gates.json` 通过，且 marker 覆盖 Qwen2 exact/standard
  LoRA、GLM5/Qwen3-MoE OLoRA-tail、Router Replay、VERL runtime 和 sidecar
  checkpoint helper；其中 trace-conflict local gate 必须包含
  `PASS r3_static:test_mlite_engine_r3_trace_metadata_conflicts_fail_before_consuming_payload`，
  证明 MLite bridge 会在消费 replay payload 前拒绝非空
  `router_replay_trace_metadata_conflicts`；standard PPO local gate 还必须包含
  `PASS r3_static:test_verl_ppo_trainer_r3_segment_trace_metadata_static`，
  证明标准 PPO 会把 `routed_experts_segments` side-channel 识别为 R3 source
  trace，并要求 `router_replay_trace_source=routed_experts_segments` 与
  `router_replay_layout=true_tokens`；该 marker 也覆盖同一 batch 同时携带
  direct `routed_experts` 与 `routed_experts_segments` 时的 source-of-truth
  冲突早失败；同时必须包含
  `PASS r3_static:test_verl_ppo_trainer_r3_metadata_without_payload_fails_static`，
  证明如果 batch 只保留 `router_replay_trace_*`、`routed_experts_digest`
  或 `router_replay_trace_metadata_conflicts`，但缺少 `routed_experts` /
  `routed_experts_segments` payload，标准 PPO 会在消费前失败；同时必须包含
  `PASS r3_static:test_verl_ppo_trainer_r3_logprob_outputs_do_not_override_rollout_trace_static`，
  证明 old-logprob/reference 输出不会覆盖 rollout `routed_experts`
  source-of-truth；同时必须包含
  `PASS r3_static:test_verl_ppo_trainer_r3_auxiliary_outputs_do_not_override_rollout_trace_static`，
  证明 reward-model 和 critic-value DataProto 输出不能回写或替换 rollout
  `routed_experts` / trace side-channel；同时必须包含
  `PASS r3_static:test_verl_ppo_trainer_r3_reward_extra_infos_do_not_override_rollout_trace_static`，
  证明 reward function 返回的 `reward_extra_infos_dict` 不能用
  `routed_experts_segments`、`routed_experts_digest` 或
  `router_replay_trace_*` 等 side-channel 覆盖 rollout trace；同时必须包含
  `PASS r3_static:test_verl_ppo_trainer_r3_post_reward_transforms_preserve_rollout_trace_static`，
  证明 `apply_kl_penalty`、`compute_rollout_correction_and_add_to_batch`
  和 `compute_advantage` 之后仍保留 rollout `routed_experts`、segments、
  digest、layout 和 `router_replay_trace_*` metadata；同时必须包含
  `PASS r3_static:test_verl_rollout_correction_bypass_mode_preserves_r3_trace_static`，
  证明 `algorithm.rollout_correction.bypass_mode=True` 只写入
  `old_log_probs = rollout_log_probs` 和 loss config，不会替换或删除
  `routed_experts`、`routed_experts_segments`、digest、layout 或
  `router_replay_trace_*` side channels；同时必须包含
  `PASS r3_static:test_verl_separation_trainer_r3_preserves_rollout_trace_static`，
  证明 fully-async/separation trainer 的 REMAX baseline、generation union、
  reward-model、old-logprob、reference-logprob、critic-value、reward extras、
  KL penalty、rollout correction 和 advantage 路径都保留 rollout R3 trace
  source-of-truth。
- `roll-r3-mlite-bridge-smoke-analysis.json` 通过，且覆盖
  `multi_turn_segment_side_channel` 与 padded segment `seq_lens` hard-fail。
- `roll-r3-real-smoke-events-validation.json` 通过，证明未来真实 smoke 的 JSONL 事件
  schema 能覆盖 extractor/analyzer 所需证据，并且五类 required events 采用
  exact-once 语义。
- 单 GPU record/replay + recompute。
- CP/THD world_size=2 layout smoke。
- EP local expert dispatch identity smoke。
- PP schedule 下 router order 稳定。
- 用户确认后的 MLite + SGLang + MoE checkpoint real smoke：
  response 中有 routed experts，training forward/recompute 消费同一 trace，
  extractor 生成完整 result candidate，analyzer 写出
  `roll-r3-real-smoke-analysis.json`；launch readiness 进入 `pass_r3_real_smoke`
  时还必须看到非模板 JSONL 的 `structured_real_events` 合同。

### 3.5 DSA / CP / THD / PP 完整化

任务：

- 默认 fused indexer 在 GLM5-like shape 上作为强证据。
- H100 torch sparse-attention fallback 只能作为 layout 证据，不替代默认 backend。
- nonzero `dsa_indexer_loss_coeff` 如果 fused wrapper 不能返回 loss 所需 top-k，
  保持 explicit `NotImplementedError`。
- PP split validator 在构建期阻断跨 stage source top-k 依赖。

验收：

- zero-loss DSA IndexShare training path forward/backward 通过。
- analyzer 强校验 backend、fallback call-count、shape、loss、replay target。

## 4. Paper-facing reproduction 阶段

目标：只在证据完整时复现论文 Fig.14；否则保持工程证据标签。

### 4.1 精确 Fig.14 route

必需输入：

- DAPO-Math-17k 数据路径。
- 真实 DAPO training entrypoint。
- DeepSeek-R1-Distill-Qwen-1.5B HF checkpoint。
- 两个 arm：
  standard LoRA 和 OLoRA-tail。
- 固定 knobs：
  500 steps、LR 1e-5、effective batch 32、rank 16、alpha 32、
  all attention + MLP projections、seed scope。

必需产物：

- 每个 arm 的 `train-result.json`：
  finite RL loss、optimizer steps、adapter delta、seed/config、init metadata。
- 每个 arm 的 adapter sidecar：
  `adapter_model.safetensors`、`adapter_config.json`、
  `megatron.lite_adapter_meta.json`。
- 每个 arm 的 eval：
  `GSM8K.json`、`MATH500.json`、`AIME22.json`、`AIME23.json`、
  `AIME24.json`、`AIME25.json`。
- `fig14-results.json` 和 finalizer 输出。

验收：

- standard LoRA 与 OLoRA-tail 使用相同数据、步数、LR、batch、rank、alpha、target、seed
  策略和 eval pipeline。
- analyzer 通过后才允许 `claims_exact_fig14_reproduction=true`。

推荐准备命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/probe_fig14_source_urls.py

PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/prepare_fig14_run_bundle.py \
  --train-entrypoint /path/to/dapo_train.sh \
  --implementation dapo \
  --dataset-path /path/to/DAPO-Math-17k \
  --seed 1234 \
  --require-existing-paths

PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/validate_fig14_run_bundle.py
```

### 4.2 Proxy route

用途：

- 在 GLM5/Qwen3-MoE 上验证 MLite OLoRA-tail、R3、adapter lifecycle、VERL sidecar、
  optimizer step 和 adapter delta 闭环。

边界：

- proxy route 只能支持工程结论：
  “MLite 的 OLoRA-tail/R3/adapter lifecycle 可以在 sparse/MoE path 上跑通”。
- proxy route 不能支持：
  “复现 Fig.14 OLoRA-tail 58.3 vs LoRA 56.3”。

验收：

- `proxy-smoke-result.json` 必须证明 finite loss、LoRA grad、optimizer step、
  adapter delta、OLoRA-tail metadata、adapter sidecar。
- proxy adapter sidecar 也必须是文件系统证据：adapter 目录存在，三件 sidecar
  文件存在且非空，两个 JSON sidecar 可解析且必须是 JSON object；proxy analyzer
  不再只相信 result JSON 里的文件名列表。
- `claims_exact_fig14_reproduction=false` 必须保持 false。

R3 推荐准备命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/intake_roll_r3_materialized_checkpoint.py \
  --route glm5 \
  --model-path /path/to/materialized/moe/checkpoint \
  --output-dir runs/20260620-mlite-olora-tail-rl-repro/roll-r3-materialized-checkpoint-bundle \
  --output runs/20260620-mlite-olora-tail-rl-repro/roll-r3-materialized-checkpoint-intake.json

PYTHONDONTWRITEBYTECODE=1 python3 runs/20260620-mlite-olora-tail-rl-repro/analyze_roll_r3_real_smoke.py \
  /path/to/user-confirmed/roll-r3-real-smoke-result.json \
  runs/20260620-mlite-olora-tail-rl-repro/roll-r3-real-smoke-analysis.json
```

`intake_roll_r3_materialized_checkpoint.py` 会内部调用 model-path validator、bundle
preparer/validator 和 launch-readiness checker，并显式传入 local LoRA gates、SGLang
R3 capability contract、bridge smoke、event contract 等 gate；它仍然只准备和聚合证据，不
启动真实运行。

## 5. 推荐执行里程碑

### M0: 资料和门禁归档

完成：

- reference audit。
- local gates。
- Qwen2 standard / OLoRA-tail local contracts。
- Fig.14 bundle/finalizer self-tests。
- adaptation deliverable audit，证明计划、`train-spec.json` 的 Fig.14 recipe、
  用户确认/GPU 边界、`train-spec.json` 与 `analysis.json` 当前 readiness、
  reference、ROLL/R3 分析、phase matrix 和 no-overclaim 边界互相一致。

输出：

- `analysis.json` 保持 `blocked_for_exact_paper_repro`，并明确缺什么。
- `mlite-lora-r3-adaptation-deliverable-audit.json` 为
  `status=pass_deliverable_audit`，同时仍保持 no-GPU/no-download/no-paper-claim。
- `fig14-no-gpu-preflight.json` 会刷新 phase matrix 并执行 deliverable audit；
  若 deliverable audit 失败，preflight 必须进入 hard failure。
- `mlite-lora-r3-adaptation-phase-matrix.json` 的 `cognition_alignment` 现在也
  直接要求 `deliverable_audit_contract=pass`，防止计划、`train-spec.json`、
  ROLL/R3 handoff、phase matrix 与 no-overclaim 边界之间出现 status-only 或滞后证据。
- `fig14_no_gpu_preflight.py` 在刷新 phase matrix 前后各运行一次 deliverable audit，
  cluster-side data handoff 也把 `audit_adaptation_deliverables.py` 放入 require-ready
  sequence；这些步骤仍然不下载数据/权重、不启动 GPU/Ray/SGLang/SLURM/训练。
- `mlite-lora-r3-status-consistency-audit.json` 现在作为 consolidated no-GPU
  preflight 的最终 hard gate，检查 phase matrix、deliverable audit、preflight、
  `analysis.json`、R3 readiness JSON 和 handoff 文档是否共享同一组当前 blocker，并
  继续保持 no-download/no-GPU/no-paper-claim 边界。该审计是 transition-aware 的：
  如果 Fig.14 data、paper-scale 结果、R3 materialized checkpoint、GLM5 storage/materialization
  readiness 或 R3 real-smoke evidence 未来变为 ready/pass，对应 blocker 必须从 phase
  matrix 移除，源 JSON 也必须同步进入匹配的 ready/pass 状态。
- `fig14_no_gpu_preflight.py` 也会独立校验 unblock runbook 的确认 marker 和
  artifacts-to-refresh 字段，避免只依赖 runbook 自身的 `status=pass`。
- `mlite-lora-r3-unblock-runbook.json/md` 由
  `prepare_adaptation_unblock_runbook.py` 生成，作为当前 phase-matrix blockers 到
  现有 guarded scripts、确认变量、ready/pass gates 和需刷新 artifacts 的索引。它本身
  不下载、不启动、不物化权重，只在 `status_consistency` 通过后生成可信 handoff。
  runbook gate 会机器校验每个 blocker entry 的 guarded steps、ready criteria、
  artifacts-to-refresh 和高风险确认 marker，包括 `CONFIRM_DATA_DOWNLOAD=yes`、
  `CONFIRM_WEIGHT_DOWNLOAD=yes`、`CONFIRM_GPU_RUN=yes`、
  `CONFIRM_REAL_SMOKE_EVIDENCE=yes`。
- `mlite-lora-r3-completion-audit.json/md` 由
  `audit_adaptation_completion.py` 生成，作为目标级最终验收入口：它把“基本复现、
  认知对齐、完整功能适配、Fig.14 复现、R3 real smoke、no-overclaim 边界”逐项映射到
  当前 JSON 证据。当前应保持 `status=in_progress_blocked_by_external_artifacts`、
  `ready_to_mark_goal_complete=false`；只有 phase matrix 完整通过、Fig.14 finalizer
  进入 `pass_paper_reproduction`、R3 launch readiness 进入 `pass_r3_real_smoke` 且无
  open blockers 时，才允许把目标级完成状态提升为 complete。
- `fig14_no_gpu_preflight.py` 现在会在 status consistency 和 unblock runbook 之后运行
  `audit_adaptation_completion.py`，把 `completion_audit_expected`、
  `completion_audit_status`、`ready_to_mark_goal_complete` 和未完成 requirement 名单写回
  `fig14-no-gpu-preflight.json`。当前 completion audit 返回
  `in_progress_blocked_by_external_artifacts` 是预期 blocker；只有 failed requirements、
  丢失 requirement coverage、或错误宣称 `ready_to_mark_goal_complete=true` 才是 hard
  failure。
- `fig14_no_gpu_preflight.py` 的 self-test 队列也纳入
  `validate_roll_r3_real_smoke_events.py --self-test`、
  `analyze_roll_r3_real_smoke.py --self-test` 和
  `extract_roll_r3_real_smoke_evidence.py --self-test`，因此 R3 real-smoke 的事件契约、
  analyzer 负例和 extractor adapter fallback 都会被最外层 no-GPU handoff gate 覆盖。
- `mlite-lora-r3-external-evidence-handoff.json/md` 由
  `prepare_adaptation_external_evidence_handoff.py` 生成，把当前 active blockers 转成
  用户决策、确认变量、review-only 命令、确认后命令、证据刷新命令和 ready gates。该
  handoff 的默认动作必须是 `review_only_no_download_no_launch`；涉及数据下载、权重物化、
  GPU/Ray/SGLang/RL 或 R3 real smoke 的命令必须带
  `CONFIRM_DATA_DOWNLOAD=yes`、`CONFIRM_WEIGHT_DOWNLOAD=yes`、
  `CONFIRM_GPU_RUN=yes` 或 `CONFIRM_REAL_SMOKE_EVIDENCE=yes`。`fig14_no_gpu_preflight.py`
  会在 completion audit 之后刷新并校验该 handoff。
- `mlite-lora-r3-refresh-chain.json/md` 由
  `audit_adaptation_refresh_chain.py` 生成，检查每条 external-evidence track 的
  runbook 与 handoff 是否共享同一组 `artifacts_to_refresh`，并确认局部
  post-evidence commands 会收敛到统一的 no-launch 聚合尾链：
  `adaptation_phase_matrix.py`、unblock runbook、external handoff、external readiness
  probe、refresh-chain、deliverable audit、status consistency audit 和 completion audit。
  该链不包含任何下载、权重物化、GPU/Ray/SGLang/SLURM 或训练启动命令；`fig14_no_gpu_preflight.py`
  已在每次 deliverable audit 前刷新并校验该链。
  R3 real-smoke track 也显式列出两个 `.draft.json` result，以保证 adapter digest
  schema 在真实 evidence 到来前后都不会从交接链里掉出。

### M1: Qwen2 exact-route GPU train-step

完成：

- 真实 DeepSeek-R1-Distill-Qwen-1.5B checkpoint on GPU。
- standard LoRA 与 OLoRA-tail 至少各一短步。
- adapter delta 和 sidecar 通过 analyzer。
- sidecar 通过必须是文件系统证据：adapter 目录存在，三件 sidecar 文件存在且非空，
  `adapter_config.json` 和 `megatron.lite_adapter_meta.json` 可解析且必须是 JSON object。
  Qwen2 exact-route smoke analyzer 以及 Fig.14 collector/analyzer/finalizer 都会拒绝空
  `adapter_model.safetensors`、非法 JSON sidecar 或 JSON list sidecar，防止坏 artifact
  被升级成 exact-route 或 paper-claim 证据。

输出：

- exact-route smoke evidence，不是 Fig.14。

### M2: GLM5/FSDP2 adapter lifecycle

完成：

- FSDP2 preflight。
- FSDP2 adapter roundtrip。
- FSDP2 OLoRA-tail。

输出：

- real FSDP2 evidence，补齐当前 fake-DTensor 到真实 runtime 的缺口。

### M3: 分布式 sparse path

完成：

- ROLL-derived R3 role/data/failure contract 已进入 MLite/VERL no-GPU audit。
- synthetic data、model-path validator、checkpoint materialization planner、
  local MoE discovery 和 materialized-checkpoint intake 都通过本地门禁。
- CP/THD Router Replay layout smoke。
- PP boundary smoke。
- EP Router Replay identity smoke。
- 默认 fused DSA backend 强证据。
- 用户确认后的 MLite + SGLang R3 real smoke。

输出：

- GLM5 sparse/MoE 路线可用于后续 RL proxy。

### M4: VERL / RL 集成

完成：

- SFT/GRPO launcher LoRA/OLoRA/sidecar knobs。
- SGLang rollout R3 flags 与 MLite actor R3 flags 对称开启，并由 readiness gate
  阻断不支持的 backend / mode。
- DAPO 或 GRPO small-loop smoke。
- adapter-only checkpoint save/load。

输出：

- MLite 后训练闭环证据。

### M5: Fig.14 双臂复现

完成：

- standard LoRA 500-step。
- OLoRA-tail 500-step。
- 六个 benchmark eval。
- result collector + finalizer。

输出：

- paper-facing reproduction package；只有这时才改写 claim。

## 6. 需要用户决策的点

- 是否优先推进 Fig.14 精确 Qwen2 route，还是先补 GLM5/FSDP2/CP/THD 完整功能。
- DAPO-Math-17k 的权威数据路径在哪里。
- DAPO training entrypoint 使用 `references/external/DAPO`、`verl-recipe`，
  还是另一个内部脚本。
- 用哪个已经 materialized 的 GLM5/Qwen3-MoE/Qwen2-MoE checkpoint 做 R3 smoke；
  如果继续用 GLM5 HF snapshot，则需要先确认 weight download，
  并提供足够的本地或集群存储空间；下载后还必须通过
  `post_materialization_validation` 的 `--require-weight-assets` model-path gate。
- 是否允许启动 MLite + SGLang + Ray/GPU 的短 R3 real smoke。未确认前只跑 no-GPU
  preflight/audit。
- 允许使用的集群、分区、GPU 形状和 walltime。
- 是否接受外部 HF/PEFT baseline 先完成 Fig.14 对照，再把 MLite exact route 作为后续替换。

## 7. 最终完成定义

完整适配完成必须同时满足：

- GLM5/Qwen3-MoE/Qwen2 的 LoRA injection、freeze、forward/backward、adapter save/load
  都有本地和真实 runtime 证据。
- OLoRA-tail 在 base load 后初始化支持 target，并在 metadata 中可追踪。
- rsLoRA scaling convention 在 config、runtime 和 adapter artifact 中一致；
  若声明论文 Figs.16-18 / rank-regime 复现，还必须有三种 alpha 规则的
  rank-sweep 结果、Qwen3-8B 216-run rank-regime 证据，和每个 arm 的真实
  optimizer-step、adapter delta、seed、RL metrics、eval outputs。
- R3 能记录 rollout expert ids，并在 training forward/recompute 中 verbatim replay。
- DSA IndexShare 与 GLM5.2 HF/MindLab 语义一致，zero-loss training path 可跑，
  nonzero indexer-loss 要么实现，要么显式 unsupported。
- VERL / checkpoint sidecar 不破坏普通 checkpoint 语义。
- FSDP2、CP/THD、PP、EP 至少有一条真实通过路径或明确 unsupported 边界。
- Fig.14 若要声明复现，必须有 DAPO 500-step 双臂、RL metrics、adapter deltas、seeds/config
  和 benchmark eval outputs。
- `mlite-lora-r3-adaptation-phase-matrix.json` 必须从
  `in_progress_blocked_by_external_artifacts` 变成完整通过；否则只能称为局部工程证据。
- `audit_adaptation_completion.py` 必须生成 `status=complete` 且
  `ready_to_mark_goal_complete=true`；若仍为
  `in_progress_blocked_by_external_artifacts`，则说明仍有未满足的目标级需求，不能宣称完整适配。
