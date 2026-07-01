# MLite GLM5 LoRA 适配计划

状态日期：2026-06-20。

本文是面向执行的中文计划，配合
`mlite/experimental/lite/docs/glm5_lora_adaptation_plan.md` 使用。目标是把
GLM5/GLM5.2 在 MLite 中的 LoRA/PEFT 能力从“局部能跑”推进到“认知一致、可复现、
可保存加载、可分布式验证”的完整状态。

如果需要按阶段推进，优先读执行版：
`mlite/experimental/lite/docs/mlite_lora_olora_adaptation_execution_plan_zh.md`。

## 0. 参考边界

### 0.1 必须固定的输入

- MindLab GLM5.2 Megatron fork：
  `references/external/Megatron-GLM5.2`，commit
  `e08610def780d9deba2778f1dbe268c0bca4d791`。
- HF GLM-5.2 config snapshot：
  `references/external/hf-zai-org-GLM-5.2/config.json`。
- 论文 Fig.14 目标模型 HF config snapshot：
  `references/external/hf-deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/config.json`。
  该配置为 `model_type="qwen2"`、`architectures=["Qwen2ForCausalLM"]`；
  MLite 现在有 `qwen2` 包身份、最小 TP=1 dense Qwen2 lite runtime slice、
  PEFT adapter lifecycle、OLoRA-tail init、HF checkpoint 映射代码和真实
  DeepSeek-R1-Distill-Qwen-1.5B safetensors 验证；
  当前 MLite 的 `qwen2_moe` 不是 dense Qwen2 的精确替代，精确 Fig.14
  仍缺分布式证据和 RL evidence。
- PEFT scaling paper notes：
  `peft-mint-repro/references/main/insights.md`。

### 0.2 三类机制不要混在一起

1. PEFT LoRA：冻结 base weight，只训练 adapter delta；需要独立保存、加载、导出。
2. GLM5 架构低秩：`q_lora_rank`、`kv_lora_rank` 是 MLA base architecture 的组成部分，
   不是 PEFT adapter。
3. 稀疏路径对齐：
   - DSA IndexShare 复用 attention top-k position。
   - R3 Router Replay 记录并回放 MoE expert id。

### 0.3 执行纪律

- MindLab commit 可以借鉴，但不能直接当真值；每个 GLM5/GLM5.2 结论都要和 HF
  config 或 HF weight map 做对照。
- 遵守 `experimental/lite/skills/` 的 MLite 维护方式：先写清 primitive/model
  contract、shape/dtype/rank 规则、unsupported boundary 和 validation，再改实现。
- 不把 Megatron-Core/MindLab 里的大块抽象硬搬进 `experimental/lite`；MLite 侧保持
  native primitive、model protocol、runtime contract 的组合方式。
- 每个 GPU/分布式结论都必须有 `runs/<run-id>/` 产物、锁定命令、analyzer 或明确失败分类；
  不能把 scheduler/container 成功说成模型逻辑成功。
- 论文 reproduction 与工程 smoke 分开报告：只有真实 RL metric、optimizer step、
  adapter delta、seed/config 齐全时，才能称为 paper-facing reproduction。

### 0.4 已确认的 GLM5.2/MTP 语义

- HF `indexer_types` 长度是 78，只覆盖主干 `num_hidden_layers=78`。
- GLM5.2 checkpoint 把第一个 MTP transformer layer 表示为 appended HF layer index 78。
- MindLab 参考实现里的 DSA 采用 1-indexed `layer_number`，MTP layer 会加主干层 offset。
  因此第一个 MTP DSA layer 对应 1-indexed 79；在 `index_skip_topk_offset=3`、
  `index_topk_freq=4` 下是 full indexer。MLite 当前把 appended MTP layer 78 判为
  full indexer 是合理的。
- 当 `indexer_types` 只覆盖主干层时，MLite helper 现在会显式使用
  `index_share_for_mtp_iteration`：GLM-5.2 的 true 配置继续按 appended MTP layer number
  派生 DSA sharing pattern；false 配置则把 implicit MTP DSA layer 视为 full indexer。
- 模型构建时的 DSA IndexShare PP split 校验现在也会把当前 rank 拥有的 appended MTP
  DSA layer ids 纳入检查；如果某个 MTP shared layer 需要跨 PP stage 读取 source top-k，
  会在构建期失败，而不是晚到 sparse attention forward 才报错。
- `eh_proj` 是 MTP checkpoint state，不是 PEFT LoRA 默认 target。

## 1. 基本复现阶段

目标：先证明 MLite 对 GLM5.2 架构、PEFT adapter、R3、DSA 的基本理解正确，再扩大到 GPU
和分布式。

### 1.1 Reference audit

任务：

- 固定 MindLab commit、HF config、PEFT notes 的路径和版本。
- 对 MindLab commit 做 changed-file audit，确认它提供的是 GLM5/DSA/MTP/CP/THD 架构参考，
  不是直接的 PEFT LoRA adapter 实现。
- 对 HF config 做字段审计：
  `index_topk_freq`、`index_skip_topk_offset`、`indexer_types`、
  `index_share_for_mtp_iteration`、`q_lora_rank`、`kv_lora_rank`、
  `num_nextn_predict_layers`、`scoring_func`、`topk_method`。
- 对 HF weight map 做结构审计：
  full DSA layer 有 `self_attn.indexer.*`，shared DSA layer 没有；
  appended MTP layer 有 `eh_proj` 和 transformer/MoE/attention weight。

验收：

- 静态测试能复现 GLM5.2 的 full/shared DSA pattern；当前本地门禁已固定
  HF config 中 `index_topk_freq=4`、`index_skip_topk_offset=3`、21 个 full
  indexer layer、57 个 shared indexer layer。
- 明确记录：`q_lora_rank/kv_lora_rank` 不能被 adapter target 逻辑误判为 PEFT LoRA。
- 明确记录：MTP appended layer 的 adapter key 使用 HF-style index 78。

### 1.2 Local static gates

任务：

- 检查 GLM5 config parser 能读取 HF 字段和 null optional 字段。
- 检查 DSA helper 的 full/shared/source-layer 计算。
- 检查 PEFT alias：
  `r -> rank`、`lora_alpha -> alpha`、`lora_dropout -> dropout`、
  `gate_proj/up_proj -> linear_fc1`。
- 检查 `use_rslora` 只改变 scaling convention：
  legacy 是 `alpha / rank`，rsLoRA 是 `alpha / sqrt(rank)`。
- 检查 unsupported target 快速失败：
  `eh_proj`、DSA indexer、embedding、LM head 不进 `all-linear`。
- 检查 Router Replay primitive 暴露 `RECORD`、`REPLAY_FORWARD`、
  `REPLAY_BACKWARD`，并且 replay active 时禁用 fused routing。

验收：

- 本地 CPU/torch 静态门通过。
- 失败信息是 actionable 的，不靠 silent fallback。

### 1.3 Local runtime smoke

任务：

- tiny GLM5 DSA IndexShare forward：full layer 产生 top-k，shared layer 复用 source layer。
- tiny GLM5/Qwen3 MoE Router Replay：record expert ids，再修改自然 router score，
  forward replay 后 expert ids 仍等于 record。
- tiny GLM5 LoRA forward/backward/freeze：base weight frozen，adapter gradient 有限且非零。
- adapter export/import round trip：
  - safetensors 只包含 LoRA A/B tensor；
  - metadata 包含 rank、alpha、dropout、target_modules、use_rslora、scaling convention；
  - strict load 能报 missing/unexpected/shape mismatch。
- OLoRA-tail 小矩阵 SVD gate：
  使用最小 singular-vector subspace，`B0=U_tail`、`A0=V_tail.T`，不乘 singular value。

验收：

- `tests/run_glm5_lora_local_gates.sh` 通过。
- 本地门禁不依赖真实 GPU，不把 TE/FlashMLA 环境问题误判成 MLite 逻辑问题。

### 1.4 单 GPU TE/CUDA smoke

任务：

- preflight：Python、Torch、CUDA、TE、DSA dependency。
- GLM5 LoRA forward/backward/optimizer step。
- adapter save/load round trip。
- Router Replay record/replay identity。
- OLoRA-tail post-load init。
- 默认 fused DSA indexer path 和必要 fallback path 分开记录。

验收：

- finite loss。
- LoRA gradient finite/nonzero。
- optimizer step 后 adapter parameter delta 非零。
- save/load 后 adapter key 和 metadata 一致。
- analyzer 输出 `pass`。

注意：任何 GPU allocation/submission 都必须先获得用户明确确认，并且 run artifact 必须落到
`runs/<run-id>/`。

## 2. 认知对齐阶段

目标：确保实现与论文、MindLab 参考、MLite 设计三者语义一致，而不是只做 key mapping。

### 2.1 PEFT LoRA 是持久策略状态

实现约束：

- Base checkpoint 保持 frozen/immutable。
- Adapter 是独立 revision，可保存、加载、迁移到 serving layout。
- Adapter artifact 禁止混入 base weight。
- Adapter config 必须能说明 rank、alpha、target、dropout、scaling convention、init history。
- `adapter_config.json` 和 `megatron.lite_adapter_meta.json` 必须是标准 JSON，且顶层必须是
  JSON object；`NaN`/`Infinity`、array/string 等坏 sidecar 要在读取
  `adapter_model.safetensors` 前给出明确错误。
- LoRA config 和 adapter sidecar 里的 `target_modules` 必须是 string 或 string sequence；
  mapping/dict、数字、非字符串元素都要在 alias expansion 前明确拒绝；GLM5 与 Qwen3-MoE
  在 primitive normalizer 和 adapter helper 的 PEFT target parser 上共享这条边界。
- LoRA config 顶层可以接受 mapping-like object，例如只读 mapping proxy，并复制到同一套
  严格校验路径；list-of-pairs 仍然不是合法 config，`target_modules` 自身也不能是
  mapping-like object。GLM5/Qwen3-MoE 的 model、protocol、primitive 和 adapter helper
  公开签名也要使用 `Mapping[str, Any]` 描述 caller-provided LoRA config，避免类型契约仍暗示
  dict-only 路径。
- enabled LoRA config 还必须至少包含一个非空 target module；rank 为正但 target 为空时，
  不能生成一个 0 tensor 的空 adapter revision。
- LoRA config、`adapter_config.json` 和 Megatron Lite metadata 里的数值/布尔字段必须类型明确：
  `r/rank` 是 integer，`lora_alpha/alpha`、`lora_dropout/dropout`、`scale` 是 finite
  JSON number，`lora_dropout/dropout` 还必须在 `[0, 1]` 概率范围内，`use_rslora`
  是 boolean；字符串数字、list、dict、bool-as-number 都要早期拒绝。
- `adapter_config.json` 里的 PEFT task 字段也要显式：如果 `task_type` 出现，只接受
  `CAUSAL_LM`；如果 `inference_mode` 出现，只接受 JSON boolean。`inference_mode=true`
  作为外部 PEFT serving artifact 仍可导入，非 boolean 要早失败。
- 非空 LoRA adapter artifact 的 `adapter_config.json` 必须携带 identity 字段
  `peft_type="LORA"`、非空 `base_model_name_or_path`、`r`、`target_modules`、
  `lora_alpha`、`lora_dropout` 和 `use_rslora`；缺失或空字符串时不能依赖
  caller config、metadata、默认 convention 或后续 shape check 才暴露。
- Adapter 保存入口也必须在创建输出目录前 strip 并拒绝空的 `base_model_name_or_path`，
  不能生成无身份 artifact，也不能把前后空白污染后的 base identity 写进 sidecar，
  导致后续导入时出现伪不一致。
- Adapter 保存入口还必须在创建输出目录前校验 `init_lora_weights` 类型，并拒绝空字符串
  init history，避免非法或无身份信息的 init 配置留下半写入的 safetensors/config 目录。
- Adapter 保存入口还必须在创建输出目录前把最终 `adapter_config.json` 和
  `megatron.lite_adapter_meta.json` payload 序列化为标准 JSON；模型 metadata 或生命周期
  字段里的 `NaN`/`Infinity` 不能留下半写入目录。
- 新 adapter 目录写入必须先落到同级临时目录，三份 artifact 文件全部写完后再提交到目标
  路径；safetensors 写入失败或 sidecar 写入失败不能留下目标 adapter 目录。
- `adapter_model.safetensors` 必须至少包含一个 LoRA A/B tensor；空 safetensors 不能被导入为
  `loaded_tensors=0` 的“成功” adapter revision。
- 覆盖已有 adapter 目录时，只能替换三份 adapter artifact 文件；写入失败必须回滚到旧的
  `adapter_model.safetensors`、`adapter_config.json` 和 `megatron.lite_adapter_meta.json`，
  并保留目录里的非 adapter 文件。
- `megatron.lite_adapter_meta.json` 也要记录同一个 `base_model_name_or_path`；如果
  两份 sidecar 同时存在但 base identity 不一致，导入必须失败。
- Megatron Lite metadata 的生命周期字段也必须类型明确：`num_tensors`、`num_parameters`
  和 `parallel.tp/ep/pp/etp` 是 integer，不接受字符串数字或 bool-as-number。
- 只要 Megatron Lite metadata sidecar 存在，lifecycle 核心字段就不能省略：
  `base_model_name_or_path`、`num_tensors`、`num_parameters`、
  `expert_lora_representation`、`lora`、`parallel`、`model` 和 `metadata`
  都必须存在；即使 `adapter_config.json` 缺失且导入依赖 caller-provided
  `lora_config`，也不能把 metadata sidecar 当成部分提示。
- nested lifecycle object 也必须完整：`lora` 要携带 rank/alpha/dropout/
  use_rslora/scaling/scale/target，`parallel` 要携带 TP/EP/ETP/PP，`model`
  要携带对应 adapter helper 写出的模型身份字段；这些 nested lifecycle 字段和用户
  `metadata` 都必须是 JSON object，不能是 null、array 或 scalar。
- 保存入口的用户 `metadata` 还必须是 mapping/object-like，并在创建输出目录前通过标准
  JSON 序列化检查；只读 mapping proxy 会复制成普通 JSON object，list-of-pairs、set、
  对象、NaN/Infinity 或非字符串 metadata object key 都不能写进 artifact，也不能留下
  半写入目录。嵌套在 list 或其它 metadata mapping 里的 mapping value 也要递归复制成普通
  JSON object。GLM5 与 Qwen3-MoE 本地 gate 现在覆盖嵌套只读 mapping proxy、嵌套用户
  metadata 中的非标准数值常量，也覆盖普通不可序列化对象。
- `model` identity 字段也必须类型明确：整数配置项必须是 JSON integer，布尔配置项必须
  是 JSON boolean，HF/config 中本来为 null 的 optional 字段只能保存为 null；不能靠
  `2.0 == 2` 或 `0 == false` 这类宽松相等通过校验。
- GLM5 adapter state 使用全局 expert key 表示，因此 `parallel.ep` 只作为 lifecycle
  metadata 做类型校验；`parallel.tp/pp/etp` 在 gather/scatter 未支持前仍保持
  unsupported 边界。
- 只要 GLM5 的 `megatron.lite_adapter_meta.json` 存在，用户自定义 `metadata` 字段
  也必须是 JSON object；即使 `adapter_config.json` 缺失且导入依赖 caller-provided
  `lora_config`，也不能放松这个 sidecar 边界。
- Qwen3-MoE 作为对照路径时也要校验 `megatron.lite_adapter_meta.json`：
  metadata format、tensor/parameter 计数、expert LoRA representation、当前并行形状、
  关键 model 字段、LoRA rank/alpha/dropout/target/use_rslora/scale/scaling convention
  和用户 metadata object 类型都必须和实际 artifact/model 一致。
- Qwen3-MoE 对照路径必须通过 model protocol 暴露 adapter lifecycle API：
  `export_lora_adapter_state`、`save_lora_adapter`、`load_lora_adapter_state`
  和 `load_lora_adapter`，不能要求上层绕过 protocol 直接调用 helper 模块；本地 CPU
  safetensors gate 需要通过这些 protocol wrapper 完成 qkv-only adapter round trip。
- Qwen3-MoE 对照路径还要覆盖 per-expert `GroupedLinearLoRA` 的 safetensors-backed
  正向 round trip：`gate_proj`、`up_proj` 和 `down_proj` 的 PEFT expert keys
  必须能导出并导回 native grouped tensor。
- 外部 adapter 目录缺少 `adapter_config.json` 时，GLM5 与 Qwen3-MoE 都必须依赖
  caller-provided expected LoRA config，并用它校验 tensor rank、target modules、alpha、
  dropout 和 rsLoRA 设置；即使 `megatron.lite_adapter_meta.json` 仍存在，也不能绕过
  caller-provided config；没有 caller config 时必须在读取 `adapter_model.safetensors`
  前失败。

验收：

- export 只写 LoRA A/B。
- strict import 能拒绝 base tensor、未知 tensor、shape mismatch。
- direct adapter-state load 与 safetensors import 都必须先拒绝非 LoRA A/B tensor；
  `strict=False` 只能忽略额外 LoRA adapter tensor，不能吞掉 base weight。
- `strict=False` 也只能忽略位于已支持 adapter surface 的额外 LoRA tensor；`lm_head`、
  DSA indexer、未知 projection 名、Qwen/GLM 互串的 LoRA-looking key 都必须在进入模块级
  shape/load 前早失败。adapter load API 还必须显式解析 boolean 或 boolean string
  `strict`，不能让 `"False"` 因 Python truthiness 误变成 strict mode。
- GLM5 shared-local-expert 导入即使在 `strict=False` 下也必须无损：
  fused gate/up 的 `lora_A` 必须一致，展开后的本地 per-expert tensor 必须完全一致，
  才能 fold 回 shared native LoRA；`strict=False` 只放松 extra-key 处理。
- direct adapter-state load 还必须先拒绝非 tensor 的 LoRA 值、非浮点 dtype、非 2D 的
  LoRA A/B tensor、任一维度为空或包含 NaN/Inf 的 LoRA tensor，再进入模块级 shape 校验。
- 每个 adapter surface 的 `lora_A.weight` 与 `lora_B.weight` 必须成对出现；半个 LoRA
  adapter 不能依赖后续 module-level missing-key 检查才失败。
- 每对 LoRA tensor 内部也必须满足 `lora_A` 行数等于 `lora_B` 列数；A/B rank 维不一致时，
  不能等到 config/metadata 或 module-level shape 校验阶段才失败。
- 同一个 adapter artifact 仍然使用单一 LoRA rank；跨 adapter surface 混用不同 rank
  要在 direct state load / safetensors import 入口早失败。
- adapter metadata 足以重现实验配置。

### 2.2 rsLoRA 是 scaling convention

实现约束：

- `use_rslora=False`：`scale = alpha / rank`。
- `use_rslora=True`：`scale = alpha / sqrt(rank)`。
- 保存时记录 `alpha_over_rank` 或 `alpha_over_sqrt_rank`，不要只记录 `alpha`。

验收：

- 同一个 adapter rank/alpha/use_rslora 组合导出后再导入，forward delta scale 不漂移。
- Qwen3-MoE 与 GLM5 使用同一套 artifact/LoRA/lifecycle metadata 边界；GLM5
  仍按当前 unsupported scope 要求 TP/PP/ETP 为 1，Qwen3 则校验当前 TP/EP/ETP/PP
  形状一致。

### 2.3 OLoRA-tail 是 post-load initialization

实现约束：

- 只能在 base weight 已加载后运行。
- 不在空 module construction 时运行。
- `initialize_lora_olora_tail` 必须通过 GLM5 model protocol 直接可调用，并有本地 CPU
  gate 证明 runtime-facing post-load pass 能初始化支持的 attention LoRA module。
- runtime/VERL 入口使用显式 `impl_cfg.lora_init` / `LORA_INIT` 开关；GLM5
  支持 `olora_tail`。Qwen3-MoE 对照路径现在也支持 TP=1/ETP=1 的
  `olora_tail` post-load 初始化，覆盖 fused qkv、o_proj 和可精确匹配的 single-local
  routed expert LoRA；TP-sharded / ETP-sharded OLoRA-tail 仍必须显式拒绝，不能
  silent ignore。
- fake FSDP-style wrapper 下也必须能在 TP=1 解开 chunk，并完成 post-load OLoRA-tail
  attention LoRA 初始化；真实 FSDP2 runtime 仍需单独验证。
- 本地 CPU DTensor gates 还要证明 OLoRA-tail 可 materialize DTensor base weight 做 SVD，
  并通过 DTensor-aware whole-parameter copy 写回 ordinary attention LoRA 和 per-expert
  grouped routed LoRA；真实 multi-rank FSDP2 仍需单独验证。
- 默认防止重复初始化；只有 `force=True` 能重跑。
- TP-sharded weight 在没有明确 gather/init/scatter 方案前保持 unsupported 或 documented
  approximation。
- OLoRA-tail SVD 前必须显式拒绝 NaN/Inf base weight；SVD 计算可在 float32 完成，但写回的
  LoRA A/B tensor 必须有限，并保持目标模块 dtype/device。

验收：

- 支持 attention、dense MLP、shared expert、可精确匹配的 routed expert。
- multi-local-expert shared LoRA 如无法精确对应单个 base weight，必须显式 skip 并记录原因。

### 2.4 R3 解决 MoE sparse-path TIM

实现约束：

- Rollout/reference forward：`RECORD`，输出 ordered `routed_experts`。
- Training forward：从 batch 装入 ordered `routed_experts`，`REPLAY_FORWARD`。
- Activation recompute/backward：使用 forward 保存的 ids，`REPLAY_BACKWARD`。
- primitive 和 protocol action 入口只接受 `RouterReplayAction` 或合法字符串：
  `record`、`replay_forward`、`replay_backward`；非法字符串/类型不能静默回到自然 routing。
- record/replay tensor 都表示 expert id，必须是 `[tokens, topk]` integer tensor；
  float/bool/complex 不能被静默 cast 成 long，否则会破坏“verbatim replay”的语义。
- `RECORD` 模式保存 rollout trace 前，也必须根据当前 router scores 校验 top-k 输出的
  shape 和 expert id range；坏 default-topk 输出不能进入 recorded trace。
- `REPLAY_BACKWARD` 消费 activation-recompute 队列前必须先校验 saved indices；校验失败时
  不能先 pop 掉队列，方便保留原始 trace 做诊断或重试。
- record/replay tensor 写入 router state 时必须是 detach/clone 快照；`get_recorded_indices`
  返回的 trace 也不能共享内部状态，避免外部 mutation 或 cleanup 污染 rollout artifact。
- legacy/debug 全局 helper（`set_replay_data`、`get_recorded_data`）也必须遵守同样的
  per-router tensor list/tuple 输入和快照隔离边界。
- MTP routers 的顺序追加在 main-model routers 后。
- `PackedBatch.routed_experts` 必须转换到实际 router input 的 token grain：
  padding、THD packing、CP local split 都要在安装 replay tensor 前完成。
- 默认 `PackedBatch.routed_experts` 是 rollout/reference 产出的 true-token grain；
  如果调试或分布式路径传入已经 padding 或 CP-split 的 replay tensor，必须通过
  `batch.extras["router_replay_layout"]` 显式声明 `full_padded` 或 `cp_local`，并在
  token 行数不匹配时早失败。

验收：

- CPU primitive/context gate 覆盖 action 字符串解析、bad action、record/replay bad
  shape、bad dtype、bad range、snapshot isolation、list/tensor layout、cleanup。
- protocol context gate 覆盖 bad action string/type、无 router 却提供 replay data、
  forward replay 缺少 `batch.routed_experts`、以及 `RECORD` 与 replay ids 混用。
- protocol context gate 覆盖 explicit THD replay layout：默认 true-token、`full_padded`
  split 到 CP-local、`cp_local` 直通，以及 invalid layout/type 的 actionable error。
- fake FSDP-style wrapper 下，protocol context 必须用解开后的模型 parallel state 做 THD/CP
  replay packing，不能因为外层 wrapper 没有 `ps` 而退回默认 `cp_size=1`。
- GLM5+MTP tiny gate 覆盖 main router 和 MTP router 顺序。
- GPU gate 覆盖 activation recompute consume queue。
- CP/THD gate 覆盖 world_size=2、cp ranks `{0,1}`、record/target shape、target_matches_expected。

### 2.5 DSA IndexShare 解决 attention top-k 复用

实现约束：

- full layer 计算并保存 top-k。
- shared layer 只从 source layer 读取 top-k。
- PP split 不能跨 stage 复用 top-k，除非未来显式实现跨 PP holder。
- `dsa_indexer_loss_coeff=0` 的训练路径必须可跑。
- `dsa_indexer_loss_coeff>0` 如果 fused wrapper 不能返回 loss 所需 top-k，必须显式
  `NotImplementedError`。

验收：

- zero-loss IndexShare 有 forward/backward smoke。
- nonzero-loss 当前要么实现 tiny deterministic parity，要么保留显式 unsupported gate。

### 2.6 MinT/lifecycle 兼容

实现约束：

- Adapter revision 是 policy identity 的一部分。
- Router replay trace 是 rollout/training artifact，不属于 adapter weight。
- DSA/MLA/MTP config 是 base model config，不属于 adapter config。
- Serving 可从 metadata 选择 legacy LoRA 或 rsLoRA scale。

验收：

- adapter 目录小而纯。
- lifecycle 字段在 `adapter_config.json` 和 `megatron.lite_adapter_meta.json` 中一致；
  冲突时 import 失败。

## 3. 完整功能适配阶段

目标：把本地门禁、单 GPU smoke、分布式 smoke、adapter lifecycle 收口成可交付能力。

### 3.1 GLM5 LoRA injection 全面覆盖

范围：

- DSA attention projections：
  `q_a_proj`、`q_b_proj`、`kv_a_proj_with_mqa`、`kv_b_proj`、`o_proj`。
- Dense MLP：
  fused gate/up surface、down surface。
- Routed MoE experts：
  expert `linear_fc1`、`linear_fc2`。
- Shared experts：
  shared gate/up/down surfaces。
- MTP transformer layers：
  appended HF-style layer index。

不默认支持：

- DSA indexer projections。
- MTP `eh_proj`。
- embedding / LM head。

完成标准：

- `lora_config=None` 时行为完全不变。
- `target_modules="all-linear"` 覆盖支持的 attention + MLP projection targets。
- `freeze_non_lora_params` 后只有 adapter trainable。
- GLM5 protocol 的 `ModelBundle` 记录 LoRA stats。

### 3.2 Adapter import/export 完整化

任务：

- GLM5-native adapter key map，不复用 Qwen3 名字硬套。
- GLM5 protocol wrapper 必须能直接完成 adapter lifecycle round trip：
  `export_lora_adapter_state`、`save_lora_adapter`、`load_lora_adapter_state`
  和 `load_lora_adapter` 需要有 CPU safetensors gate 覆盖。
- MLite runtime 必须能从 `ModelHandle` 透传 adapter lifecycle helper：
  `export_lora_adapter_state`、`save_lora_adapter`、`load_lora_adapter`，使
  VERL/训练侧不用直接抓 protocol internals。runtime 透传边界还必须在调用 model protocol
  前校验 protocol、model config、非空且不含 None 的 model chunks sequence，以及
  parallel_state，避免坏 handle 进入协议内部才以属性错误或 shape 错误失败。
- VERL checkpoint 需要支持显式 adapter sidecar：`save_contents` /
  `load_contents` 包含 `lora_adapter` 时，可只保存/加载 PEFT-style adapter
  目录；普通 model/optimizer checkpoint 语义不应被默认改变。本地 gate 覆盖
  adapter-only save/load 和 user kwargs/init metadata 透传。SFT/GRPO launcher
  还需要暴露环境变量入口：`CHECKPOINT_SAVE_CONTENTS`、
  `CHECKPOINT_LOAD_CONTENTS`、`CHECKPOINT_SAVE_LORA_ADAPTER` 和
  `LORA_ADAPTER_DIR_NAME`，并用 dry-run gate 证明它们映射到对应 Hydra 字段。
  `save_contents` / `load_contents` 还必须按精确 content key 归一化，支持 list、
  tuple、单个字符串和 bracket/comma 字符串；key 周围的空白和简单 shell quote 要被
  归一化，mapping/dict 不能被当作可迭代 key 集合接受。不能让 `"not_model"` 这类
  substring 误触发 model checkpoint；`save_lora_adapter` / `load_lora_adapter` 也必须按显式
  boolean 或 boolean string 解析，不能让 `"False"` 因 Python truthiness 误启用 sidecar。
  `lora_adapter_dir_name` 只能是 checkpoint 目录内的相对路径，绝对路径和 `..`
  path traversal 要在调用 runtime adapter save/load 前失败。
  `lora_adapter_kwargs` 和其中的 `metadata` 必须是 mapping/object，不能把 list-of-pairs
  或其它非 JSON object 隐式转换成 adapter metadata。当同一次 checkpoint 同时请求
  model/optimizer 和 LoRA sidecar 时，sidecar path、kwargs 和 runtime adapter API
  capability 必须在任何 full checkpoint 读写副作用前完成校验；sidecar save 还必须在
  full checkpoint 写入前确认 `engine.impl_cfg.lora` 或
  `checkpoint_config.lora_adapter_kwargs.lora_config` 里存在 enabled LoRA config。
  sidecar load 也必须在任何 full checkpoint load 副作用前确认选中的 adapter path
  已存在且是目录。
  adapter-only save 如果 adapter 保存失败，且外层 checkpoint 目录是本次新建的，则清理该
  wrapper 目录；已有 checkpoint 目录必须保留。
  `load_contents` 不包含 MLite 拥有的 `model`、`optimizer` 或 LoRA adapter sidecar
  时，load 必须在参数 offload 或 CUDA reload 前直接跳过。
- `adapter_config.json` + `megatron.lite_adapter_meta.json` 双 metadata 校验。
- strict/non-strict load 语义明确：
  strict 拒 missing/unexpected，non-strict 只允许显式 extra adapter tensor；非 LoRA
  A/B tensor 在 direct state load 和 directory import 入口都必须早失败。
- grouped expert / shared local expert 表示可导出为 PEFT per-expert keys；
  导入时只有本地 per-expert tensor 完全一致才 fold 回 shared native LoRA。
- GLM5 与 Qwen3-MoE adapter helper 都需要能解开 FSDP-style wrapper，例如
  `_fsdp_wrapped_module`，并在 TP=1 下完成本地 safetensors save/load round trip。
- GLM5 与 Qwen3-MoE adapter export 在遇到 DTensor/FSDP2 风格参数时必须先
  materialize full tensor，不能把 `to_local()` 的本地 shard 当成完整 adapter 保存；
  本地 fake-DTensor gates 已覆盖两条 ordinary LinearLoRA export 路径和 routed expert
  export 路径，真实 FSDP2 runtime 仍需单独验证。
- GLM5 与 Qwen3-MoE adapter load 也必须处理 DTensor/FSDP2 风格 LoRA 参数：
  普通 Tensor adapter state 不能直接 `.data.copy_()` 进 DTensor 参数。本地 CPU
  DTensor gates 已证明 ordinary GLM5 attention LoRA 和 Qwen3 qkv LoRA state 会先按
  目标参数的 mesh/placements distribute 后再复制。本地 CPU DTensor gates 也覆盖
  GLM5 与 Qwen3-MoE per-expert grouped LoRA import：先组装 rank-local grouped tensor，
  再整参复制到 DTensor，避免 DTensor indexed assignment。真实 multi-rank EP/FSDP2
  验证仍需单独完成。
- TP/PP/ETP adapter save/load 在未实现 gather/scatter 前显式 unsupported；目录级
  `save_lora_adapter` / `load_lora_adapter` 也必须在读写 artifact 前早失败，并用
  missing-path gate 证明 load 不会先读 adapter 文件。

完成标准：

- tiny CPU safetensors round trip 通过。
- fake FSDP-style wrapper 的 TP=1 adapter state/save/load round trip 通过。
- real TE tiny model round trip 通过。
- 真实 FSDP2 TP=1 save/load smoke 通过。

### 3.3 OLoRA-tail 完整化

任务：

- 支持 unsharded attention/MLP/shared expert/MTP transformer LoRA。
- 支持 per-expert grouped routed expert 的精确初始化。
- 为 TP-sharded weights 设计 gather/init/scatter 或 documented local approximation。
- 大矩阵加入 memory guard 或 randomized SVD 方案。

完成标准：

- 小矩阵数值 gate。
- tiny GLM5 runtime gate。
- Qwen3-MoE proxy CPU gate 覆盖 fused qkv/o_proj 与 single-local routed expert
  OLoRA-tail 初始化；multi-local shared expert LoRA 继续显式 skip。
- fake FSDP-style wrapper 的 TP=1 post-load OLoRA-tail gate。
- B300A/GB300 TE smoke 证明真实模块可初始化。
- metadata 正确记录 `init_lora_weights="olora_tail"`，load 不会默认重跑 init。

### 3.4 R3 Runtime 完整化

任务：

- GLM5/Qwen3 protocol `_forward_step` 贯通 `routed_experts`。
- batch packing 支持 true-token 到 padded/THD/CP-local replay tensor 的转换。
- activation recompute wrapper 可靠切换 `REPLAY_BACKWARD` 并清理状态。
- EP local expert dispatch 与 R3 replay 做 identity smoke。
- PP schedule 下 main/MTP router order 保持稳定。

完成标准：

- CPU primitive + model gates 通过。
- B300A/GB300 单 GPU record/replay + recompute 通过。
- CP/THD world_size=2 layout smoke 通过。
- EP/PP smoke 至少覆盖构建、失败边界和一个成功路径。

### 3.5 DSA/CP/THD/PP 完整化

任务：

- 默认 fused indexer backend 在 GLM5-like shape 上通过。
- H100 fallback path 只能作为 layout 证据，不替代默认 backend 验收。
- CP/THD sparse attention fallback 必须显式记录 backend 和 call-count。
- PP boundary：valid split 通过，invalid cross-PP IndexShare split 失败且错误信息明确。

完成标准：

- B300A/GB300 默认 backend CP/THD 通过，作为最终强证据。
- H100 torch fallback CP/THD 只能标注为“layout smoke evidence”。
- analyzer 对 backend、shape、loss、replay target、fallback call-count 做硬检查。

### 3.6 Paper-facing reproduction

任务：

- 先做小模型 OLoRA-tail reproduction：
  rank 16、alpha 32、constant LR 1e-5、attention+MLP projections。
  论文 Fig.14 的精确配置是 DeepSeek-R1-Distill-Qwen-1.5B +
  DAPO-Math-17k；已固定的目标 HF config 为 `model_type="qwen2"`、
  `architectures=["Qwen2ForCausalLM"]`。当前 MLite registry 已能识别
  dense `qwen2` 身份，并已有最小 TP=1 lite runtime slice、dense-Qwen2
  adapter save/load/import/export、OLoRA-tail init 和 HF checkpoint load/export
  映射；真实 exact HF safetensors 验证现在覆盖 339 个 BF16 tensor，包括
  q/k/v bias 到 native fused qkv bias 的 pack/unpack；full BF16 model load
  smoke 也已通过 Qwen2 protocol 加载真实 checkpoint，并检查首尾层 fused
  tensor。这个 slice 仍只证明本地 forward/backward、LoRA freeze、PEFT-style
  adapter round-trip、SVD-tail initialization、checkpoint tensor 映射和真实
  model load，还不包含分布式 runtime 或 RL metrics。
  因此这一步必须先分成两条线：
  1. MLite 近似工程复现：在当前已支持的 GLM5/Qwen3-MoE 路径上证明
     OLoRA-tail、optimizer step、adapter delta、metadata 和评估采集闭环；
  2. 论文精确复现：基于已验证的真实 DeepSeek-R1-Distill-Qwen-1.5B
     checkpoint 映射继续补齐 runtime/RL 后续能力，或明确使用 HF/PEFT
     外部基线承接 Fig.14，再把结果和 MLite 证据分开报告。
- 再做 Qwen3/GLM5 MoE R3 TIM reproduction：
  比较自然 routing 与 R3 replay 的 prob-diff/KL/grad norm。
- 最后才考虑 paper-scale 500-step RL。

完成标准：

- 不把工程 smoke 说成论文复现。
- 论文级结论必须有 RL metric、optimizer-step、adapter-delta、seed/config 证据。
- 如果只是 runtime/infra 通过，报告中明确称为 engineering evidence。

## 4. 推荐执行顺序

1. 锁定 reference audit 与现有本地门禁，确保没有概念漂移。
2. 跑本地 `run_glm5_lora_local_gates.sh`，作为每次代码改动后的第一道门。
3. 修完 CP/THD H100 torch fallback layout smoke，只把它作为 R3/THD layout 证据。
4. 争取 B300A/GB300 默认 backend CP/THD，通过后才算强分布式证据。
5. 补 Qwen3 GPU parity、EP smoke、FSDP2 TP=1 adapter save/load。
6. 设计 TP/PP/ETP adapter gather/scatter，或者继续显式 unsupported。
7. 做 OLoRA-tail 小模型 reproduction。
8. 进入 paper-facing RL reproduction。

## 5. 风险和阻塞

- 容器依赖：H100 环境缺 cuDNN DSA namespace 和 `flash_mla` 时，只能做 fallback layout smoke。
- 调度形状：B300A 单节点没有 2 GPU，GB300/Pyxis/Enroot 可能先失败在容器启动。
- DSA indexer loss：nonzero loss 目前仍受 fused wrapper 能否返回 top-k/loss 所限。
- TP-sharded OLoRA-tail 和 distributed adapter save/load 需要单独设计。
- 论文 reproduction 需要真实 RL step 和 adapter delta 证据，不能用 rollout 成功替代。

## 6. 最终完成定义

适配完成必须同时满足：

- GLM5 LoRA 可以训练、保存、加载、导出，legacy LoRA 和 rsLoRA scaling 都语义稳定。
- OLoRA-tail 可在 base load 后初始化所有支持 target，并在 metadata 中可追踪。
- R3 可对 GLM5/Qwen3 记录 rollout expert ids，并在 training forward 与 recompute 中回放。
- DSA IndexShare 与 HF/MindLab config 语义一致，zero-loss training path 可跑；
  nonzero indexer-loss 要么实现，要么显式 documented unsupported。
- local static、CPU runtime、单 GPU TE/CUDA、分布式 CP/THD/PP/EP smoke 都有 run artifact
  和 analyzer 证据。
