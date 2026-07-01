# GLM5 LoRA Adaptation Plan

Status date: 2026-06-20.

This plan tracks the Megatron Lite adaptation work needed for GLM5 LoRA/PEFT
support, using the MindLab GLM5.2 Megatron fork and the PEFT scaling paper as
references. It separates three often-confused mechanisms:

- PEFT LoRA: trainable adapter deltas on frozen base weights.
- GLM5 architectural low-rank projections: `q_lora_rank` and `kv_lora_rank` in
  MLA are part of the base model architecture, not PEFT adapters.
- Sparse-path alignment: DSA IndexShare reuses attention top-k positions, while
  R3 Router Replay records/replays MoE expert ids.

## Reference Inputs

- MindLab GLM5.2 fork:
  `references/external/Megatron-GLM5.2`, commit
  `e08610def780d9deba2778f1dbe268c0bca4d791`.
- HF GLM-5.2 config snapshot:
  `references/external/hf-zai-org-GLM-5.2/config.json`.
- Exact Fig.14 target HF config snapshot:
  `references/external/hf-deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/config.json`.
  It reports `model_type="qwen2"` and `architectures=["Qwen2ForCausalLM"]`;
  MLite now has a `qwen2` package identity plus a minimal TP=1 dense Qwen2
  lite runtime slice, PEFT adapter lifecycle, OLoRA-tail init, and HF checkpoint
  mapping code validated against the real DeepSeek-R1-Distill-Qwen-1.5B
  safetensors for this target family. Current `qwen2_moe` support is not an
  exact dense-Qwen2 substitute, and exact Fig.14 still needs distributed
  evidence and RL evidence.
- PEFT paper notes:
  `peft-mint-repro/references/main/insights.md`.

Use these as read-only references. The implementation should stay native to
Megatron Lite primitives and avoid importing large Megatron-Core-only
abstractions into `experimental/lite`.

## Phase 0: Current Alignment

Already adapted in the current branch:

- GLM5 config recognizes HF GLM-5.2 DSA fields:
  `index_topk_freq`, `index_skip_topk_offset`, `index_topk_pattern`,
  `index_share_for_mtp_iteration`, `indexer_types`, `scoring_func`, and
  `topk_method`.
- The local HF GLM-5.2 reference snapshot is covered by a static audit: full
  DSA layers have `self_attn.indexer.*` checkpoint tensors, shared layers do
  not, MTP is represented as appended `model.layers.78.*`, and `eh_proj` is
  checkpoint state rather than a LoRA adapter target. The same audit now pins
  the real HF config values for the sparse architecture (`index_topk_freq=4`,
  `index_skip_topk_offset=3`, 21 full indexer layers, 57 shared indexer layers)
  and for MLA low-rank projections (`q_lora_rank=2048`, `kv_lora_rank=512`) so
  they cannot drift into PEFT adapter target logic.
- The MindLab GLM5.2 fork is pinned by a static audit to commit
  `e08610def780d9deba2778f1dbe268c0bca4d791`. That commit is treated as the
  source for GLM5 base architecture semantics (DSA, MLA, MTP, CP/THD), not as a
  direct PEFT adapter implementation: the changed-file audit intentionally
  verifies that the commit does not add PEFT/LoRA adapter files. The PEFT
  semantics come from the scaling paper notes and are adapted into native
  Megatron Lite modules.
- GLM5 DSA IndexShare has local layer helpers, shared/full indexer selection,
  a per-forward top-k holder, and PP split validation.
- Base LoRA primitives support PEFT-style aliases and rsLoRA scaling through
  `use_rslora`.
- GLM5 has base PEFT LoRA injection surfaces for DSA attention projections,
  dense MLP, routed MoE experts, and shared experts. The GLM5 protocol accepts
  `ImplConfig.lora`, normalizes PEFT aliases, freezes non-adapter parameters,
  and reports LoRA trainable-parameter stats. A CPU protocol smoke now
  monkeypatches model construction to verify that `build_model` passes the
  normalized LoRA config into `Glm5Model`, freezes base parameters, leaves
  adapter parameters trainable, and records `lora_stats` in the returned
  `ModelBundle`.
- GLM5 exposes a thin `Glm5ForCausalLM` smoke-test/tooling facade that delegates
  to native `Glm5Model` while converting public hidden-state tensors between
  BSH and the native SBH layout. GLM5 checkpoint and adapter helpers unwrap this
  facade before walking parameters, so exported names remain native/HF-mapped
  rather than gaining an extra `model.` prefix.
- GLM5 checkpoint loading accepts both real HF GLM-5.2 per-expert tensors and
  grouped expert proxy tensors such as `experts.gate_up_proj`,
  `experts.gate_and_up_projs`, `experts.down_proj`, and `experts.down_projs`.
  The grouped path resolves through the same loader helpers used by runtime
  import, not only through isolated tests.
- GLM5 has a PEFT adapter import/export module for the main transformer layers
  with GLM5-native DSA keys, dense MLP keys, routed expert keys, and shared
  expert keys. The protocol exposes `export_lora_adapter_state`,
  `save_lora_adapter`, `load_lora_adapter_state`, and `load_lora_adapter`.
  A CPU safetensors gate now exercises those GLM5 protocol wrappers directly
  on a shared routed-expert adapter round trip. Separate CPU safetensors gates
  now wrap fake GLM5 and Qwen3-MoE chunks in FSDP-style `_fsdp_wrapped_module`
  shells and prove adapter state export/import plus directory save/load still
  unwrap to the native LoRA tensors at TP=1.
  Adapter config validation covers rank, alpha, dropout, `use_rslora`,
  target modules, `init_lora_weights` type, missing keys, and unexpected keys
  in strict mode. Adapter tensor loads also validate per-key tensor shapes
  before copying into native LoRA parameters. Under EP, strict load validates
  all global expert adapter keys while copying only the local expert shard.
  Routed expert LoRA supports the current Megatron Lite
  `shared_local_expert_group` representation by expanding shared local expert
  A/B tensors into PEFT per-expert keys on export, and folding them back on
  import only when the local per-expert adapter tensors are identical.
  Adapter metadata records rank, alpha, dropout, target modules, `use_rslora`,
  the explicit scaling convention, the resulting LoRA scale, and the native
  routed-expert LoRA representation so rsLoRA recipes and shared-expert layouts
  remain portable across adapter revisions. When both `adapter_config.json` and
  `megatron.lite_adapter_meta.json` are present, import also validates that
  duplicated lifecycle fields agree across the two files, including rank, alpha,
  dropout, target modules, `use_rslora`, and `init_lora_weights`.
  Adapter export rejects any state key that is not a PEFT LoRA tensor
  (`lora_A.weight` or `lora_B.weight`), so base weights cannot be silently
  written into adapter artifacts. Adapter state must also be non-empty:
  `adapter_model.safetensors` with zero LoRA tensors is rejected instead of
  being imported as a successful `loaded_tensors=0` revision.
  Adapter target-module validation canonicalizes PEFT/Megatron Lite aliases such
  as `linear_qkv`, `linear_fc1`, `q_a`, and `down_proj` before comparing adapter
  metadata with tensor keys. `target_modules="all-linear"` is accepted during
  model construction and adapter import as a PEFT-compatible shorthand for the
  supported attention and MLP projection targets; it does not include DSA
  indexer, embedding, LM head, or MTP EH projection tensors. GLM5 adapter
  import/export now rejects unsupported explicit targets such as `eh_proj` or
  DSA indexer names instead of carrying unknown target names through metadata.
  Enabled LoRA configs also require at least one non-empty target module, so a
  rank-positive config cannot produce an empty zero-tensor adapter revision.
  Adapter save now also rejects artifacts whose requested target_modules expand
  to a different set than the actually exported adapter tensors, so a partial
  fake or misconfigured module tree cannot be labeled as `all-linear`.
  Adapter import/export also covers MTP transformer layers using HF-style
  appended layer indices `num_hidden_layers + mtp_idx`.
- PEFT MLP target names `gate_proj` and `up_proj` are canonicalized to the
  fused Megatron Lite `linear_fc1` surface. Because the native module is fused,
  adapter metadata expands either spelling to the paired `gate_proj` +
  `up_proj` tensors instead of pretending one half can be trained alone.
- GLM5 has an OLoRA-tail post-load initialization API for supported unsharded
  attention, dense MLP, and shared-expert LoRA modules. The protocol exposes
  `initialize_lora_olora_tail`, and adapter save metadata can record
  `init_lora_weights="olora_tail"`. A CPU gate now calls the protocol wrapper
  directly to prove the runtime-facing post-load pass reaches a supported
  attention LoRA module. MTP transformer-layer LoRA modules are
  visible to the same OLoRA-tail pass, with a CPU runtime gate proving appended
  MTP layer indexing and attention/dense-MLP initialization. A local CPU gate
  now also wraps a fake GLM5 chunk in an FSDP-style `_fsdp_wrapped_module`
  shell and proves the post-load OLoRA-tail pass still unwraps to the native
  attention LoRA tensor at TP=1. Local CPU DTensor gates now prove OLoRA-tail
  can materialize a DTensor base weight for SVD and write initialized ordinary
  and per-expert grouped LoRA factors back through DTensor-aware whole-parameter
  copies. Routed expert OLoRA-tail now supports exact
  initialization for per-expert grouped LoRA tensors and shared LoRA when
  there is only one local expert; shared LoRA
  spanning multiple different local expert base weights remains explicitly
  skipped because a single shared adapter has no exact per-weight tail subspace.
  The `glm5.lite` package exports `Glm5Model` lazily so CPU-only adapter helper
  tests can import `lora_adapter` without requiring Transformer Engine.
- Qwen3-MoE adapter export/import records and validates `use_rslora`,
  `lora_dropout`, target aliases including `all-linear`, LoRA scale/scaling
  convention metadata, and `init_lora_weights` metadata shape/type. It uses
  the same fused gate/up target convention as GLM5: PEFT `gate_proj` or
  `up_proj` expands to paired `gate_proj` + `up_proj` adapter tensors. Its
  strict load path tracks consumed tensor keys, validates per-key tensor
  shapes before copying, rejects unexpected adapter tensors, and applies the
  same LoRA-only export/load guard as GLM5. It also rejects unknown target
  modules, rejects save attempts whose requested `target_modules` do not match
  exported adapter tensors, and rejects PEFT configs that require unsupported
  non-adapter-only behavior:
  `bias != "none"`, `fan_in_fan_out=True`, or non-empty `modules_to_save`.
  The Qwen3-MoE protocol now exposes the same public adapter lifecycle wrappers
  (`export_lora_adapter_state`, `save_lora_adapter`,
  `load_lora_adapter_state`, and `load_lora_adapter`) so callers do not need to
  bypass the model protocol to exercise the comparison path. A CPU safetensors
  gate now round-trips a qkv-only Qwen3 adapter through those protocol wrappers.
  A separate CPU safetensors gate round-trips per-expert `GroupedLinearLoRA`
  tensors through Qwen3 PEFT expert keys for `gate_proj`, `up_proj`, and
  `down_proj`.
- R3 Router Replay has a Megatron Lite primitive, router hooks, GLM5/Qwen3
  model-level opt-in flags, and a forward-level protocol context driven by
  `PackedBatch.routed_experts` or `PackedBatch.extras["router_replay_action"]`.
  It also has model-scoped set/clear helpers and activation-checkpoint
  recompute wiring that temporarily switches replay routers to
  `REPLAY_BACKWARD`. Forward replay only queues ids for backward recompute after
  `wrap_checkpoint` marks the router, so ordinary replay does not accumulate
  stale per-batch expert ids. The legacy global router registry uses weak
  references so stale routers from discarded model instances do not pollute
  later debug or static helper calls. The protocol-context gate now also
  rejects invalid action values/types, replay data on model chunks without
  router replay instances, forward replay without `batch.routed_experts`, and
  `RECORD` requests that simultaneously provide replay ids.

Current known gaps:

- The first B300A single-GPU TE/CUDA gate passed under
  `runs/20260619-mlite-glm5-lora-te-gpu-smoke`: preflight, TE adapter
  round-trip, DSA dependency import, LoRA forward/backward/optimizer/save-load,
  Router Replay record/replay identity, and OLoRA-tail initialization all have
  collected metrics and `analyze_results.py` status `pass`.
- The first successful GLM5 DSA forward/backward GPU modes used
  `MLITE_DSA_INDEXER_TOPK_BACKEND=torch`, proving the LoRA/FlashMLA sparse
  attention path on real B300A/TE before fused-indexer diagnosis. After the
  SM100 DSA indexer reproducer narrowed the issue to unsupported 1-head tiny
  shapes, `gpu_forward` and `router_replay` were rerun under
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260619-mlite-glm5-lora-fused-indexer-rerun`
  with `MLITE_DSA_INDEXER_TOPK_BACKEND=`. Both modes passed through the default
  fused indexer path on the 64-attention-head / 32-index-head smoke config.
- A dedicated SM100 DSA indexer diagnostic run under
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260619-mlite-glm5-dsa-indexer-repro`
  passed. It reproduced `cudaErrorInvalidValue` for the original
  `tiny_1head_128s` fused indexer shape, but both `smoke_32head_128s` and
  `glm5_32head_512s` passed on default fused dispatch and generic cuDNN. For
  those 32-head shapes, fused and torch reference top-k selected identical
  per-row sets; only the internal top-k order differed. Full GLM5 LoRA
  `gpu_forward`/`router_replay` now also pass without the torch top-k fallback.
- A follow-up Router Replay activation-recompute gate passed under
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260620-mlite-glm5-router-replay-recompute`
  with `MLITE_DSA_INDEXER_TOPK_BACKEND=`. It proves GLM5 sparse-MoE Router
  Replay through real TE/CUDA activation recompute: forward replay queued one
  saved expert-id tensor per router (`[1, 1]`), backward recompute consumed
  both queues (`[0, 0]`), replay token counts matched recorded ids
  (`max_count_diff=0.0`), and LoRA gradients were finite/nonzero.
- GLM5 routed expert adapter import/export has a CPU torch gate for the
  `shared_local_expert_group` representation, including strict rejection of
  non-identical local per-expert tensors that cannot be folded into shared
  native LoRA parameters. A safetensors-backed CPU gate also saves and reloads
  this representation as a PEFT adapter artifact, including
  `expert_lora_representation` metadata.
- R3 now has a torch runtime gate for the RouterReplay primitive, forward-level
  protocol context, and the distinction between ordinary replay and
  checkpoint-backed backward replay queues. The protocol context gate covers
  one-router and multi-router replay data, list input,
  `[routers, tokens, topk]` and `[tokens, routers, topk]` tensor layouts, and
  explicit `REPLAY_BACKWARD` consumption of saved indices. A checkpoint-wrapper
  runtime gate now proves that activation recompute temporarily switches replay
  routers to `REPLAY_BACKWARD`, consumes the saved forward indices, and restores
  the prior action afterward. A tiny GLM5 CPU model-level gate now records
  routed expert ids, changes the natural router scores, replays the recorded
  ids, and verifies the model-level router token counts still follow the
  replayed ids. Companion CPU GLM5+MTP gates now prove router replay instances
  append MTP sparse-layer routers after main-model sparse-layer routers, replay
  tensor installation/cleanup follows that same list order, and a real tiny
  GLM5+MTP forward records/replays both main and MTP router ids in that order
  even when the natural router scores are changed. The same controlled-score
  CPU gate also covers Qwen3-MoE model record/replay identity. B300A GLM5
  sparse-MoE smokes now prove both
  record/replay identity and activation-recompute replay consumption through
  the model with real TE/CUDA. The protocol now converts true-token
  `PackedBatch.routed_experts` into the padded / CP-local THD token grain
  before installing replay ids on routers; CPU runtime gates cover both
  TP-aligned padding and CP=2 zigzag splitting. Remaining R3 coverage is Qwen3
  GPU parity, real-model SP/CP packing smoke, EP routing replay, and pipeline
  schedules.
- The CP=2 + packed THD Router Replay GPU gate is prepared under
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260620-mlite-glm5-cp-thd-router-replay`.
  The B300A and GB300 attempts were blocked by scheduler/container startup
  issues, not model assertions. H100 attempts reached both ranks and proved the
  launcher/source overlay path, but the default H100 container is missing cuDNN
  DSA helper namespaces and `flash_mla`. A clearly gated
  `MLITE_DSA_SPARSE_ATTN_BACKEND=torch` forward-only fallback is now covered by
  the local GLM5 LoRA gates so a later H100x2 run can isolate CP/THD/R3 layout
  semantics from those sparse-attention runtime dependencies. The CP/THD smoke
  records the selected DSA backends, `dsa_kernels.py` source path, and fallback
  call-count delta; the analyzer rejects a torch sparse-attention run if the
  fallback was requested but not called. Any such run is layout smoke evidence
  only; it is not default FlashMLA/fused-indexer or backward evidence.
- OLoRA-tail now has CPU torch runtime gates for the small-matrix SVD-tail
  convention, double-init guard, explicit `force=True` reinitialization, exact
  routed expert initialization for per-expert grouped LoRA, and exact routed
  expert initialization for a single local shared expert. Multi-local-expert
  shared LoRA is intentionally skipped with an explicit reason. A B300A
  TE/CUDA smoke now proves OLoRA-tail reaches supported GLM5 LoRA modules and
  the appended MTP transformer layer. TP-sharded initialization is still a
  follow-up.
- DSA IndexShare training with zero indexer loss has a CPU runtime gate proving
  full layers store top-k indices and shared layers reuse them from the same
  forward holder. A companion gate verifies that a shared layer fails before
  sparse attention when its source top-k is absent, preserving the Cross-PP
  invalid-split boundary. Zero-indexer-loss training now routes through top-k +
  sparse attention, matching the MindLab reference intent more closely than the
  previous unconditional fused-indexer-loss path. Nonzero indexer loss is still
  not complete because the current fused training wrapper does not expose top-k
  for the loss path. A CPU gate verifies this unsupported mode fails before
  dispatching the top-k kernel, with an explicit `NotImplementedError`.
  MTP appended layers now honor `index_share_for_mtp_iteration` when
  `indexer_types` covers only the main decoder layers: GLM-5.2 keeps sharing
  enabled and derives the appended MTP DSA pattern from the MindLab-style
  1-indexed layer number, while a config with the flag disabled treats those
  implicit MTP DSA layers as full indexer layers. The DSA IndexShare PP split
  validator now also includes the appended MTP DSA layer ids on ranks that own
  MTP, so a shared MTP DSA layer whose source top-k lives on another PP stage
  fails at construction time instead of failing later inside sparse attention.
- A tiny GLM5 CPU forward/backward gate still covers structural behavior with
  test-only Transformer Engine and DSA kernel stubs, while the B300A smoke now
  covers real TE/CUDA LoRA forward/backward/freeze/update and adapter
  save/load. Distributed variants, real-model packed THD/CP, and
  production-shape fused indexer remain follow-ups.

## Phase 1: Basic Reproduction

The first reproduction target is not a full paper-scale RL run. It is a ladder
of checks that proves each semantic mechanism is represented correctly before
spending GPU time.

1. Reference audit.
   - Pin the MindLab commit and HF config paths in any run notes. The static
     gate `test_glm5_reference_inputs_pin_mindlab_commit_and_peft_boundaries`
     now checks the local MindLab clone commit and its GLM5/DSA/MTP changed
     files.
   - Verify that HF shared DSA layers have no `self_attn.indexer.*` weights,
     while full DSA layers do.
   - Record the DSA pattern from HF: first three full layers, then repeated
     full/shared/shared/shared with `index_topk_freq=4` and
     `index_skip_topk_offset=3`.
   - Record that HF MTP is represented as appended layer index
     `num_hidden_layers` (`model.layers.78.*` for GLM-5.2) and contains
     `eh_proj`, which is checkpoint state but not a PEFT LoRA target.
   - Record the source boundary: MindLab GLM5.2 commit gives DSA/MLA/MTP
     architecture evidence; rsLoRA, OLoRA-tail, R3, and MinT lifecycle
     semantics are paper-derived and adapted locally.

2. Local static checks.
   - Parse GLM5 config fields from the HF-style config.
   - Parse the local HF GLM-5.2 weight map and assert full/shared DSA indexer
     tensor presence matches `indexer_types`.
   - Assert HF MTP checkpoint state uses appended layer index 78 and that
     `eh_proj` is excluded from GLM5 PEFT adapter targets.
   - Confirm DSA helpers produce the expected full/shared/source-layer pattern.
   - Confirm rsLoRA scale is `alpha / sqrt(rank)` only when `use_rslora=True`.
   - Confirm PEFT `gate_proj` / `up_proj` aliases reach fused `linear_fc1` and
     exported GLM5 adapter metadata records both gate and up tensors.
   - Confirm Router Replay exposes `RECORD`, `REPLAY_FORWARD`, and
     `REPLAY_BACKWARD`, and disables fused routing when replay is active.

3. Minimal runtime smoke, after a torch/TE environment is available.
   - Instantiate tiny GLM5-like configs with DSA IndexShare and verify shared
     layers reuse the source layer top-k.
   - Instantiate tiny Qwen3-MoE and GLM5 MoE routers, record expert ids, replay
     them, and assert the replayed ids are identical.
   - Round-trip a tiny adapter export/import with `use_rslora` metadata.

4. Multi-GPU smoke, only after user confirmation.
   - TP=1/PP=1/EP=1 sanity first.
   - Then FSDP2 with TP=1.
   - Keep TP/ETP as negative gate tests until GLM5 native DSA supports tensor
     parallelism.
   - Then PP layouts that both pass and intentionally fail DSA source-layer
     validation.
   - Then EP routing replay with local expert dispatch.

5. Paper-facing reproduction, optional and later.
   - Reproduce OLoRA-tail on a small public model/config first.
   - Use paper knobs for the small reproduction: rank 16, alpha 32, constant LR
     1e-5, all attention plus MLP projections.
   - The exact Fig.14 target is DeepSeek-R1-Distill-Qwen-1.5B on
     DAPO-Math-17k. The pinned target HF config has `model_type="qwen2"` and
     `architectures=["Qwen2ForCausalLM"]`. Current MLite registry support does
     include a dense `qwen2` identity, a minimal TP=1 lite runtime slice, and
     dense-Qwen2 adapter save/load/import/export, OLoRA-tail init, and HF
     checkpoint load/export mapping at TP=1. The real exact HF safetensors
     validation now covers 339 BF16 tensors, including q/k/v bias pack/unpack
     into native fused qkv bias; the full BF16 model load smoke also loads the
     real checkpoint through the Qwen2 protocol and checks fused tensors for the
     first and last layers. This slice proves local forward/backward with LoRA
     freezing, PEFT-style adapter round-trip, SVD-tail initialization,
     checkpoint tensor mapping, and real model load, but it does not yet provide
     distributed runtime or RL metrics. Split this
     into either a GLM5/Qwen3-MoE MLite proxy smoke labeled as engineering
     evidence, or finish dense-Qwen2 runtime/RL support/use an external HF/PEFT
     baseline before claiming the exact paper reproduction.
   - Treat paper-scale 500-step RL metrics as a separate experiment, not as a
     prerequisite for the code adaptation PR.

## Phase 2: Cognitive Alignment

These are the mental contracts the implementation should preserve.

1. PEFT LoRA is persistent policy state.
   - It should be represented as explicit adapter parameters and metadata.
   - It must be exportable independently from the frozen base.

2. rsLoRA is a scaling convention, not a new module type.
   - Legacy default remains `alpha / rank`.
   - PEFT-compatible rsLoRA uses `alpha / sqrt(rank)`.
   - Adapter configs must record the convention, otherwise rank/learning-rate
     transfer claims are not reproducible.

3. OLoRA-tail is post-load initialization.
   - It uses the smallest singular-vector subspace of the pretrained base
     weight.
   - It sets `B0 = U_tail`, `A0 = V_tail.T`.
   - It must not include singular-value scaling.
   - It should not run during empty module construction because it needs loaded
     base weights.
   - Runtime and VERL entry points should use explicit `impl_cfg.lora_init` /
     `LORA_INIT` knobs. GLM5 supports `olora_tail`; the Qwen3-MoE control path
     now supports TP=1/ETP=1 post-load `olora_tail` initialization for fused
     qkv, `o_proj`, and exactly matched single-local routed expert LoRA.
     TP-sharded or ETP-sharded OLoRA-tail remains explicitly unsupported. Dense
     Qwen2 now also supports TP=1 post-load `olora_tail` for fused qkv, output
     projection, fused gate/up, and down projection LoRA.

4. R3 Router Replay fixes MoE sparse-path mismatch.
   - It records rollout expert ids.
   - It replays the same expert ids during training forward and recompute.
   - It is independent from DSA IndexShare.

5. DSA IndexShare fixes attention indexer reuse.
   - It stores attention top-k positions from full indexer layers.
   - Shared layers reuse positions from the earlier source layer.
   - Cross-PP source reuse is invalid unless explicitly implemented.

## Phase 3: Complete Functional Adaptation

### 3.1 GLM5 LoRA Injection

Initial `lora_config` support is implemented for GLM5 model construction and
threaded through:

- DSA attention PEFT targets:
  - `q_a_proj`
  - `q_b_proj`
  - `kv_a_proj_with_mqa`
  - `kv_b_proj`
  - `o_proj`
- Dense MLP targets:
  - gate/up projection surface
  - down projection surface
- MoE expert targets:
  - expert `linear_fc1`
  - expert `linear_fc2`
- Shared expert targets:
  - shared expert up/gate/down surfaces, if represented separately from routed
    expert `Experts`.
- Optional explicit-only target:
  - DSA indexer projections. Do not train them by default until the loss and
    checkpoint semantics are validated.

Acceptance criteria:

- Default GLM5 construction is unchanged when `lora_config=None`.
- PEFT aliases `r`, `lora_alpha`, and `lora_dropout` work.
- `use_rslora` reaches every instantiated GLM5 LoRA module.
- `freeze_non_lora_params` leaves only adapter tensors trainable.
- Protocol `build_model` records LoRA freeze/trainable stats in the returned
  `ModelBundle` when `ImplConfig.lora` is enabled.
- Static and torch runtime tests cover at least one attention, dense MLP, MoE,
  and shared expert target.

Remaining for this subsection:

- Real Transformer Engine runtime coverage is still required; the local CPU
  LoRA gate uses test-only TE/DSA stubs.
- DSA indexer projections remain intentionally unsupported as default LoRA
  targets.

### 3.2 GLM5 Adapter Checkpoint Mapping

GLM5-specific main-layer adapter import/export is implemented rather than
reusing Qwen3 names blindly.

Required pieces:

- A canonical target-module map from GLM5 local modules to PEFT/HF names.
- `adapter_config.json` fields:
  - `peft_type`
  - `base_model_name_or_path`
  - `r`
  - `lora_alpha`
  - `lora_dropout`
  - `target_modules`
  - `use_rslora`
  - `init_lora_weights`, including `olora_tail` once supported
- Safetensors export for adapter tensors only.
- Import validation for rank, alpha, dropout, `use_rslora`, target modules, and
  tensor shapes.
- Correct routed expert representation handling: native shared local expert
  LoRA exports as PEFT per-expert tensors, imports only compatible local
  per-expert tensors, records the representation in metadata, and round-trips
  through `adapter_model.safetensors` without Transformer Engine.
- Adapter metadata records the rsLoRA/legacy scaling convention alongside rank,
  alpha, dropout, target modules, and the effective LoRA scale.
- A CPU fake-structure adapter round-trip covers `target_modules="all-linear"`
  with an MTP transformer layer using appended HF-style layer indices; this
  validates import/export mapping without requiring Transformer Engine.
- Adapter target validation fails fast for unsupported explicit target names;
  the supported adapter surfaces are attention and MLP projections only.
- CPU adapter round-trip covers both legacy scaling and rsLoRA
  `alpha_over_sqrt_rank` metadata for shared routed expert LoRA.
- LoRA config, `adapter_config.json`, and Megatron Lite metadata reject
  ambiguous scalar types early: `r`/`rank` must be integer values,
  `lora_alpha`/`alpha`, `lora_dropout`/`dropout`, and `scale` must be finite
  JSON numbers, `lora_dropout`/`dropout` must also stay in the `[0, 1]`
  probability range, and `use_rslora` must be a boolean. Numeric strings,
  bool-as-number values, lists, and objects are not accepted before mismatch
  checks.
- PEFT task fields in `adapter_config.json` are explicit too: when `task_type`
  is present it must be `CAUSAL_LM`, and when `inference_mode` is present it
  must be a JSON boolean. `inference_mode=true` remains importable for external
  PEFT serving artifacts, but non-boolean values fail early.
- Megatron Lite lifecycle metadata also keeps explicit scalar boundaries:
  `num_tensors`, `num_parameters`, and `parallel.tp`/`parallel.pp`/
  `parallel.ep`/`parallel.etp` must be integer values, not numeric strings or
  bool-as-number values.
- Once a Megatron Lite metadata sidecar is present, its lifecycle core is not
  optional: `base_model_name_or_path`, `num_tensors`, `num_parameters`,
  `expert_lora_representation`, `lora`, `parallel`, `model`, and `metadata`
  must all be present, even when `adapter_config.json` is absent and import
  relies on caller-provided expected LoRA config.
- The nested lifecycle objects are complete records, not partial hints:
  `lora` must carry rank/alpha/dropout/use_rslora/scaling/scale/targets,
  `parallel` must carry TP/EP/ETP/PP, and `model` must carry the model identity
  fields written by the corresponding adapter helper. These nested lifecycle
  fields, plus user `metadata`, must be JSON objects rather than `null`,
  arrays, or scalars.
- User `metadata` supplied to save APIs must also be mapping/object-like and pass
  standard JSON serialization before the output directory is created; read-only
  mapping proxies are copied into plain JSON objects, while list-of-pairs, sets,
  arbitrary objects, NaN, Infinity, or non-string metadata object keys cannot
  enter the artifact or leave a half-written adapter directory. Mapping values
  nested inside lists or other metadata mappings are recursively copied into
  plain JSON objects. GLM5 and Qwen3-MoE gates cover nested read-only mapping
  proxies, non-standard numeric constants inside nested user metadata, and
  non-serializable objects.
- Save APIs also serialize the final `adapter_config.json` and
  `megatron.lite_adapter_meta.json` payloads as standard JSON before creating
  the output directory, so non-finite model metadata or lifecycle fields cannot
  leave a half-written adapter directory.
- New adapter directories are staged in a sibling temporary directory and only
  committed after all three artifact files are written; safetensors or sidecar
  write failures cannot leave the target adapter directory behind.
- Overwriting an existing adapter directory may only replace the three adapter
  artifact files. If installation fails, save must restore the previous
  `adapter_model.safetensors`, `adapter_config.json`, and
  `megatron.lite_adapter_meta.json` while preserving unrelated files in the
  directory.
- `model` identity fields are type-strict too: integer config fields must stay
  JSON integers, boolean fields must stay JSON booleans, and optional fields
  that are `null` in the base config must remain `null`; loose equality such as
  `2.0 == 2` or `0 == false` is not accepted.
- If `megatron.lite_adapter_meta.json` is present, CPU import parses it before
  reading `adapter_model.safetensors`, then validates its format,
  tensor/parameter counts, native routed-expert LoRA representation, key GLM5
  model fields, LoRA rank/alpha/target modules/scale/scaling convention,
  supported TP/PP/ETP metadata, GLM5 EP metadata scalar type, and user
  metadata object type. The
  Qwen3-MoE adapter path now validates the same adapter artifact and LoRA
  semantics: format, tensor/parameter counts, expert LoRA representation,
  current TP/EP/ETP/PP shape, key model fields, user metadata object type,
  rank, alpha, dropout, `use_rslora`, target modules, scaling convention, and
  effective scale.
- When both `adapter_config.json` and Megatron Lite metadata are present, CPU
  import rejects inconsistent duplicate LoRA/lifecycle fields rather than
  choosing one source silently.
- CPU safetensors import gate rejects mismatched `adapter_config.json` rank,
  alpha, dropout, `use_rslora`, target modules, and invalid
  `init_lora_weights` values without Transformer Engine.
- For non-empty LoRA adapter artifacts, `adapter_config.json` must carry
  identity/scaling fields `peft_type="LORA"`, non-empty
  `base_model_name_or_path`, `r`, `target_modules`, `lora_alpha`,
  `lora_dropout`, and `use_rslora`; missing or empty values fail before relying
  on caller config, metadata, default conventions, or module-specific tensor
  shape checks.
- Adapter save APIs must also strip and reject empty `base_model_name_or_path`
  before creating the output directory, so save cannot produce an
  identity-less artifact or preserve whitespace-polluted base identities that
  will later disagree during import.
- Adapter save APIs must validate `init_lora_weights` before creating the
  output directory, including rejecting empty string init-history values, so
  invalid or identity-less init metadata cannot leave a half-written
  safetensors/config directory.
- `megatron.lite_adapter_meta.json` should record the same
  `base_model_name_or_path`; when both sidecars are present, import must reject
  base-identity disagreement.
- If an external adapter directory lacks `adapter_config.json`, GLM5 and
  Qwen3-MoE CPU import still require an enabled caller-provided expected LoRA
  config, then validate tensor rank, target modules, alpha, dropout, and
  rsLoRA settings against it. The local gates now cover the metadata-only
  sidecar case as well, so `megatron.lite_adapter_meta.json` cannot silently
  replace the caller's expected LoRA config. Without caller config, import must
  fail before reading `adapter_model.safetensors`.
- CPU safetensors import gate also rejects PEFT configs that require unsupported
  non-adapter-only behavior: `bias != "none"`, `fan_in_fan_out=True`, or
  non-empty `modules_to_save`.
- CPU safetensors import gate rejects non-standard or non-object JSON sidecars
  before reading `adapter_model.safetensors`, so `NaN`/`Infinity` constants or
  a top-level array/string in `adapter_config.json` or
  `megatron.lite_adapter_meta.json` produce explicit errors instead of an
  incidental file or attribute error.
- LoRA config and adapter sidecar `target_modules` must be either a string or a
  sequence of strings. Mapping values, scalars, and non-string entries are
  rejected with explicit `TypeError`s before alias expansion. GLM5 and
  Qwen3-MoE share this PEFT target-module type boundary in both the primitive
  normalizer and the adapter helper PEFT target parser.
- Top-level LoRA config normalization accepts mapping-like objects such as
  read-only mapping proxies by copying them into the same validated config path
  as plain dicts. List-of-pairs and mapping-like `target_modules` values remain
  rejected, so object-like config support does not weaken target-module
  semantics. Public GLM5/Qwen3-MoE model, protocol, primitive, and adapter
  helper signatures now annotate caller-provided LoRA configs as
  `Mapping[str, Any]`, matching the runtime contract instead of implying a
  dict-only path.
- Qwen3-MoE parity gates now reject the same unsupported target-module,
  unexported-target, non-object sidecar, LoRA metadata, and Megatron Lite
  metadata lifecycle cases used by the GLM5 adapter path, so Qwen3 remains a
  clean PEFT/R3 comparison target rather than a looser adapter format.
- CPU safetensors import and direct adapter-state load gates reject any
  non-adapter tensor, even when state loading is requested with `strict=False`.
  Direct adapter-state gates also reject non-tensor LoRA values, non-floating
  dtypes, non-2D LoRA A/B tensors, and LoRA tensors with any empty dimension
  or NaN/Inf values before module-specific shape checks.
- Each adapter surface must include paired `lora_A.weight` and
  `lora_B.weight`; half-present LoRA adapters must fail before module-level
  missing-key handling.
- Each LoRA tensor pair must also satisfy `lora_A` rows equal `lora_B` columns;
  A/B rank disagreements must fail before config/metadata or module-level shape
  validation.
- A single adapter artifact still uses one LoRA rank; mixed ranks across
  adapter surfaces must fail at direct state load / safetensors import entry.
- `strict=False` may only ignore extra LoRA tensors on supported adapter
  surfaces; `lm_head`, DSA indexer, unknown projection names, and crossed
  Qwen/GLM LoRA-looking keys fail before module-specific shape/load handling.
- GLM5 and Qwen3-MoE CPU adapter-state load gates report missing required
  tensors, reject unexpected tensors in strict mode, and permit extra tensors
  only with explicit `strict=False`; adapter load APIs parse explicit boolean
  strings such as `"false"` instead of relying on Python truthiness. The
  Qwen3-MoE gate also proves explicit shape mismatch rejection before tensor
  copy/slice for qkv, projection, and shared-expert adapter tensors.
- GLM5 shared-local-expert import must remain lossless even with
  `strict=False`: fused gate/up `lora_A` tensors must match, and expanded
  per-expert tensors must be identical before they can fold back into shared
  native LoRA parameters. `strict=False` only relaxes extra-key handling.
- Public adapter APIs reject unsupported parallel scopes without Transformer
  Engine before attempting tensor import/export: GLM5 currently rejects
  TP/PP/ETP, while Qwen3-MoE rejects PP/ETP and keeps TP/EP on the explicit
  gather/slice adapter path. The local gates now cover the directory-level
  `save_lora_adapter`/`load_lora_adapter` APIs as well as state APIs; load
  uses a missing adapter path to prove unsupported parallel scopes fail before
  artifact reads.
- FSDP2-style wrapper-safe save/load behavior at TP=1 is covered by local
  fake-wrapper gates for GLM5 and Qwen3-MoE; real FSDP2 runtime validation is
  still required. Explicit TP/PP/ETP unsupported gates remain in place for
  adapter import/export until distributed adapter gather/scatter is implemented.
- GLM5 and Qwen3-MoE adapter export must materialize DTensor/FSDP2-style
  parameters through their full tensor representation before writing adapter
  state; saving a `to_local()` shard as the whole adapter is invalid. Local
  fake-DTensor gates cover both ordinary LinearLoRA export paths and routed
  expert export paths; real FSDP2 runtime validation is still required.
- GLM5 and Qwen3-MoE adapter load must also handle DTensor/FSDP2-style LoRA
  parameters: ordinary Tensor adapter state cannot be copied directly into a
  DTensor parameter with `.data.copy_()`. Local CPU DTensor gates now prove
  ordinary GLM5 attention LoRA and Qwen3 qkv LoRA state load by distributing
  the source tensor onto the target parameter's mesh/placements first. Local
  CPU DTensor gates also cover GLM5 and Qwen3-MoE per-expert grouped LoRA
  imports by assembling the rank-local grouped tensor and copying the whole
  DTensor parameter, avoiding unsupported DTensor indexed assignment. Real
  multi-rank EP/FSDP2 validation is still required.
- The MLite runtime now exposes thin adapter helper forwarding from a
  `ModelHandle` to the model protocol: `export_lora_adapter_state`,
  `save_lora_adapter`, and `load_lora_adapter`. This keeps VERL and training
  loops out of protocol internals while preserving protocol-owned validation.
  The forwarding boundary also validates that adapter helpers have protocol,
  model config, a non-empty sequence of non-None model chunks, and parallel
  state before dispatching to the model protocol, so bad runtime handles fail
  with an actionable contract error instead of a protocol-internal attribute or
  shape failure.
- The VERL MLite engine now supports explicit LoRA adapter sidecar
  checkpoints. `checkpoint.save_contents` / `checkpoint.load_contents` may
  include `lora_adapter` for adapter-only saves or full checkpoint plus a
  PEFT-style adapter directory, without changing ordinary model/optimizer
  checkpoint behavior by default. The SFT and GRPO launchers expose this through
  `CHECKPOINT_SAVE_CONTENTS`, `CHECKPOINT_LOAD_CONTENTS`,
  `CHECKPOINT_SAVE_LORA_ADAPTER`, and `LORA_ADAPTER_DIR_NAME`; dry-run gates
  prove those environment knobs map to the expected checkpoint Hydra fields.
  Checkpoint content values are normalized as exact keys, including list/tuple,
  single-string, and bracket/comma-string forms; whitespace and simple shell
  quotes around keys are stripped, and mapping/dict values are rejected rather
  than treated as iterable keys. Strings such as `not_model` cannot
  accidentally trigger model checkpointing by substring match.
  `save_lora_adapter` and `load_lora_adapter` also parse only explicit booleans
  or boolean strings, so `"False"` cannot enable sidecar checkpoints through
  Python truthiness. `lora_adapter_dir_name` must be a relative path inside the
  checkpoint directory; absolute paths and `..` traversal fail before runtime
  adapter save/load is called. `lora_adapter_kwargs` and nested `metadata` must
  be mapping/object values, not list-of-pairs or other non-JSON-object values
  that Python could otherwise coerce with `dict(...)`. When a checkpoint
  requests both model/optimizer state and a LoRA sidecar, the sidecar path,
  kwargs, and runtime adapter API capability are validated before any full
  checkpoint write/load side effects. Sidecar saves also require an enabled
  LoRA config from `engine.impl_cfg.lora` or
  `checkpoint_config.lora_adapter_kwargs.lora_config` before any full
  checkpoint write side effects. Sidecar loads also require the selected adapter
  path to exist as a directory before any full checkpoint load side effects. For
  adapter-only saves, a newly-created checkpoint wrapper directory is removed if
  adapter saving fails; pre-existing checkpoint directories are preserved.
  Load requests whose content set contains no MLite-owned state (`model`,
  `optimizer`, or LoRA adapter sidecar) are skipped before parameter offload or
  CUDA reload side effects.
  Local gates cover adapter-only save/load and user kwargs/init metadata
  forwarding.

Acceptance criteria:

- Exported GLM5 adapters can be reloaded into an equivalent Megatron Lite model.
- Missing/unexpected adapter keys are reported with actionable messages.
- Qwen3 adapter parity gates cover `adapter_config.json` and
  `megatron.lite_adapter_meta.json` validation rather than staying looser than
  GLM5.

Remaining for this subsection:

- Execute the real-model `target_modules="all-linear"` + MTP round trip in an
  environment with torch/TE/safetensors. The no-TE fake-structure gate covers
  adapter key mapping but not real Transformer Engine module construction.
- Promote the enabled-MTP tiny GLM5 round trip from gate to required CI once the
  target CI image has those dependencies.
- Keep TP/PP/ETP adapter save/load as unsupported until GLM5 native DSA and
  distributed adapter gather/scatter have supported paths.

### 3.3 OLoRA-Tail Initialization

Implement OLoRA-tail as an explicit post-base-load pass.

Initial scope:

- TP=1 is implemented for unsharded GLM5 LoRA modules.
- TP=1/ETP=1 is implemented for the Qwen3-MoE control path as an engineering
  proxy: fused qkv/o_proj LoRA and single-local routed expert LoRA can be
  initialized from the native base weights; shared LoRA spanning multiple local
  experts is still skipped rather than approximated.
- FSDP2-style wrapper unwrapping is covered by a fake local CPU gate; real
  FSDP2 runtime validation is still required.
- DTensor base-weight materialization and DTensor LoRA parameter writes are
  covered by local CPU gates for ordinary attention LoRA and per-expert grouped
  routed LoRA. Real multi-rank FSDP2 validation is still required.
- Attention, dense MLP, and shared-expert projections have an API-level
  implementation.
- Routed grouped expert weights are initialized when the native LoRA parameter
  can be matched exactly to a single base weight: per-expert grouped LoRA, or
  shared LoRA with exactly one local expert. Shared LoRA over multiple local
  experts is skipped rather than approximated.

Algorithm:

1. For each selected base weight `W0`, compute SVD.
2. Select the smallest `rank` singular-vector columns.
3. Set LoRA `B` to `U_tail`.
4. Set LoRA `A` to `V_tail.T`.
5. Do not multiply by singular values.
6. Record metadata so exported adapters show `init_lora_weights=olora_tail`.
7. Reject non-finite base weights before SVD; SVD may run in fp32 for stability,
   but written LoRA A/B tensors must be finite and keep the target module
   dtype/device.

Follow-up scope:

- TP-sharded weights by gather/init/scatter or equivalent local-SVD rule with a
  documented approximation.
- TP-sharded grouped MoE experts, and an explicit design decision for whether
  multi-local-expert shared LoRA should remain skipped or gain a documented
  approximation.
- Real GLM5+TE runtime validation for MTP transformer-layer OLoRA-tail
  initialization beyond the current fake-chunk CPU gate.
- Memory-guarded randomized SVD for very large matrices.

Acceptance criteria:

- Static tests assert the OLoRA-tail API, metadata, SVD-tail contract, and
  no-silent-double-init guard.
- A deterministic torch unit test compares initialized LoRA tensors against a
  known small-matrix SVD-tail subspace and verifies singular values are not
  applied.
- A regression gate rejects non-finite SVD output before it can be copied into
  adapter parameters.
- A tiny runtime test shows nonzero adapter deltas after init.
- Initialization is never silently applied twice; explicit `force=True`
  reinitializes from the current base weights.

### 3.4 R3 Runtime Integration

Complete Router Replay beyond primitive/model wiring.

Runtime data contract:

- `ModelOutputs.routed_experts` is a list ordered by router instantiation.
- `PackedBatch.routed_experts` carries the same ordered list for replay. CPU
  GLM5+MTP structure and forward-behavior gates now cover main-model routers
  followed by MTP routers.
- Record and replay tensors encode expert ids and must have shape
  `[tokens, topk]` with integer tensor dtypes. Float, bool, or complex tensors
  are rejected instead of being silently cast to `long`, preserving the paper's
  verbatim replay semantics.
- `RECORD` mode validates the default-topk output against the current router
  scores before saving a rollout trace: shape and expert-id range errors fail
  at the source rather than becoming recorded replay artifacts.
- `REPLAY_BACKWARD` validates saved indices before consuming the activation-
  recompute queue. Validation failures preserve the queued trace for diagnosis
  or retry instead of popping it first.
- Record/replay tensors are snapshot when written into router state, and
  `get_recorded_indices` returns a snapshot too. External mutation or later
  cleanup must not alter a rollout trace that has already been handed to the
  runtime/artifact layer.
- Legacy/debug global helpers such as `set_replay_data` and
  `get_recorded_data` follow the same contract: callers pass a list/tuple of
  per-router tensors and receive snapshot-isolated recorded traces.
- By default `PackedBatch.routed_experts` uses the true-token grain emitted by
  rollout/reference passes. Protocol packing pads and CP-splits those tensors
  before installing them on routers. If a diagnostic or distributed path passes
  pre-packed replay tensors, it must set `batch.extras["router_replay_layout"]`
  to `full_padded` or `cp_local`; row-count mismatches fail before router state
  is installed.
- Each installed tensor shape is `[local_tokens, topk]` at the same token grain
  as the router input after packing/SP/CP handling.
- MTP routers are appended after main-model routers in model order.
- Primitive and protocol action setters accept only `RouterReplayAction` values
  or the valid strings `record`, `replay_forward`, and `replay_backward`.
  Invalid strings/types fail early instead of silently falling back to natural
  routing.

Action contract:

- Rollout/reference pass: set action `RECORD`, collect output
  `routed_experts`, clear action.
- Training forward: load `PackedBatch.routed_experts`, set action
  `REPLAY_FORWARD`, clear action after forward.
- Activation recompute/backward: set action `REPLAY_BACKWARD` with the saved
  forward indices.
- Non-R3 mode: no output `routed_experts`, no replay state, fused routing stays
  enabled.

Implementation tasks:

- Model instance methods for setting replay data/actions are implemented for
  GLM5 and Qwen3-MoE.
- Thread `routed_experts` through protocol `_forward_step` and batch packing.
- Add primitive/context runtime gate for record, forward replay, backward
  replay, action string parsing and bad-action errors, multi-router list/tensor
  data layouts, record/replay dtype/range/shape validation errors, snapshot
  isolation, and action cleanup.
- Add protocol-context coverage for explicit THD replay layouts: default
  true-token replay, `full_padded` split to CP-local, `cp_local` passthrough,
  and invalid layout value/type errors.
- Add fake FSDP-style wrapper coverage proving protocol-context replay packing
  uses the unwrapped model's parallel state for THD/CP layout instead of
  falling back to the outer wrapper's default `cp_size=1`.
- Add tiny GLM5 model-level CPU gate for record/replay identity with controlled
  router scores.
- Add static buffer support only if CUDA graph capture requires it.
- Keep lifecycle cleanup in place to avoid stale global instances across model
  rebuilds.
- A local checkpoint-wrapper test proves activation recompute consumes
  `REPLAY_BACKWARD` indices in the same order as forward replay. The remaining
  gate is model-level identity under real activation recompute and distributed
  schedules.

Acceptance criteria:

- RouterReplay primitive/context gate proves record, `REPLAY_FORWARD`,
  `REPLAY_BACKWARD`, multi-router list/tensor layout handling, bad shape/range
  errors, and cleanup semantics.
- Record/replay identity tests pass on GLM5 and Qwen3-MoE tiny CPU model gates;
  real distributed identity gates remain follow-ups.
- Replay mode fails fast when tensors are missing, wrong count, or wrong shape.
- Fused router is disabled only while replay is active.

### 3.5 DSA IndexShare Training Completeness

Current support is enough for config/readiness and no-indexer-loss paths, but
complete training requires:

- Fused training kernel or wrapper returns top-k indices needed by the DSA
  indexer loss.
- Non-fused fallback path computes the same loss for validation.
- Packed THD and CP paths use the same holder key semantics as inference.
- MTP respects `index_share_for_mtp_iteration`; the local config gate covers
  both GLM-5.2's sharing-enabled appended layer semantics and the disabled
  fallback where implicit MTP DSA layers are full indexer layers. The local
  construction gate also verifies that MTP DSA layer ids participate in the
  same PP split validator as main decoder layers.

Acceptance criteria:

- Training with `dsa_indexer_loss_coeff=0` works with IndexShare enabled; the
  local CPU gate covers full-layer top-k storage and shared-layer top-k reuse.
- Training with `dsa_indexer_loss_coeff>0` either works or fails with the
  current explicit `NotImplementedError`; the local CPU gate covers the
  explicit-failure path.
- Once implemented, fused and non-fused indexer-loss values match on a tiny
  deterministic case.

### 3.6 MinT/Serving Compatibility

Keep exported adapters compatible with a policy-lifecycle view:

- Base checkpoint remains immutable.
- Adapter revision carries all PEFT metadata.
- Router replay traces are rollout/training artifacts, not adapter weights.
- DSA IndexShare config belongs to the base model config, not adapter config.

Acceptance criteria:

- Adapter export is small and contains no base weights.
- Serving can choose legacy LoRA or rsLoRA scale from metadata.
- OLoRA-tail metadata describes initialization history without re-running it at
  load time unless explicitly requested.

## Verification Matrix

Run locally:

```bash
cd /Users/lmei/Documents/NVIDIA/Megatron/mlite/experimental/lite
TORCH_PYTHON_BIN=/Users/lmei/anaconda3/envs/cua/bin/python \
  tests/run_glm5_lora_local_gates.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  megatron/lite/model/glm5/config.py \
  megatron/lite/model/glm5/lite/model.py \
  megatron/lite/model/glm5/lite/protocol.py \
  megatron/lite/model/glm5/lite/lora_adapter.py \
  megatron/lite/model/qwen3_moe/lite/model.py \
  megatron/lite/model/qwen3_moe/lite/protocol.py \
  megatron/lite/model/qwen3_moe/lite/lora_adapter.py \
  megatron/lite/primitive/modules/attention/dsa.py \
  megatron/lite/primitive/modules/router_replay.py \
  megatron/lite/primitive/modules/router.py \
  megatron/lite/primitive/utils/moe.py \
  megatron/lite/primitive/modules/experts.py \
  megatron/lite/primitive/modules/gqa.py \
  megatron/lite/primitive/modules/lora.py \
  tests/unit/primitive/test_peft_dsa_router_replay_static.py \
  tests/unit/primitive/test_router_replay_runtime.py \
  tests/unit/model/test_glm5_lite_static.py \
  tests/unit/model/test_glm5_lora_adapter_runtime.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=megatron/lite \
  pytest -q tests/unit/primitive/test_peft_dsa_router_replay_static.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=megatron/lite \
  pytest -q tests/unit/model/test_glm5_lite_static.py \
  -k 'config_reads or ignores_null or indexshare or reference_inputs or preserves_mtp or uses_shared_mla or parallel_scope'
```

Completed in local torch/CPU gates:

- Tiny GLM5 DSA IndexShare forward covers zero-loss top-k reuse through patched
  kernel entry points, and the tiny GLM5 CPU smoke covers base and MTP
  forward/backward through test-only TE/DSA stubs.
- Tiny GLM5 LoRA forward/backward/freeze covers structural adapter gradients
  and frozen-base behavior through test-only TE/DSA stubs.
- RouterReplay primitive/context runtime gate:
  `pytest -q tests/unit/primitive/test_router_replay_runtime.py`.
- RouterReplay protocol-context THD/CP layout gate: true-token rollout
  `PackedBatch.routed_experts` is padded to the local THD layout and CP-split
  with the same zigzag rule used for `input_ids`.
- Tiny Qwen3-MoE and GLM5 Router Replay model-level record/replay.
- GLM5 adapter export/import round trip:
  `pytest -q tests/unit/model/test_glm5_lora_adapter_runtime.py`.

Completed on B300A after explicit confirmation:

- Single-GPU smoke under
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260619-mlite-glm5-lora-te-gpu-smoke`.
- `MODE=preflight`, `MODE=te_adapter`, `MODE=dsa_deps`,
  `MODE=gpu_forward`, `MODE=router_replay`, and `MODE=olora_tail`.
- Final analyzer status: `pass`.
- Initial GLM5 DSA forward modes used `MLITE_DSA_INDEXER_TOPK_BACKEND=torch`,
  then `gpu_forward`/`router_replay` were rerun without the torch fallback under
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260619-mlite-glm5-lora-fused-indexer-rerun`.
- Router Replay activation recompute passed under
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260620-mlite-glm5-router-replay-recompute`
  with analyzer status `pass`; replay queues were `[1, 1]` before backward and
  `[0, 0]` after backward, with finite nonzero LoRA gradients.

GPU follow-ups:

- Real FSDP2 smoke at TP=1, plus TP/ETP unsupported-gate smoke. The local
  fake-wrapper gate only proves adapter unwrapping/save-load semantics.
- PP validation smoke and EP Router Replay smoke. A prepared PP DSA boundary
  gate lives at
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260620-mlite-glm5-pp-boundary`.
  The first 2-node B300A launch shape was rejected by SLURM before allocation
  with `Requested topology configuration is not available`. A confirmed
  single-node 2-rank shared-B300A fallback then passed with analyzer status
  `pass`: valid PP layout `E|t*2,L` built on both ranks with LoRA stats, and
  invalid layout `E,t|t,L` failed on rank 1 with the expected
  `Cross-PP top-k sharing is not supported` DSA IndexShare boundary message.
  Caveat: this is CUDA/TE PP split-construction evidence, not multi-node PP
  communication evidence.
- CP/THD packed sequence real-model smoke. The CPU protocol contract is now
  covered locally. A prepared GLM5 protocol-forward GPU gate lives at
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260620-mlite-glm5-cp-thd-router-replay`;
  it has local artifact/guard checks. The default B300A single-node 2-GPU
  launch failed before allocation because each `b300a_preprod` node exposes
  only one B300A, and the 2-node B300A shape also fails `sbatch --test-only`.
  The confirmed `galaxy_gb300_preprod_pairx2` 2-node GB300 run allocated, but
  failed before model execution: rank 0 on `galaxy-ts4-055` could not import
  the NGC container because enroot layer downloads returned 401, and rank 1
  then timed out connecting to TCPStore. Current scheduler candidates are
  single-node `gb300nvl72_preprod` with 2 x GB300 (test-only pass, queued
  later), `H100x2` with 2 x H100 (test-only pass, fastest fallback), or
  `b100x4_preprod` with 2 x B100 (test-only pass, Blackwell-family fallback).
  These candidates were refreshed with no allocation at
  `2026-06-19T18:48:39Z` on `dlcluster-login-01`. A confirmed single-node
  `gb300nvl72_preprod` attempt then allocated on `gb300-nvl-022-compute01`,
  but failed before Python/model execution because Pyxis could not start the
  container. After that failure, no-allocation test-only checks showed `H100x2`
  as the fastest fallback (`2u2g-gen-0857`, projected start
  `2026-06-19T19:36:04Z`) and `b100x4_preprod` as the Blackwell-family
  fallback (`4u4g-gen-0178`, projected start `2026-06-19T21:27:05Z`). Any
  actual fallback launch still requires explicit confirmation before
  allocation. A confirmed `H100x2` attempt reached container startup, Python,
  and both NCCL ranks on `2u2g-gen-0857`, but failed before metrics because the
  H100 container did not expose the cuDNN DSA namespace required by the default
  fused indexer top-k path. The next useful diagnostic is the same `H100x2`
  shape with `MLITE_DSA_INDEXER_TOPK_BACKEND=torch`, which isolates CP/THD/R3
  replay layout from fused-indexer availability but should be labeled as torch
  top-k fallback evidence, not default fused-indexer evidence. The first
  confirmed H100 torch-top-k attempt failed before distributed startup because
  the launcher source-overlay tar stream carried macOS AppleDouble/xattr
  metadata and both ranks extracted into one shared overlay directory. The
  launcher has since been fixed to use `COPYFILE_DISABLE=1 tar` and
  rank-local source-overlay directories; the next H100 torch-top-k retry should
  use that fixed launcher. That fixed-launcher retry reached both ranks with
  `MLITE_DSA_INDEXER_TOPK_BACKEND=torch`, but still failed before metrics
  because the packed DSA sparse-attention helper `build_flat_topk_idxs`
  requires the same missing cuDNN DSA namespace. The next local code-side check
  was whether this helper can gain a torch fallback under the torch top-k mode,
  preserving default fused-indexer semantics while enabling H100 CP/THD/R3
  fallback evidence. That fallback is now implemented locally:
  `build_flat_topk_idxs` uses the torch compactify path on CUDA only when
  `MLITE_DSA_INDEXER_TOPK_BACKEND=torch`, the default CUDA path remains the
  cuDNN DSA compactify wrapper, and local gates cover the compacted flat-index
  layout. A rerun then showed that merely adding `dsa_kernels.py` to a
  `PYTHONPATH` overlay was insufficient because the regular `megatron` package
  continued to load from the staged `/home` source tree. The launcher now
  copies the staged source into a rank-local `/tmp` source tree and extracts the
  overlay into that tree before running, so the next H100 torch-top-k rerun can
  exercise the updated helper. Acceptance for a full default-backend CP/THD
  gate still requires actual CP ranks `{0, 1}` and
  `target_matches_expected=true` for the padded / CP-local replay ids.
- Production-like cuDNN SM100 DSA indexer-forward reproducer. A dedicated
  diagnostic run artifact is prepared at
  `/Users/lmei/Documents/NVIDIA/Megatron/runs/20260619-mlite-glm5-dsa-indexer-repro`;
  it compares the torch reference top-k path, default fused dispatch, and the
  generic cuDNN DSA wrapper on the known tiny and GLM5-like shapes. The
  diagnostic run passed: 1-head fused shapes fail with `cudaErrorInvalidValue`,
  while 32-head smoke/GLM5-like shapes pass and match the torch reference at
  the per-row top-k set level.
- Optional OLoRA-tail small RL reproduction.

Prepared run artifacts for the first GLM5 LoRA TE/GPU gate are kept at
`/Users/lmei/Documents/NVIDIA/Megatron/runs/20260619-mlite-glm5-lora-te-gpu-smoke`.
The run remains guarded by `CONFIRM_GPU_RUN=yes`; its `status.md` records job
IDs, failures, fixes, collected metrics, and caveats.

## Stop Criteria

The adaptation is complete when:

- GLM5 PEFT LoRA can be trained, saved, loaded, and exported with rsLoRA and
  legacy scaling.
- OLoRA-tail can initialize supported GLM5 LoRA targets after base load and is
  represented in adapter metadata; Qwen3-MoE has a TP=1/ETP=1 proxy path for
  fused qkv/o_proj and exactly matched single-local routed expert LoRA.
- R3 can record rollout expert ids and replay them through training forward and
  recompute for both Qwen3-MoE and GLM5.
- DSA IndexShare matches HF/MindLab config semantics and has a validated
  training path or an explicit documented limitation for indexer-loss training.
- Local static tests, torch runtime smoke, and approved GPU smoke all pass.
