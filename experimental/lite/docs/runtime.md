# Runtime

The public runtime entrypoint is `megatron.lite.runtime`.

```python
from megatron.lite.runtime import MegatronLiteConfig, ParallelConfig, RuntimeConfig, create_runtime

cfg = RuntimeConfig(
    backend="mlite",
    hf_path="/path/to/hf-model",
    backend_cfg=MegatronLiteConfig(
        model_name="qwen3_moe",
        impl="lite",
        parallel=ParallelConfig(tp=1, pp=1, cp=1, ep=1),
    ),
)
runtime = create_runtime(cfg)
handle = runtime.build_model()
```

## API Tiers

All runtime backends implement the pretraining tier:

- `build_model`
- `save_checkpoint`
- `load_checkpoint`
- `train_mode`
- `eval_mode`
- `forward_backward`
- `zero_grad`
- `optimizer_step`
- `lr_scheduler_step`

The lite runtime also implements `export_weights` and `to` when the underlying
model and optimizer support those operations.

## Fatal Checkpoint Loads

`MegatronLiteRuntime.load_checkpoint` separates read-only preflight from final
live-state application. Metadata, sidecar, topology, and schema failures found
during preflight raise their ordinary `Exception` and leave `handle` usable.

If model, optimizer, RNG, or registered extra-state application has started and
the load then fails, it raises `CheckpointLoadFatalError` and permanently sets
`handle.poisoned` to `True`. The target state may be partially restored on this
rank or a distributed peer. The handle must be discarded and rebuilt; every
later lite-runtime operation using it (including save/load, mode changes,
forward/backward, optimizer/scheduler steps, export, and offload) fails closed.
The original error is available as `handle.poison_reason`.

Non-`Exception` control flow (`KeyboardInterrupt`, `SystemExit`, cancellation,
or another `BaseException`) is never converted or swallowed: the source rank
re-raises the original object. Because the exact interruption point cannot be
proven safe, every participating runtime fails closed and poisons its handle;
peer ranks receive `CheckpointLoadFatalError`. Discard the handle even if the
interrupt appeared to occur during preflight. `ModelHandle` is also deliberately
uncopyable and unpicklable so an alias cannot bypass this shared-state poison
boundary.

Registered extra-state validators and target adapters are trusted transaction
code. Their `validate`, optional `validate_step`, `snapshot`, and `fingerprint`
callbacks must be side-effect free and may return only canonical plain-data
fingerprints; `apply` and `restore` must mutate only their own target. Coupled
state belongs in one composite target. MLite scans all registered fingerprints
around every callback, but cannot dynamically prove that the very first
successful `fingerprint()` call itself was pure; registering an adapter asserts
that boundary explicitly.

```python
from megatron.lite.runtime import CheckpointLoadFatalError

try:
    runtime.load_checkpoint(handle, checkpoint_path)
except CheckpointLoadFatalError:
    # Never reuse handle here.
    handle = runtime.build_model()
    raise
```

The `mbridge` runtime implements the same runtime contract through the legacy
`mbridge` package and Megatron-Core optimizer/checkpoint helpers. The benchmark
example currently uses this backend for validated reference runs.

The `bridge` runtime is the real Megatron-Bridge path. It imports
`megatron.bridge` lazily from `build_model()`, so config construction and dry-run
examples can execute without Megatron-Bridge installed.

## Config Types

`RuntimeConfig` selects the backend and carries the Hugging Face model path.

`MegatronLiteConfig` carries `mlite` backend settings:

- `model_name`: `qwen3_moe` or `qwen3_5` for new configs. `qwen3` remains
  accepted as a legacy alias for `qwen3_moe` only; dense Qwen3 is not included.
- `impl`: currently only `lite`.
- `parallel`: tensor, expert, pipeline, virtual pipeline, and context sizes.
- `optimizer`: Megatron-Core optimizer settings.
- `impl_cfg`: model-specific options consumed by each model protocol.

`BridgeConfig` carries shared `mbridge` and `bridge` backend settings:

- `model_name`: optional model identifier used for benchmark metadata.
- `parallel`: tensor, expert, pipeline, virtual pipeline, and context sizes.
- `optimizer`: Megatron-Core optimizer settings.
- `override_ddp_config`, `override_transformer_config`, and
  `override_optimizer_config`: explicit reference-backend/Core override maps.
- `param_offload` and `optimizer_offload`: offload model/optimizer state between
  train/eval contexts.

## Backend Registry

The built-in backend keys are `mlite`, `mbridge`, and `bridge`. Model
implementations for the native runtime remain selected through
`MegatronLiteConfig.impl`, which currently supports `impl="lite"`.

Custom runtime backends can be registered with:

```python
from megatron.lite.runtime import register_runtime

register_runtime("my_backend", "my_package.my_runtime")
```

The target module must expose `create(hf_path, cfg)`.
