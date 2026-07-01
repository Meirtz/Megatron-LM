#!/usr/bin/env python3
"""DeepSeek-V4 forward-logit parity: mlite (native) vs HF transformers (the trusted reference).

The Megatron-Bridge/MCore path can't be the DSv4 reference: bridge maps `mlp.router.tid2eid`
but the pinned MCore has no such hash-routing buffer, so the first `num_hash_layers` MoE layers
route wrong. mlite DOES load + use tid2eid. This script feeds ONE fixed input through both the
HF model (trust_remote_code) and the mlite native model on the same toy checkpoint, and reports
logit cosine + per-position cosine, to (a) validate mlite ds4 and (b) explain the ~0.998 that the
bridge-vs-transformers comparison showed.

Run (1 GPU): python hf_vs_mlite_ds4.py <toy_hf_dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_dir", nargs="?", default="/home/scratch.lmei_other/dsv4-flash-toy")
    parser.add_argument("seq_len", nargs="?", type=int, default=128)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--attention-backend",
        default=os.environ.get("DS4_ATTENTION_BACKEND"),
        choices=("auto", "flash", "fused", "unfused", "local", None),
        help="Optional DS4 attention backend override. Use local/unfused for the independent fallback proxy.",
    )
    return parser.parse_args()


ARGS = _args()
HF = ARGS.hf_dir
SEQ = ARGS.seq_len
ATTENTION_BACKEND = ARGS.attention_backend
SEED = 42

cfg = json.load(open(f"{HF}/config.json"))
vocab = cfg["vocab_size"]
g = torch.Generator().manual_seed(SEED)
ids_1d = torch.randint(0, vocab, (SEQ,), generator=g)  # shared fixed input
print(
    f"[parity] toy={HF} seq={SEQ} vocab={vocab} num_hash_layers={cfg.get('num_hash_layers')} "
    f"attention_backend={ATTENTION_BACKEND or 'default'}",
    flush=True,
)


def _mlite_logits() -> torch.Tensor:
    from megatron.lite.runtime import RuntimeConfig, create_runtime
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
    from megatron.lite.runtime.contracts.data import PackedBatch

    bcfg = MegatronLiteConfig(
        model_name="deepseek_v4",
        impl="lite",
        hf_path=HF,
        parallel=ParallelConfig(tp=1, etp=1, ep=1, pp=1, cp=1),
        optimizer=OptimizerConfig(lr=0.0, min_lr=0.0, weight_decay=0.0),
        load_hf_weights=True,
        # This toy config advertises one MTP layer, but the toy safetensors do
        # not contain MTP weights. Keep P1 focused on main-logit CSA/DSA/hash/mHC
        # parity; MTP weight/train coverage belongs to P3/P6 with a matching ckpt.
        impl_cfg={
            "optimizer": "fsdp2",
            "deterministic": True,
            "mtp_enable": False,
            "attention_backend_override": ATTENTION_BACKEND,
        },
    )
    from examples.bench.correctness import _forward_logits

    rt = create_runtime(RuntimeConfig(backend="mlite", hf_path=HF, backend_cfg=bcfg))
    handle = rt.build_model()
    # DS4's MLite runtime contract uses PackedBatch, not a raw dict. Keep this
    # proxy to one sequence so the reference input, seed, and sequence length
    # stay controlled for precision alignment.
    ids_cuda = ids_1d.cuda()
    batch = PackedBatch(
        input_ids=ids_cuda,
        labels=None,
        seq_lens=torch.tensor([SEQ], dtype=torch.int64, device=ids_cuda.device),
        loss_mask=None,
    )
    logits = _forward_logits(rt, handle, batch)
    if logits is None:
        raise RuntimeError("mlite _forward_logits returned None.")
    logits = logits.detach().float().cpu()
    print(f"[parity] mlite raw logits shape = {tuple(logits.shape)}", flush=True)
    # Squeeze a leading batch dim (BSH path returns [1, SEQ, V]); do NOT blind-reshape —
    # mlite pads the vocab dim (e.g. to a multiple of 128), so reshape(SEQ, true_vocab) would
    # misalign every position. Keep the model's own [SEQ, V_pad]; the caller slices to true vocab.
    logits = logits.reshape(-1, logits.shape[-1])  # [SEQ, V_pad]
    return logits


def _hf_logits() -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    # Load fp32: the toy DSv4 modeling upcasts RMSNorm/mHC to fp32 and then feeds a linear
    # (q_a_proj). In bf16 that's a dtype mismatch (crash); an autocast(bf16) band-aid silently
    # corrupted the fp32-sensitive ops (RoPE / DSA indexer / softmax) -> garbage logits.
    # fp32 weights make the whole graph consistent with no autocast. This is the trusted reference;
    # mlite runs bf16, so expect a small (not orthogonal) gap.
    model = (
        AutoModelForCausalLM.from_pretrained(HF, trust_remote_code=True, torch_dtype=torch.float32)
        .cuda()
        .eval()
    )
    with torch.no_grad():
        out = model(input_ids=ids_1d.unsqueeze(0).cuda())
    hl = out.logits[0].detach().float().cpu()  # [SEQ, V]
    print(
        f"[parity] HF raw logits shape = {tuple(hl.shape)}  "
        f"finite={bool(torch.isfinite(hl).all())} min={hl.min():.3f} max={hl.max():.3f} "
        f"row0_argmax={int(hl[0].argmax())}",
        flush=True,
    )
    return hl


print("[parity] HF forward ...", flush=True)
hl = _hf_logits()
print("[parity] mlite forward ...", flush=True)
ml = _mlite_logits()

# Align: both are [SEQ, V*]; mlite may pad vocab. Compare over the true vocab columns only.
V = min(hl.shape[-1], ml.shape[-1], vocab)
vocab_padded = ml.shape[-1] != hl.shape[-1]
if ml.shape[-1] != hl.shape[-1]:
    print(f"[parity] NOTE vocab dim differs (HF={hl.shape[-1]} mlite={ml.shape[-1]}); comparing first {V} cols", flush=True)
ml = ml[:, :V].contiguous()
hl = hl[:, :V].contiguous()

cos = torch.nn.functional.cosine_similarity(ml.flatten(), hl.flatten(), dim=0).item()
maxabs = (ml - hl).abs().max().item()
pcos = torch.nn.functional.cosine_similarity(ml, hl, dim=-1)  # [SEQ]
# argmax (greedy next-token) agreement
agree = (ml.argmax(-1) == hl.argmax(-1)).float().mean().item()
metrics = {
    "hf_dir": HF,
    "seq_len": SEQ,
    "seed": SEED,
    "vocab": vocab,
    "compared_vocab": V,
    "hf_shape": list(hl.shape),
    "mlite_shape": list(ml.shape),
    "vocab_padded": bool(vocab_padded),
    "hf_logits_finite": bool(torch.isfinite(hl).all()),
    "mlite_logits_finite": bool(torch.isfinite(ml).all()),
    "logit_cosine": float(cos),
    "max_abs_logit_diff": float(maxabs),
    "per_pos_cosine_min": float(pcos.min().item()),
    "per_pos_cosine_mean": float(pcos.mean().item()),
    "argmax_agreement_pct": float(agree * 100.0),
    "mtp_enable": False,
    "attention_backend": ATTENTION_BACKEND or "default",
    "threshold_policy": "baseline: no numeric pass/fail threshold; hard fail only on crash, nonfinite, missing marker, or missing metrics",
}
if ARGS.json_out:
    ARGS.json_out.parent.mkdir(parents=True, exist_ok=True)
    ARGS.json_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[parity] wrote metrics json = {ARGS.json_out}", flush=True)
print(
    f"\n[PARITY RESULT] mlite vs HF DSv4 toy:\n"
    f"  logit cosine        = {cos:.6f}\n"
    f"  max abs logit diff  = {maxabs:.4f}\n"
    f"  per-pos cosine min  = {pcos.min().item():.6f}  mean = {pcos.mean().item():.6f}\n"
    f"  argmax agreement    = {agree*100:.2f}%\n"
    f"  per-pos cosine[:8]  = {[round(x,4) for x in pcos[:8].tolist()]}",
    flush=True,
)
print("PARITY_DONE", flush=True)
