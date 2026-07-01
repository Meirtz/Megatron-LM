"""Static and CPU smoke tests for native GLM-5 lite."""

from __future__ import annotations

from pathlib import Path


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _hf_glm52_reference_dir() -> Path:
    return _workspace_root() / "references" / "external" / "hf-zai-org-GLM-5.2"


def _mindlab_glm52_reference_dir() -> Path:
    return _workspace_root() / "references" / "external" / "Megatron-GLM5.2"


def _peft_paper_notes_path() -> Path:
    return _workspace_root() / "peft-mint-repro" / "references" / "main" / "insights.md"


def _install_te_rmsnorm_stub_for_cpu_dsa(torch) -> None:
    from contextlib import nullcontext
    import sys
    import types

    try:
        import transformer_engine.pytorch as te  # noqa: F401
    except (ModuleNotFoundError, OSError):
        te = None
    if te is not None:
        try:
            te.RMSNorm(2)
            import transformer_engine.pytorch.cpp_extensions  # noqa: F401
            import transformer_engine.pytorch.permutation  # noqa: F401
            import transformer_engine.pytorch.router  # noqa: F401
            return
        except Exception:
            te = None

    class _RMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-5, **kwargs):
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            variance = x.float().pow(2).mean(dim=-1, keepdim=True)
            return (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight

    class _Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=False, **kwargs):
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
            self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
            torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

        def forward(self, x):
            bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
            return torch.nn.functional.linear(x, self.weight.to(dtype=x.dtype), bias)

    class _LayerNormLinear(_Linear):
        def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            eps=1e-5,
            zero_centered_gamma=False,
            **kwargs,
        ):
            super().__init__(in_features, out_features, bias=bias, **kwargs)
            self.layer_norm_weight = torch.nn.Parameter(torch.ones(in_features))
            self.layer_norm_bias = None
            self.eps = eps
            self.zero_centered_gamma = zero_centered_gamma

        def forward(self, x):
            weight = self.layer_norm_weight
            if self.zero_centered_gamma:
                weight = weight + 1.0
            variance = x.float().pow(2).mean(dim=-1, keepdim=True)
            normed = (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype)
            normed = normed * weight.to(device=x.device, dtype=x.dtype)
            return super().forward(normed)

    class _GroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, in_features, out_features, bias=False, **kwargs):
            super().__init__()
            del kwargs
            self.num_gemms = int(num_gemms)
            self.bias_enabled = bool(bias)
            for idx in range(self.num_gemms):
                weight = torch.nn.Parameter(torch.empty(out_features, in_features))
                torch.nn.init.normal_(weight, mean=0.0, std=0.02)
                self.register_parameter(f"weight{idx}", weight)
                if self.bias_enabled:
                    self.register_parameter(f"bias{idx}", torch.nn.Parameter(torch.zeros(out_features)))

        def forward(self, x, m_splits):
            pieces = []
            start = 0
            for idx, split in enumerate(m_splits):
                split = int(split)
                part = x[start : start + split]
                start += split
                weight = getattr(self, f"weight{idx}").to(dtype=x.dtype)
                bias = None
                if self.bias_enabled:
                    bias = getattr(self, f"bias{idx}").to(dtype=x.dtype)
                pieces.append(torch.nn.functional.linear(part, weight, bias))
            return torch.cat(pieces, dim=0) if pieces else x.new_empty(0, 0)

    class _DotProductAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            del args
            self.num_attention_heads = kwargs.get("num_attention_heads")
            self.num_gqa_groups = kwargs.get("num_gqa_groups", self.num_attention_heads)

        def forward(self, *args, **kwargs):
            del kwargs
            q, k, v = args[:3]
            if q.dim() == 3:
                q = q.unsqueeze(1)
                k = k.unsqueeze(1)
                v = v.unsqueeze(1)
                squeeze_batch = True
            else:
                squeeze_batch = False
            if q.dim() != 4:
                raise ValueError(f"CPU attention stub expects [S, B, H, D], got {tuple(q.shape)}")
            if k.shape[2] != q.shape[2]:
                repeat = q.shape[2] // k.shape[2]
                k = k.repeat_interleave(repeat, dim=2)
                v = v.repeat_interleave(repeat, dim=2)
            q_bhsd = q.permute(1, 2, 0, 3).float()
            k_bhds = k.permute(1, 2, 3, 0).float()
            scores = q_bhsd.matmul(k_bhds) * (q.shape[-1] ** -0.5)
            causal = torch.tril(torch.ones(q.shape[0], k.shape[0], device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.view(1, 1, q.shape[0], k.shape[0]), float("-inf"))
            probs = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
            out = probs.matmul(v.permute(1, 2, 0, 3))
            out = out.permute(2, 0, 1, 3).contiguous()
            return out.squeeze(1) if squeeze_batch else out

    def _general_gemm(a, b, *args, out_dtype=None, layout="TN", out=None, bias=None, **kwargs):
        del args, kwargs
        if layout == "TN":
            result = b.to(dtype=out_dtype or b.dtype).matmul(a.to(dtype=out_dtype or b.dtype).t())
        elif layout == "NN":
            result = b.to(dtype=out_dtype or b.dtype).matmul(a.to(dtype=out_dtype or b.dtype))
        elif layout == "NT":
            result = b.to(dtype=out_dtype or b.dtype).t().matmul(a.to(dtype=out_dtype or b.dtype))
        else:
            return None
        if bias is not None:
            result = result + bias.to(dtype=result.dtype)
        if out is not None:
            out.copy_(result)
            result = out
        return (result,)

    def _not_available(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError("CPU Transformer Engine fused stub was called.")

    root = sys.modules.get("transformer_engine") or types.ModuleType("transformer_engine")
    pytorch = sys.modules.get("transformer_engine.pytorch") or types.ModuleType(
        "transformer_engine.pytorch"
    )
    pytorch.__path__ = []
    pytorch.RMSNorm = getattr(pytorch, "RMSNorm", _RMSNorm)
    pytorch.Linear = getattr(pytorch, "Linear", _Linear)
    pytorch.LayerNormLinear = getattr(pytorch, "LayerNormLinear", _LayerNormLinear)
    pytorch.GroupedLinear = getattr(pytorch, "GroupedLinear", _GroupedLinear)
    pytorch.DotProductAttention = getattr(pytorch, "DotProductAttention", _DotProductAttention)
    pytorch.fp8_autocast = getattr(pytorch, "fp8_autocast", lambda **kwargs: nullcontext())
    root.pytorch = pytorch
    sys.modules["transformer_engine"] = root
    sys.modules["transformer_engine.pytorch"] = pytorch
    if not hasattr(pytorch, "RMSNorm"):
        pytorch.RMSNorm = _RMSNorm

    cpp_extensions = types.ModuleType("transformer_engine.pytorch.cpp_extensions")
    cpp_extensions.general_gemm = _general_gemm
    sys.modules["transformer_engine.pytorch.cpp_extensions"] = cpp_extensions

    permutation = types.ModuleType("transformer_engine.pytorch.permutation")
    permutation.moe_permute = _not_available
    permutation.moe_permute_and_pad_with_probs = _not_available
    permutation.moe_permute_with_probs = _not_available
    permutation.moe_unpermute = _not_available
    sys.modules["transformer_engine.pytorch.permutation"] = permutation

    router = types.ModuleType("transformer_engine.pytorch.router")
    router.fused_compute_score_for_moe_aux_loss = _not_available
    router.fused_moe_aux_loss = _not_available
    router.fused_topk_with_score_function = _not_available
    sys.modules["transformer_engine.pytorch.router"] = router

    common = types.ModuleType("transformer_engine.common")
    recipe = types.ModuleType("transformer_engine.common.recipe")
    recipe.DelayedScaling = lambda *args, **kwargs: ("DelayedScaling", args, kwargs)
    recipe.Format = types.SimpleNamespace(HYBRID="HYBRID")
    common.recipe = recipe
    root.common = common
    sys.modules["transformer_engine.common"] = common
    sys.modules["transformer_engine.common.recipe"] = recipe

    dsa_module = sys.modules.get("megatron.lite.primitive.modules.attention.dsa")
    if dsa_module is not None:
        dsa_module.RMSNorm = _RMSNorm


def _tiny_config_kwargs():
    return dict(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=16,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )


def test_glm5_registry_resolves_lite():
    from megatron.lite.model.registry import (
        get_train_runtime_module,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    runtime_name = resolve_runtime_model_name("glm5", "lite")
    assert runtime_name == "glm5"
    module = get_train_runtime_module(runtime_name)
    assert module.__name__ == "megatron.lite.model.glm5.lite.protocol"
    assert resolve_model_type_from_hf({"model_type": "glm_moe_dsa"}) == "glm5"


def test_glm5_config_reads_hf_architecture_fields():
    from megatron.lite.model.glm5.config import Glm5Config

    indexer_types = [
        "full" if idx < 3 or ((idx + 1 - 3) % 4 == 0) else "shared" for idx in range(78)
    ]
    cfg = Glm5Config._from_hf_dict(
        {
            "model_type": "glm_moe_dsa",
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_head_dim": 256,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 2048,
            "index_topk_freq": 4,
            "index_skip_topk_offset": 3,
            "index_share_for_mtp_iteration": True,
            "indexer_rope_interleave": True,
            "indexer_types": indexer_types,
            "first_k_dense_replace": 3,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "vocab_size": 154880,
            "rope_interleave": True,
            "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
        }
    )

    assert cfg.q_lora_rank == 2048
    assert cfg.kv_lora_rank == 512
    assert cfg.index_topk == 2048
    assert cfg.index_topk_freq == 4
    assert cfg.index_skip_topk_offset == 3
    assert cfg.index_share_for_mtp_iteration is True
    assert cfg.indexer_rope_interleave is True
    assert cfg.indexer_types[:4] == ["full", "full", "full", "shared"]
    assert cfg.num_nextn_predict_layers == 1
    assert cfg.rope_interleave is True
    assert cfg.rope_theta == 8_000_000.0
    assert cfg.is_moe_layer(2) is False
    assert cfg.is_moe_layer(3) is True
    assert len(cfg.indexer_types) == cfg.num_hidden_layers
    assert cfg.dsa_indexer_type(78) == "full"
    assert cfg.dsa_source_compute_layer(78) == 78


def test_glm5_config_honors_mtp_index_share_flag_when_indexer_types_end_at_main_layers():
    from megatron.lite.model.glm5.config import Glm5Config

    main_indexer_types = [
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
    ]
    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": len(main_indexer_types),
            "num_nextn_predict_layers": 3,
            "index_topk_freq": 4,
            "index_skip_topk_offset": 3,
            "indexer_types": main_indexer_types,
            "index_share_for_mtp_iteration": True,
        }
    )

    mtp_layer_ids = [6, 7, 8]
    assert [cfg.dsa_indexer_type(idx) for idx in mtp_layer_ids] == [
        "full",
        "shared",
        "shared",
    ]
    assert [cfg.dsa_source_compute_layer(idx) for idx in mtp_layer_ids] == [6, 6, 6]

    no_share_cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": len(main_indexer_types),
            "num_nextn_predict_layers": 3,
            "index_topk_freq": 4,
            "index_skip_topk_offset": 3,
            "indexer_types": main_indexer_types,
            "index_share_for_mtp_iteration": False,
        }
    )

    assert [no_share_cfg.dsa_indexer_type(idx) for idx in mtp_layer_ids] == [
        "full",
        "full",
        "full",
    ]
    assert [no_share_cfg.dsa_source_compute_layer(idx) for idx in mtp_layer_ids] == mtp_layer_ids


def test_glm5_config_ignores_null_hf_optional_fields():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            "model_type": "glm_moe_dsa",
            "indexer_rope_first": None,
            "indexer_use_hadamard": None,
            "mlp_layer_types": None,
        }
    )

    assert cfg.indexer_rope_first is True
    assert cfg.indexer_use_hadamard is False
    assert cfg.mlp_layer_types is None


def test_glm5_config_tracks_dsa_indexshare_pattern():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 8,
            "index_topk_freq": 4,
            "index_skip_topk_offset": 3,
            "indexer_types": [
                "full",
                "full",
                "full",
                "shared",
                "shared",
                "shared",
                "full",
                "shared",
            ],
        }
    )

    assert [cfg.dsa_indexer_type(idx) for idx in range(8)] == [
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
    ]
    assert [cfg.is_dsa_skip_topk_layer(idx) for idx in range(8)] == [
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
    ]
    assert [cfg.dsa_source_compute_layer(idx) for idx in range(8)] == [
        0,
        1,
        2,
        2,
        2,
        2,
        6,
        6,
    ]


def test_glm5_config_derives_dsa_indexshare_pattern_without_hf_indexer_types():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 8,
            "index_topk_freq": 4,
            "index_skip_topk_offset": 3,
        }
    )

    assert [cfg.dsa_indexer_type(idx) for idx in range(8)] == [
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
    ]
    assert cfg.dsa_source_compute_layer(3) == 2
    assert cfg.dsa_source_compute_layer(7) == 6


def test_glm5_hf_reference_weight_map_matches_dsa_indexshare_and_mtp_layout():
    import json

    reference_dir = _hf_glm52_reference_dir()
    config_path = reference_dir / "config.json"
    index_path = reference_dir / "model.safetensors.index.json"
    if not config_path.exists() or not index_path.exists():
        import pytest

        pytest.skip("HF GLM-5.2 reference snapshot is not available in this workspace.")

    config = json.loads(config_path.read_text())
    weight_map = json.loads(index_path.read_text())["weight_map"]
    indexer_types = config["indexer_types"]
    num_layers = int(config["num_hidden_layers"])

    assert config["model_type"] == "glm_moe_dsa"
    assert num_layers == 78
    assert int(config["num_nextn_predict_layers"]) == 1
    assert len(indexer_types) == num_layers
    assert indexer_types[:7] == ["full", "full", "full", "shared", "shared", "shared", "full"]

    for layer_idx, indexer_type in enumerate(indexer_types):
        prefix = f"model.layers.{layer_idx}.self_attn.indexer."
        has_indexer_weights = any(key.startswith(prefix) for key in weight_map)
        assert has_indexer_weights is (indexer_type == "full")

    mtp_layer_idx = num_layers
    assert f"model.layers.{mtp_layer_idx}.eh_proj.weight" in weight_map
    assert any(
        key.startswith(f"model.layers.{mtp_layer_idx}.self_attn.indexer.")
        for key in weight_map
    )
    assert f"model.layers.{mtp_layer_idx}.self_attn.q_a_proj.weight" in weight_map
    assert f"model.layers.{mtp_layer_idx}.mlp.experts.0.gate_proj.weight" in weight_map
    assert not any("lora" in key.lower() for key in weight_map)

    root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    glm5_adapter_text = (root / "model" / "glm5" / "lite" / "lora_adapter.py").read_text()
    adapter_targets = glm5_adapter_text.split("_TARGET_MODULE_EXPANSIONS", 1)[0]
    assert "eh_proj" not in adapter_targets


def test_glm52_reference_config_pins_sparse_arch_low_rank_and_peft_targets():
    import ast
    from collections import Counter
    import json

    from megatron.lite.model.glm5.config import Glm5Config

    reference_dir = _hf_glm52_reference_dir()
    config_path = reference_dir / "config.json"
    if not config_path.exists():
        import pytest

        pytest.skip("HF GLM-5.2 reference config is not available in this workspace.")

    hf_config = json.loads(config_path.read_text())
    cfg = Glm5Config._from_hf_dict(hf_config)

    assert hf_config["model_type"] == "glm_moe_dsa"
    assert cfg.num_hidden_layers == 78
    assert cfg.num_nextn_predict_layers == 1
    assert cfg.q_lora_rank == hf_config["q_lora_rank"] == 2048
    assert cfg.kv_lora_rank == hf_config["kv_lora_rank"] == 512
    assert cfg.index_topk_freq == hf_config["index_topk_freq"] == 4
    assert cfg.index_skip_topk_offset == hf_config["index_skip_topk_offset"] == 3
    assert cfg.index_share_for_mtp_iteration is True
    assert cfg.scoring_func == "sigmoid"
    assert cfg.topk_method == "noaux_tc"

    indexer_types = hf_config["indexer_types"]
    assert len(indexer_types) == cfg.num_hidden_layers
    assert Counter(indexer_types) == Counter({"full": 21, "shared": 57})
    assert [idx for idx, kind in enumerate(indexer_types) if kind == "full"] == [
        0,
        1,
        2,
        *range(6, 75, 4),
    ]

    mtp_layer_idx = cfg.num_hidden_layers
    assert cfg.dsa_indexer_type(mtp_layer_idx) == "full"
    assert cfg.dsa_source_compute_layer(mtp_layer_idx) == mtp_layer_idx

    root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    adapter_text = (root / "model" / "glm5" / "lite" / "lora_adapter.py").read_text()
    adapter_semantics_text = (
        root / "model" / "glm5" / "lite" / "adapter_semantics.py"
    ).read_text()
    primitive_lora_text = (root / "primitive" / "modules" / "lora.py").read_text()

    def literal_assignment(source: str, name: str):
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            ):
                return ast.literal_eval(node.value)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == name and node.value is not None:
                    return ast.literal_eval(node.value)
        raise AssertionError(f"missing literal assignment {name}")

    attention_targets = set(
        literal_assignment(adapter_semantics_text, "ATTENTION_ADAPTER_TARGETS")
    )
    mlp_targets = set(literal_assignment(adapter_semantics_text, "MLP_ADAPTER_TARGETS"))
    glm5_adapter_targets = attention_targets | mlp_targets
    default_lora_targets = set(literal_assignment(primitive_lora_text, "_DEFAULT_TARGET_MODULES"))

    assert "ADAPTER_TARGET_MODULES as _ADAPTER_TARGET_MODULES" in adapter_text
    assert glm5_adapter_targets == {
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert default_lora_targets == {"linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"}
    assert not glm5_adapter_targets.intersection(
        {
            "q_lora_rank",
            "kv_lora_rank",
            "eh_proj",
            "indexer",
            "embed_tokens",
            "lm_head",
        }
    )


def test_glm5_reference_inputs_pin_mindlab_commit_and_peft_boundaries():
    import subprocess

    reference_dir = _mindlab_glm52_reference_dir()
    paper_notes_path = _peft_paper_notes_path()
    if not reference_dir.exists() or not paper_notes_path.exists():
        import pytest

        pytest.skip("GLM5.2 MindLab fork or PEFT paper notes are not available.")

    commit = subprocess.check_output(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert commit == "e08610def780d9deba2778f1dbe268c0bca4d791"

    changed_files = set(
        subprocess.check_output(
            ["git", "-C", str(reference_dir), "show", "--name-only", "--format=", commit],
            text=True,
        ).splitlines()
    )
    assert "megatron/core/transformer/experimental_attention_variant/dsa.py" in changed_files
    assert "megatron/core/transformer/multi_latent_attention.py" in changed_files
    assert "megatron/core/transformer/transformer_config.py" in changed_files
    assert "tests/unit_tests/transformer/experimental_attention_variant/test_attention_variant_dsa.py" in changed_files
    assert not any(
        "lora" in path.lower() or "peft" in path.lower() for path in changed_files
    )

    dsa_text = (
        reference_dir
        / "megatron"
        / "core"
        / "transformer"
        / "experimental_attention_variant"
        / "dsa.py"
    ).read_text()
    config_text = (
        reference_dir / "megatron" / "core" / "transformer" / "transformer_config.py"
    ).read_text()
    router_replay_text = (
        reference_dir / "megatron" / "core" / "transformer" / "moe" / "router_replay.py"
    ).read_text()
    paper_notes = paper_notes_path.read_text()

    assert '_HOLDER_ATTR = "_dsa_index_share_topk_holder"' in dsa_text
    assert 'getattr(self.config, "dsa_indexer_topk_freq", 1) or 1' in dsa_text
    assert 'getattr(self.config, "dsa_indexer_skip_topk_offset", 0) or 0' in dsa_text
    assert "self.skip_topk = self.index_share and is_dsa_skip_topk_layer" in dsa_text
    assert "self.source_layer = (" in dsa_text
    assert "Cross-PP top-k sharing is not supported" in dsa_text
    assert "topk_holder[self.layer_number] = topk_indices" in dsa_text
    assert "latent_v_channels = int(getattr(self.config, \"kv_lora_rank\"" in dsa_text

    assert "dsa_indexer_topk_freq: int = 1" in config_text
    assert "dsa_indexer_skip_topk_offset: int = 0" in config_text
    assert "dsa_indexer_loss_coeff: Optional[float] = None" in config_text
    assert "moe_enable_routing_replay: bool = False" in config_text
    assert "q_lora_rank: int = 512" in config_text
    assert "kv_lora_rank: int = 512" in config_text

    assert "class RouterReplayAction" in router_replay_text
    assert 'RECORD = "record"' in router_replay_text
    assert 'REPLAY_FORWARD = "replay_forward"' in router_replay_text
    assert 'REPLAY_BACKWARD = "replay_backward"' in router_replay_text
    assert "replay_backward_list" in router_replay_text

    assert "PEFT (LoRA) is not just a cheap fine-tune" in paper_notes
    assert "R3 = Router Replay" in paper_notes
    assert "OLoRA-tail:" in paper_notes
    assert "with NO singular-value scaling" in paper_notes
    assert "scaling convention" in paper_notes
    assert "MinT" in paper_notes


def test_glm5_config_preserves_mtp_aliases_and_layer_types():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            **_tiny_config_kwargs(),
            "num_nextn_predict": 1,
            "mtp_loss_scaling_factor": 0.2,
            "mlp_layer_types": ["dense", "sparse", "sparse"],
        }
    )

    assert cfg.num_nextn_predict_layers == 1
    assert cfg.mtp_loss_scaling_factor == 0.2
    assert cfg.is_moe_layer(2) is True


def test_glm5_lite_does_not_import_wrappers_or_sibling_models():
    root = Path(__file__).resolve().parents[3] / "megatron" / "lite" / "model" / "glm5" / "lite"
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "megatron.lite.model.qwen" not in text
        assert "mbridge" not in text
        assert "MCore" not in text
        assert "megatron.core" not in text


def test_glm5_lite_uses_shared_mla_and_dsa_primitive():
    root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    model_text = (root / "model" / "glm5" / "lite" / "model.py").read_text()
    primitive_text = (root / "primitive" / "modules" / "attention" / "dsa.py").read_text()
    kernel_text = (root / "primitive" / "kernels" / "dsa_kernels.py").read_text()

    assert "DynamicSparseAttention" in model_text
    assert (
        "from megatron.lite.primitive.modules.attention.mla import MultiLatentAttention"
        in primitive_text
    )
    assert "class DynamicSparseAttention" in primitive_text
    assert "class MultiLatentAttention" not in primitive_text
    assert "class DSAIndexer" in primitive_text
    assert "megatron.core" not in primitive_text
    assert "dsa_kernels.fused_indexer_sparse_attn" in primitive_text
    assert "dsa_kernels.dsa_sparse_attn" in primitive_text
    assert "dsa_kernels.indexer_topk" in primitive_text
    assert "value_dim" in kernel_text
    assert "from cudnn.deepseek_sparse_attention import DSA" in kernel_text
    assert "from cudnn import DSA" in kernel_text
    assert "cudnn.deepseek_sparse_attention.indexer_forward._interface_sm90" in kernel_text
    assert "cudnn.deepseek_sparse_attention.indexer_forward._interface" in kernel_text
    assert "torch.cuda.get_device_capability(device)" in kernel_text
    assert "torch.topk" not in primitive_text
    assert "torch.softmax" not in primitive_text
    assert "torch.matmul" not in primitive_text


def test_glm5_lite_wires_dsa_indexshare_layer_metadata():
    root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    model_text = (root / "model" / "glm5" / "lite" / "model.py").read_text()
    primitive_text = (root / "primitive" / "modules" / "attention" / "dsa.py").read_text()
    attention_init_text = (root / "primitive" / "modules" / "attention" / "__init__.py").read_text()

    assert "DSA_INDEX_SHARE_TOPK_HOLDER_ATTR" in attention_init_text
    assert "Glm5DSAAttention(config, ps, layer_idx, lora_config=lora_config)" in model_text
    assert "dsa_indexer_type=config.dsa_indexer_type(layer_idx)" in model_text
    assert "dsa_source_layer_idx=config.dsa_source_compute_layer(layer_idx)" in model_text
    assert "self.dsa_index_share_layer_indices = _dsa_index_share_local_layer_indices" in model_text
    assert "_validate_dsa_index_share_pipeline_split(config, self.dsa_index_share_layer_indices)" in model_text
    assert "_new_dsa_index_share_topk_holder" in model_text
    assert "dsa_index_share_topk_holder=dsa_index_share_topk_holder" in model_text

    assert "dsa_index_share_enabled" in primitive_text
    assert "self.skip_topk = dsa_indexer_type == \"shared\"" in primitive_text
    assert "self.indexer: DSAIndexer | None = None" in primitive_text
    assert "if not self.skip_topk:" in primitive_text
    assert "_load_index_share_topk" in primitive_text
    assert "_store_index_share_topk" in primitive_text
    assert "index_share_segment_idx=idx" in primitive_text


def test_glm5_dsa_indexshare_pipeline_split_includes_mtp_layers():
    import pytest
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import (
        _dsa_index_share_local_layer_indices,
        _validate_dsa_index_share_pipeline_split,
    )

    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 1,
            "index_topk_freq": 3,
            "index_skip_topk_offset": 1,
            "index_share_for_mtp_iteration": True,
        }
    )
    assert cfg.dsa_indexer_type(4) == "shared"
    assert cfg.dsa_source_compute_layer(4) == 3

    local_with_source = _dsa_index_share_local_layer_indices(cfg, [3], has_mtp=True)
    assert local_with_source == [3, 4]
    _validate_dsa_index_share_pipeline_split(cfg, local_with_source)

    local_without_source = _dsa_index_share_local_layer_indices(cfg, [], has_mtp=True)
    assert local_without_source == [4]
    with pytest.raises(AssertionError, match="Cross-PP top-k sharing"):
        _validate_dsa_index_share_pipeline_split(cfg, local_without_source)

    no_share_cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 1,
            "index_topk_freq": 3,
            "index_skip_topk_offset": 1,
            "index_share_for_mtp_iteration": False,
        }
    )
    local_no_share = _dsa_index_share_local_layer_indices(no_share_cfg, [], has_mtp=True)
    assert local_no_share == [4]
    assert no_share_cfg.dsa_indexer_type(4) == "full"
    _validate_dsa_index_share_pipeline_split(no_share_cfg, local_no_share)


def test_glm5_dsa_kernel_routes_indexer_forward_by_sm(monkeypatch):
    from megatron.lite.primitive.kernels import dsa_kernels

    sm90_entry = object()
    sm100_entry = object()

    monkeypatch.setattr(dsa_kernels, "_load_indexer_fwd_sm90", lambda: sm90_entry)
    monkeypatch.setattr(dsa_kernels, "_load_indexer_fwd_sm100", lambda: sm100_entry)

    monkeypatch.setattr(dsa_kernels.torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert dsa_kernels._select_indexer_forward(None) is sm90_entry

    monkeypatch.setattr(dsa_kernels.torch.cuda, "get_device_capability", lambda device: (10, 0))
    assert dsa_kernels._select_indexer_forward(None) is sm100_entry

    monkeypatch.setattr(dsa_kernels.torch.cuda, "get_device_capability", lambda device: (8, 0))
    assert dsa_kernels._select_indexer_forward(None) is None


def test_glm5_dsa_indexer_topk_torch_backend_masks_invalid_slots(monkeypatch):
    import torch

    from megatron.lite.primitive.kernels import dsa_kernels

    monkeypatch.setenv("MLITE_DSA_INDEXER_TOPK_BACKEND", "torch")
    q = torch.ones(3, 1, 1, 2, dtype=torch.bfloat16)
    k = torch.ones(3, 1, 2, dtype=torch.bfloat16)
    weights = torch.ones(3, 1, 1, dtype=torch.bfloat16)

    indices, lengths = dsa_kernels.indexer_topk(q, k, weights, topk=2, ratio=1)

    assert indices.shape == (1, 3, 2)
    assert lengths.tolist() == [[1, 2, 2]]
    assert indices[0, 0, 1].item() == -1


def test_glm5_dsa_build_flat_topk_uses_torch_compact_backend(monkeypatch):
    import torch

    from megatron.lite.primitive.kernels import dsa_kernels

    monkeypatch.setenv("MLITE_DSA_INDEXER_TOPK_BACKEND", "torch")
    local_idxs = torch.tensor(
        [
            [[2, -1, 0], [1, -1, -1]],
            [[0, 2, -1], [-1, 1, 2]],
        ],
        dtype=torch.int32,
    )

    flat_idxs, topk_length = dsa_kernels.build_flat_topk_idxs(
        local_idxs,
        batch_size=2,
        seqlen_kv=3,
        compact=True,
    )

    assert flat_idxs.tolist() == [
        [4, 0, -1],
        [1, 5, -1],
        [2, -1, -1],
        [3, 5, -1],
    ]
    assert topk_length.tolist() == [2, 2, 1, 2]


def test_glm5_dsa_sparse_attn_torch_backend_matches_reference(monkeypatch):
    import torch

    from megatron.lite.primitive.kernels import dsa_kernels

    monkeypatch.setenv("MLITE_DSA_SPARSE_ATTN_BACKEND", "torch")
    query = torch.tensor(
        [
            [
                [[0.2, 0.1, 0.4], [0.0, 0.3, 0.1]],
                [[0.5, -0.2, 0.0], [0.4, 0.1, -0.3]],
            ],
            [
                [[-0.1, 0.6, 0.2], [0.2, -0.4, 0.3]],
                [[0.3, 0.2, -0.5], [-0.2, 0.5, 0.4]],
            ],
        ],
        dtype=torch.float32,
    )
    kv = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.3, -0.1, 0.2]],
            [[0.4, 0.0, -0.2], [-0.5, 0.2, 0.1]],
            [[0.2, -0.3, 0.5], [0.1, 0.4, -0.4]],
        ],
        dtype=torch.float32,
    )
    local_idxs = torch.tensor(
        [
            [[0, 2], [-1, -1]],
            [[1, -1], [0, 2]],
        ],
        dtype=torch.int32,
    )
    flat_idxs, topk_length = dsa_kernels.build_flat_topk_idxs(
        local_idxs, batch_size=2, seqlen_kv=3, compact=True
    )
    calls_before = dsa_kernels.torch_sparse_attn_fallback_call_count()

    out = dsa_kernels.dsa_sparse_attn(
        query,
        kv,
        torch.full((2,), -1.0e20, dtype=torch.float32),
        flat_idxs,
        softmax_scale=0.5,
        topk_length=topk_length,
        value_dim=2,
    )

    q_flat = query.reshape(4, 2, 3)
    kv_flat = kv.reshape(6, 3)
    expected_rows = []
    for row in range(q_flat.shape[0]):
        row_heads = []
        valid_idxs = flat_idxs[row, : topk_length[row].item()]
        valid_idxs = valid_idxs[valid_idxs >= 0].long()
        for head in range(q_flat.shape[1]):
            if valid_idxs.numel() == 0:
                row_heads.append(torch.zeros(2))
                continue
            logits = torch.mv(kv_flat[valid_idxs], q_flat[row, head]) * 0.5
            probs = torch.softmax(logits, dim=0)
            row_heads.append(probs @ kv_flat[valid_idxs, :2])
        expected_rows.append(torch.stack(row_heads, dim=0))
    expected = torch.stack(expected_rows, dim=0).reshape(2, 2, 4)

    assert torch.allclose(out, expected, atol=1e-6)
    assert dsa_kernels.torch_sparse_attn_fallback_call_count() == calls_before + 1


def test_glm5_dsa_training_with_indexer_loss_uses_fused_kernel(monkeypatch):
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention, build_rope_cache

    calls = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        kv_offset=0,
        calculate_per_token_loss=False,
        value_dim=None,
    ):
        del attn_sink, q_indexer, k_indexer, weights, softmax_scale, indexer_softmax_scale
        calls["training"] = {
            "query_shape": tuple(query.shape),
            "kv_shape": tuple(kv_full.shape),
            "window_shape": tuple(window_idxs.shape),
            "indexer_topk": indexer_topk,
            "ratio": ratio,
            "loss_coeff": loss_coeff,
            "sparse_loss": sparse_loss,
            "kv_offset": kv_offset,
            "calculate_per_token_loss": calculate_per_token_loss,
            "value_dim": value_dim,
        }
        return query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        ), torch.zeros((), device=query.device, dtype=torch.float32)

    monkeypatch.setattr(
        dsa._dsa_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    attn = DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1e-5,
        indexer_loss_coeff=0.1,
    )
    attn.train()
    x = torch.randn(1, 4, 16)
    cos, sin = build_rope_cache(dim=4, max_position_embeddings=4, rope_theta=1_000_000.0)
    position_ids = torch.arange(4).unsqueeze(0)

    out = attn(x, cos=cos, sin=sin, position_ids=position_ids)

    assert out.shape == (1, 4, 16)
    assert calls["training"] == {
        "query_shape": (4, 1, 2, 8),
        "kv_shape": (4, 1, 8),
        "window_shape": (1, 4, 0),
        "indexer_topk": 2,
        "ratio": 1,
        "loss_coeff": 0.1,
        "sparse_loss": False,
        "kv_offset": 0,
        "calculate_per_token_loss": False,
        "value_dim": 4,
    }


def test_glm5_dsa_training_zero_loss_uses_topk_sparse_attention(monkeypatch):
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention, build_rope_cache

    calls = {}

    def fake_indexer_topk(q_indexer, k_indexer, weights, topk, ratio, indexer_softmax_scale=1.0):
        del q_indexer, k_indexer, weights, indexer_softmax_scale
        calls["indexer"] = {"topk": topk, "ratio": ratio}
        idx = torch.zeros((1, 4, topk), dtype=torch.int32)
        return idx, torch.full((1, 4), topk, dtype=torch.int32)

    def fake_dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length=None,
        indexer_topk=0,
        value_dim=None,
    ):
        del kv_full, attn_sink, topk_idxs, softmax_scale, indexer_topk
        calls["sparse"] = {"topk_length_is_set": topk_length is not None, "value_dim": value_dim}
        return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", fake_indexer_topk)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", fake_dsa_sparse_attn)

    attn = DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1e-5,
    )
    attn.train()
    x = torch.randn(1, 4, 16)
    cos, sin = build_rope_cache(dim=4, max_position_embeddings=4, rope_theta=1_000_000.0)
    position_ids = torch.arange(4).unsqueeze(0)

    out = attn(x, cos=cos, sin=sin, position_ids=position_ids)

    assert out.shape == (1, 4, 16)
    assert calls["indexer"] == {"topk": 2, "ratio": 1}
    assert calls["sparse"] == {"topk_length_is_set": True, "value_dim": 4}


def test_glm5_dsa_eval_forward_uses_fused_sparse_attention(monkeypatch):
    import torch

    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention, build_rope_cache

    calls = {}

    def fake_indexer_topk(q_indexer, k_indexer, weights, topk, ratio, indexer_softmax_scale=1.0):
        del q_indexer, k_indexer, weights, indexer_softmax_scale
        calls["indexer"] = {"topk": topk, "ratio": ratio}
        idx = torch.zeros((1, 4, topk), dtype=torch.int32)
        return idx, torch.full((1, 4), topk, dtype=torch.int32)

    def fake_dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length=None,
        indexer_topk=0,
        value_dim=None,
    ):
        del kv_full, attn_sink, topk_idxs, softmax_scale, indexer_topk
        calls["sparse"] = {"topk_length_is_set": topk_length is not None, "value_dim": value_dim}
        return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", fake_indexer_topk)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", fake_dsa_sparse_attn)

    attn = DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1e-5,
    )
    attn.eval()
    x = torch.randn(1, 4, 16)
    cos, sin = build_rope_cache(dim=4, max_position_embeddings=4, rope_theta=1_000_000.0)
    position_ids = torch.arange(4).unsqueeze(0)

    with torch.no_grad():
        out = attn(x, cos=cos, sin=sin, position_ids=position_ids)

    assert out.shape == (1, 4, 16)
    assert calls["indexer"] == {"topk": 2, "ratio": 1}
    assert calls["sparse"] == {"topk_length_is_set": True, "value_dim": 4}


def test_glm5_dsa_indexshare_loss_fails_explicitly_before_topk_kernel():
    import pytest
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention, build_rope_cache

    attn = DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1e-5,
        indexer_loss_coeff=0.1,
        dsa_index_share_enabled=True,
        dsa_indexer_type="full",
        layer_idx=0,
        dsa_source_layer_idx=0,
    )
    attn.train()
    x = torch.randn(1, 4, 16)
    cos, sin = build_rope_cache(dim=4, max_position_embeddings=4, rope_theta=1_000_000.0)
    position_ids = torch.arange(4).unsqueeze(0)

    with pytest.raises(NotImplementedError, match="dsa_indexer_loss_coeff > 0"):
        attn(
            x,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            dsa_index_share_topk_holder={},
        )


def test_glm5_dsa_indexshare_training_zero_loss_reuses_full_layer_topk():
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention, build_rope_cache
    from megatron.lite.primitive.modules.attention import dsa

    calls = {"indexer": 0, "sparse_topk": []}
    expected_topk = torch.tensor([[[0, 1], [0, 1], [1, 2], [2, 3]]], dtype=torch.int32)
    expected_topk_len = torch.full((1, 4), 2, dtype=torch.int32)

    def fake_indexer_topk(q_indexer, k_indexer, weights, topk, ratio, indexer_softmax_scale=1.0):
        del q_indexer, k_indexer, weights, ratio, indexer_softmax_scale
        calls["indexer"] += 1
        assert topk == 2
        return expected_topk, expected_topk_len

    def fake_build_flat_topk_idxs(topk_indices, *, batch_size, seqlen_kv, compact=True):
        assert batch_size == 1
        assert seqlen_kv == 4
        assert compact is True
        calls["sparse_topk"].append(topk_indices)
        return topk_indices, expected_topk_len

    def fake_dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length=None,
        indexer_topk=0,
        value_dim=None,
    ):
        del kv_full, attn_sink, topk_idxs, softmax_scale, topk_length, indexer_topk
        return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

    original_indexer_topk = dsa._dsa_kernels.indexer_topk
    original_build_flat_topk_idxs = dsa._dsa_kernels.build_flat_topk_idxs
    original_dsa_sparse_attn = dsa._dsa_kernels.dsa_sparse_attn
    dsa._dsa_kernels.indexer_topk = fake_indexer_topk
    dsa._dsa_kernels.build_flat_topk_idxs = fake_build_flat_topk_idxs
    dsa._dsa_kernels.dsa_sparse_attn = fake_dsa_sparse_attn
    try:
        full_attn = DynamicSparseAttention(
            hidden_size=16,
            num_attention_heads=2,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            rms_norm_eps=1e-5,
            dsa_index_share_enabled=True,
            dsa_indexer_type="full",
            layer_idx=0,
            dsa_source_layer_idx=0,
        )
        shared_attn = DynamicSparseAttention(
            hidden_size=16,
            num_attention_heads=2,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            rms_norm_eps=1e-5,
            dsa_index_share_enabled=True,
            dsa_indexer_type="shared",
            layer_idx=1,
            dsa_source_layer_idx=0,
        )
        full_attn.train()
        shared_attn.train()
        x = torch.randn(1, 4, 16)
        cos, sin = build_rope_cache(dim=4, max_position_embeddings=4, rope_theta=1_000_000.0)
        position_ids = torch.arange(4).unsqueeze(0)
        holder = {}

        full_out = full_attn(
            x,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            dsa_index_share_topk_holder=holder,
        )
        shared_out = shared_attn(
            x,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            dsa_index_share_topk_holder=holder,
        )
    finally:
        dsa._dsa_kernels.indexer_topk = original_indexer_topk
        dsa._dsa_kernels.build_flat_topk_idxs = original_build_flat_topk_idxs
        dsa._dsa_kernels.dsa_sparse_attn = original_dsa_sparse_attn

    assert full_out.shape == (1, 4, 16)
    assert shared_out.shape == (1, 4, 16)
    assert calls["indexer"] == 1
    assert list(holder) == [0]
    assert holder[0] is expected_topk
    assert len(calls["sparse_topk"]) == 2
    assert calls["sparse_topk"][0] is expected_topk
    assert calls["sparse_topk"][1] is expected_topk


def test_glm5_dsa_indexshare_shared_layer_requires_source_topk_in_holder():
    import pytest
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention, build_rope_cache

    shared_attn = DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1e-5,
        dsa_index_share_enabled=True,
        dsa_indexer_type="shared",
        layer_idx=1,
        dsa_source_layer_idx=0,
    )
    x = torch.randn(1, 4, 16)
    cos, sin = build_rope_cache(dim=4, max_position_embeddings=4, rope_theta=1_000_000.0)
    position_ids = torch.arange(4).unsqueeze(0)

    with pytest.raises(AssertionError, match="Cross-PP top-k sharing is not supported"):
        shared_attn(
            x,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            dsa_index_share_topk_holder={},
        )


def test_glm5_lite_model_exports_hf_style_state_names():
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM

    model = Glm5ForCausalLM(Glm5Config(**_tiny_config_kwargs()))
    keys = set(model.state_dict())

    assert "model.embed_tokens.weight" in keys
    assert "model.layers.0.self_attn.q_a_proj.weight" in keys
    assert "model.layers.0.mlp.gate_proj.weight" in keys
    assert "model.layers.1.mlp.gate.weight" in keys
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in keys
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in keys
    assert "lm_head.weight" in keys


def test_glm5_checkpoint_exports_and_saves_hf_style_weights(tmp_path):
    import torch
    from safetensors import safe_open

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import (
        export_hf_weights,
        save_hf_weights,
        save_weights,
    )
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Glm5Config(**_tiny_config_kwargs())
    model = Glm5ForCausalLM(cfg)
    ps = ParallelState()
    model.model.layers[1].moe.router.e_score_correction_bias.copy_(
        torch.tensor([0.25, -0.5, 1.0])
    )

    exported = dict(export_hf_weights(model, cfg, ps))
    state = model.state_dict()

    assert torch.equal(
        exported["model.layers.1.mlp.experts.2.gate_proj.weight"],
        state["model.layers.1.mlp.experts.2.gate_proj.weight"],
    )
    assert torch.equal(
        exported["model.layers.1.mlp.gate.e_score_correction_bias"],
        state["model.layers.1.mlp.gate.e_score_correction_bias"],
    )
    assert "model.layers.1.mlp.experts.gate_up_proj" not in exported

    hf_dir = tmp_path / "hf"
    save_hf_weights(model, str(hf_dir), cfg, ps)
    with safe_open(str(hf_dir / "model.safetensors"), framework="pt", device="cpu") as handle:
        assert torch.equal(
            handle.get_tensor("model.layers.1.mlp.experts.2.down_proj.weight"),
            state["model.layers.1.mlp.experts.2.down_proj.weight"],
        )
        assert torch.equal(
            handle.get_tensor("model.layers.1.mlp.gate.e_score_correction_bias"),
            state["model.layers.1.mlp.gate.e_score_correction_bias"],
        )

    loaded = Glm5ForCausalLM(cfg)
    from megatron.lite.model.glm5.lite.checkpoint import load_hf_weights

    load_hf_weights(loaded, str(hf_dir), cfg, ps)
    assert torch.equal(
        loaded.state_dict()["model.layers.1.mlp.gate.e_score_correction_bias"],
        state["model.layers.1.mlp.gate.e_score_correction_bias"],
    )

    hf_bf16_dir = tmp_path / "hf_bf16"
    save_hf_weights(model, str(hf_bf16_dir), cfg, ps, export_dtype=torch.bfloat16)
    with safe_open(str(hf_bf16_dir / "model.safetensors"), framework="pt", device="cpu") as handle:
        floating_dtypes = {
            handle.get_tensor(key).dtype
            for key in handle.keys()
            if handle.get_tensor(key).is_floating_point()
        }
        assert floating_dtypes == {torch.bfloat16}

    loaded_bf16 = Glm5ForCausalLM(cfg)
    load_hf_weights(loaded_bf16, str(hf_bf16_dir), cfg, ps)
    assert torch.equal(
        loaded_bf16.state_dict()["model.layers.1.mlp.experts.2.up_proj.weight"],
        state["model.layers.1.mlp.experts.2.up_proj.weight"].to(torch.bfloat16).to(torch.float32),
    )

    native_dir = tmp_path / "native"
    save_weights(model, str(native_dir), cfg, ps)
    with safe_open(str(native_dir / "model.safetensors"), framework="pt", device="cpu") as handle:
        assert torch.equal(
            handle.get_tensor("model.layers.1.mlp.experts.0.up_proj.weight"),
            state["model.layers.1.mlp.experts.0.up_proj.weight"],
        )


def test_glm5_checkpoint_loader_skips_shared_indexer_weights(tmp_path):
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import export_hf_weights, load_hf_weights
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Glm5Config(**_tiny_config_kwargs(), indexer_types=["full", "shared"])
    ps = ParallelState()
    model = Glm5ForCausalLM(cfg, ps=ps)
    exported = dict(export_hf_weights(model, cfg, ps))

    assert any(key.startswith("model.layers.0.self_attn.indexer.") for key in exported)
    assert not any(key.startswith("model.layers.1.self_attn.indexer.") for key in exported)

    save_safetensors(exported, str(tmp_path))
    loaded = Glm5ForCausalLM(cfg, ps=ps)
    load_hf_weights(loaded, str(tmp_path), cfg, ps)

    state = model.state_dict()
    loaded_state = loaded.state_dict()
    assert torch.equal(
        loaded_state["model.layers.1.self_attn.q_a_proj.weight"],
        state["model.layers.1.self_attn.q_a_proj.weight"],
    )


def test_glm5_checkpoint_exports_and_loads_mtp_layers(tmp_path):
    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import export_hf_weights, load_hf_weights
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1)
    ps = ParallelState()
    model = Glm5ForCausalLM(cfg, ps=ps, mtp_enable=True)
    state = model.state_dict()

    assert "model.mtp.layers.0.eh_proj.weight" in state
    assert "model.mtp.layers.0.transformer_layer.input_layernorm.weight" in state

    exported = dict(export_hf_weights(model, cfg, ps))
    assert "model.layers.2.eh_proj.weight" in exported
    assert "model.layers.2.enorm.weight" in exported
    assert "model.layers.2.hnorm.weight" in exported
    assert "model.layers.2.final_layernorm.weight" in exported
    assert "model.layers.2.input_layernorm.weight" in exported
    assert "model.layers.2.mlp.gate.weight" in exported

    save_safetensors(exported, str(tmp_path))
    loaded = Glm5ForCausalLM(cfg, ps=ps, mtp_enable=True)
    load_hf_weights(loaded, str(tmp_path), cfg, ps)
    assert torch.equal(
        loaded.state_dict()["model.mtp.layers.0.eh_proj.weight"],
        state["model.mtp.layers.0.eh_proj.weight"],
    )


def test_glm5_initialize_weights_resets_all_router_weights():
    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM, Glm5Router

    model = Glm5ForCausalLM(
        Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1), mtp_enable=True
    )
    routers = [module for module in model.modules() if isinstance(module, Glm5Router)]
    assert len(routers) == 2
    for router in routers:
        router.weight.data.fill_(float("nan"))
        router.e_score_correction_bias.data.fill_(float("nan"))

    model.initialize_weights()

    for router in routers:
        assert torch.isfinite(router.weight).all()
        assert torch.equal(
            router.e_score_correction_bias, torch.zeros_like(router.e_score_correction_bias)
        )


def test_glm5_hf_loader_resolves_grouped_expert_tensors():
    import torch

    from megatron.lite.model.glm5.lite.checkpoint import _resolve_hf_tensor

    class FakeReader:
        def __init__(self, tensors):
            self.tensors = tensors
            self.index = {name: "model.safetensors" for name in tensors}

        def get_tensor(self, name):
            return self.tensors[name]

    hf_gate_up = torch.arange(3 * 12 * 16, dtype=torch.float32).reshape(3, 12, 16)
    hf_down = torch.arange(3 * 16 * 6, dtype=torch.float32).reshape(3, 16, 6)
    hf_reader = FakeReader(
        {
            "model.layers.1.mlp.experts.gate_up_proj": hf_gate_up,
            "model.layers.1.mlp.experts.down_proj": hf_down,
        }
    )

    gate_target = torch.empty(6, 16)
    down_target = torch.empty(16, 6)
    assert torch.equal(
        _resolve_hf_tensor(hf_reader, "model.layers.1.mlp.experts.2.gate_proj.weight", gate_target),
        hf_gate_up[2, :6, :],
    )
    assert torch.equal(
        _resolve_hf_tensor(hf_reader, "model.layers.1.mlp.experts.2.up_proj.weight", gate_target),
        hf_gate_up[2, 6:, :],
    )
    assert torch.equal(
        _resolve_hf_tensor(hf_reader, "model.layers.1.mlp.experts.2.down_proj.weight", down_target),
        hf_down[2],
    )

    automodel_gate_up = torch.arange(3 * 16 * 12, dtype=torch.float32).reshape(3, 16, 12)
    automodel_down = torch.arange(3 * 6 * 16, dtype=torch.float32).reshape(3, 6, 16)
    automodel_reader = FakeReader(
        {
            "model.layers.1.mlp.experts.gate_and_up_projs": automodel_gate_up,
            "model.layers.1.mlp.experts.down_projs": automodel_down,
        }
    )
    assert torch.equal(
        _resolve_hf_tensor(
            automodel_reader, "model.layers.1.mlp.experts.1.gate_proj.weight", gate_target
        ),
        automodel_gate_up[1, :, :6].T,
    )
    assert torch.equal(
        _resolve_hf_tensor(
            automodel_reader, "model.layers.1.mlp.experts.1.up_proj.weight", gate_target
        ),
        automodel_gate_up[1, :, 6:].T,
    )
    assert torch.equal(
        _resolve_hf_tensor(
            automodel_reader, "model.layers.1.mlp.experts.1.down_proj.weight", down_target
        ),
        automodel_down[1].T,
    )


def test_glm5_hf_loader_slices_full_expert_tensors_for_proxy_targets():
    import torch

    from megatron.lite.model.glm5.lite.checkpoint import _slice_to_target_shape

    source = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8)
    target = torch.empty(4, 8)
    assert torch.equal(_slice_to_target_shape(source, target), source[:4])

    source3d = torch.arange(256 * 6 * 8, dtype=torch.float32).reshape(256, 6, 8)
    target3d = torch.empty(4, 6, 8)
    assert torch.equal(_slice_to_target_shape(source3d, target3d), source3d[:4])


def test_glm5_hf_loader_resolves_packed_experts_from_single_safetensor(tmp_path):
    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import _resolve_named_parameter_tensor
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader, save_safetensors
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Glm5Config(**_tiny_config_kwargs())
    gate_up = torch.arange(3 * 12 * 16, dtype=torch.float32).reshape(3, 12, 16)
    down = torch.arange(3 * 16 * 6, dtype=torch.float32).reshape(3, 16, 6)
    save_safetensors(
        {
            "model.layers.1.mlp.experts.gate_up_proj": gate_up,
            "model.layers.1.mlp.experts.down_proj": down,
        },
        str(tmp_path),
    )
    reader = SafeTensorReader(str(tmp_path))

    resolved_gate_up = _resolve_named_parameter_tensor(
        reader,
        "model.layers.1.mlp.experts.gate_up_proj",
        torch.empty(3, 12, 16),
        config=cfg,
        ps=ParallelState(),
    )
    resolved_down = _resolve_named_parameter_tensor(
        reader,
        "model.layers.1.mlp.experts.down_proj",
        torch.empty(3, 16, 6),
        config=cfg,
        ps=ParallelState(),
    )

    assert torch.equal(resolved_gate_up, gate_up)
    assert torch.equal(resolved_down, down)


def test_glm5_protocol_parallel_scope_allows_pp_cp_ep_and_rejects_tp_etp():
    import pytest

    from megatron.lite.model.glm5.lite.protocol import _validate_parallel_scope
    from megatron.lite.runtime.contracts import ParallelConfig

    _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=1, cp=2, pp=1, vpp=1))
    _validate_parallel_scope(ParallelConfig(tp=1, ep=2, etp=1, cp=1, pp=2, vpp=2))
    with pytest.raises(NotImplementedError):
        _validate_parallel_scope(ParallelConfig(tp=2, ep=1, etp=1, cp=1, pp=1, vpp=1))
    with pytest.raises(NotImplementedError):
        _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=2, cp=1, pp=1, vpp=1))


def test_glm5_impl_config_accepts_runtime_mtp_fields():
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.protocol import ImplConfig

    cfg = Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1)

    assert ImplConfig(mtp_enable=False, mtp_enable_train=False).mtp_enable is False
    assert ImplConfig(mtp_enable=True, mtp_enable_train=True).mtp_enable_train is True
    assert cfg.num_nextn_predict_layers == 1


def test_glm5_pipeline_layer_split_handles_non_divisible_pp():
    from types import SimpleNamespace

    from megatron.lite.model.glm5.lite.model import _build_glm5_pipeline_layers

    def split_for_rank(pp_rank):
        ps = SimpleNamespace(pp_size=4, pp_rank=pp_rank)
        return _build_glm5_pipeline_layers(5, ps)

    assert [split_for_rank(rank) for rank in range(4)] == [[], [0, 1], [2, 3], [4]]


def test_glm5_protocol_uses_mlite_optimizer_api():
    from megatron.lite.model.glm5.lite.protocol import ImplConfig

    protocol_path = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "glm5"
        / "lite"
        / "protocol.py"
    )
    protocol_text = protocol_path.read_text()

    assert ImplConfig().optimizer == "dist_opt"
    assert "build_dist_opt_training_optimizer" in protocol_text


def test_glm5_protocol_build_model_freezes_base_params_and_reports_lora_stats(monkeypatch):
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import model as glm5_model
    from megatron.lite.model.glm5.lite import protocol
    from megatron.lite.model.glm5.lite.protocol import ImplConfig
    from megatron.lite.primitive.parallel import ParallelState

    captured_lora_configs = []

    class FakeGlm5Model(torch.nn.Module):
        def __init__(self, model_cfg, train_cfg, ps, **kwargs):
            super().__init__()
            del model_cfg, train_cfg, ps
            captured_lora_configs.append(kwargs["lora_config"])
            self.layers = []
            self.base = torch.nn.Linear(2, 2, bias=False)
            self.lora_adapter = torch.nn.Parameter(torch.ones(2, 2))

        def cuda(self, device=None):
            del device
            return self

    monkeypatch.setattr(protocol, "init_parallel", lambda parallel: ParallelState())
    monkeypatch.setattr(glm5_model, "Glm5Model", FakeGlm5Model)

    bundle = protocol.build_model(
        Glm5Config(**_tiny_config_kwargs()),
        impl_cfg=ImplConfig(
            optimizer=None,
            lora={
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "target_modules": "all-linear",
                "use_rslora": True,
            },
        ),
    )

    assert len(captured_lora_configs) == 1
    assert captured_lora_configs[0].enabled is True
    assert captured_lora_configs[0].use_rslora is True
    chunk = bundle.chunks[0]
    assert chunk.base.weight.requires_grad is False
    assert chunk.lora_adapter.requires_grad is True
    assert bundle.extras["lora_config"].rank == 2
    assert bundle.extras["lora_stats"]["chunks"] == [
        {
            "lora_tensors": 1,
            "lora_numel": 4,
            "frozen_tensors": 1,
            "frozen_numel": 4,
            "trainable_tensors": 1,
            "trainable_numel": 4,
        }
    ]


def test_glm5_protocol_lora_init_runs_as_post_load_hook(monkeypatch):
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import model as glm5_model
    from megatron.lite.model.glm5.lite import protocol
    from megatron.lite.model.glm5.lite.protocol import ImplConfig
    from megatron.lite.primitive.parallel import ParallelState

    captured = {}

    class FakeGlm5Model(torch.nn.Module):
        def __init__(self, model_cfg, train_cfg, ps, **kwargs):
            super().__init__()
            del model_cfg, train_cfg, ps
            captured["lora_config"] = kwargs["lora_config"]
            self.layers = []
            self.base = torch.nn.Linear(2, 2, bias=False)
            self.lora_adapter = torch.nn.Parameter(torch.ones(2, 2))

        def cuda(self, device=None):
            del device
            return self

    def fake_initialize_lora_olora_tail(chunks, model_cfg, ps):
        captured["chunks"] = chunks
        captured["model_cfg"] = model_cfg
        captured["ps"] = ps
        return {"init_lora_weights": "olora_tail", "initialized_modules": 3}

    monkeypatch.setattr(protocol, "init_parallel", lambda parallel: ParallelState())
    monkeypatch.setattr(glm5_model, "Glm5Model", FakeGlm5Model)
    monkeypatch.setattr(protocol, "initialize_lora_olora_tail", fake_initialize_lora_olora_tail)

    model_cfg = Glm5Config(**_tiny_config_kwargs())
    bundle = protocol.build_model(
        model_cfg,
        impl_cfg=ImplConfig(
            optimizer=None,
            lora={"rank": 2, "alpha": 4, "target_modules": "all-linear"},
            lora_init="olora_tail",
        ),
    )

    assert bundle.optimizer is None
    assert bundle.extras["lora_init"] == "olora_tail"
    assert captured["lora_config"].enabled is True
    post_load_hook = bundle.extras["post_model_load_hook"]
    updates = post_load_hook()

    assert updates == {
        "extras": {
            "lora_init_result": {
                "init_lora_weights": "olora_tail",
                "initialized_modules": 3,
            }
        }
    }
    assert captured["chunks"] == bundle.chunks
    assert captured["model_cfg"] is model_cfg
    assert isinstance(captured["ps"], ParallelState)


def test_glm5_protocol_lora_init_accepts_mapping_config(monkeypatch):
    import torch
    from types import MappingProxyType

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import model as glm5_model
    from megatron.lite.model.glm5.lite import protocol
    from megatron.lite.model.glm5.lite.protocol import ImplConfig
    from megatron.lite.primitive.parallel import ParallelState

    captured = {}

    class FakeGlm5Model(torch.nn.Module):
        def __init__(self, model_cfg, train_cfg, ps, **kwargs):
            super().__init__()
            del model_cfg, train_cfg, ps
            captured["lora_config"] = kwargs["lora_config"]
            self.layers = []
            self.base = torch.nn.Linear(2, 2, bias=False)
            self.lora_adapter = torch.nn.Parameter(torch.ones(2, 2))

        def cuda(self, device=None):
            del device
            return self

    def fake_initialize_lora_olora_tail(chunks, model_cfg, ps):
        captured["chunks"] = chunks
        captured["model_cfg"] = model_cfg
        captured["ps"] = ps
        return {"init_lora_weights": "olora_tail", "initialized_modules": 1}

    monkeypatch.setattr(protocol, "init_parallel", lambda parallel: ParallelState())
    monkeypatch.setattr(glm5_model, "Glm5Model", FakeGlm5Model)
    monkeypatch.setattr(protocol, "initialize_lora_olora_tail", fake_initialize_lora_olora_tail)

    model_cfg = Glm5Config(**_tiny_config_kwargs())
    bundle = protocol.build_model(
        model_cfg,
        impl_cfg=ImplConfig(
            optimizer=None,
            lora=MappingProxyType(
                {
                    "rank": 2,
                    "alpha": 4,
                    "target_modules": "all-linear",
                    "init_lora_weights": "olora_tail",
                }
            ),
        ),
    )

    assert bundle.extras["lora_init"] == "olora_tail"
    assert captured["lora_config"].enabled is True
    assert captured["lora_config"].rank == 2
    assert bundle.extras["post_model_load_hook"]() == {
        "extras": {
            "lora_init_result": {
                "init_lora_weights": "olora_tail",
                "initialized_modules": 1,
            }
        }
    }
    assert captured["chunks"] == bundle.chunks
    assert captured["model_cfg"] is model_cfg
    assert isinstance(captured["ps"], ParallelState)


def test_glm5_protocol_lora_init_requires_enabled_lora():
    import pytest

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import protocol
    from megatron.lite.model.glm5.lite.protocol import ImplConfig

    with pytest.raises(ValueError, match="requires enabled LoRA"):
        protocol.build_model(
            Glm5Config(**_tiny_config_kwargs()),
            impl_cfg=ImplConfig(optimizer=None, lora={"rank": 0}, lora_init="olora_tail"),
        )


def test_qwen3_moe_lora_init_runs_as_post_load_hook(monkeypatch):
    import torch

    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol
    from megatron.lite.model.qwen3_moe.lite.protocol import ImplConfig
    from megatron.lite.primitive.parallel import ParallelState

    captured = {}

    class FakeQwen3MoEModel(torch.nn.Module):
        def __init__(self, model_cfg, ps, **kwargs):
            super().__init__()
            del model_cfg, ps
            captured["lora_config"] = kwargs["lora_config"]
            self.layers = []
            self.base = torch.nn.Linear(2, 2, bias=False)
            self.lora_adapter = torch.nn.Parameter(torch.ones(2, 2))

        def to(self, *args, **kwargs):
            del args, kwargs
            return self

        def cuda(self, device=None):
            del device
            return self

    def fake_initialize_lora_olora_tail(chunks, model_cfg, ps):
        captured["chunks"] = chunks
        captured["model_cfg"] = model_cfg
        captured["ps"] = ps
        return {"init_lora_weights": "olora_tail", "initialized_modules": 2}

    monkeypatch.setattr(protocol, "init_parallel", lambda parallel: ParallelState())
    monkeypatch.setattr(protocol, "Qwen3MoEModel", FakeQwen3MoEModel)
    monkeypatch.setattr(protocol, "initialize_lora_olora_tail", fake_initialize_lora_olora_tail)

    model_cfg = Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        layer_types=["full_attention"],
    )
    bundle = protocol.build_model(
        model_cfg,
        impl_cfg=ImplConfig(
            optimizer=None,
            lora={"rank": 2, "alpha": 4, "target_modules": "linear_qkv"},
            lora_init="olora_tail",
        ),
    )

    assert bundle.optimizer is None
    assert bundle.extras["lora_init"] == "olora_tail"
    assert captured["lora_config"].enabled is True
    post_load_hook = bundle.extras["post_model_load_hook"]
    assert post_load_hook is not None

    assert post_load_hook() == {
        "extras": {
            "lora_init_result": {
                "init_lora_weights": "olora_tail",
                "initialized_modules": 2,
            }
        }
    }
    assert captured["chunks"] == bundle.chunks
    assert captured["model_cfg"] is model_cfg
    assert isinstance(captured["ps"], ParallelState)


def test_qwen3_moe_lora_init_rejects_unsupported_values():
    import pytest

    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol
    from megatron.lite.model.qwen3_moe.lite.protocol import ImplConfig

    model_cfg = Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        layer_types=["full_attention"],
    )

    with pytest.raises(NotImplementedError, match="not implemented in MLite yet"):
        protocol.build_model(
            model_cfg,
            impl_cfg=ImplConfig(
                optimizer=None,
                lora={"rank": 2, "init_lora_weights": "pissa"},
            ),
        )

    with pytest.raises(ValueError, match="requires enabled LoRA"):
        protocol.build_model(
            model_cfg,
            impl_cfg=ImplConfig(optimizer=None, lora={"rank": 0}, lora_init="olora_tail"),
        )


def test_glm5_lite_tiny_cpu_forward_backward(monkeypatch):
    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.modules.attention import dsa

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        kv_offset=0,
        calculate_per_token_loss=False,
        value_dim=None,
    ):
        del (
            kv_full,
            attn_sink,
            window_idxs,
            q_indexer,
            k_indexer,
            weights,
            indexer_topk,
            ratio,
            softmax_scale,
            indexer_softmax_scale,
            loss_coeff,
            sparse_loss,
            kv_offset,
            calculate_per_token_loss,
        )
        return query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        ), torch.zeros((), device=query.device, dtype=torch.float32)

    def fake_indexer_topk(q_indexer, k_indexer, weights, topk, ratio, indexer_softmax_scale=1.0):
        del q_indexer, k_indexer, weights, ratio, indexer_softmax_scale
        return (
            torch.zeros((1, 5, topk), dtype=torch.int32),
            torch.full((1, 5), topk, dtype=torch.int32),
        )

    def fake_build_flat_topk_idxs(topk_indices, *, batch_size, seqlen_kv, compact=True):
        del batch_size, seqlen_kv, compact
        return topk_indices, torch.full(topk_indices.shape[:2], topk_indices.shape[-1])

    def fake_dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length=None,
        indexer_topk=0,
        value_dim=None,
    ):
        del kv_full, attn_sink, topk_idxs, softmax_scale, topk_length, indexer_topk
        return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

    monkeypatch.setattr(
        dsa._dsa_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )
    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", fake_indexer_topk)
    monkeypatch.setattr(dsa._dsa_kernels, "build_flat_topk_idxs", fake_build_flat_topk_idxs)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", fake_dsa_sparse_attn)

    torch.manual_seed(1234)
    model = Glm5ForCausalLM(Glm5Config(**_tiny_config_kwargs())).float()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    labels = torch.randint(0, model.config.vocab_size, (2, 5))

    output = model(input_ids=input_ids, labels=labels)

    assert output["hidden_states"].shape == (2, 5, model.config.hidden_size)
    assert output["loss"].ndim == 0
    output["loss"].backward()
    grad_norm = sum(
        param.grad.detach().float().norm() for param in model.parameters() if param.grad is not None
    )
    assert torch.isfinite(grad_norm)

    mtp_model = Glm5ForCausalLM(
        Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1),
        mtp_enable=True,
        mtp_enable_train=True,
    ).float()
    mtp_output = mtp_model(
        input_ids=input_ids, labels=labels, loss_mask=torch.ones_like(labels, dtype=torch.float32)
    )

    assert len(mtp_output["mtp_hidden_states"]) == 1
    assert mtp_output["mtp_hidden_states"][0].shape == (2, 5, mtp_model.config.hidden_size)
    assert mtp_output["mtp_loss"].ndim == 0
    assert "mtp_logits" not in mtp_output
    mtp_output["loss"].backward()
    mtp_grad_norm = sum(
        param.grad.detach().float().norm()
        for param in mtp_model.parameters()
        if param.grad is not None
    )
    assert torch.isfinite(mtp_grad_norm)

    with torch.no_grad():
        mtp_infer_output = mtp_model(input_ids=input_ids)
    assert len(mtp_infer_output["mtp_logits"]) == 1
    assert mtp_infer_output["mtp_logits"][0].shape == (2, 5, mtp_model.config.vocab_size)


def test_glm5_lora_tiny_cpu_forward_backward_freezes_base_params(monkeypatch):
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.modules.lora import freeze_non_lora_params

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        kv_offset=0,
        calculate_per_token_loss=False,
        value_dim=None,
    ):
        del (
            kv_full,
            attn_sink,
            window_idxs,
            q_indexer,
            k_indexer,
            weights,
            indexer_topk,
            ratio,
            softmax_scale,
            indexer_softmax_scale,
            loss_coeff,
            sparse_loss,
            kv_offset,
            calculate_per_token_loss,
        )
        return query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        ), torch.zeros((), device=query.device, dtype=torch.float32)

    def fake_indexer_topk(q_indexer, k_indexer, weights, topk, ratio, indexer_softmax_scale=1.0):
        del q_indexer, k_indexer, weights, ratio, indexer_softmax_scale
        return (
            torch.zeros((1, 5, topk), dtype=torch.int32),
            torch.full((1, 5), topk, dtype=torch.int32),
        )

    def fake_build_flat_topk_idxs(topk_indices, *, batch_size, seqlen_kv, compact=True):
        del batch_size, seqlen_kv, compact
        return topk_indices, torch.full(topk_indices.shape[:2], topk_indices.shape[-1])

    def fake_dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length=None,
        indexer_topk=0,
        value_dim=None,
    ):
        del kv_full, attn_sink, topk_idxs, softmax_scale, topk_length, indexer_topk
        return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

    monkeypatch.setattr(
        dsa._dsa_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )
    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", fake_indexer_topk)
    monkeypatch.setattr(dsa._dsa_kernels, "build_flat_topk_idxs", fake_build_flat_topk_idxs)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", fake_dsa_sparse_attn)

    torch.manual_seed(4321)
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "use_rslora": True,
    }
    model = Glm5ForCausalLM(Glm5Config(**_tiny_config_kwargs()), lora_config=lora_config).float()

    stats = freeze_non_lora_params(model)
    assert stats["lora_tensors"] > 0
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(("lora" in name.lower() or "adapter" in name.lower()) for name in trainable)
    assert any("self_attention.q_a_lora.lora_a" in name for name in trainable)
    assert any("mlp.gate_up_lora.lora_b" in name for name in trainable)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    labels = torch.randint(0, model.config.vocab_size, (2, 5))
    output = model(input_ids=input_ids, labels=labels)

    assert output["loss"].ndim == 0
    output["loss"].backward()

    lora_grads = [
        (name, param.grad)
        for name, param in model.named_parameters()
        if param.requires_grad and ("lora" in name.lower() or "adapter" in name.lower())
    ]
    assert lora_grads
    present_lora_grads = [(name, grad) for name, grad in lora_grads if grad is not None]
    assert present_lora_grads
    assert all(torch.isfinite(grad.detach()).all() for _, grad in present_lora_grads)
    assert any(
        name.endswith("lora_b") and grad.detach().float().abs().sum().item() > 0
        for name, grad in present_lora_grads
    )
    frozen_grads = [
        name
        for name, param in model.named_parameters()
        if not param.requires_grad and param.grad is not None
    ]
    assert frozen_grads == []


def test_glm5_router_replay_tiny_cpu_record_replay_identity(monkeypatch):
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import model as glm5_model
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.modules.router_replay import RouterReplayAction

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        kv_offset=0,
        calculate_per_token_loss=False,
        value_dim=None,
    ):
        del (
            kv_full,
            attn_sink,
            window_idxs,
            q_indexer,
            k_indexer,
            weights,
            indexer_topk,
            ratio,
            softmax_scale,
            indexer_softmax_scale,
            loss_coeff,
            sparse_loss,
            kv_offset,
            calculate_per_token_loss,
        )
        return query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        ), torch.zeros((), device=query.device, dtype=torch.float32)

    def fake_indexer_topk(q_indexer, k_indexer, weights, topk, ratio, indexer_softmax_scale=1.0):
        del q_indexer, k_indexer, weights, ratio, indexer_softmax_scale
        return (
            torch.zeros((1, 5, topk), dtype=torch.int32),
            torch.full((1, 5), topk, dtype=torch.int32),
        )

    def fake_build_flat_topk_idxs(topk_indices, *, batch_size, seqlen_kv, compact=True):
        del batch_size, seqlen_kv, compact
        return topk_indices, torch.full(topk_indices.shape[:2], topk_indices.shape[-1])

    def fake_dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length=None,
        indexer_topk=0,
        value_dim=None,
    ):
        del kv_full, attn_sink, topk_idxs, softmax_scale, topk_length, indexer_topk
        return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

    monkeypatch.setattr(
        dsa._dsa_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )
    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", fake_indexer_topk)
    monkeypatch.setattr(dsa._dsa_kernels, "build_flat_topk_idxs", fake_build_flat_topk_idxs)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", fake_dsa_sparse_attn)

    current_scores = {"value": None}
    routed_indices_seen = []

    def fake_topk_routing_with_score_function(
        logits,
        topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
        fused=False,
        dense_output=False,
        router_replay=None,
    ):
        del use_pre_softmax, scaling_factor, score_function, expert_bias, fused
        scores = current_scores["value"].to(device=logits.device, dtype=logits.dtype)
        assert scores.shape == logits.shape

        def default_topk(score_tensor, topk_value, _num_groups, _group_topk):
            del _num_groups, _group_topk
            return score_tensor.topk(topk_value, dim=1)

        if router_replay is not None:
            values, indices = router_replay.get_replay_topk(
                scores,
                topk,
                num_groups,
                group_topk,
                default_topk,
            )
        else:
            values, indices = default_topk(scores, topk, num_groups, group_topk)
        routed_indices_seen.append(indices.detach().clone())
        if dense_output:
            return values, indices
        routing_probs = torch.zeros_like(scores).scatter(1, indices, values)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(
            1, indices, torch.ones_like(values, dtype=torch.bool)
        )
        return routing_probs, routing_map

    monkeypatch.setattr(
        glm5_model,
        "topk_routing_with_score_function",
        fake_topk_routing_with_score_function,
    )

    torch.manual_seed(1234)
    model = Glm5ForCausalLM(
        Glm5Config(**_tiny_config_kwargs()),
        router_replay=True,
    ).float()
    routers = model.router_replay_instances()
    assert len(routers) == 1
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    num_tokens = input_ids.numel()
    record_scores = torch.tensor(
        [[0.9, 0.1, 0.8], [0.1, 0.7, 0.6]],
        dtype=torch.float32,
    ).repeat(num_tokens // 2, 1)
    replay_natural_scores = torch.tensor(
        [[0.1, 0.95, 0.2], [0.2, 0.1, 0.95]],
        dtype=torch.float32,
    ).repeat(num_tokens // 2, 1)
    current_scores["value"] = record_scores

    model.set_router_replay_action(RouterReplayAction.RECORD)
    record_output = model(input_ids=input_ids)
    recorded = record_output["routed_experts"][0].detach().clone()
    model.clear_router_replay_action()
    assert recorded.shape == (num_tokens, model.config.num_experts_per_tok)
    torch.testing.assert_close(recorded, routed_indices_seen[-1])
    natural_replay_indices = replay_natural_scores.topk(model.config.num_experts_per_tok, dim=1).indices
    assert not torch.equal(recorded, natural_replay_indices)

    model.model.layers[1].moe.router.local_tokens_per_expert.zero_()
    current_scores["value"] = replay_natural_scores
    model.set_router_replay_data([recorded])
    model.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    replay_output = model(input_ids=input_ids)
    model.clear_router_replay_action()

    assert "routed_experts" not in replay_output
    torch.testing.assert_close(routed_indices_seen[-1], recorded)
    expected_counts = torch.zeros(model.config.num_experts, dtype=torch.float32)
    expected_counts.scatter_add_(
        0,
        recorded.reshape(-1),
        torch.ones(recorded.numel(), dtype=torch.float32),
    )
    torch.testing.assert_close(
        model.model.layers[1].moe.router.local_tokens_per_expert,
        expected_counts,
    )


def test_glm5_router_replay_preserves_duplicate_topk_slots_for_padding():
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5SigmoidTopKRouter
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
    from megatron.lite.primitive.parallel import ParallelState

    replay = RouterReplay(layer_idx=1)
    cfg = Glm5Config(**_tiny_config_kwargs())
    router = Glm5SigmoidTopKRouter(cfg, ParallelState(), router_replay=replay).float()
    router.eval()
    with torch.no_grad():
        router.gate.weight.fill_(0.01)
        router.local_tokens_per_expert.zero_()

    x = torch.randn(3, cfg.hidden_size)
    duplicate_replay = torch.tensor([[0, 0], [2, 2], [1, 1]], dtype=torch.long)
    replay.set_target_indices(duplicate_replay)
    replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    try:
        topk_scores, topk_indices = router(x)
    finally:
        replay.clear_router_replay_action()

    torch.testing.assert_close(topk_indices, duplicate_replay)
    assert int(topk_indices.max().item()) < cfg.num_experts
    assert topk_scores.shape == duplicate_replay.shape
    assert torch.isfinite(topk_scores).all()
    torch.testing.assert_close(
        router.local_tokens_per_expert,
        torch.tensor([1.0, 1.0, 1.0]),
    )


def test_glm5_router_replay_instances_append_mtp_routers_after_main_layers():
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.modules.router_replay import RouterReplay

    RouterReplay.clear_global_router_replay_instances()
    try:
        cfg = Glm5Config(**{**_tiny_config_kwargs(), "num_nextn_predict_layers": 1})
        model = Glm5ForCausalLM(
            cfg,
            mtp_enable=True,
            router_replay=True,
        ).float()

        routers = model.router_replay_instances()
        assert [router.layer_idx for router in routers] == [1, 2]

        main_replay = torch.zeros(3, cfg.num_experts_per_tok, dtype=torch.long)
        mtp_replay = torch.ones(3, cfg.num_experts_per_tok, dtype=torch.long)
        model.set_router_replay_data([main_replay, mtp_replay])

        torch.testing.assert_close(routers[0].target_topk_idx, main_replay)
        torch.testing.assert_close(routers[1].target_topk_idx, mtp_replay)
        assert routers[0].target_topk_idx is not main_replay
        assert routers[1].target_topk_idx is not mtp_replay

        try:
            model.set_router_replay_data([main_replay])
        except ValueError as exc:
            assert "expected 2 tensors" in str(exc)
        else:
            raise AssertionError("set_router_replay_data accepted a missing MTP replay tensor.")

        model.clear_router_replay_indices()
        assert all(router.target_topk_idx is None for router in routers)
    finally:
        RouterReplay.clear_global_router_replay_instances()


def test_glm5_mtp_router_replay_tiny_cpu_record_replay_identity(monkeypatch):
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import model as glm5_model
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction

    def fake_indexer_topk(q_indexer, k_indexer, weights, topk, ratio, indexer_softmax_scale=1.0):
        del q_indexer, k_indexer, weights, ratio, indexer_softmax_scale
        return (
            torch.zeros((1, 5, topk), dtype=torch.int32),
            torch.full((1, 5), topk, dtype=torch.int32),
        )

    def fake_build_flat_topk_idxs(topk_indices, *, batch_size, seqlen_kv, compact=True):
        del batch_size, seqlen_kv, compact
        return topk_indices, torch.full(topk_indices.shape[:2], topk_indices.shape[-1])

    def fake_dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length=None,
        indexer_topk=0,
        value_dim=None,
    ):
        del kv_full, attn_sink, topk_idxs, softmax_scale, topk_length, indexer_topk
        return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", fake_indexer_topk)
    monkeypatch.setattr(dsa._dsa_kernels, "build_flat_topk_idxs", fake_build_flat_topk_idxs)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", fake_dsa_sparse_attn)

    score_queues: list[torch.Tensor] = []
    routed_indices_seen: list[torch.Tensor] = []

    def fake_topk_routing_with_score_function(
        logits,
        topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
        fused=False,
        dense_output=False,
        router_replay=None,
    ):
        del use_pre_softmax, scaling_factor, score_function, expert_bias, fused
        assert score_queues, "test did not provide router scores for this call"
        scores = score_queues.pop(0).to(device=logits.device, dtype=logits.dtype)
        assert scores.shape == logits.shape

        def default_topk(score_tensor, topk_value, _num_groups, _group_topk):
            del _num_groups, _group_topk
            return score_tensor.topk(topk_value, dim=1)

        if router_replay is not None:
            values, indices = router_replay.get_replay_topk(
                scores,
                topk,
                num_groups,
                group_topk,
                default_topk,
            )
        else:
            values, indices = default_topk(scores, topk, num_groups, group_topk)
        routed_indices_seen.append(indices.detach().clone())
        if dense_output:
            return values, indices
        routing_probs = torch.zeros_like(scores).scatter(1, indices, values)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(
            1, indices, torch.ones_like(values, dtype=torch.bool)
        )
        return routing_probs, routing_map

    monkeypatch.setattr(
        glm5_model,
        "topk_routing_with_score_function",
        fake_topk_routing_with_score_function,
    )

    RouterReplay.clear_global_router_replay_instances()
    try:
        torch.manual_seed(1234)
        cfg = Glm5Config(**{**_tiny_config_kwargs(), "num_nextn_predict_layers": 1})
        model = Glm5ForCausalLM(
            cfg,
            mtp_enable=True,
            router_replay=True,
        ).float()
        routers = model.router_replay_instances()
        assert [router.layer_idx for router in routers] == [1, 2]
        input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
        num_tokens = input_ids.numel()

        main_record_scores = torch.tensor(
            [[0.9, 0.1, 0.8], [0.1, 0.7, 0.6]],
            dtype=torch.float32,
        ).repeat(num_tokens // 2, 1)
        mtp_record_scores = torch.tensor(
            [[0.2, 0.9, 0.8], [0.8, 0.7, 0.1]],
            dtype=torch.float32,
        ).repeat(num_tokens // 2, 1)
        main_natural_scores = torch.tensor(
            [[0.1, 0.95, 0.2], [0.2, 0.1, 0.95]],
            dtype=torch.float32,
        ).repeat(num_tokens // 2, 1)
        mtp_natural_scores = torch.tensor(
            [[0.95, 0.2, 0.1], [0.1, 0.95, 0.2]],
            dtype=torch.float32,
        ).repeat(num_tokens // 2, 1)

        score_queues[:] = [main_record_scores, mtp_record_scores]
        model.set_router_replay_action(RouterReplayAction.RECORD)
        record_output = model(input_ids=input_ids)
        model.clear_router_replay_action()
        assert score_queues == []
        recorded = [tensor.detach().clone() for tensor in record_output["routed_experts"]]
        assert [tuple(tensor.shape) for tensor in recorded] == [
            (num_tokens, cfg.num_experts_per_tok),
            (num_tokens, cfg.num_experts_per_tok),
        ]
        torch.testing.assert_close(recorded[0], routed_indices_seen[-2])
        torch.testing.assert_close(recorded[1], routed_indices_seen[-1])
        assert not torch.equal(recorded[0], main_natural_scores.topk(cfg.num_experts_per_tok, dim=1).indices)
        assert not torch.equal(recorded[1], mtp_natural_scores.topk(cfg.num_experts_per_tok, dim=1).indices)

        for layer in model.model.layers:
            if layer.moe is not None:
                layer.moe.router.local_tokens_per_expert.zero_()
        assert model.model.mtp is not None
        for mtp_layer in model.model.mtp.layers:
            layer = mtp_layer.transformer_layer
            if layer.moe is not None:
                layer.moe.router.local_tokens_per_expert.zero_()

        score_queues[:] = [main_natural_scores, mtp_natural_scores]
        model.set_router_replay_data(recorded)
        model.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        replay_output = model(input_ids=input_ids)
        model.clear_router_replay_action()

        assert score_queues == []
        assert "routed_experts" not in replay_output
        torch.testing.assert_close(routed_indices_seen[-2], recorded[0])
        torch.testing.assert_close(routed_indices_seen[-1], recorded[1])
    finally:
        RouterReplay.clear_global_router_replay_instances()


def test_qwen3_moe_router_replay_tiny_cpu_record_replay_identity(monkeypatch):
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel
    from megatron.lite.primitive.modules import router as router_module
    from megatron.lite.primitive.modules.router_replay import RouterReplayAction
    from megatron.lite.primitive.parallel import ParallelState

    current_scores = {"value": None}
    routed_indices_seen = []

    def fake_topk_routing_with_score_function(
        logits,
        topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
        fused=False,
        dense_output=False,
        router_replay=None,
    ):
        del use_pre_softmax, scaling_factor, score_function, expert_bias, fused
        scores = current_scores["value"].to(device=logits.device, dtype=logits.dtype)
        assert scores.shape == logits.shape

        def default_topk(score_tensor, topk_value, _num_groups, _group_topk):
            del _num_groups, _group_topk
            return score_tensor.topk(topk_value, dim=1)

        if router_replay is not None:
            values, indices = router_replay.get_replay_topk(
                scores,
                topk,
                num_groups,
                group_topk,
                default_topk,
            )
        else:
            values, indices = default_topk(scores, topk, num_groups, group_topk)
        routed_indices_seen.append(indices.detach().clone())
        if dense_output:
            return values, indices
        routing_probs = torch.zeros_like(scores).scatter(1, indices, values)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(
            1, indices, torch.ones_like(values, dtype=torch.bool)
        )
        return routing_probs, routing_map

    monkeypatch.setattr(
        router_module,
        "topk_routing_with_score_function",
        fake_topk_routing_with_score_function,
    )

    cfg = Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        num_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=6,
        max_position_embeddings=16,
        layer_types=["full_attention"],
    )
    torch.manual_seed(1234)
    model = Qwen3MoEModel(cfg, ParallelState(), use_deepep=False, router_replay=True).float()
    routers = model.router_replay_instances()
    assert len(routers) == 1
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    num_tokens = input_ids.numel()
    record_scores = torch.tensor(
        [[0.9, 0.1, 0.8], [0.1, 0.7, 0.6]],
        dtype=torch.float32,
    ).repeat(num_tokens // 2, 1)
    replay_natural_scores = torch.tensor(
        [[0.1, 0.95, 0.2], [0.2, 0.1, 0.95]],
        dtype=torch.float32,
    ).repeat(num_tokens // 2, 1)
    current_scores["value"] = record_scores

    model.set_router_replay_action(RouterReplayAction.RECORD)
    record_output = model(input_ids=input_ids)
    recorded = record_output["routed_experts"][0].detach().clone()
    model.clear_router_replay_action()

    assert recorded.shape == (num_tokens, cfg.num_experts_per_tok)
    torch.testing.assert_close(recorded, routed_indices_seen[-1])
    natural_replay_indices = replay_natural_scores.topk(cfg.num_experts_per_tok, dim=1).indices
    assert not torch.equal(recorded, natural_replay_indices)

    current_scores["value"] = replay_natural_scores
    model.set_router_replay_data([recorded])
    model.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    replay_output = model(input_ids=input_ids)
    model.clear_router_replay_action()

    assert "routed_experts" not in replay_output
    torch.testing.assert_close(routed_indices_seen[-1], recorded)
    assert replay_output["hidden_states"].shape == (5, 2, cfg.hidden_size)


def test_qwen3_moe_router_replay_instances_append_mtp_routers_after_main_layers():
    import torch

    _install_te_rmsnorm_stub_for_cpu_dsa(torch)
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel
    from megatron.lite.primitive.modules.router_replay import RouterReplay
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Qwen3MoEConfig(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        num_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=6,
        max_position_embeddings=16,
        num_nextn_predict_layers=1,
        layer_types=["full_attention", "full_attention"],
    )

    RouterReplay.clear_global_router_replay_instances()
    try:
        model = Qwen3MoEModel(
            cfg,
            ParallelState(),
            use_deepep=False,
            mtp_enable=True,
            router_replay=True,
        ).float()

        routers = model.router_replay_instances()
        assert [router.layer_idx for router in routers] == [0, 1, 2]

        main0_replay = torch.zeros(3, cfg.num_experts_per_tok, dtype=torch.long)
        main1_replay = torch.ones(3, cfg.num_experts_per_tok, dtype=torch.long)
        mtp_replay = torch.full((3, cfg.num_experts_per_tok), 2, dtype=torch.long)
        model.set_router_replay_data([main0_replay, main1_replay, mtp_replay])

        torch.testing.assert_close(routers[0].target_topk_idx, main0_replay)
        torch.testing.assert_close(routers[1].target_topk_idx, main1_replay)
        torch.testing.assert_close(routers[2].target_topk_idx, mtp_replay)
        assert routers[2].target_topk_idx is not mtp_replay

        try:
            model.set_router_replay_data([main0_replay, main1_replay])
        except ValueError as exc:
            assert "expected 3 tensors" in str(exc)
        else:
            raise AssertionError("set_router_replay_data accepted a missing Qwen3-MoE MTP replay tensor.")

        model.clear_router_replay_indices()
    finally:
        RouterReplay.clear_global_router_replay_instances()
