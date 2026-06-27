# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.mlite


@pytest.fixture(autouse=True)
def _te_import_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


def _split_grouped_qkvg():
    from megatron.lite.primitive.modules import split_grouped_qkvg

    return split_grouped_qkvg


def _moe_aux_scaler():
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    return MoEAuxLossAutoScaler


def _router_and_parallel_state(monkeypatch):
    from megatron.lite.primitive.modules.router import TopKRouter
    from megatron.lite.primitive.parallel import ParallelState

    del monkeypatch
    return TopKRouter, ParallelState


def _router_config():
    return SimpleNamespace(
        hidden_size=4, num_experts=4, num_experts_per_tok=2, router_aux_loss_coef=0.1
    )


def _walk_grad_fn_names(tensor: torch.Tensor) -> set[str]:
    names: set[str] = set()
    stack = [tensor.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None:
            continue
        names.add(type(fn).__name__)
        stack.extend(parent for parent, _idx in fn.next_functions)
    return names


def test_gqa_split_grouped_qkvg_preserves_q_gate_kv_order():
    split_grouped_qkvg = _split_grouped_qkvg()
    qkv = torch.arange(24).reshape(1, 24)

    query, gate, key, value = split_grouped_qkvg(qkv, num_heads=4, num_kv_heads=2, head_dim=2)

    assert query.shape == (1, 4, 2)
    assert gate.shape == (1, 4, 2)
    assert key.shape == (1, 2, 2)
    assert value.shape == (1, 2, 2)
    assert torch.equal(query, torch.tensor([[[0, 1], [2, 3], [12, 13], [14, 15]]]))
    assert torch.equal(gate, torch.tensor([[[4, 5], [6, 7], [16, 17], [18, 19]]]))
    assert torch.equal(key, torch.tensor([[[8, 9], [20, 21]]]))
    assert torch.equal(value, torch.tensor([[[10, 11], [22, 23]]]))


def test_moe_aux_loss_auto_scaler_threads_scaled_aux_gradient():
    MoEAuxLossAutoScaler = _moe_aux_scaler()
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor([0.25]))
    output = torch.randn(3, requires_grad=True)
    aux_loss = torch.tensor(2.0, requires_grad=True)

    scaled_output = MoEAuxLossAutoScaler.apply(output * 2.0, aux_loss)
    scaled_output.sum().backward()

    torch.testing.assert_close(output.grad, torch.full_like(output, 2.0))
    torch.testing.assert_close(aux_loss.grad, torch.tensor(0.25))
    MoEAuxLossAutoScaler.main_loss_backward_scale = None


def test_topk_router_returns_finite_scores_and_valid_expert_indices(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=False)
    hidden = torch.randn(5, 4)

    scores, indices = router(hidden)

    assert scores.shape == (5, 2)
    assert indices.shape == (5, 2)
    assert scores.dtype == hidden.dtype
    assert torch.isfinite(scores).all()
    assert indices.min().item() >= 0
    assert indices.max().item() < config.num_experts


def test_topk_router_scores_are_normalized_and_deterministic_in_eval(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=False)
    hidden = torch.randn(5, config.hidden_size)

    router.eval()
    scores_1, indices_1 = router(hidden)
    scores_2, indices_2 = router(hidden)

    torch.testing.assert_close(scores_1.sum(dim=-1), torch.ones(hidden.size(0)))
    torch.testing.assert_close(scores_1, scores_2, atol=0, rtol=0)
    assert torch.equal(indices_1, indices_2)


def test_topk_router_does_not_attach_aux_scaler_in_eval(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=True)
    hidden = torch.randn(5, config.hidden_size)

    router.eval()
    scores, _indices = router(hidden)

    assert not any("MoEAuxLoss" in name for name in _walk_grad_fn_names(scores))


def test_topk_router_aux_loss_contributes_gate_gradient(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=True)
    hidden = torch.randn(8, config.hidden_size)

    router.train()
    scores, _indices = router(hidden)
    scores.sum().backward()
    grad_with_aux = router.gate.weight.grad.detach().clone()

    router.zero_grad()
    saved_coeff = router.aux_loss_coeff
    router.aux_loss_coeff = 0.0
    scores_no_aux, _indices = router(hidden)
    scores_no_aux.sum().backward()
    grad_no_aux = router.gate.weight.grad.detach().clone()
    router.aux_loss_coeff = saved_coeff

    assert torch.isfinite(grad_with_aux).all()
    assert torch.isfinite(grad_no_aux).all()
    assert (grad_with_aux - grad_no_aux).abs().sum().item() > 0.0


def test_dsa_index_share_schedule_and_state():
    from megatron.lite.primitive.modules.attention.dsa import (
        DSAIndexShareState,
        dsa_indexer_type_for_layer,
        is_dsa_skip_topk_layer,
        source_dsa_compute_layer,
    )

    assert is_dsa_skip_topk_layer(3, skip_topk_offset=3, topk_freq=4) is False
    assert is_dsa_skip_topk_layer(4, skip_topk_offset=3, topk_freq=4) is True
    assert dsa_indexer_type_for_layer(7, skip_topk_offset=3, topk_freq=4) == "full"
    assert source_dsa_compute_layer(6, skip_topk_offset=3, topk_freq=4) == 3

    state = DSAIndexShareState({3: 1})
    topk = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int32)
    state.save_topk(3, topk, sequence_key=0)
    assert state.cached_tensor_count == 1
    assert torch.equal(state.get_topk(6, 3, sequence_key=0), topk)
    assert state.cached_tensor_count == 0
    with pytest.raises(AssertionError, match="source layer 3"):
        state.get_topk(5, 3, sequence_key=1)


def test_dsa_index_share_state_releases_final_packed_consumer():
    from megatron.lite.primitive.modules.attention.dsa import DSAIndexShareState

    state = DSAIndexShareState({3: 2})
    segment_0 = torch.tensor([[[0, 1]]], dtype=torch.int32)
    segment_1 = torch.tensor([[[1, 0]]], dtype=torch.int32)
    state.save_topk(3, segment_0, sequence_key=0)
    state.save_topk(3, segment_1, sequence_key=1)
    assert state.cached_tensor_count == 2

    assert torch.equal(state.get_topk(4, 3, sequence_key=0), segment_0)
    assert torch.equal(state.get_topk(4, 3, sequence_key=1), segment_1)
    assert state.cached_tensor_count == 2
    assert torch.equal(state.get_topk(5, 3, sequence_key=0), segment_0)
    assert state.cached_tensor_count == 1
    assert torch.equal(state.get_topk(5, 3, sequence_key=1), segment_1)
    assert state.cached_tensor_count == 0


def test_dsa_index_share_state_rejects_unconsumed_source_save():
    from megatron.lite.primitive.modules.attention.dsa import DSAIndexShareState

    state = DSAIndexShareState()
    assert state.needs_topk(3) is False
    with pytest.raises(AssertionError, match="no shared consumer"):
        state.save_topk(3, torch.zeros(1, 1, 1, dtype=torch.int32))


def test_glm5_counts_local_index_share_consumers_including_repeated_mtp():
    from megatron.lite.model.glm5.lite.model import (
        _local_dsa_index_share_consumer_counts,
        _validate_dsa_index_share_recompute,
    )

    def layer(*, shared: bool, source_layer: int):
        dsa = SimpleNamespace(skip_topk=shared, index_share_source_layer=source_layer)
        return SimpleNamespace(self_attention=SimpleNamespace(self_attention=dsa))

    trunk_layers = [
        layer(shared=False, source_layer=3),
        layer(shared=True, source_layer=3),
        layer(shared=True, source_layer=3),
    ]
    mtp = SimpleNamespace(
        layers=[SimpleNamespace(transformer_layer=layer(shared=True, source_layer=7))],
        repeated_layer=True,
        num_layers=3,
    )

    consumer_counts = _local_dsa_index_share_consumer_counts(trunk_layers, mtp)
    assert consumer_counts == {3: 2, 7: 3}
    _validate_dsa_index_share_recompute(consumer_counts, ["moe", "attn_proj"])
    _validate_dsa_index_share_recompute({}, ["full", "self_attn", "dsa"])
    for unsafe_mode in ("full", "self_attn", "dsa"):
        with pytest.raises(ValueError, match="group-aware checkpoint"):
            _validate_dsa_index_share_recompute(consumer_counts, [unsafe_mode])


def test_dsv4_fused_dsa_legacy_two_output_api_cpu_mock(monkeypatch):
    """DSv4's public fused wrapper and direct Function keep two outputs."""
    from megatron.lite.primitive.kernels import dsa_kernels

    def fake_forward(ctx, *args):
        query = args[0]
        ctx.input_count = len(args)
        output = query * 2.0
        loss = query.float().sum() * 0.0
        topk = torch.zeros(query.shape[:2] + (1,), dtype=torch.int32)
        return output, loss, topk

    def fake_backward(ctx, grad_output, grad_loss, grad_topk=None):
        del grad_loss, grad_topk
        grads = [None] * ctx.input_count
        grads[0] = grad_output * 2.0
        return tuple(grads)

    with_topk_func = dsa_kernels._FusedIndexerSparseAttnWithTopKFunc
    monkeypatch.setattr(with_topk_func, "forward", staticmethod(fake_forward))
    monkeypatch.setattr(with_topk_func, "backward", staticmethod(fake_backward))

    query = torch.ones(2, 1, 1, 2, requires_grad=True)
    args = (
        query,
        torch.ones(2, 1, 2),
        torch.zeros(1),
        torch.empty(1, 2, 0, dtype=torch.int32),
        torch.ones(2, 1, 1, 2),
        torch.ones(2, 1, 2),
        torch.ones(2, 1, 1),
        1,
        1,
        0.5,
        1.0,
        0.0,
        False,
        0,
        False,
        2,
    )

    direct_result = dsa_kernels.FusedIndexerSparseAttnFunc.apply(*args)
    assert isinstance(direct_result, tuple)
    assert len(direct_result) == 2

    output, indexer_loss = dsa_kernels.fused_indexer_sparse_attn(*args)
    assert output.shape == query.shape
    assert indexer_loss.ndim == 0
    (output.sum() + indexer_loss).backward()
    torch.testing.assert_close(query.grad, torch.full_like(query, 2.0))

    with_topk_result = dsa_kernels.fused_indexer_sparse_attn_with_topk(*args)
    assert len(with_topk_result) == 3
    assert with_topk_result[2].dtype == torch.int32


def test_dsa_index_share_pipeline_guard_rejects_cross_stage_sources():
    from megatron.lite.primitive.modules.attention.dsa import (
        validate_dsa_index_share_pipeline_split,
    )

    validate_dsa_index_share_pipeline_split(
        [0, 1, 2, 3],
        topk_freq=4,
        skip_topk_offset=3,
    )
    with pytest.raises(ValueError, match="cannot cross pipeline stages"):
        validate_dsa_index_share_pipeline_split(
            [3, 4, 5],
            topk_freq=4,
            skip_topk_offset=3,
        )
    with pytest.raises(ValueError, match="must execute before"):
        validate_dsa_index_share_pipeline_split(
            [3, 2],
            topk_freq=4,
            skip_topk_offset=3,
        )
