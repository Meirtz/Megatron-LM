"""Runtime gates for RouterReplay primitive and protocol context semantics."""

from __future__ import annotations

import pytest


def _default_topk(scores, topk, num_groups, group_topk):
    del num_groups, group_topk
    return scores.topk(topk, dim=1)


def test_router_replay_compact_dtype_matches_roll_reference_bounds():
    torch = pytest.importorskip("torch")

    from megatron.lite.primitive.modules.router_replay import get_routed_experts_dtype

    assert get_routed_experts_dtype(0) == torch.uint8
    assert get_routed_experts_dtype(255) == torch.uint8
    assert get_routed_experts_dtype(256) == torch.int16
    assert get_routed_experts_dtype(32767) == torch.int16

    with pytest.raises(TypeError, match="max_expert_idx must be an int"):
        get_routed_experts_dtype(torch.tensor(1))
    with pytest.raises(ValueError, match="must be non-negative"):
        get_routed_experts_dtype(-1)
    with pytest.raises(ValueError, match="exceeds int16 range"):
        get_routed_experts_dtype(32768)


def test_padded_routed_experts_to_list_compacts_roll_style_batch():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import padded_routed_experts_to_list

    padded = torch.tensor(
        [
            [
                [[1, 0], [4, 3]],
                [[2, 1], [5, 4]],
                [[3, 2], [6, 5]],
                [[99, 99], [99, 99]],
            ],
            [
                [[9, 8], [12, 11]],
                [[10, 11], [13, 14]],
                [[88, 88], [88, 88]],
                [[77, 77], [77, 77]],
            ],
        ],
        dtype=torch.long,
    )

    first, second = padded_routed_experts_to_list(
        padded,
        torch.tensor([3, 2]),
        num_routers=2,
    )

    assert first.dtype == torch.uint8
    assert second.dtype == torch.uint8
    torch.testing.assert_close(
        first.to(torch.long),
        torch.tensor([[1, 0], [2, 1], [3, 2], [9, 8], [10, 11]]),
    )
    torch.testing.assert_close(
        second.to(torch.long),
        torch.tensor([[4, 3], [5, 4], [6, 5], [12, 11], [13, 14]]),
    )

    int16_padded = padded.clone()
    int16_padded[0, 0, 0, 0] = 256
    (int16_first, _) = padded_routed_experts_to_list(
        int16_padded,
        torch.tensor([1, 1]),
        num_routers=2,
    )
    assert int16_first.dtype == torch.int16

    with pytest.raises(ValueError, match="router dimension mismatch"):
        padded_routed_experts_to_list(padded, torch.tensor([3, 2]), num_routers=3)
    with pytest.raises(ValueError, match="seq_lens cannot exceed"):
        padded_routed_experts_to_list(padded, torch.tensor([5, 2]), num_routers=2)
    with pytest.raises(TypeError, match="integer tensor dtype"):
        padded_routed_experts_to_list(padded.float(), torch.tensor([3, 2]), num_routers=2)
    negative_padded = padded.clone()
    negative_padded[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="non-negative expert ids"):
        padded_routed_experts_to_list(negative_padded, torch.tensor([3, 2]), num_routers=2)


def test_concat_routed_experts_segments_matches_roll_multi_turn_contract():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import concat_routed_experts_segments

    first_turn_padded = torch.tensor(
        [
            [
                [[1, 0], [4, 3]],
                [[2, 1], [5, 4]],
                [[99, 99], [99, 99]],
            ],
            [
                [[8, 7], [6, 5]],
                [[88, 88], [88, 88]],
                [[77, 77], [77, 77]],
            ],
        ],
        dtype=torch.long,
    )
    second_turn_tokens_first = torch.tensor(
        [
            [[9, 0], [12, 3]],
            [[10, 1], [13, 4]],
            [[11, 2], [15, 6]],
        ],
        dtype=torch.long,
    )
    third_turn_per_router = [
        torch.tensor([[256, 0]], dtype=torch.long),
        torch.tensor([[14, 5]], dtype=torch.long),
    ]

    first, second = concat_routed_experts_segments(
        [first_turn_padded, second_turn_tokens_first, third_turn_per_router],
        [torch.tensor([2, 1]), None, None],
        num_routers=2,
    )

    assert first.dtype == torch.int16
    assert second.dtype == torch.uint8
    torch.testing.assert_close(
        first.to(torch.long),
        torch.tensor([[1, 0], [2, 1], [8, 7], [9, 0], [10, 1], [11, 2], [256, 0]]),
    )
    torch.testing.assert_close(
        second.to(torch.long),
        torch.tensor([[4, 3], [5, 4], [6, 5], [12, 3], [13, 4], [15, 6], [14, 5]]),
    )

    with pytest.raises(ValueError, match="non-empty"):
        concat_routed_experts_segments([])
    with pytest.raises(ValueError, match="requires seq_lens"):
        concat_routed_experts_segments([first_turn_padded], num_routers=2)
    with pytest.raises(ValueError, match="requires num_routers"):
        concat_routed_experts_segments([second_turn_tokens_first])
    with pytest.raises(ValueError, match="topk mismatch"):
        concat_routed_experts_segments(
            [
                [torch.tensor([[1, 0]]), torch.tensor([[2, 0]])],
                [torch.tensor([[1, 0, 2]]), torch.tensor([[2, 0]])],
            ],
            num_routers=2,
        )
    with pytest.raises(ValueError, match="non-negative expert ids"):
        concat_routed_experts_segments(
            [[torch.tensor([[1, -1]]), torch.tensor([[2, 0]])]],
            num_routers=2,
        )


def test_routed_experts_digest_canonicalizes_supported_layouts():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import routed_experts_digest

    first = torch.tensor([[1, 0], [2, 1], [8, 7]], dtype=torch.long)
    second = torch.tensor([[4, 3], [5, 4], [6, 5]], dtype=torch.long)
    per_router = [first, second]
    digest = routed_experts_digest(per_router, num_routers=2)

    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64

    routers_first = torch.stack(per_router, dim=0)
    tokens_first = torch.stack(per_router, dim=1)
    padded = torch.tensor(
        [
            [
                [[1, 0], [4, 3]],
                [[2, 1], [5, 4]],
                [[99, 99], [99, 99]],
            ],
            [
                [[8, 7], [6, 5]],
                [[88, 88], [88, 88]],
                [[77, 77], [77, 77]],
            ],
        ],
        dtype=torch.long,
    )
    per_router_uint8 = [tensor.to(torch.uint8) for tensor in per_router]
    per_router_int16 = [tensor.to(torch.int16) for tensor in per_router]

    assert routed_experts_digest(routers_first, num_routers=2) == digest
    assert routed_experts_digest(tokens_first, num_routers=2) == digest
    assert (
        routed_experts_digest(padded, seq_lens=torch.tensor([2, 1]), num_routers=2)
        == digest
    )
    assert routed_experts_digest(per_router_uint8, num_routers=2) == digest
    assert routed_experts_digest(per_router_int16, num_routers=2) == digest

    changed = [first.clone(), second.clone()]
    changed[1][1, 0] = 7
    assert routed_experts_digest(changed, num_routers=2) != digest

    ambiguous = torch.stack([first[:2], second[:2]], dim=0)
    with pytest.raises(ValueError, match="ambiguous"):
        routed_experts_digest(ambiguous, num_routers=2)


def test_routed_experts_segments_digest_matches_concatenated_contract():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import (
        concat_routed_experts_segments,
        routed_experts_digest,
        routed_experts_segments_digest,
    )

    first_turn_padded = torch.tensor(
        [
            [
                [[1, 0], [4, 3]],
                [[2, 1], [5, 4]],
                [[99, 99], [99, 99]],
            ],
            [
                [[8, 7], [6, 5]],
                [[88, 88], [88, 88]],
                [[77, 77], [77, 77]],
            ],
        ],
        dtype=torch.long,
    )
    second_turn_tokens_first = torch.tensor(
        [
            [[9, 0], [12, 3]],
            [[10, 1], [13, 4]],
            [[11, 2], [15, 6]],
        ],
        dtype=torch.long,
    )
    third_turn_per_router = [
        torch.tensor([[16, 0]], dtype=torch.long),
        torch.tensor([[14, 5]], dtype=torch.long),
    ]
    segments = [first_turn_padded, second_turn_tokens_first, third_turn_per_router]
    segment_seq_lens = [torch.tensor([2, 1]), None, None]
    concatenated = concat_routed_experts_segments(
        segments,
        segment_seq_lens,
        num_routers=2,
        compact=False,
    )

    assert routed_experts_segments_digest(
        segments,
        segment_seq_lens,
        num_routers=2,
    ) == routed_experts_digest(concatenated, num_routers=2)


def test_router_replay_trace_contract_builds_audit_safe_extras():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import routed_experts_digest
    from megatron.lite.primitive.modules import (
        build_router_replay_trace_contract as public_contract_builder,
    )
    from megatron.lite.primitive.modules.router_replay import (
        ROUTER_REPLAY_TRACE_CONFLICTS_KEY,
        build_router_replay_trace_contract,
        normalize_routed_experts_digest,
    )

    routed_experts = [
        torch.tensor([[1, 0], [2, 1], [8, 7]], dtype=torch.long),
        torch.tensor([[4, 3], [5, 4], [6, 5]], dtype=torch.long),
    ]
    digest = routed_experts_digest(routed_experts, num_routers=2)
    bare_upper_digest = digest.removeprefix("sha256:").upper()

    contract = build_router_replay_trace_contract(
        collection_path="user_defined_rollout_loop",
        trace_source="routed_experts_segments",
        routed_experts_digest=bare_upper_digest,
        router_replay_layout="true_tokens",
    )
    assert contract.routed_experts_digest == digest
    assert normalize_routed_experts_digest(bare_upper_digest) == digest

    extras = contract.to_packed_batch_extras()
    assert extras == {
        "router_replay_layout": "true_tokens",
        "router_replay_trace_collection_path": "user_defined_rollout_loop",
        "router_replay_trace_source": "routed_experts_segments",
        "router_replay_trace_preserved_through_scheduler": True,
        "router_replay_postprocess_wrote_trace_to_batch": True,
        "routed_experts_digest": digest,
        ROUTER_REPLAY_TRACE_CONFLICTS_KEY: [],
    }

    payload = contract.to_contract_dict()
    assert payload["status"] == "pass_router_replay_trace_contract"
    assert payload["paper_exact"] is False
    assert payload["claims_paper_results"] is False
    assert payload["claims_real_smoke"] is False
    assert payload["starts_sglang"] is False
    assert payload["launches_training"] is False
    assert all(payload["invariants"].values())

    public_contract = public_contract_builder(
        collection_path="router_generate_request",
        trace_source="routed_experts",
        routed_experts_digest=digest,
        router_replay_layout="full_padded",
    )
    assert public_contract.to_packed_batch_extras()["router_replay_layout"] == "full_padded"


def test_router_replay_trace_contract_rejects_unsafe_or_inconsistent_metadata():
    pytest.importorskip("torch")

    from megatron.lite.primitive.modules.router_replay import (
        build_router_replay_trace_contract,
        normalize_router_replay_layout,
        normalize_router_replay_trace_collection_path,
        normalize_router_replay_trace_source,
    )

    digest = "sha256:" + "a" * 64

    with pytest.raises(ValueError, match="routed_experts_digest must be a sha256 digest"):
        build_router_replay_trace_contract(
            collection_path="router_generate_request",
            trace_source="routed_experts",
            routed_experts_digest="not-a-digest",
        )
    with pytest.raises(ValueError, match="must be one of"):
        normalize_router_replay_trace_collection_path("plain_generate")
    with pytest.raises(ValueError, match="must be one of"):
        normalize_router_replay_trace_source("attention_positions")
    with pytest.raises(ValueError, match="must be one of"):
        normalize_router_replay_layout("diagonal")
    with pytest.raises(ValueError, match="routed_experts_segments traces must use"):
        build_router_replay_trace_contract(
            collection_path="user_defined_rollout_loop",
            trace_source="routed_experts_segments",
            routed_experts_digest=digest,
            router_replay_layout="full_padded",
        )
    with pytest.raises(ValueError, match="empty trace metadata conflicts"):
        build_router_replay_trace_contract(
            collection_path="router_generate_request",
            trace_source="routed_experts",
            routed_experts_digest=digest,
            metadata_conflicts=["digest_mismatch"],
        )
    with pytest.raises(ValueError, match="trace preservation through scheduler"):
        build_router_replay_trace_contract(
            collection_path="router_generate_request",
            trace_source="routed_experts",
            routed_experts_digest=digest,
            trace_preserved_through_scheduler=False,
        )
    with pytest.raises(ValueError, match="local no-launch contract"):
        build_router_replay_trace_contract(
            collection_path="router_generate_request",
            trace_source="routed_experts",
            routed_experts_digest=digest,
            starts_sglang=True,
        )
    with pytest.raises(ValueError, match="must not claim paper results"):
        build_router_replay_trace_contract(
            collection_path="router_generate_request",
            trace_source="routed_experts",
            routed_experts_digest=digest,
            claims_paper_results=True,
        )


def test_router_replay_records_and_replays_forward_and_backward_indices():
    torch = pytest.importorskip("torch")

    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction

    router = RouterReplay(layer_idx=7)
    scores = torch.tensor(
        [
            [0.1, 0.8, 0.4],
            [0.7, 0.2, 0.5],
            [0.0, 0.6, 0.9],
        ]
    )

    with router.action_context(RouterReplayAction.RECORD):
        record_values, record_indices = router.get_replay_topk(
            scores, 2, None, None, _default_topk
        )

    expected_values, expected_indices = scores.topk(2, dim=1)
    torch.testing.assert_close(record_values, expected_values)
    torch.testing.assert_close(record_indices, expected_indices)
    torch.testing.assert_close(router.get_recorded_indices(), expected_indices)
    assert router.router_replay_action is None

    replay_indices = torch.tensor([[2, 0], [1, 2], [0, 1]])
    router.set_target_indices(replay_indices)
    router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    replay_values, replayed = router.get_replay_topk(scores, 2, None, None, _default_topk)

    torch.testing.assert_close(replayed, replay_indices)
    torch.testing.assert_close(replay_values, scores.gather(1, replay_indices))
    assert router.replay_backward_list == []

    router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    with pytest.raises(RuntimeError, match="no saved forward indices"):
        router.get_replay_topk(scores, 2, None, None, _default_topk)

    router.set_target_indices(replay_indices, save_for_backward=True)

    backward_values, backward_indices = router.get_replay_topk(
        scores, 2, None, None, _default_topk
    )

    torch.testing.assert_close(backward_indices, replay_indices)
    torch.testing.assert_close(backward_values, scores.gather(1, replay_indices))
    assert router.replay_backward_list == []


def test_router_replay_accepts_string_actions_and_rejects_bad_actions():
    torch = pytest.importorskip("torch")

    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction

    router = RouterReplay(layer_idx=9)
    scores = torch.tensor([[0.1, 0.8, 0.4], [0.7, 0.2, 0.5]])
    replay_indices = torch.tensor([[2, 0], [1, 2]])

    with router.action_context("record"):
        _, recorded = router.get_replay_topk(scores, 2, None, None, _default_topk)
    torch.testing.assert_close(router.get_recorded_indices(), recorded)
    assert router.router_replay_action is None

    router.set_target_indices(replay_indices)
    router.set_router_replay_action("replay_forward")
    values, indices = router.get_replay_topk(scores, 2, None, None, _default_topk)
    assert router.router_replay_action == RouterReplayAction.REPLAY_FORWARD
    torch.testing.assert_close(indices, replay_indices)
    torch.testing.assert_close(values, scores.gather(1, replay_indices))

    with pytest.raises(ValueError, match="Unsupported RouterReplay action"):
        router.set_router_replay_action("replay_sideways")

    with pytest.raises(TypeError, match="RouterReplay action must be"):
        router.set_router_replay_action(3)

    with pytest.raises(ValueError, match="Unsupported RouterReplay action"):
        with router.action_context("replay_sideways"):
            pass


def test_router_replay_record_rejects_bad_recorded_indices():
    torch = pytest.importorskip("torch")

    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction

    router = RouterReplay(layer_idx=11)
    scores = torch.tensor([[0.1, 0.8, 0.4], [0.7, 0.2, 0.5]])

    with pytest.raises(ValueError, match=r"recorded indices must have shape \[tokens, topk\]"):
        router.record_indices(torch.tensor([0, 1]))

    with pytest.raises(TypeError, match="recorded indices must use an integer tensor dtype"):
        router.record_indices(torch.tensor([[0.0, 1.0], [1.0, 2.0]]))

    def bad_record_topk(scores, topk, num_groups, group_topk):
        del topk, num_groups, group_topk
        return scores[:, :2], torch.tensor([[0.0, 1.0], [1.0, 2.0]])

    with router.action_context(RouterReplayAction.RECORD):
        with pytest.raises(TypeError, match="recorded indices must use an integer tensor dtype"):
            router.get_replay_topk(scores, 2, None, None, bad_record_topk)

    def out_of_range_record_topk(scores, topk, num_groups, group_topk):
        del topk, num_groups, group_topk
        return scores[:, :2], torch.tensor([[0, 4], [1, 2]])

    with router.action_context(RouterReplayAction.RECORD):
        with pytest.raises(ValueError, match="recorded indices are out of range"):
            router.get_replay_topk(scores, 2, None, None, out_of_range_record_topk)


def test_router_replay_snapshots_record_and_replay_tensors():
    torch = pytest.importorskip("torch")

    from megatron.lite.primitive.modules.router_replay import RouterReplay

    router = RouterReplay(layer_idx=12)

    replay_indices = torch.tensor([[2, 0], [1, 2]])
    router.set_target_indices(replay_indices, save_for_backward=True)
    replay_indices.fill_(0)
    expected_replay = torch.tensor([[2, 0], [1, 2]])
    torch.testing.assert_close(router.target_topk_idx, expected_replay)
    torch.testing.assert_close(router.replay_backward_list[0], expected_replay)

    router.target_topk_idx.fill_(1)
    torch.testing.assert_close(router.replay_backward_list[0], expected_replay)

    recorded_indices = torch.tensor([[0, 1], [2, 1]])
    router.record_indices(recorded_indices)
    recorded_indices.fill_(2)
    first_read = router.get_recorded_indices()
    expected_recorded = torch.tensor([[0, 1], [2, 1]])
    torch.testing.assert_close(first_read, expected_recorded)
    first_read.fill_(0)
    torch.testing.assert_close(router.get_recorded_indices(), expected_recorded)


def test_router_replay_global_helpers_validate_and_snapshot_data():
    torch = pytest.importorskip("torch")

    from megatron.lite.primitive.modules.router_replay import RouterReplay

    RouterReplay.clear_global_router_replay_instances()
    try:
        first_router = RouterReplay(layer_idx=0)
        second_router = RouterReplay(layer_idx=1)

        first = torch.tensor([[0, 1], [1, 0]])
        second = torch.tensor([[2, 0], [1, 2]])
        RouterReplay.set_replay_data((first, second))
        first.fill_(2)
        torch.testing.assert_close(first_router.target_topk_idx, torch.tensor([[0, 1], [1, 0]]))
        torch.testing.assert_close(second_router.target_topk_idx, second)

        with pytest.raises(TypeError, match="expects a list or tuple"):
            RouterReplay.set_replay_data(torch.stack([first, second], dim=0))

        with pytest.raises(ValueError, match="does not match router replay instances"):
            RouterReplay.set_replay_data([second])

        first_router.record_indices(torch.tensor([[0, 1], [1, 0]]))
        second_router.record_indices(torch.tensor([[2, 0], [1, 2]]))
        recorded = RouterReplay.get_recorded_data()
        torch.testing.assert_close(recorded[0], torch.tensor([[0, 1], [1, 0]]))
        torch.testing.assert_close(recorded[1], torch.tensor([[2, 0], [1, 2]]))
        recorded[0].fill_(2)
        torch.testing.assert_close(
            RouterReplay.get_recorded_data()[0], torch.tensor([[0, 1], [1, 0]])
        )

        RouterReplay.clear_global_indices()
        assert first_router.target_topk_idx is None
        assert second_router.target_topk_idx is None
        assert first_router.get_recorded_indices() is None
        assert second_router.get_recorded_indices() is None
    finally:
        RouterReplay.clear_global_router_replay_instances()


def test_router_replay_rejects_missing_bad_shape_and_out_of_range_replay_data():
    torch = pytest.importorskip("torch")

    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction

    router = RouterReplay()
    scores = torch.randn(3, 4)

    router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    with pytest.raises(RuntimeError, match="requires target_topk_idx"):
        router.get_replay_topk(scores, 2, None, None, _default_topk)

    with pytest.raises(ValueError, match=r"shape \[tokens, topk\]"):
        router.set_target_indices(torch.tensor([1, 2]))

    with pytest.raises(TypeError, match="integer tensor dtype"):
        router.set_target_indices(torch.tensor([[0.0, 1.0], [1.0, 2.0]]))

    with pytest.raises(TypeError, match="integer tensor dtype"):
        router.set_target_indices(torch.tensor([[True, False], [False, True]]))

    router.set_target_indices(torch.zeros(2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="shape mismatch"):
        router.get_replay_topk(scores, 2, None, None, _default_topk)

    router.set_target_indices(torch.tensor([[0, 4], [1, 2], [2, 3]]))
    with pytest.raises(ValueError, match="out of range"):
        router.get_replay_topk(scores, 2, None, None, _default_topk)

    router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    router.set_target_indices(torch.tensor([[0, 4], [1, 2], [2, 3]]), save_for_backward=True)
    with pytest.raises(ValueError, match="out of range"):
        router.get_replay_topk(scores, 2, None, None, _default_topk)
    assert len(router.replay_backward_list) == 1


def test_router_replay_context_sets_data_actions_and_cleans_up():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers

        def router_replay_instances(self):
            return self._routers

    router = RouterReplay()
    model = FakeModel([router])
    replay_indices = torch.tensor([[1, 0], [0, 1]])
    batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        routed_experts=replay_indices,
    )

    with router_replay_context(model, batch):
        assert router.router_replay_action == RouterReplayAction.REPLAY_FORWARD
        torch.testing.assert_close(router.target_topk_idx, replay_indices)
        assert router.replay_backward_list == []
    assert router.router_replay_action is None
    torch.testing.assert_close(router.target_topk_idx, replay_indices)

    router.enable_replay_backward_queue()
    with router_replay_context(model, batch):
        assert router.router_replay_action == RouterReplayAction.REPLAY_FORWARD
        assert len(router.replay_backward_list) == 1
    assert router.router_replay_action is None
    torch.testing.assert_close(router.target_topk_idx, replay_indices)

    record_batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        extras={"record_routed_experts": True},
    )
    router.record_indices(replay_indices)
    with router_replay_context(model, record_batch):
        assert router.router_replay_action == RouterReplayAction.RECORD
    assert router.router_replay_action is None
    assert router.target_topk_idx is None
    assert router.recorded_topk_idx is None
    assert router.replay_backward_list == []


def test_router_replay_context_rejects_bad_actions_and_conflicting_data():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers

        def router_replay_instances(self):
            return self._routers

    replay_indices = torch.tensor([[1, 0], [0, 1]])
    model = FakeModel([RouterReplay()])

    bad_string_action_batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        extras={"router_replay_action": "replay_sideways"},
    )
    with pytest.raises(ValueError, match="Unsupported router_replay_action"):
        with router_replay_context(model, bad_string_action_batch):
            pass

    bad_type_action_batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        extras={"router_replay_action": 3},
    )
    with pytest.raises(TypeError, match="RouterReplayAction or string"):
        with router_replay_context(model, bad_type_action_batch):
            pass

    no_router_model = FakeModel([])
    no_router_batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        routed_experts=replay_indices,
    )
    with pytest.raises(ValueError, match="no router replay instances"):
        with router_replay_context(no_router_model, no_router_batch):
            pass

    missing_replay_batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        extras={"router_replay_action": "replay_forward"},
    )
    with pytest.raises(ValueError, match="forward replay requires batch.routed_experts"):
        with router_replay_context(model, missing_replay_batch):
            pass

    record_with_replay_data_batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        routed_experts=replay_indices,
        extras={"router_replay_action": "record"},
    )
    with pytest.raises(ValueError, match="RECORD does not accept batch.routed_experts"):
        with router_replay_context(model, record_with_replay_data_batch):
            pass

    float_replay_batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        routed_experts=replay_indices.to(torch.float32),
    )
    with pytest.raises(TypeError, match="integer tensor dtype"):
        with router_replay_context(model, float_replay_batch):
            pass

    negative_replay_batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        routed_experts=torch.tensor([[1, -1], [0, 1]]),
    )
    with pytest.raises(ValueError, match="non-negative expert ids"):
        with router_replay_context(model, negative_replay_batch):
            pass


def test_router_replay_context_handles_multi_router_list_and_tensor_layouts():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers

        def router_replay_instances(self):
            return self._routers

    routers = [RouterReplay(layer_idx=0), RouterReplay(layer_idx=1)]
    model = FakeModel(routers)
    first = torch.tensor([[0, 1], [1, 0], [2, 1]])
    second = torch.tensor([[2, 0], [1, 2], [0, 1]])

    list_batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([2, 3, 4]),
        seq_lens=torch.tensor([3]),
        routed_experts=[first, second],
    )
    with router_replay_context(model, list_batch):
        assert all(router.router_replay_action == RouterReplayAction.REPLAY_FORWARD for router in routers)
        torch.testing.assert_close(routers[0].target_topk_idx, first)
        torch.testing.assert_close(routers[1].target_topk_idx, second)

    stacked_routers_first = torch.stack([first, second], dim=0)
    tensor_batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([2, 3, 4]),
        seq_lens=torch.tensor([3]),
        routed_experts=stacked_routers_first,
    )
    with router_replay_context(model, tensor_batch):
        torch.testing.assert_close(routers[0].target_topk_idx, first)
        torch.testing.assert_close(routers[1].target_topk_idx, second)

    stacked_tokens_first = torch.stack([first, second], dim=1)
    tensor_batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([2, 3, 4]),
        seq_lens=torch.tensor([3]),
        routed_experts=stacked_tokens_first,
    )
    with router_replay_context(model, tensor_batch):
        torch.testing.assert_close(routers[0].target_topk_idx, first)
        torch.testing.assert_close(routers[1].target_topk_idx, second)

    ambiguous_first = first[:2]
    ambiguous_second = second[:2]
    ambiguous_batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        routed_experts=torch.stack([ambiguous_first, ambiguous_second], dim=0),
    )
    with pytest.raises(ValueError, match="Router Replay tensor data is ambiguous"):
        with router_replay_context(model, ambiguous_batch):
            pass

    bad_batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([2, 3, 4]),
        seq_lens=torch.tensor([3]),
        routed_experts=[first],
    )
    with pytest.raises(ValueError, match="expected 2 tensors"):
        with router_replay_context(model, bad_batch):
            pass

    bad_entry_batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([2, 3, 4]),
        seq_lens=torch.tensor([3]),
        routed_experts=[first, "not-a-tensor"],
    )
    with pytest.raises(TypeError, match="replay data entries must be tensors"):
        with router_replay_context(model, bad_entry_batch):
            pass

    negative_entry_batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([2, 3, 4]),
        seq_lens=torch.tensor([3]),
        routed_experts=[first, torch.tensor([[2, 0], [-1, 2], [0, 1]])],
    )
    with pytest.raises(ValueError, match="non-negative expert ids"):
        with router_replay_context(model, negative_entry_batch):
            pass


def test_router_replay_context_rejects_bad_per_router_entry_shape_before_pack():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers

        def router_replay_instances(self):
            return self._routers

    model = FakeModel([RouterReplay(layer_idx=0), RouterReplay(layer_idx=1)])
    first = torch.tensor([[0, 1], [1, 0], [2, 1]])
    bad_entry_shape_batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([2, 3, 4]),
        seq_lens=torch.tensor([3]),
        routed_experts=[first, torch.tensor([2, 0, 1])],
    )

    with pytest.raises(ValueError, match=r"replay data entries must have shape \[tokens, topk\]"):
        with router_replay_context(model, bad_entry_shape_batch):
            pass


def test_router_replay_rejects_mismatched_per_router_token_counts_before_pack():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import (
        concat_routed_experts_segments,
        router_replay_context,
    )
    from megatron.lite.primitive.modules.router_replay import RouterReplay
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers

        def router_replay_instances(self):
            return self._routers

    first = torch.tensor([[0, 1], [1, 0], [2, 1]])
    second_short = torch.tensor([[2, 0], [1, 2]])

    with pytest.raises(ValueError, match="same token count for every router"):
        concat_routed_experts_segments(
            [[first, second_short]],
            num_routers=2,
        )

    model = FakeModel([RouterReplay(layer_idx=0), RouterReplay(layer_idx=1)])
    batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([2, 3, 4]),
        seq_lens=torch.tensor([3]),
        routed_experts=[first, second_short],
    )

    with pytest.raises(ValueError, match="same token count for every router"):
        with router_replay_context(model, batch):
            pass


def test_router_replay_context_accepts_padded_batch_routed_experts():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers

        def router_replay_instances(self):
            return self._routers

    routers = [RouterReplay(layer_idx=0), RouterReplay(layer_idx=1)]
    model = FakeModel(routers)
    padded = torch.tensor(
        [
            [
                [[1, 0], [4, 3]],
                [[2, 1], [5, 4]],
                [[99, 99], [99, 99]],
            ],
            [
                [[8, 7], [6, 5]],
                [[88, 88], [88, 88]],
                [[77, 77], [77, 77]],
            ],
        ],
        dtype=torch.long,
    )
    batch = PackedBatch(
        input_ids=torch.arange(3),
        labels=torch.arange(10, 13),
        seq_lens=torch.tensor([2, 1]),
        routed_experts=padded,
    )

    with router_replay_context(model, batch):
        assert all(router.router_replay_action == RouterReplayAction.REPLAY_FORWARD for router in routers)
        assert routers[0].target_topk_idx.dtype == torch.uint8
        assert routers[1].target_topk_idx.dtype == torch.uint8
        torch.testing.assert_close(
            routers[0].target_topk_idx.to(torch.long),
            torch.tensor([[1, 0], [2, 1], [8, 7]]),
        )
        torch.testing.assert_close(
            routers[1].target_topk_idx.to(torch.long),
            torch.tensor([[4, 3], [5, 4], [6, 5]]),
        )


def test_router_replay_context_explicit_backward_replay_consumes_saved_indices():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers

        def router_replay_instances(self):
            return self._routers

    routers = [RouterReplay(layer_idx=0), RouterReplay(layer_idx=1)]
    model = FakeModel(routers)
    first = torch.tensor([[1, 0], [2, 1]])
    second = torch.tensor([[0, 2], [1, 0]])
    scores = torch.tensor(
        [
            [0.1, 0.8, 0.4],
            [0.7, 0.2, 0.5],
        ]
    )
    batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        routed_experts=[first, second],
        extras={"router_replay_action": "replay_backward"},
    )

    with router_replay_context(model, batch):
        assert all(router.router_replay_action == RouterReplayAction.REPLAY_BACKWARD for router in routers)
        assert [len(router.replay_backward_list) for router in routers] == [1, 1]
        first_values, first_indices = routers[0].get_replay_topk(
            scores, 2, None, None, _default_topk
        )
        second_values, second_indices = routers[1].get_replay_topk(
            scores, 2, None, None, _default_topk
        )
        torch.testing.assert_close(first_indices, first)
        torch.testing.assert_close(first_values, scores.gather(1, first))
        torch.testing.assert_close(second_indices, second)
        torch.testing.assert_close(second_values, scores.gather(1, second))
        assert [router.replay_backward_list for router in routers] == [[], []]

    assert all(router.router_replay_action is None for router in routers)


def test_router_replay_context_pads_true_token_replay_data_to_thd_layout():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers
            self.ps = ParallelState(tp_size=4)

        def router_replay_instances(self):
            return self._routers

    router = RouterReplay(layer_idx=0)
    model = FakeModel([router])
    replay_indices = torch.tensor(
        [
            [1, 0],
            [2, 1],
            [0, 2],
            [1, 2],
            [2, 0],
        ]
    )
    batch = PackedBatch(
        input_ids=torch.arange(5),
        labels=torch.arange(10, 15),
        seq_lens=torch.tensor([3, 2]),
        routed_experts=replay_indices,
    )

    with router_replay_context(model, batch):
        assert router.router_replay_action == RouterReplayAction.REPLAY_FORWARD
        expected = torch.tensor(
            [
                [1, 0],
                [2, 1],
                [0, 2],
                [0, 0],
                [1, 2],
                [2, 0],
                [0, 0],
                [0, 0],
            ]
        )
        torch.testing.assert_close(router.target_topk_idx, expected)


def test_router_replay_context_splits_padded_replay_data_to_cp_local_layout():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers
            self.ps = ParallelState(cp_size=2, cp_rank=1)

        def router_replay_instances(self):
            return self._routers

    router = RouterReplay(layer_idx=0)
    model = FakeModel([router])
    replay_indices = torch.tensor(
        [
            [1, 0],
            [2, 1],
            [0, 2],
            [1, 1],
            [2, 0],
            [0, 1],
            [1, 2],
            [2, 2],
        ]
    )
    batch = PackedBatch(
        input_ids=torch.arange(8),
        labels=torch.arange(10, 18),
        seq_lens=torch.tensor([3, 5]),
        routed_experts=replay_indices,
    )

    with router_replay_context(model, batch):
        expected = torch.tensor(
            [
                [2, 1],
                [0, 2],
                [0, 1],
                [1, 2],
                [2, 2],
                [0, 0],
            ]
        )
        torch.testing.assert_close(router.target_topk_idx, expected)


def test_router_replay_context_respects_explicit_thd_replay_layout():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers
            self.ps = ParallelState(cp_size=2, cp_rank=1)

        def router_replay_instances(self):
            return self._routers

    router = RouterReplay(layer_idx=0)
    model = FakeModel([router])

    # seq_lens=[2] has true_tokens=2, full_padded=4, cp_local=2. The default
    # PackedBatch contract is true-token replay data, so this is padded then
    # split to CP rank 1.
    true_token_replay = torch.tensor([[1, 0], [2, 1]])
    true_token_batch = PackedBatch(
        input_ids=torch.arange(2),
        labels=torch.arange(10, 12),
        seq_lens=torch.tensor([2]),
        routed_experts=true_token_replay,
    )
    with router_replay_context(model, true_token_batch):
        expected = torch.tensor([[2, 1], [0, 0]])
        torch.testing.assert_close(router.target_topk_idx, expected)

    cp_local_replay = torch.tensor([[7, 6], [5, 4]])
    cp_local_batch = PackedBatch(
        input_ids=torch.arange(2),
        labels=torch.arange(10, 12),
        seq_lens=torch.tensor([2]),
        routed_experts=cp_local_replay,
        extras={"router_replay_layout": "cp_local"},
    )
    with router_replay_context(model, cp_local_batch):
        torch.testing.assert_close(router.target_topk_idx, cp_local_replay)

    full_padded_replay = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    full_padded_batch = PackedBatch(
        input_ids=torch.arange(2),
        labels=torch.arange(10, 12),
        seq_lens=torch.tensor([2]),
        routed_experts=full_padded_replay,
        extras={"router_replay_layout": "full_padded"},
    )
    with router_replay_context(model, full_padded_batch):
        expected = torch.tensor([[1, 2], [2, 3]])
        torch.testing.assert_close(router.target_topk_idx, expected)

    bad_full_padded_batch = PackedBatch(
        input_ids=torch.arange(2),
        labels=torch.arange(10, 12),
        seq_lens=torch.tensor([2]),
        routed_experts=true_token_replay,
        extras={"router_replay_layout": "full_padded"},
    )
    with pytest.raises(ValueError, match="router_replay_layout='full_padded'"):
        with router_replay_context(model, bad_full_padded_batch):
            pass

    bad_layout_batch = PackedBatch(
        input_ids=torch.arange(2),
        labels=torch.arange(10, 12),
        seq_lens=torch.tensor([2]),
        routed_experts=true_token_replay,
        extras={"router_replay_layout": "diagonal"},
    )
    with pytest.raises(ValueError, match="Unsupported router_replay_layout"):
        with router_replay_context(model, bad_layout_batch):
            pass

    bad_layout_type_batch = PackedBatch(
        input_ids=torch.arange(2),
        labels=torch.arange(10, 12),
        seq_lens=torch.tensor([2]),
        routed_experts=true_token_replay,
        extras={"router_replay_layout": 3},
    )
    with pytest.raises(TypeError, match="router_replay_layout must be a string"):
        with router_replay_context(model, bad_layout_type_batch):
            pass


def test_router_replay_context_uses_unwrapped_model_parallel_state_for_thd_layout():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.contracts.data import PackedBatch

    class InnerModel:
        def __init__(self, routers):
            self._routers = routers
            self.ps = ParallelState(cp_size=2, cp_rank=1)

        def router_replay_instances(self):
            return self._routers

    class FakeFSDPWrapper:
        def __init__(self, module):
            self._fsdp_wrapped_module = module

    router = RouterReplay(layer_idx=0)
    model = FakeFSDPWrapper(InnerModel([router]))
    replay_indices = torch.tensor([[1, 0], [2, 1]])
    batch = PackedBatch(
        input_ids=torch.arange(2),
        labels=torch.arange(10, 12),
        seq_lens=torch.tensor([2]),
        routed_experts=replay_indices,
    )

    with router_replay_context(model, batch):
        expected = torch.tensor([[2, 1], [0, 0]])
        torch.testing.assert_close(router.target_topk_idx, expected)


def test_router_replay_checkpoint_recompute_switches_to_backward_replay():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.protocol_utils import router_replay_context
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
    from megatron.lite.primitive.recompute import wrap_checkpoint
    from megatron.lite.runtime.contracts.data import PackedBatch

    class FakeModel:
        def __init__(self, routers):
            self._routers = routers

        def router_replay_instances(self):
            return self._routers

    class FakeCheckpointedRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router_replay = RouterReplay(layer_idx=0)
            self.actions = []
            self.indices_seen = []

        def forward(self, scores):
            self.actions.append(self.router_replay.router_replay_action)
            values, indices = self.router_replay.get_replay_topk(
                scores, 2, None, None, _default_topk
            )
            self.indices_seen.append(indices.detach().clone())
            return values.sum(dim=1)

    module = FakeCheckpointedRouter()
    model = FakeModel([module.router_replay])
    wrap_checkpoint(module, preserve_rng_state=False)
    assert module.router_replay.replay_backward_enabled is True

    replay_indices = torch.tensor([[2, 0], [1, 2]])
    scores = torch.tensor(
        [
            [0.1, 0.8, 0.4],
            [0.7, 0.2, 0.5],
        ],
        requires_grad=True,
    )
    batch = PackedBatch(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, 3]),
        seq_lens=torch.tensor([2]),
        routed_experts=replay_indices,
    )

    with router_replay_context(model, batch):
        out = module(scores)
        assert module.actions == [RouterReplayAction.REPLAY_FORWARD]
        assert len(module.router_replay.replay_backward_list) == 1

    assert module.router_replay.router_replay_action is None
    out.sum().backward()

    assert module.actions == [
        RouterReplayAction.REPLAY_FORWARD,
        RouterReplayAction.REPLAY_BACKWARD,
    ]
    torch.testing.assert_close(module.indices_seen[0], replay_indices)
    torch.testing.assert_close(module.indices_seen[1], replay_indices)
    assert module.router_replay.replay_backward_list == []
    assert module.router_replay.router_replay_action is None
    assert scores.grad is not None
