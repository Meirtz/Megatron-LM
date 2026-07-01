# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static R3 bridge checks for the VERL MLite engine."""

from __future__ import annotations

import ast
import contextlib
import copy
import functools
import json
import sys
import types
from collections.abc import Mapping
from pathlib import Path


ENGINE = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "verl"
    / "verl_mlite"
    / "engine"
    / "mlite_engine.py"
)
WORKER = Path(__file__).resolve().parents[6] / "verl" / "verl" / "workers" / "engine_workers.py"
AGENT_LOOP = (
    Path(__file__).resolve().parents[6]
    / "verl"
    / "verl"
    / "experimental"
    / "agent_loop"
    / "agent_loop.py"
)
FULLY_ASYNC_ROLLOUTER = (
    Path(__file__).resolve().parents[6]
    / "verl"
    / "verl"
    / "experimental"
    / "fully_async_policy"
    / "fully_async_rollouter.py"
)
PPO_TRAINER = (
    Path(__file__).resolve().parents[6]
    / "verl"
    / "verl"
    / "trainer"
    / "ppo"
    / "ray_trainer.py"
)
SEPARATION_TRAINER = (
    Path(__file__).resolve().parents[6]
    / "verl"
    / "verl"
    / "experimental"
    / "separation"
    / "ray_trainer.py"
)
ROLLOUT_CORR_HELPER = (
    Path(__file__).resolve().parents[6]
    / "verl"
    / "verl"
    / "trainer"
    / "ppo"
    / "rollout_corr_helper.py"
)
PADDING_UTILS = (
    Path(__file__).resolve().parents[6]
    / "verl"
    / "verl"
    / "workers"
    / "utils"
    / "padding.py"
)
SGLANG_ROLLOUT_SERVER = (
    Path(__file__).resolve().parents[6]
    / "verl"
    / "verl"
    / "workers"
    / "rollout"
    / "sglang_rollout"
    / "async_sglang_server.py"
)
VLLM_ROLLOUT_SERVER = (
    Path(__file__).resolve().parents[6]
    / "verl"
    / "verl"
    / "workers"
    / "rollout"
    / "vllm_rollout"
    / "vllm_async_server.py"
)
REAL_SMOKE_EVENTS_VALIDATOR = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "validate_roll_r3_real_smoke_events.py"
)
REAL_SMOKE_EXTRACTOR = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "extract_roll_r3_real_smoke_evidence.py"
)
REAL_SMOKE_BUNDLE_PREPARER = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "prepare_roll_r3_real_smoke_bundle.py"
)
REAL_SMOKE_BUNDLE_VALIDATOR = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "validate_roll_r3_real_smoke_bundle.py"
)
REAL_SMOKE_RESULT_TEMPLATE = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "roll-r3-real-smoke-result.template.json"
)
REAL_SMOKE_EVENTS_TEMPLATE = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "roll-r3-real-smoke-events.template.jsonl"
)
R3_LAUNCH_READINESS = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "check_roll_r3_launch_readiness.py"
)
UNBLOCK_RUNBOOK_PREPARER = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "prepare_adaptation_unblock_runbook.py"
)
EXTERNAL_HANDOFF_PREPARER = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "prepare_adaptation_external_evidence_handoff.py"
)
DELIVERABLE_AUDIT = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "audit_adaptation_deliverables.py"
)
STATUS_CONSISTENCY_AUDIT = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "audit_adaptation_status_consistency.py"
)
REAL_SMOKE_ANALYZER = (
    Path(__file__).resolve().parents[6]
    / "runs"
    / "20260620-mlite-olora-tail-rl-repro"
    / "analyze_roll_r3_real_smoke.py"
)


def _strip_annotations(node: ast.AST) -> ast.AST:
    for child in ast.walk(node):
        if isinstance(child, ast.arg):
            child.annotation = None
        elif isinstance(child, ast.FunctionDef):
            child.returns = None
    return node


def test_mlite_engine_r3_replay_requires_routed_experts_unless_recording():
    text = ENGINE.read_text(encoding="utf-8")

    assert 'key="enable_routing_replay", default=None' in text
    assert 'key="router_replay_action", default=None' in text
    assert '"record_routed_experts"' in text
    assert '"router_replay_record"' in text
    assert 'for key in ("record_routed_experts", "router_replay_record"):' in text
    assert 'router_replay_action == "record"' in text
    assert "enable_routing_replay is True and not record_routed_experts" in text
    assert "R3/replay train path requires routed_experts" in text
    assert "routed_experts_segments" in text
    assert "router_replay_trace_collection_path" in text
    assert "router_replay_trace_preserved_through_scheduler" in text
    assert "router_replay_postprocess_wrote_trace_to_batch" in text
    assert "routed_experts_digest" in text
    assert "_normalize_router_replay_digest" in text
    assert "_router_replay_digest_for_extras" in text
    assert "routed_experts_digest does not match" in text
    assert 'layout == "true_tokens" and true_tokens == int(routed_experts.size(0))' in text
    assert "_ROUTER_REPLAY_LAYOUT_ALIASES" in text
    assert "_normalize_router_replay_layout" in text
    assert "requires router_replay_layout" in text
    assert "true_tokens, full_padded, or cp_local" in text
    assert "_ROUTER_REPLAY_TRACE_COLLECTION_PATHS" in text
    assert "_ROUTER_REPLAY_TRACE_SOURCES" in text
    assert "_required_router_replay_trace_metadata" in text
    assert "router_replay_trace_collection_path when replay payload is provided" in text
    assert "router_replay_trace_source={expected_source!r}" in text
    assert "{key}=True when replay payload is provided" in text
    assert "Use record_routed_experts=True only for record-mode passes." in text


def test_mlite_engine_r3_config_guard_rejects_unsupported_layouts():
    text = ENGINE.read_text(encoding="utf-8")

    assert "def _router_replay_mode" in text
    assert "router_replay_mode == \"R2\"" in text
    assert "does not support router_replay.mode=R2 yet" in text
    assert "router_replay_mode == \"R3\"" in text
    assert "def _sequence_packing_enabled" in text
    assert "def _config_has_key" in text
    assert "def _config_like" in text
    assert "_CONFIG_MISSING = object()" in text
    assert "MegatronLiteEngine R3 does not support sequence packing" in text
    assert "Disable sequence packing until routed_experts remapping is implemented." in text
    assert "engine.router_replay.mode=R3 requires" in text
    assert "engine.impl_cfg.router_replay=True" in text
    assert "impl_cfg[\"router_replay\"] = True" in text
    assert "unknown_string_is_true=True" in text


def test_mlite_engine_r3_replay_flags_use_config_boolean_parsing():
    text = ENGINE.read_text(encoding="utf-8")

    assert "enable_routing_replay = MegatronLiteEngine._bool_config_value(" in text
    assert 'key="enable_routing_replay"' in text
    assert "record_routed_experts = router_replay_action == \"record\"" in text
    assert "extras[key] = MegatronLiteEngine._bool_config_value(value, key=key)" in text
    assert "extras[key] = bool(value)" not in text
    assert "if enable_routing_replay is False:" in text


def test_mlite_engine_r3_trace_metadata_reaches_packed_batch_extras():
    text = ENGINE.read_text(encoding="utf-8")

    assert "_ROUTER_REPLAY_TRACE_STRING_EXTRA_KEYS" in text
    assert "_ROUTER_REPLAY_TRACE_BOOL_EXTRA_KEYS" in text
    assert "_ROUTER_REPLAY_DIGEST_EXTRA_KEY" in text
    assert "_ROUTER_REPLAY_TRACE_CONFLICT_EXTRA_KEY" in text
    assert "router_replay_trace_metadata_conflicts" in text
    assert "conflicts_present" in text
    assert "to be empty before" in text
    assert "routed_experts_digest(" in text
    assert "*_ROUTER_REPLAY_TRACE_STRING_EXTRA_KEYS" in text
    assert "for key in _ROUTER_REPLAY_TRACE_BOOL_EXTRA_KEYS:" in text
    assert "extras=self._router_replay_extras(" in text
    assert "seq_lens=seq_lens" in text
    assert 'if segment_replay_payload_present:' in text
    assert 'extras["router_replay_layout"] = "true_tokens"' in text
    assert 'extras["router_replay_layout"] = "true_tokens"' in text


def test_mlite_engine_r3_trace_metadata_conflicts_fail_before_consuming_payload():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronLiteEngine"
    )
    helper = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_required_router_replay_trace_metadata"
    )
    helper = _strip_annotations(copy.deepcopy(helper))
    helper.decorator_list = []
    support_names = {
        "_ROUTER_REPLAY_TRACE_BOOL_EXTRA_KEYS",
        "_ROUTER_REPLAY_TRACE_CONFLICT_EXTRA_KEY",
        "_ROUTER_REPLAY_TRACE_COLLECTION_PATHS",
        "_ROUTER_REPLAY_TRACE_SOURCES",
    }
    support_defs = [
        _strip_annotations(copy.deepcopy(node))
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
    ]
    module = ast.Module(body=[*support_defs, helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Mapping": Mapping,
        "tu": types.SimpleNamespace(
            get_non_tensor_data=lambda data, key, default=None: data.get(key, default)
        ),
        "MegatronLiteEngine": types.SimpleNamespace(_bool_config_value=lambda value, key: bool(value)),
    }
    exec(compile(module, str(ENGINE), "exec"), namespace)
    require_metadata = namespace["_required_router_replay_trace_metadata"]

    valid = {
        "router_replay_trace_collection_path": "router_generate_request",
        "router_replay_trace_source": "routed_experts",
        "router_replay_trace_preserved_through_scheduler": True,
        "router_replay_postprocess_wrote_trace_to_batch": True,
    }
    for empty_conflicts in (None, "", {}, [], ()):
        data = dict(valid)
        if empty_conflicts is not None:
            data["router_replay_trace_metadata_conflicts"] = empty_conflicts
        assert require_metadata(data, expected_source="routed_experts") == valid

    for conflicts in (["sglang_version"], {"sglang_version": ["0.4.0", "0.4.1"]}):
        data = dict(valid)
        data["router_replay_trace_metadata_conflicts"] = conflicts
        try:
            require_metadata(data, expected_source="routed_experts")
        except ValueError as exc:
            assert "router_replay_trace_metadata_conflicts to be empty" in str(exc)
        else:
            raise AssertionError("conflicting router replay trace metadata must fail early")


def test_verl_worker_r3_symmetry_helpers_execute_without_torch_import():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    helper_names = {
        "_config_get",
        "_config_bool_enabled",
        "_engine_router_replay_mode",
        "_actor_router_replay_mode",
        "_rollout_router_replay_enabled",
        "_sequence_packing_enabled",
        "_validate_router_replay_symmetry",
    }
    helper_defs = [
        _strip_annotations(copy.deepcopy(node))
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(WORKER), "exec"), namespace)

    actor_router_replay_mode = namespace["_actor_router_replay_mode"]
    rollout_router_replay_enabled = namespace["_rollout_router_replay_enabled"]
    sequence_packing_enabled = namespace["_sequence_packing_enabled"]
    validate_router_replay_symmetry = namespace["_validate_router_replay_symmetry"]

    dict_actor = {
        "strategy": "mlite",
        "engine": {"router_replay": {"mode": "R3"}},
    }
    namespace_actor = types.SimpleNamespace(
        strategy="mlite",
        engine=types.SimpleNamespace(router_replay=types.SimpleNamespace(mode="R3")),
    )
    megatron_actor = types.SimpleNamespace(
        strategy="megatron",
        megatron=types.SimpleNamespace(router_replay=types.SimpleNamespace(mode="R2")),
    )
    disabled_actor = types.SimpleNamespace(strategy="mlite", engine=types.SimpleNamespace())

    assert actor_router_replay_mode(dict_actor) == "R3"
    assert actor_router_replay_mode(namespace_actor) == "R3"
    assert actor_router_replay_mode(megatron_actor) == "R2"
    assert actor_router_replay_mode(disabled_actor) == "disabled"
    assert rollout_router_replay_enabled({"enable_rollout_routing_replay": "yes"}) is True
    assert rollout_router_replay_enabled({"enable_rollout_routing_replay": "disabled"}) is False
    assert sequence_packing_enabled({"use_sequence_packing": "yes"}) is True
    assert sequence_packing_enabled(
        {
            "engine": {
                "impl_cfg": {
                    "sequence_packing": True,
                }
            }
        }
    ) is True
    assert sequence_packing_enabled(types.SimpleNamespace(sequence_packing="false")) is False

    assert validate_router_replay_symmetry(
        dict_actor, {"name": "sglang", "enable_rollout_routing_replay": "true"}
    ) == "R3"
    assert validate_router_replay_symmetry(
        disabled_actor, {"name": "vllm", "enable_rollout_routing_replay": False}
    ) == "disabled"

    try:
        validate_router_replay_symmetry(
            dict_actor, {"name": "vllm", "enable_rollout_routing_replay": True}
        )
    except ValueError as exc:
        assert "rollout.name='sglang'" in str(exc)
    else:
        raise AssertionError("R3 must reject non-SGLang rollout backends.")

    try:
        validate_router_replay_symmetry(
            dict_actor, {"name": "sglang", "enable_rollout_routing_replay": False}
        )
    except ValueError as exc:
        assert "enable_rollout_routing_replay=True" in str(exc)
    else:
        raise AssertionError("R3 must reject rollout replay disabled.")

    try:
        validate_router_replay_symmetry(
            disabled_actor, {"name": "sglang", "enable_rollout_routing_replay": True}
        )
    except ValueError as exc:
        assert "requires actor router_replay.mode=R3" in str(exc)
    else:
        raise AssertionError("Rollout replay must reject actor replay disabled.")


def test_verl_worker_r3_symmetry_rejects_sequence_packing_without_torch_import():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    helper_names = {
        "_config_get",
        "_config_bool_enabled",
        "_engine_router_replay_mode",
        "_actor_router_replay_mode",
        "_rollout_router_replay_enabled",
        "_sequence_packing_enabled",
        "_validate_router_replay_symmetry",
    }
    helper_defs = [
        _strip_annotations(copy.deepcopy(node))
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(WORKER), "exec"), namespace)

    validate_router_replay_symmetry = namespace["_validate_router_replay_symmetry"]
    actor = {
        "strategy": "mlite",
        "engine": {"router_replay": {"mode": "R3"}},
    }
    rollout = {"name": "sglang", "enable_rollout_routing_replay": True}

    actor_with_top_level_packing = {
        **actor,
        "use_sequence_packing": True,
    }
    try:
        validate_router_replay_symmetry(actor_with_top_level_packing, rollout)
    except ValueError as exc:
        assert "does not support actor sequence packing" in str(exc)
    else:
        raise AssertionError("R3 must reject actor sequence packing.")

    actor_with_nested_packing = {
        "strategy": "mlite",
        "engine": {
            "router_replay": {"mode": "R3"},
            "impl_cfg": {"sequence_packing": "yes"},
        },
    }
    try:
        validate_router_replay_symmetry(actor_with_nested_packing, rollout)
    except ValueError as exc:
        assert "does not support actor sequence packing" in str(exc)
    else:
        raise AssertionError("R3 must reject nested actor sequence packing.")

    rollout_with_packing = {
        "name": "sglang",
        "enable_rollout_routing_replay": True,
        "enable_sequence_packing": "on",
    }
    try:
        validate_router_replay_symmetry(actor, rollout_with_packing)
    except ValueError as exc:
        assert "does not support rollout sequence packing" in str(exc)
    else:
        raise AssertionError("R3 must reject rollout sequence packing.")

    rollout_with_disabled_packing = {
        "name": "sglang",
        "enable_rollout_routing_replay": True,
        "sequence_packing": "false",
    }
    assert validate_router_replay_symmetry(actor, rollout_with_disabled_packing) == "R3"


def test_verl_worker_rejects_r2_and_unknown_router_replay_modes_without_torch_import():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    helper_names = {
        "_config_get",
        "_config_bool_enabled",
        "_engine_router_replay_mode",
        "_actor_router_replay_mode",
        "_rollout_router_replay_enabled",
        "_sequence_packing_enabled",
        "_validate_router_replay_symmetry",
    }
    helper_defs = [
        _strip_annotations(copy.deepcopy(node))
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(WORKER), "exec"), namespace)

    validate_router_replay_symmetry = namespace["_validate_router_replay_symmetry"]
    rollout_disabled = {"name": "sglang", "enable_rollout_routing_replay": False}

    actor_r2 = {
        "strategy": "mlite",
        "engine": {"router_replay": {"mode": "R2"}},
    }
    try:
        validate_router_replay_symmetry(actor_r2, rollout_disabled)
    except NotImplementedError as exc:
        assert "router_replay.mode=R2 is not implemented" in str(exc)
        assert "router_replay.mode=R3 with SGLang routed_experts" in str(exc)
    else:
        raise AssertionError("R2 router replay must fail before worker construction.")

    actor_unknown = {
        "strategy": "mlite",
        "engine": {"router_replay": {"mode": "R4"}},
    }
    try:
        validate_router_replay_symmetry(actor_unknown, rollout_disabled)
    except ValueError as exc:
        assert "Unsupported actor router_replay.mode='R4'" in str(exc)
    else:
        raise AssertionError("Unknown router replay mode must fail before worker construction.")


def test_verl_worker_routing_replay_decorator_assigns_flag_without_torch_import():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    support_names = {
        "_ROUTING_REPLAY_SIDE_CHANNEL_KEYS",
        "_clear_routing_replay_side_channels",
        "_with_routing_replay_flag",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "functools": functools,
        "tu": types.SimpleNamespace(
            assign_non_tensor_data=lambda data, key, value: data.__setitem__(key, value),
            pop=lambda data, key, default=None: data.pop(key, default),
        ),
    }
    exec(compile(module, str(WORKER), "exec"), namespace)
    with_routing_replay_flag = namespace["_with_routing_replay_flag"]

    calls = []

    def endpoint(self, data, *args, **kwargs):
        calls.append((self.enable_routing_replay, dict(data), args, kwargs))
        return "ok"

    enabled_worker = types.SimpleNamespace(enable_routing_replay=True)
    disabled_worker = types.SimpleNamespace(enable_routing_replay=False)

    replay_data = {}
    assert with_routing_replay_flag(True)(endpoint)(
        enabled_worker, replay_data, "arg", key="value"
    ) == "ok"
    assert replay_data == {"enable_routing_replay": True}
    assert calls[-1] == (True, {"enable_routing_replay": True}, ("arg",), {"key": "value"})

    ref_data = {
        "input_ids": "keep",
        "routed_experts": "drop",
        "routed_experts_segments": "drop",
        "routed_experts_segment_seq_lens": "drop",
        "routed_experts_num_routers": "drop",
        "routed_experts_digest": "drop",
        "router_replay_action": "drop",
        "record_routed_experts": "drop",
        "router_replay_record": "drop",
        "router_replay_layout": "drop",
        "router_replay_trace_collection_path": "drop",
        "router_replay_trace_source": "drop",
        "router_replay_trace_preserved_through_scheduler": "drop",
        "router_replay_postprocess_wrote_trace_to_batch": "drop",
        "router_replay_trace_metadata_conflicts": "drop",
    }
    assert with_routing_replay_flag(False)(endpoint)(enabled_worker, ref_data) == "ok"
    assert ref_data == {"input_ids": "keep", "enable_routing_replay": False}
    assert calls[-1] == (True, {"input_ids": "keep", "enable_routing_replay": False}, (), {})

    untouched_data = {}
    assert with_routing_replay_flag(True)(endpoint)(disabled_worker, untouched_data) == "ok"
    assert untouched_data == {}
    assert calls[-1] == (False, {}, (), {})


def test_verl_worker_r3_disabled_roles_clear_trace_conflict_side_channel_static():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    support_names = {
        "_ROUTING_REPLAY_SIDE_CHANNEL_KEYS",
        "_clear_routing_replay_side_channels",
        "_with_routing_replay_flag",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "functools": functools,
        "tu": types.SimpleNamespace(
            assign_non_tensor_data=lambda data, key, value: data.__setitem__(key, value),
            pop=lambda data, key, default=None: data.pop(key, default),
        ),
    }
    exec(compile(module, str(WORKER), "exec"), namespace)

    side_channel_keys = namespace["_ROUTING_REPLAY_SIDE_CHANNEL_KEYS"]
    assert "router_replay_trace_metadata_conflicts" in side_channel_keys
    assert "routed_experts_digest" in side_channel_keys

    stale_trace = {key: "drop" for key in side_channel_keys}
    stale_trace["input_ids"] = "keep"
    namespace["_clear_routing_replay_side_channels"](stale_trace)
    assert stale_trace == {"input_ids": "keep"}

    calls = []

    def endpoint(self, data):
        calls.append(dict(data))
        return "ok"

    data = {key: "drop" for key in side_channel_keys}
    data["input_ids"] = "keep"
    worker = types.SimpleNamespace(enable_routing_replay=True)
    assert namespace["_with_routing_replay_flag"](False)(endpoint)(worker, data) == "ok"
    assert data == {"input_ids": "keep", "enable_routing_replay": False}
    assert calls == [{"input_ids": "keep", "enable_routing_replay": False}]


def test_verl_worker_r3_role_endpoints_bind_expected_replay_flags_static():
    text = WORKER.read_text(encoding="utf-8")

    assert "@_with_routing_replay_flag(enabled=False)" in text
    assert "@_with_routing_replay_flag(enabled=True)" in text

    tree = ast.parse(text)
    worker_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker"
    )

    expected = {
        "compute_ref_log_prob": False,
        "compute_log_prob": True,
        "update_actor": True,
    }
    observed: dict[str, bool] = {}
    for node in worker_class.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in expected:
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "_with_routing_replay_flag"
            ):
                enabled_keywords = [
                    keyword
                    for keyword in decorator.keywords
                    if keyword.arg == "enabled" and isinstance(keyword.value, ast.Constant)
                ]
                assert len(enabled_keywords) == 1
                observed[node.name] = bool(enabled_keywords[0].value.value)

    assert observed == expected


def test_verl_agent_loop_r3_postprocess_writes_trace_metadata_static():
    text = AGENT_LOOP.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS" in text
    assert '"router_replay_layout": "full_padded"' in text
    assert '"router_replay_trace_collection_path": "user_defined_rollout_loop"' in text
    assert '"router_replay_trace_source": "routed_experts"' in text
    assert '"router_replay_trace_preserved_through_scheduler": True' in text
    assert '"router_replay_postprocess_wrote_trace_to_batch": True' in text
    assert 'if "routed_experts" in optional_outputs:' in text
    assert "for key, value in ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS.items():" in text
    assert "non_tensor_batch[key] = np.full(len(inputs), value, dtype=object)" in text

    test_text = (
        Path(__file__).resolve().parents[6]
        / "verl"
        / "tests"
        / "experimental"
        / "agent_loop"
        / "test_agent_loop_extra_fields_schema_on_cpu.py"
    ).read_text(encoding="utf-8")
    assert "test_agent_loop_postprocess_writes_router_replay_trace_metadata_on_cpu" in test_text
    assert "for key in expected_metadata:" in test_text
    assert "assert key not in plain.non_tensor_batch" in test_text


def test_verl_fully_async_r3_partial_rollout_preserves_trace_extra_fields_static():
    text = FULLY_ASYNC_ROLLOUTER.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_TRACE_CONSISTENT_EXTRA_FIELDS" in text
    assert "ROUTER_REPLAY_TRACE_SUM_EXTRA_FIELDS" in text
    assert "ROUTER_REPLAY_TRACE_MAX_EXTRA_FIELDS" in text
    assert "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD" in text
    assert "_ROUTED_EXPERTS_DIGEST_VERSION" in text
    assert "def _routed_experts_digest(" in text
    assert '"responses_with_routed_experts"' in text
    assert '"valid_tokens"' in text
    assert '"max_expert_id"' in text
    assert "def _merge_router_replay_extra_fields(" in text
    assert "def _sync_router_replay_extra_fields_with_routed_experts(" in text
    assert "_merge_router_replay_extra_fields(final_output.extra_fields, output.extra_fields)" in text
    assert "_sync_router_replay_extra_fields_with_routed_experts(final_output)" in text
    assert 'output.extra_fields["routed_experts_shape"] = shape' in text
    assert 'output.extra_fields["routed_experts_dtype"] = str(routed_experts.dtype).removeprefix("torch.")' in text
    assert 'output.extra_fields["routed_experts_digest"] = _routed_experts_digest(routed_experts)' in text

    test_text = (
        Path(__file__).resolve().parents[6]
        / "verl"
        / "tests"
        / "experimental"
        / "agent_loop"
        / "test_agent_loop_extra_fields_schema_on_cpu.py"
    ).read_text(encoding="utf-8")
    assert "test_fully_async_partial_rollout_preserves_router_replay_extra_fields_on_cpu" in test_text
    assert 'output.extra_fields["responses_with_routed_experts"] == 2' in test_text
    assert 'output.extra_fields["routed_experts_shape"] == [3, 3, 2]' in test_text
    assert 'output.extra_fields["routed_experts_digest"].startswith("sha256:")' in test_text
    assert '"router_replay_trace_metadata_conflicts" not in output.extra_fields' in test_text


def test_roll_r3_real_smoke_events_rejects_trace_metadata_conflicts_static():
    text = REAL_SMOKE_EVENTS_VALIDATOR.read_text(encoding="utf-8")

    assert "def _find_router_replay_trace_metadata_conflicts(" in text
    assert "router_replay_trace_metadata_conflicts at " in text
    assert "R3 trace metadata conflicts must be absent" in text
    assert "trace_metadata_conflict_rejected" in text
    assert "training.old_logprob_did_not_return_routed_experts" in text
    assert "training.rollout_trace_source_of_truth_preserved" in text
    assert "old_logprob_trace_source_contract" in text
    assert '"router_replay_trace_metadata_conflicts": ["sglang_version"]' in text

    extractor_text = REAL_SMOKE_EXTRACTOR.read_text(encoding="utf-8")
    assert "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD" in extractor_text
    assert "router replay trace metadata conflict at {item['path']}" in extractor_text
    assert '"router_replay_trace_metadata_conflicts": trace_metadata_conflicts' in extractor_text
    assert "trace_metadata_conflict_sample" in extractor_text
    assert "old_logprob_did_not_return_routed_experts" in extractor_text
    assert "rollout_trace_source_of_truth_preserved" in extractor_text

    analyzer_text = REAL_SMOKE_ANALYZER.read_text(encoding="utf-8")
    assert "extraction.router_replay_trace_metadata_conflicts must be an empty list" in analyzer_text
    assert "training.old_logprob_did_not_return_routed_experts must be true" in analyzer_text
    assert "training.rollout_trace_source_of_truth_preserved must be true" in analyzer_text
    assert "bad_old_logprob_trace_source_contract" in analyzer_text


def test_roll_r3_real_smoke_evidence_aliases_cover_real_carriers_static():
    extractor_text = REAL_SMOKE_EXTRACTOR.read_text(encoding="utf-8")
    validator_text = REAL_SMOKE_EVENTS_VALIDATOR.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_NON_TENSOR_ALIASES" in extractor_text
    assert "NON_TENSOR_BATCH_ALIAS_PREFIXES" in extractor_text
    assert "PACKED_BATCH_EXTRAS_ALIASES" in extractor_text
    assert "PACKED_BATCH_EXTRAS_ALIAS_PREFIXES" in extractor_text
    assert '"non_tensor_batch"' in extractor_text
    assert '"batch.non_tensor_batch"' in extractor_text
    assert '"data.non_tensor_batch"' in extractor_text
    assert '"output.non_tensor_batch"' in extractor_text
    assert '"packed_batch.extras"' in extractor_text
    assert '"runtime_batch.extras"' in extractor_text
    assert '"batch_bridge.packed_batch_extras"' in extractor_text
    assert '"routed_experts_digest": "rollout.routed_experts_digest"' in extractor_text
    assert '"routed_experts_digest": "batch_bridge.routed_experts_digest"' in extractor_text
    assert '"router_replay_trace_collection_path": "rollout.trace_collection_path"' in extractor_text
    assert "carrier_alias_paths" in extractor_text

    assert "carrier_alias_contract" in validator_text
    assert '"meta_info.extra_fields.routed_experts_digest" not in extra_fields_alias_result["known_aliases"]' in validator_text
    assert '"non_tensor_batch.routed_experts_digest" not in carrier_alias_result["known_aliases"]' in validator_text
    assert '"non_tensor_batch.router_replay_trace_collection_path" not in carrier_alias_result["known_aliases"]' in validator_text
    assert '"packed_batch.extras.routed_experts_digest" not in carrier_alias_result["known_aliases"]' in validator_text
    assert '"runtime_batch.extras.routed_experts_digest" not in carrier_alias_result["known_aliases"]' in validator_text


def test_roll_r3_real_smoke_extractor_mtp_router_count_static():
    text = REAL_SMOKE_EXTRACTOR.read_text(encoding="utf-8")

    assert "def _contains_enabled_mtp(" in text
    assert 'normalized in {"mtp_enable", "enable_mtp"}' in text
    assert "def _expected_router_count_for_bundle(" in text
    assert "expected_router_replay_instances_without_mtp" in text
    assert "expected_mtp_router_replay_instances_if_enabled" in text
    assert "expected_router_replay_instances_with_mtp_if_enabled" in text
    assert "if _contains_enabled_mtp(train_spec) or _contains_enabled_mtp(summary):" in text
    assert "expected_with_mtp = route_fields.get(" in text
    assert "return expected_with_mtp" in text
    assert "expected_without_mtp = route_fields.get(" in text
    assert '"expected_router_replay_instances": expected_routers or 0' in text
    assert "without_mtp_expected_router_count" in text
    assert "mtp_expected_router_count" in text
    assert 'json.dumps({"impl_cfg": {"mtp_enable": True}})' in text


def test_roll_r3_real_smoke_extractor_expected_router_count_source_of_truth_static():
    text = REAL_SMOKE_EXTRACTOR.read_text(encoding="utf-8")

    assert "bundle_expected_routers = _get_path(" in text
    assert "bundle_model_validation=model_validation_path" in text
    assert "observed_expected_routers = _get_path(" in text
    assert "expected_router_source_failures" in text
    assert "training.expected_router_replay_instances must match bundle model-path validation" in text
    assert "route must match bundle model-path validation route" in text
    assert "model.checkpoint_path must match bundle model-path validation model_path" in text
    assert '"bundle_expected_router_replay_instances": bundle_expected_routers' in text
    assert '"observed_expected_router_replay_instances": observed_expected_routers' in text
    assert "wrong_expected_events" in text
    assert "route_mismatch_bundle" in text
    assert "checkpoint_mismatch_bundle" in text
    assert "bundle_expected_router_count_source_of_truth" in text
    assert "bundle_route_source_of_truth" in text
    assert "bundle_checkpoint_path_source_of_truth" in text


def test_roll_r3_real_smoke_events_bundle_router_count_source_of_truth_static():
    text = REAL_SMOKE_EVENTS_VALIDATOR.read_text(encoding="utf-8")

    assert "def _bundle_expected_router_count(" in text
    assert "def _bundle_expected_num_experts(" in text
    assert "def _bundle_expected_topk(" in text
    assert "expected_router_replay_instances_without_mtp" in text
    assert "expected_router_replay_instances_with_mtp_if_enabled" in text
    assert "bundle_expected_router_replay_instances" in text
    assert "observed_expected_router_replay_instances" in text
    assert "bundle_expected_num_experts" in text
    assert "observed_model_num_experts" in text
    assert "bundle_expected_topk" in text
    assert "observed_rollout_topk" in text
    assert "training.expected_router_replay_instances must match bundle model-path validation" in text
    assert "model.num_experts must match bundle model-path validation num_experts" in text
    assert "rollout.topk must match bundle model-path validation num_experts_per_tok" in text
    assert "valid_with_bundle_router_count" in text
    assert "mtp_bundle_router_count_mismatch" in text
    assert "bundle_router_count_source_of_truth" in text
    assert "bundle_num_experts_source_of_truth" in text
    assert "bundle_topk_source_of_truth" in text
    assert "--bundle-model-validation" in text
    assert "--mtp-enable" in text

    analyzer = REAL_SMOKE_ANALYZER.read_text(encoding="utf-8")
    assert "bundle_model_validation: Path | None = None" in analyzer
    assert "bundle_model_validation=bundle_model_validation" in analyzer
    assert "def _bundle_model_num_experts_source_of_truth_failures(" in analyzer
    assert "def _bundle_model_topk_source_of_truth_failures(" in analyzer
    assert "def _bundle_model_route_source_of_truth_failures(" in analyzer
    assert "def _bundle_model_checkpoint_path_source_of_truth_failures(" in analyzer
    assert "bad_bundle_router_count_source_of_truth" in analyzer
    assert "bad_bundle_num_experts_source_of_truth" in analyzer
    assert "bad_bundle_topk_source_of_truth" in analyzer
    assert "bad_bundle_route_source_of_truth" in analyzer
    assert "bad_bundle_checkpoint_path_source_of_truth" in analyzer
    assert "training.expected_router_replay_instances must match bundle model-path validation" in analyzer
    assert "model.num_experts must match bundle model-path validation num_experts" in analyzer
    assert "rollout.topk must match bundle model-path validation num_experts_per_tok" in analyzer
    assert "route must match bundle model-path validation route" in analyzer
    assert "model.family must match bundle model-path validation route" in analyzer
    assert "model.checkpoint_path must match bundle model-path validation model_path" in analyzer


def test_roll_r3_real_smoke_training_router_count_matches_rollout_static():
    validator = REAL_SMOKE_EVENTS_VALIDATOR.read_text(encoding="utf-8")
    analyzer = REAL_SMOKE_ANALYZER.read_text(encoding="utf-8")
    readiness = R3_LAUNCH_READINESS.read_text(encoding="utf-8")
    preparer = REAL_SMOKE_BUNDLE_PREPARER.read_text(encoding="utf-8")
    bundle_validator = REAL_SMOKE_BUNDLE_VALIDATOR.read_text(encoding="utf-8")

    for text in (validator, analyzer, preparer, bundle_validator):
        assert "training.router_replay_instances must equal rollout.routers" in text

    assert "router_count_matches_rollout_contract" in validator
    assert "bad_router_count_sample" in analyzer
    assert "bad_router_count_training_contract" in analyzer
    assert "rollout_routers" in analyzer
    assert "router_replay_instances_match_rollout_routers" in analyzer
    assert "router_replay_instances_match_rollout_routers" in readiness
    assert "training.router_replay_instances must equal rollout.routers" in readiness
    assert "router_replay_instances = training.get(\"router_replay_instances\")" in preparer
    assert "if router_replay_instances != routers:" in preparer


def test_roll_r3_real_smoke_bundle_collects_with_bundle_router_count_source_of_truth_static():
    preparer = REAL_SMOKE_BUNDLE_PREPARER.read_text(encoding="utf-8")
    validator = REAL_SMOKE_BUNDLE_VALIDATOR.read_text(encoding="utf-8")
    readiness = R3_LAUNCH_READINESS.read_text(encoding="utf-8")
    template = json.loads(REAL_SMOKE_RESULT_TEMPLATE.read_text(encoding="utf-8"))
    event_template_lines = [
        json.loads(line)
        for line in REAL_SMOKE_EVENTS_TEMPLATE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_template = {}
    for event in event_template_lines:
        for key, value in event.items():
            if key in {"event", "template_only"}:
                continue
            if isinstance(value, dict):
                event_template.setdefault(key, {}).update(value)
            else:
                event_template.setdefault(key, value)

    for text in (preparer, validator, readiness):
        assert "collect_evidence_requires_bundle_router_count_source_of_truth" in text

    assert "event_validator_args=(" in preparer
    assert "analyzer_args=(" in preparer
    assert "--bundle-model-validation" in preparer
    assert '"$model_validation"' in preparer
    assert "--mtp-enable" in preparer
    assert "MTP_ENABLE" in preparer
    assert "training.expected_router_replay_instances must match bundle model-path validation" in readiness
    assert "--bundle-model-validation" in validator
    assert "--mtp-enable" in validator
    assert "MTP_ENABLE" in validator
    assert "REQUIRED_RESULT_TEMPLATE_DIGEST_PATHS" in validator
    assert "REQUIRED_TEMPLATE_TRACE_PATHS" in validator
    assert "REAL_SMOKE_EVENTS_TEMPLATE" in validator
    assert "_events_template_contract_failures" in validator
    assert "_sha256_digest_like" in validator
    assert "_result_template_digest_failures" in validator
    assert "R3 real-smoke result template {dotted_path} must be a sha256 fill placeholder" in validator
    assert "R3 real-smoke events template {dotted_path} must be a sha256 digest" in validator
    assert "roll-r3-real-smoke-events.template.jsonl is missing" in validator
    assert "bad_result_template_missing_trace_source" in validator
    assert "bad_events_template_missing_required_event" in validator
    assert "bad_events_template_bad_digest" in validator
    assert "bad_events_template_missing_layout" in validator

    assert "sha256" in template["rollout"]["routed_experts_digest"]
    assert "sha256" in template["batch_bridge"]["routed_experts_digest"]
    assert "sha256" in template["training"]["routed_experts_digest"]
    assert "sha256" in template["training"]["recompute_routed_experts_digest"]
    assert "sha256" in event_template["rollout"]["routed_experts_digest"]
    assert "sha256" in event_template["batch_bridge"]["routed_experts_digest"]
    assert "sha256" in event_template["training"]["routed_experts_digest"]
    assert "sha256" in event_template["training"]["recompute_routed_experts_digest"]
    required_trace_extra_keys = {
        "router_replay_trace_collection_path",
        "router_replay_trace_source",
        "router_replay_trace_preserved_through_scheduler",
        "router_replay_postprocess_wrote_trace_to_batch",
    }
    for payload in (template, event_template):
        assert payload["rollout"]["trace_collection_path"]
        assert (
            payload["batch_bridge"]["trace_source"] in {"routed_experts", "routed_experts_segments"}
            or "<fill:" in payload["batch_bridge"]["trace_source"]
        )
        assert payload["batch_bridge"]["router_replay_layout"] in {"true_tokens", "full_padded", "cp_local"}
        assert required_trace_extra_keys.issubset(set(payload["batch_bridge"]["packed_batch_extras_trace_keys"]))


def test_roll_r3_handoff_ready_markers_pin_strict_digest_shape_static():
    strict_marker = (
        "routed_experts_digest values match across rollout, batch_bridge, training, and recompute "
        "after sha256 normalization (optional sha256: prefix, 64-hex payload)"
    )
    strict_readiness_contract = (
        "rollout/batch_bridge/training/recompute routed_experts_digest values must match "
        "after sha256 normalization (optional sha256: prefix, 64-hex payload)"
    )

    readiness = R3_LAUNCH_READINESS.read_text(encoding="utf-8")
    assert strict_readiness_contract in readiness
    assert (
        "rollout/batch_bridge/training/recompute routed_experts_digest values must be matching sha256 digests"
        not in readiness
    )

    for source in (
        UNBLOCK_RUNBOOK_PREPARER,
        EXTERNAL_HANDOFF_PREPARER,
        DELIVERABLE_AUDIT,
        STATUS_CONSISTENCY_AUDIT,
    ):
        text = source.read_text(encoding="utf-8")
        assert strict_marker in text
        assert "routed_experts_digest values match across rollout, batch_bridge, training, and recompute\"," not in text


def test_verl_ppo_trainer_r3_standard_rollout_writes_missing_trace_metadata_static():
    text = PPO_TRAINER.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS" in text
    assert '"router_replay_layout": "full_padded"' in text
    assert '"router_replay_trace_collection_path": "router_generate_request"' in text
    assert '"router_replay_trace_source": "routed_experts"' in text
    assert '"router_replay_trace_preserved_through_scheduler": True' in text
    assert '"router_replay_postprocess_wrote_trace_to_batch": True' in text
    assert "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS_BY_SOURCE" in text
    assert '"routed_experts_segments": {' in text
    assert '"router_replay_layout": "true_tokens"' in text
    assert '"router_replay_trace_source": "routed_experts_segments"' in text
    assert "def _router_replay_source_trace(data: DataProto) -> str | None:" in text
    assert "ROUTER_REPLAY_TRACE_COLLECTION_PATHS" in text
    assert "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD" in text
    assert "ROUTER_REPLAY_SOURCE_TRACE_BATCH_FIELDS" in text
    assert "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_NON_TENSOR_FIELDS" in text
    assert "ROUTER_REPLAY_SEGMENT_TRACE_ANCILLARY_NON_TENSOR_FIELDS" in text
    assert "ROUTER_REPLAY_TRACE_METADATA_ONLY_NON_TENSOR_FIELDS" in text
    assert "def _uniform_router_replay_metadata(data: DataProto, key: str) -> Any:" in text
    assert "def _router_replay_conflicts_empty(value: Any) -> bool:" in text
    assert "def _router_replay_digest_is_sha256(value: Any) -> bool:" in text
    assert "def _router_replay_metadata_without_source_fields(data: DataProto) -> set[str]:" in text
    assert "def _router_replay_segment_side_channel_fields_without_segments" in text
    assert "def _ensure_router_replay_trace_metadata(data: DataProto) -> None:" in text
    assert "def _ensure_log_prob_output_preserves_router_replay_source(" in text
    assert "trace_source = _router_replay_source_trace(data)" in text
    assert "if trace_source is None:" in text
    assert "metadata-only fields={sorted(metadata_only_fields)!r}" in text
    assert "expected_fields = ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS_BY_SOURCE[trace_source]" in text
    assert "if key not in data.non_tensor_batch:" in text
    assert "data.non_tensor_batch[key] = np.full(len(data), value, dtype=object)" in text
    assert "must be uniform across a router replay batch" in text
    assert "router_replay_trace_metadata_conflicts must be empty" in text
    assert "routed_experts_digest must be a sha256 digest before standard " in text
    assert "router_replay_trace_collection_path must identify a trace-preserving " in text
    assert "rollout path ({valid}) before standard PPO consumes Router Replay trace" in text
    assert "before standard PPO consumes {trace_source}" in text
    assert "_ensure_router_replay_trace_metadata(batch)" in text
    assert text.count("_ensure_router_replay_trace_metadata(batch)") >= 2
    assert (
        "self._balance_batch(batch, metrics=metrics)\n"
        "                        _ensure_router_replay_trace_metadata(batch)"
    ) in text
    assert "already carries rollout routed_experts" in text
    assert "as the source of truth; {output_kind} may consume rollout trace" in text
    assert 'output_kind="log-prob/reference passes"' in text
    assert 'role="old_log_prob"' in text
    assert 'role="ref_log_prob"' in text

    tree = ast.parse(text)
    support_names = {
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS_BY_SOURCE",
        "ROUTER_REPLAY_TRACE_COLLECTION_PATHS",
        "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD",
        "ROUTER_REPLAY_SOURCE_TRACE_BATCH_FIELDS",
        "ROUTER_REPLAY_SOURCE_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_SEGMENT_TRACE_ANCILLARY_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_BATCH_FIELDS",
        "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_PRESERVED_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_METADATA_ONLY_NON_TENSOR_FIELDS",
        "_router_replay_metadata_values",
        "_router_replay_metadata_equal",
        "_uniform_router_replay_metadata",
        "_router_replay_conflicts_empty",
        "_router_replay_digest_is_sha256",
        "_router_replay_source_trace",
        "_router_replay_conflicting_source_trace_fields",
        "_router_replay_metadata_without_source_fields",
        "_router_replay_segment_side_channel_fields_without_segments",
        "_ensure_router_replay_trace_metadata",
        "_has_router_replay_source_trace",
        "_ensure_log_prob_output_preserves_router_replay_source",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"np": __import__("numpy")}
    exec(compile(module, str(PPO_TRAINER), "exec"), namespace)
    ensure_metadata = namespace["_ensure_router_replay_trace_metadata"]
    default_metadata = namespace["ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS"]
    metadata_by_source = namespace["ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS_BY_SOURCE"]
    source_trace = namespace["_router_replay_source_trace"]
    np = namespace["np"]

    class FakeData:
        def __init__(self, batch, non_tensor_batch=None, length=2):
            self.batch = batch
            self.non_tensor_batch = {} if non_tensor_batch is None else dict(non_tensor_batch)
            self._length = length

        def __len__(self):
            return self._length

    plain = FakeData(batch={})
    ensure_metadata(plain)
    assert plain.non_tensor_batch == {}

    routed = FakeData(batch={"routed_experts": object()})
    assert source_trace(routed) == "routed_experts"
    ensure_metadata(routed)
    for key, value in default_metadata.items():
        assert routed.non_tensor_batch[key].tolist() == [value, value]
        assert routed.non_tensor_batch[key].dtype == object

    segments = FakeData(
        batch={},
        non_tensor_batch={"routed_experts_segments": object()},
    )
    assert source_trace(segments) == "routed_experts_segments"
    ensure_metadata(segments)
    for key, value in metadata_by_source["routed_experts_segments"].items():
        assert segments.non_tensor_batch[key].tolist() == [value, value]
        assert segments.non_tensor_batch[key].dtype == object

    both = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={"routed_experts_segments": object()},
    )
    assert source_trace(both) == "routed_experts_segments"
    try:
        ensure_metadata(both)
    except ValueError as exc:
        message = str(exc)
        assert "exactly one trace payload source" in message
        assert "routed_experts_segments" in message
        assert "routed_experts" in message
    else:
        raise AssertionError("standard PPO must reject mixed Router Replay trace sources")

    existing = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "router_replay_trace_collection_path": np.array(
                ["user_defined_rollout_loop"], dtype=object
            )
        },
        length=1,
    )
    ensure_metadata(existing)
    assert existing.non_tensor_batch["router_replay_trace_collection_path"].tolist() == [
        "user_defined_rollout_loop"
    ]
    for key, value in default_metadata.items():
        if key == "router_replay_trace_collection_path":
            continue
        assert existing.non_tensor_batch[key].tolist() == [value]

    valid_digest = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "routed_experts_digest": np.array(
                ["sha256:" + "a" * 64, "sha256:" + "a" * 64], dtype=object
            )
        },
    )
    ensure_metadata(valid_digest)

    mixed_digest = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "routed_experts_digest": np.array(
                ["sha256:" + "a" * 64, "sha256:" + "b" * 64], dtype=object
            )
        },
    )
    try:
        ensure_metadata(mixed_digest)
    except ValueError as exc:
        assert "routed_experts_digest must be uniform" in str(exc)
    else:
        raise AssertionError("mixed routed_experts_digest values must fail early")

    bad_digest = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={"routed_experts_digest": np.array(["not-a-digest"], dtype=object)},
        length=1,
    )
    try:
        ensure_metadata(bad_digest)
    except ValueError as exc:
        assert "routed_experts_digest must be a sha256 digest" in str(exc)
    else:
        raise AssertionError("invalid routed_experts_digest must fail early")

    mixed_path = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "router_replay_trace_collection_path": np.array(
                ["router_generate_request", "user_defined_rollout_loop"], dtype=object
            )
        },
    )
    try:
        ensure_metadata(mixed_path)
    except ValueError as exc:
        assert "router_replay_trace_collection_path must be uniform" in str(exc)
    else:
        raise AssertionError("mixed router replay collection paths must fail early")

    bad_conflict = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "router_replay_trace_metadata_conflicts": np.array(
                [[], ["sglang_version"]], dtype=object
            )
        },
    )
    try:
        ensure_metadata(bad_conflict)
    except ValueError as exc:
        assert "router_replay_trace_metadata_conflicts must be uniform" in str(exc)
    else:
        raise AssertionError("mixed router replay conflict metadata must fail early")

    non_empty_conflict = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "router_replay_trace_metadata_conflicts": np.array(
                [["sglang_version"], ["sglang_version"]], dtype=object
            )
        },
    )
    try:
        ensure_metadata(non_empty_conflict)
    except ValueError as exc:
        assert "router_replay_trace_metadata_conflicts must be empty" in str(exc)
    else:
        raise AssertionError("non-empty router replay conflict metadata must fail early")

    bad_source = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "router_replay_trace_source": np.array(
                ["routed_experts_segments", "routed_experts_segments"], dtype=object
            )
        },
    )
    try:
        ensure_metadata(bad_source)
    except ValueError as exc:
        assert "router_replay_trace_source must be 'routed_experts'" in str(exc)
    else:
        raise AssertionError("standard PPO routed_experts must reject segment source")

    bad_segment_source = FakeData(
        batch={},
        non_tensor_batch={
            "routed_experts_segments": object(),
            "router_replay_trace_source": np.array(
                ["routed_experts", "routed_experts"], dtype=object
            ),
        },
    )
    try:
        ensure_metadata(bad_segment_source)
    except ValueError as exc:
        assert "router_replay_trace_source must be 'routed_experts_segments'" in str(exc)
    else:
        raise AssertionError("standard PPO routed_experts_segments must reject tensor source")

    bad_segment_layout = FakeData(
        batch={},
        non_tensor_batch={
            "routed_experts_segments": object(),
            "router_replay_layout": np.array(["full_padded", "full_padded"], dtype=object),
        },
    )
    try:
        ensure_metadata(bad_segment_layout)
    except ValueError as exc:
        assert "router_replay_layout must be 'true_tokens'" in str(exc)
    else:
        raise AssertionError("standard PPO routed_experts_segments must require true-token layout")

    bad_path = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "router_replay_trace_collection_path": np.array(
                ["plain_generate", "plain_generate"], dtype=object
            )
        },
    )
    try:
        ensure_metadata(bad_path)
    except ValueError as exc:
        assert "trace-preserving rollout path" in str(exc)
    else:
        raise AssertionError("unknown router replay trace collection path must fail early")


def test_verl_ppo_trainer_r3_segment_trace_metadata_static():
    text = PPO_TRAINER.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS_BY_SOURCE" in text
    assert "def _router_replay_source_trace(data: DataProto) -> str | None:" in text
    assert 'if "routed_experts_segments" in data.non_tensor_batch:' in text
    assert 'return "routed_experts_segments"' in text
    assert '"router_replay_layout": "true_tokens"' in text
    assert '"router_replay_trace_collection_path": "user_defined_rollout_loop"' in text
    assert '"router_replay_trace_source": "routed_experts_segments"' in text

    tree = ast.parse(text)
    support_names = {
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS_BY_SOURCE",
        "ROUTER_REPLAY_TRACE_COLLECTION_PATHS",
        "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD",
        "ROUTER_REPLAY_SOURCE_TRACE_BATCH_FIELDS",
        "ROUTER_REPLAY_SOURCE_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_SEGMENT_TRACE_ANCILLARY_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_PRESERVED_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_METADATA_ONLY_NON_TENSOR_FIELDS",
        "_router_replay_metadata_values",
        "_router_replay_metadata_equal",
        "_uniform_router_replay_metadata",
        "_router_replay_conflicts_empty",
        "_router_replay_digest_is_sha256",
        "_router_replay_source_trace",
        "_router_replay_conflicting_source_trace_fields",
        "_router_replay_metadata_without_source_fields",
        "_router_replay_segment_side_channel_fields_without_segments",
        "_ensure_router_replay_trace_metadata",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"np": __import__("numpy")}
    exec(compile(module, str(PPO_TRAINER), "exec"), namespace)
    ensure_metadata = namespace["_ensure_router_replay_trace_metadata"]
    source_trace = namespace["_router_replay_source_trace"]
    metadata_by_source = namespace["ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS_BY_SOURCE"]
    np = namespace["np"]

    class FakeData:
        def __init__(self, batch, non_tensor_batch=None, length=2):
            self.batch = batch
            self.non_tensor_batch = {} if non_tensor_batch is None else dict(non_tensor_batch)
            self._length = length

        def __len__(self):
            return self._length

    segment_trace = FakeData(
        batch={},
        non_tensor_batch={
            "routed_experts_segments": object(),
            "routed_experts_segment_seq_lens": object(),
            "routed_experts_num_routers": 1,
        },
    )
    assert source_trace(segment_trace) == "routed_experts_segments"
    ensure_metadata(segment_trace)
    for key, value in metadata_by_source["routed_experts_segments"].items():
        assert segment_trace.non_tensor_batch[key].tolist() == [value, value]

    both_trace_shapes = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={"routed_experts_segments": object()},
    )
    assert source_trace(both_trace_shapes) == "routed_experts_segments"
    try:
        ensure_metadata(both_trace_shapes)
    except ValueError as exc:
        message = str(exc)
        assert "exactly one trace payload source" in message
        assert "routed_experts_segments" in message
        assert "routed_experts" in message
    else:
        raise AssertionError("segment trace must reject mixed Router Replay source payloads")

    bad_source = FakeData(
        batch={},
        non_tensor_batch={
            "routed_experts_segments": object(),
            "router_replay_trace_source": np.array(["routed_experts"], dtype=object),
        },
        length=1,
    )
    try:
        ensure_metadata(bad_source)
    except ValueError as exc:
        assert "router_replay_trace_source must be 'routed_experts_segments'" in str(exc)
    else:
        raise AssertionError("segment trace must reject direct routed_experts metadata")

    bad_layout = FakeData(
        batch={},
        non_tensor_batch={
            "routed_experts_segments": object(),
            "router_replay_layout": np.array(["full_padded"], dtype=object),
        },
        length=1,
    )
    try:
        ensure_metadata(bad_layout)
    except ValueError as exc:
        assert "router_replay_layout must be 'true_tokens'" in str(exc)
    else:
        raise AssertionError("segment trace must require true-token replay layout")

    bad_conflict = FakeData(
        batch={},
        non_tensor_batch={
            "routed_experts_segments": object(),
            "router_replay_trace_metadata_conflicts": np.array([["segment_conflict"]], dtype=object),
        },
        length=1,
    )
    try:
        ensure_metadata(bad_conflict)
    except ValueError as exc:
        assert "router_replay_trace_metadata_conflicts must be empty" in str(exc)
    else:
        raise AssertionError("segment trace metadata conflicts must fail early")


def test_verl_ppo_trainer_r3_metadata_without_payload_fails_static():
    text = PPO_TRAINER.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_TRACE_METADATA_ONLY_NON_TENSOR_FIELDS" in text
    assert "def _router_replay_metadata_without_source_fields(data: DataProto) -> set[str]:" in text
    assert "def _router_replay_conflicting_source_trace_fields(data: DataProto) -> set[str]:" in text
    assert "Router Replay batch must carry exactly one trace payload source" in text
    assert "def _router_replay_segment_side_channel_fields_without_segments" in text
    assert "Router Replay segment side-channel fields require " in text
    assert "orphan segment fields={sorted(segment_side_channel_fields)!r}" in text
    assert "Router Replay trace metadata fields are present without a " in text
    assert "R3 requires routed_experts or " in text
    assert "routed_experts_segments to move with metadata" in text

    tree = ast.parse(text)
    support_names = {
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS_BY_SOURCE",
        "ROUTER_REPLAY_TRACE_COLLECTION_PATHS",
        "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD",
        "ROUTER_REPLAY_SOURCE_TRACE_BATCH_FIELDS",
        "ROUTER_REPLAY_SOURCE_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_SEGMENT_TRACE_ANCILLARY_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_BATCH_FIELDS",
        "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_PRESERVED_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_METADATA_ONLY_NON_TENSOR_FIELDS",
        "_router_replay_metadata_values",
        "_router_replay_metadata_equal",
        "_uniform_router_replay_metadata",
        "_router_replay_conflicts_empty",
        "_router_replay_digest_is_sha256",
        "_router_replay_source_trace",
        "_router_replay_conflicting_source_trace_fields",
        "_router_replay_metadata_without_source_fields",
        "_router_replay_segment_side_channel_fields_without_segments",
        "_ensure_router_replay_trace_metadata",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"np": __import__("numpy")}
    exec(compile(module, str(PPO_TRAINER), "exec"), namespace)
    ensure_metadata = namespace["_ensure_router_replay_trace_metadata"]
    metadata_without_source = namespace["_router_replay_metadata_without_source_fields"]
    np = namespace["np"]

    class FakeData:
        def __init__(self, batch, non_tensor_batch=None, length=1):
            self.batch = batch
            self.non_tensor_batch = {} if non_tensor_batch is None else dict(non_tensor_batch)
            self._length = length

        def __len__(self):
            return self._length

    plain = FakeData(batch={})
    assert metadata_without_source(plain) == set()
    ensure_metadata(plain)

    traced = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={"router_replay_trace_source": np.array(["routed_experts"], dtype=object)},
    )
    assert metadata_without_source(traced) == set()
    ensure_metadata(traced)

    mixed_source = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={"routed_experts_segments": object()},
    )
    try:
        ensure_metadata(mixed_source)
    except ValueError as exc:
        message = str(exc)
        assert "exactly one trace payload source" in message
        assert "routed_experts_segments" in message
        assert "routed_experts" in message
    else:
        raise AssertionError("mixed Router Replay source payloads must fail early")

    for key in (
        "router_replay_trace_source",
        "router_replay_trace_collection_path",
        "routed_experts_digest",
        "router_replay_trace_metadata_conflicts",
    ):
        ghost = FakeData(
            batch={},
            non_tensor_batch={key: np.array(["routed_experts"], dtype=object)},
        )
        try:
            ensure_metadata(ghost)
        except ValueError as exc:
            message = str(exc)
            assert "metadata fields are present without" in message
            assert "routed_experts or routed_experts_segments" in message
            assert key in message
        else:
            raise AssertionError(f"metadata-only Router Replay field {key} must fail early")

    for key in ("routed_experts_segment_seq_lens", "routed_experts_num_routers"):
        orphan_segment = FakeData(batch={}, non_tensor_batch={key: object()})
        try:
            ensure_metadata(orphan_segment)
        except ValueError as exc:
            message = str(exc)
            assert "segment side-channel fields require routed_experts_segments" in message
            assert key in message
        else:
            raise AssertionError(f"orphan Router Replay segment side-channel {key} must fail")

        direct_trace_with_segment_side_channel = FakeData(
            batch={"routed_experts": object()},
            non_tensor_batch={key: object()},
        )
        try:
            ensure_metadata(direct_trace_with_segment_side_channel)
        except ValueError as exc:
            message = str(exc)
            assert "segment side-channel fields require routed_experts_segments" in message
            assert key in message
        else:
            raise AssertionError(
                f"direct routed_experts must reject stale segment side-channel {key}"
            )


def test_verl_ppo_trainer_r3_logprob_outputs_do_not_override_rollout_trace_static():
    text = PPO_TRAINER.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_SOURCE_TRACE_BATCH_FIELDS" in text
    assert "ROUTER_REPLAY_SOURCE_TRACE_NON_TENSOR_FIELDS" in text
    assert "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_BATCH_FIELDS" in text
    assert "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_NON_TENSOR_FIELDS" in text
    assert "def _has_router_replay_source_trace(data: DataProto) -> bool:" in text
    assert "def _ensure_data_proto_output_preserves_router_replay_source(" in text
    assert "def _ensure_log_prob_output_preserves_router_replay_source(" in text
    assert "already carries rollout routed_experts" in text
    assert "as the source of truth; {output_kind} may consume rollout trace" in text
    assert 'output_kind="log-prob/reference passes"' in text
    assert 'role="old_log_prob"' in text
    assert 'role="ref_log_prob"' in text

    tree = ast.parse(text)
    support_names = {
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD",
        "ROUTER_REPLAY_SOURCE_TRACE_BATCH_FIELDS",
        "ROUTER_REPLAY_SOURCE_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_BATCH_FIELDS",
        "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_NON_TENSOR_FIELDS",
        "_has_router_replay_source_trace",
        "_ensure_data_proto_output_preserves_router_replay_source",
        "_ensure_log_prob_output_preserves_router_replay_source",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(PPO_TRAINER), "exec"), namespace)
    ensure_log_prob_preserves_source = namespace[
        "_ensure_log_prob_output_preserves_router_replay_source"
    ]

    class FakeData:
        def __init__(self, batch, non_tensor_batch=None):
            self.batch = batch
            self.non_tensor_batch = {} if non_tensor_batch is None else dict(non_tensor_batch)

    rollout_source = FakeData(batch={"routed_experts": object()})
    clean_log_prob = FakeData(batch={"old_log_probs": object()})
    ensure_log_prob_preserves_source(rollout_source, clean_log_prob, role="old_log_prob")

    r2_record_output_without_rollout_source = FakeData(batch={"routed_experts": object()})
    ensure_log_prob_preserves_source(
        FakeData(batch={}),
        r2_record_output_without_rollout_source,
        role="old_log_prob",
    )

    bad_old_log_prob = FakeData(batch={"routed_experts": object()})
    try:
        ensure_log_prob_preserves_source(
            rollout_source, bad_old_log_prob, role="old_log_prob"
        )
    except ValueError as exc:
        assert "old_log_prob must not return router replay trace fields" in str(exc)
        assert "source of truth" in str(exc)
        assert "tensor fields=['routed_experts']" in str(exc)
    else:
        raise AssertionError("old_log_prob must not replace rollout routed_experts")

    segment_source = FakeData(
        batch={},
        non_tensor_batch={"routed_experts_segments": object()},
    )
    bad_ref_log_prob = FakeData(
        batch={"ref_log_prob": object()},
        non_tensor_batch={"routed_experts_digest": "sha256:" + "a" * 64},
    )
    try:
        ensure_log_prob_preserves_source(segment_source, bad_ref_log_prob, role="ref_log_prob")
    except ValueError as exc:
        assert "ref_log_prob must not return router replay trace fields" in str(exc)
        assert "non-tensor fields=['routed_experts_digest']" in str(exc)
    else:
        raise AssertionError("ref_log_prob must not return router replay side channels")


def test_verl_ppo_trainer_r3_auxiliary_outputs_do_not_override_rollout_trace_static():
    text = PPO_TRAINER.read_text(encoding="utf-8")

    assert "def _ensure_data_proto_output_preserves_router_replay_source(" in text
    assert "as the source of truth; {output_kind} may consume rollout trace" in text
    assert 'output_kind="reward-model passes"' in text
    assert 'output_kind="critic value passes"' in text
    assert 'role="reward_model"' in text
    assert 'role="critic_values"' in text
    assert (
        "_ensure_data_proto_output_preserves_router_replay_source(\n"
        "                                gen_baseline_output,\n"
        "                                baseline_reward,\n"
        '                                role="reward_model",'
    ) in text
    assert (
        "_ensure_data_proto_output_preserves_router_replay_source(\n"
        "                                batch,\n"
        "                                batch_reward,\n"
        '                                role="reward_model",'
    ) in text
    assert (
        "_ensure_data_proto_output_preserves_router_replay_source(\n"
        "                                batch,\n"
        "                                values,\n"
        '                                role="critic_values",'
    ) in text
    assert text.index('role="reward_model"') < text.index("batch = batch.union(batch_reward)")
    assert text.index('role="critic_values"') < text.index("batch = batch.union(values)")

    tree = ast.parse(text)
    support_names = {
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD",
        "ROUTER_REPLAY_SOURCE_TRACE_BATCH_FIELDS",
        "ROUTER_REPLAY_SOURCE_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_BATCH_FIELDS",
        "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_NON_TENSOR_FIELDS",
        "_has_router_replay_source_trace",
        "_ensure_data_proto_output_preserves_router_replay_source",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(PPO_TRAINER), "exec"), namespace)
    ensure_output_preserves_source = namespace[
        "_ensure_data_proto_output_preserves_router_replay_source"
    ]

    class FakeData:
        def __init__(self, batch, non_tensor_batch=None):
            self.batch = batch
            self.non_tensor_batch = {} if non_tensor_batch is None else dict(non_tensor_batch)

    rollout_source = FakeData(batch={"routed_experts": object()})
    ensure_output_preserves_source(
        rollout_source,
        FakeData(batch={"rm_scores": object()}),
        role="reward_model",
        output_kind="reward-model passes",
    )
    ensure_output_preserves_source(
        rollout_source,
        FakeData(batch={"values": object()}),
        role="critic_values",
        output_kind="critic value passes",
    )
    ensure_output_preserves_source(
        FakeData(batch={}),
        FakeData(batch={"routed_experts": object()}),
        role="reward_model",
        output_kind="reward-model passes",
    )

    for role, output_kind, bad_output in (
        (
            "reward_model",
            "reward-model passes",
            FakeData(batch={"routed_experts": object()}),
        ),
        (
            "critic_values",
            "critic value passes",
            FakeData(non_tensor_batch={"router_replay_trace_source": "routed_experts"}, batch={}),
        ),
    ):
        try:
            ensure_output_preserves_source(
                rollout_source,
                bad_output,
                role=role,
                output_kind=output_kind,
            )
        except ValueError as exc:
            assert f"{role} must not return router replay trace fields" in str(exc)
            assert "source of truth" in str(exc)
            assert output_kind in str(exc)
        else:
            raise AssertionError(f"{role} must not replace rollout router replay trace")


def test_verl_ppo_trainer_r3_post_reward_transforms_preserve_rollout_trace_static():
    text = PPO_TRAINER.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_PRESERVED_NON_TENSOR_FIELDS" in text
    assert "def _snapshot_router_replay_trace_fields(data: DataProto)" in text
    assert "def _ensure_router_replay_trace_fields_survive(" in text
    assert "must preserve Router Replay trace fields once standard PPO" in text
    assert text.count("router_replay_trace_snapshot = _snapshot_router_replay_trace_fields(batch)") >= 3
    assert (
        "batch, kl_metrics = apply_kl_penalty(\n"
        "                                batch, kl_ctrl=self.kl_ctrl_in_reward"
    ) in text
    assert 'role="kl_penalty"' in text
    assert "batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)" in text
    assert 'role="rollout_correction"' in text
    assert "batch = compute_advantage(" in text
    assert 'role="compute_advantage"' in text

    tree = ast.parse(text)
    support_names = {
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS_BY_SOURCE",
        "ROUTER_REPLAY_TRACE_COLLECTION_PATHS",
        "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD",
        "ROUTER_REPLAY_SOURCE_TRACE_BATCH_FIELDS",
        "ROUTER_REPLAY_SOURCE_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_SEGMENT_TRACE_ANCILLARY_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_PRESERVED_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_METADATA_ONLY_NON_TENSOR_FIELDS",
        "_router_replay_metadata_values",
        "_router_replay_metadata_equal",
        "_uniform_router_replay_metadata",
        "_router_replay_conflicts_empty",
        "_router_replay_digest_is_sha256",
        "_router_replay_source_trace",
        "_router_replay_conflicting_source_trace_fields",
        "_router_replay_metadata_without_source_fields",
        "_router_replay_segment_side_channel_fields_without_segments",
        "_ensure_router_replay_trace_metadata",
        "_snapshot_router_replay_trace_fields",
        "_ensure_router_replay_trace_fields_survive",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"np": __import__("numpy")}
    exec(compile(module, str(PPO_TRAINER), "exec"), namespace)
    snapshot_trace = namespace["_snapshot_router_replay_trace_fields"]
    ensure_trace_survives = namespace["_ensure_router_replay_trace_fields_survive"]
    np = namespace["np"]

    class FakeData:
        def __init__(self, batch, non_tensor_batch=None, length=2):
            self.batch = batch
            self.non_tensor_batch = {} if non_tensor_batch is None else dict(non_tensor_batch)
            self._length = length

        def __len__(self):
            return self._length

    plain = FakeData(batch={})
    ensure_trace_survives(plain, snapshot_trace(plain), role="compute_advantage")

    traced = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "router_replay_trace_collection_path": np.array(
                ["router_generate_request", "router_generate_request"], dtype=object
            ),
            "router_replay_trace_source": np.array(["routed_experts", "routed_experts"], dtype=object),
            "router_replay_trace_preserved_through_scheduler": np.array([True, True], dtype=object),
            "router_replay_postprocess_wrote_trace_to_batch": np.array([True, True], dtype=object),
            "routed_experts_digest": np.array(["sha256:" + "a" * 64, "sha256:" + "a" * 64], dtype=object),
            "router_replay_trace_metadata_conflicts": np.array([[], []], dtype=object),
        },
    )
    traced_snapshot = snapshot_trace(traced)
    ensure_trace_survives(traced, traced_snapshot, role="kl_penalty")

    missing_routed = FakeData(batch={}, non_tensor_batch=traced.non_tensor_batch)
    try:
        ensure_trace_survives(missing_routed, traced_snapshot, role="rollout_correction")
    except ValueError as exc:
        assert "rollout_correction must preserve Router Replay trace fields" in str(exc)
        assert "missing batch fields=['routed_experts']" in str(exc)
    else:
        raise AssertionError("post-reward transforms must not drop routed_experts")

    missing_trace_metadata = FakeData(batch=traced.batch, non_tensor_batch={})
    try:
        ensure_trace_survives(missing_trace_metadata, traced_snapshot, role="compute_advantage")
    except ValueError as exc:
        assert "compute_advantage must preserve Router Replay trace fields" in str(exc)
        assert "router_replay_trace_source" in str(exc)
    else:
        raise AssertionError("post-reward transforms must not drop R3 metadata")

    bad_conflicts = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={
            "router_replay_trace_collection_path": np.array(["router_generate_request"], dtype=object),
            "router_replay_trace_source": np.array(["routed_experts"], dtype=object),
            "router_replay_trace_preserved_through_scheduler": np.array([True], dtype=object),
            "router_replay_postprocess_wrote_trace_to_batch": np.array([True], dtype=object),
            "router_replay_trace_metadata_conflicts": np.array([["late_conflict"]], dtype=object),
        },
        length=1,
    )
    try:
        ensure_trace_survives(bad_conflicts, snapshot_trace(bad_conflicts), role="kl_penalty")
    except ValueError as exc:
        assert "router_replay_trace_metadata_conflicts must be empty" in str(exc)
    else:
        raise AssertionError("post-reward transforms must still reject R3 metadata conflicts")


def test_verl_ppo_trainer_r3_reward_extra_infos_do_not_override_rollout_trace_static():
    text = PPO_TRAINER.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_REWARD_EXTRA_FORBIDDEN_NON_TENSOR_FIELDS" in text
    assert "def _ensure_reward_extra_infos_preserves_router_replay_source(" in text
    assert "reward_extra_infos_dict must not return router replay trace fields" in text
    assert "reward extras may add ordinary " in text
    assert "metrics but must not overwrite replay trace side channels" in text
    assert "_ensure_reward_extra_infos_preserves_router_replay_source(" in text
    assert (
        "_ensure_reward_extra_infos_preserves_router_replay_source(\n"
        "                                batch, reward_extra_infos_dict\n"
        "                            )\n"
        "                            batch.non_tensor_batch.update"
    ) in text

    tree = ast.parse(text)
    support_names = {
        "ROUTER_REPLAY_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_TRACE_METADATA_CONFLICTS_FIELD",
        "ROUTER_REPLAY_SOURCE_TRACE_BATCH_FIELDS",
        "ROUTER_REPLAY_SOURCE_TRACE_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_LOG_PROB_OUTPUT_FORBIDDEN_NON_TENSOR_FIELDS",
        "ROUTER_REPLAY_REWARD_EXTRA_FORBIDDEN_NON_TENSOR_FIELDS",
        "_has_router_replay_source_trace",
        "_ensure_reward_extra_infos_preserves_router_replay_source",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(PPO_TRAINER), "exec"), namespace)
    ensure_reward_extras_preserve_source = namespace[
        "_ensure_reward_extra_infos_preserves_router_replay_source"
    ]

    class FakeData:
        def __init__(self, batch, non_tensor_batch=None):
            self.batch = batch
            self.non_tensor_batch = {} if non_tensor_batch is None else dict(non_tensor_batch)

    rollout_source = FakeData(batch={"routed_experts": object()})
    ensure_reward_extras_preserve_source(
        rollout_source,
        {"reward_model_score": [1.0, 0.5]},
    )
    ensure_reward_extras_preserve_source(
        FakeData(batch={}),
        {"routed_experts_digest": ["sha256:" + "a" * 64]},
    )

    for bad_key in (
        "routed_experts_digest",
        "router_replay_trace_source",
        "router_replay_trace_metadata_conflicts",
        "routed_experts_segments",
    ):
        try:
            ensure_reward_extras_preserve_source(
                rollout_source,
                {bad_key: ["conflicting reward-side value"]},
            )
        except ValueError as exc:
            assert "reward_extra_infos_dict must not return router replay trace fields" in str(exc)
            assert "source of truth" in str(exc)
            assert f"non-tensor fields=['{bad_key}']" in str(exc)
        else:
            raise AssertionError(f"reward extra {bad_key} must not overwrite R3 trace")


def test_verl_separation_trainer_r3_preserves_rollout_trace_static():
    text = SEPARATION_TRAINER.read_text(encoding="utf-8")

    for helper in (
        "_ensure_data_proto_output_preserves_router_replay_source",
        "_ensure_log_prob_output_preserves_router_replay_source",
        "_ensure_reward_extra_infos_preserves_router_replay_source",
        "_ensure_router_replay_trace_fields_survive",
        "_ensure_router_replay_trace_metadata",
        "_snapshot_router_replay_trace_fields",
    ):
        assert helper in text

    assert (
        "batch = batch.union(gen_batch_output)\n"
        "        _ensure_router_replay_trace_metadata(batch)"
    ) in text
    assert (
        "self._balance_batch(batch, metrics=metrics)\n"
        "            _ensure_router_replay_trace_metadata(batch)"
    ) in text
    assert text.count("_ensure_data_proto_output_preserves_router_replay_source(") >= 3
    assert text.count("_ensure_log_prob_output_preserves_router_replay_source(") >= 2
    assert (
        "_ensure_log_prob_output_preserves_router_replay_source(\n"
        "                    batch,\n"
        "                    old_log_prob,\n"
        '                    role="old_log_prob",'
    ) in text
    assert text.index(
        "_ensure_log_prob_output_preserves_router_replay_source(\n"
        "                    batch,\n"
        "                    old_log_prob,"
    ) < text.index(
        'if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:'
    )
    assert (
        "_ensure_log_prob_output_preserves_router_replay_source(\n"
        "                    batch,\n"
        "                    ref_log_prob,\n"
        '                    role="ref_log_prob",'
    ) in text
    assert (
        "_ensure_data_proto_output_preserves_router_replay_source(\n"
        "                    batch,\n"
        "                    batch_reward,\n"
        '                    role="reward_model",'
    ) in text
    assert (
        "_ensure_data_proto_output_preserves_router_replay_source(\n"
        "                    batch,\n"
        "                    values,\n"
        '                    role="critic_values",'
    ) in text
    assert (
        "_ensure_reward_extra_infos_preserves_router_replay_source(\n"
        "                    batch, reward_extra_infos_dict\n"
        "                )\n"
        "                batch.non_tensor_batch.update"
    ) in text
    assert text.count("router_replay_trace_snapshot = _snapshot_router_replay_trace_fields(batch)") >= 3
    assert 'role="kl_penalty"' in text
    assert 'role="rollout_correction"' in text
    assert 'role="compute_advantage"' in text
    assert text.index('role="kl_penalty"') < text.index("metrics.update(kl_metrics)")
    assert text.index('role="rollout_correction"') < text.index("metrics.update(is_metrics)")
    assert text.index('role="compute_advantage"') > text.index("batch = compute_advantage(")


def test_verl_rollout_correction_bypass_mode_preserves_r3_trace_static():
    text = ROLLOUT_CORR_HELPER.read_text(encoding="utf-8")

    assert "ROUTER_REPLAY_BYPASS_PRESERVED_BATCH_FIELDS" in text
    assert "ROUTER_REPLAY_BYPASS_PRESERVED_NON_TENSOR_FIELDS" in text
    assert "def _snapshot_router_replay_bypass_preserved_fields(" in text
    assert "def _ensure_router_replay_bypass_preserved_fields(" in text
    assert "router_replay_trace_snapshot = _snapshot_router_replay_bypass_preserved_fields(batch)" in text
    assert "_ensure_router_replay_bypass_preserved_fields(batch, router_replay_trace_snapshot)" in text
    assert '"routed_experts_segments"' in text
    assert '"routed_experts_digest"' in text
    assert '"router_replay_trace_collection_path"' in text
    assert '"router_replay_trace_source"' in text
    assert '"router_replay_trace_metadata_conflicts"' in text
    assert "bypass_mode must preserve Router Replay trace fields" in text

    tree = ast.parse(text)
    support_names = {
        "ROUTER_REPLAY_BYPASS_PRESERVED_BATCH_FIELDS",
        "ROUTER_REPLAY_BYPASS_PRESERVED_NON_TENSOR_FIELDS",
        "_snapshot_router_replay_bypass_preserved_fields",
        "_ensure_router_replay_bypass_preserved_fields",
        "apply_bypass_mode",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"Any": object}
    exec(compile(module, str(ROLLOUT_CORR_HELPER), "exec"), namespace)

    snapshot_trace = namespace["_snapshot_router_replay_bypass_preserved_fields"]
    ensure_trace_preserved = namespace["_ensure_router_replay_bypass_preserved_fields"]
    apply_bypass_mode = namespace["apply_bypass_mode"]

    class FakeData:
        def __init__(self, batch, non_tensor_batch=None):
            self.batch = dict(batch)
            self.non_tensor_batch = {} if non_tensor_batch is None else dict(non_tensor_batch)

    @contextlib.contextmanager
    def fake_open_dict(config):
        yield config

    fake_omegaconf = types.ModuleType("omegaconf")
    fake_omegaconf.open_dict = fake_open_dict
    old_omegaconf = sys.modules.get("omegaconf")
    sys.modules["omegaconf"] = fake_omegaconf
    try:
        rollout_log_probs = object()
        routed_experts = object()
        routed_experts_segments = object()
        trace_conflicts = object()
        batch = FakeData(
            batch={
                "rollout_log_probs": rollout_log_probs,
                "routed_experts": routed_experts,
            },
            non_tensor_batch={
                "routed_experts_segments": routed_experts_segments,
                "routed_experts_digest": "sha256:" + "a" * 64,
                "router_replay_layout": "true_tokens",
                "router_replay_trace_collection_path": "router_generate_request",
                "router_replay_trace_source": "routed_experts",
                "router_replay_trace_preserved_through_scheduler": True,
                "router_replay_postprocess_wrote_trace_to_batch": True,
                "router_replay_trace_metadata_conflicts": trace_conflicts,
            },
        )
        rollout_corr_config = object()
        policy_loss_config = {}

        apply_bypass_mode(batch, rollout_corr_config, policy_loss_config)

        assert batch.batch["old_log_probs"] is rollout_log_probs
        assert batch.batch["routed_experts"] is routed_experts
        assert batch.non_tensor_batch["routed_experts_segments"] is routed_experts_segments
        assert batch.non_tensor_batch["router_replay_trace_metadata_conflicts"] is trace_conflicts
        assert policy_loss_config["rollout_correction"] is rollout_corr_config
        assert policy_loss_config["loss_mode"] == "bypass_mode"
    finally:
        if old_omegaconf is None:
            del sys.modules["omegaconf"]
        else:
            sys.modules["omegaconf"] = old_omegaconf

    trace_source = FakeData(
        batch={"routed_experts": object()},
        non_tensor_batch={"routed_experts_digest": "sha256:" + "b" * 64},
    )
    trace_snapshot = snapshot_trace(trace_source)
    ensure_trace_preserved(trace_source, trace_snapshot)

    trace_source.batch["routed_experts"] = object()
    try:
        ensure_trace_preserved(trace_source, trace_snapshot)
    except ValueError as exc:
        assert "bypass_mode must preserve Router Replay trace fields" in str(exc)
        assert "batch.routed_experts" in str(exc)
    else:
        raise AssertionError("bypass mode guard must reject replaced rollout routed_experts")

    trace_source = FakeData(
        batch={},
        non_tensor_batch={"routed_experts_digest": "sha256:" + "c" * 64},
    )
    trace_snapshot = snapshot_trace(trace_source)
    del trace_source.non_tensor_batch["routed_experts_digest"]
    try:
        ensure_trace_preserved(trace_source, trace_snapshot)
    except ValueError as exc:
        assert "missing fields=['non_tensor_batch.routed_experts_digest']" in str(exc)
    else:
        raise AssertionError("bypass mode guard must reject dropped replay side channels")


def test_verl_sglang_rollout_r3_capability_contract_static():
    text = SGLANG_ROLLOUT_SERVER.read_text(encoding="utf-8")

    assert "if self.config.enable_rollout_routing_replay:" in text
    assert 'version.parse("0.5.6.post3")' in text
    assert "enable_rollout_routing_replay requires sglang >= 0.5.6.post3" in text
    assert 'args.update({"enable_return_routed_experts": True})' in text
    assert 'request.update({"return_routed_experts": True})' in text
    assert (
        "from sglang.srt.layers.moe.routed_experts_capturer "
        "import extract_routed_experts_from_meta_info"
    ) in text
    assert "def _shape_sglang_routed_experts(" in text
    assert "def _routed_experts_shape_from_hf_config(" in text
    assert "def _compact_sglang_routed_experts_dtype(" in text
    assert "def _sglang_routed_experts_digest(" in text
    assert "def _sglang_routed_experts_trace_extra_fields(" in text
    assert "_ROUTED_EXPERTS_DIGEST_VERSION" in text
    assert "hashlib.sha256()" in text
    assert '"routed_experts_digest": _sglang_routed_experts_digest(routed_experts)' in text
    assert "torch.as_tensor(routed_experts)" in text
    assert "torch.uint8" in text
    assert "torch.int16" in text
    assert "RouterReplay compact storage range" in text
    assert "SGLang routed_experts must contain integer expert ids" in text
    assert '("num_experts", "n_routed_experts", "moe_num_experts", "num_local_experts")' in text
    assert "SGLang routed_experts expert ids must be non-negative" in text
    assert "SGLang routed_experts expert ids must be less than hf_config num_experts" in text
    assert "SGLang routed_experts last dimension must match hf_config.num_experts_per_tok" in text
    assert "SGLang routed_experts layer dimension must match hf_config.num_hidden_layers" in text
    assert "num_hidden_layers * num_experts_per_tok" in text
    assert "_shape_sglang_routed_experts(" in text
    assert "extract_routed_experts_from_meta_info(output)" in text
    assert "SGLang response is missing " in text
    assert "routed_experts. Verify that the server was launched with " in text
    assert "enable_return_routed_experts=True" in text
    assert "return_routed_experts=True" in text
    assert "'num_hidden_layers' or 'num_experts_per_tok'" in text
    assert "TokenOutput(" in text
    assert "routed_experts=routed_experts" in text
    assert "extra_fields.update(" in text
    assert '"routed_experts_dtype": str(routed_experts.dtype).removeprefix("torch.")' in text
    assert '"routed_experts_shape": shape' in text
    assert '"responses_with_routed_experts": 1' in text
    assert '"enable_rollout_routing_replay": True' in text


def test_verl_vllm_rollout_rejects_r3_router_replay_static():
    text = VLLM_ROLLOUT_SERVER.read_text(encoding="utf-8")

    assert "def _reject_router_replay_for_vllm() -> None:" in text
    assert "vLLM rollout does not support router replay R3 in VERL/MLite" in text
    assert "ROLL R3 requires SGLang rollout with routed_experts" in text
    assert "set rollout.name='sglang'" in text
    assert 'args.update({"enable_return_routed_experts": True})' not in text
    assert text.count("_reject_router_replay_for_vllm()") >= 4

    tree = ast.parse(text)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_reject_router_replay_for_vllm"
    )
    module = ast.Module(body=[_strip_annotations(copy.deepcopy(helper))], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(VLLM_ROLLOUT_SERVER), "exec"), namespace)
    try:
        namespace["_reject_router_replay_for_vllm"]()
    except NotImplementedError as exc:
        assert "SGLang rollout with routed_experts" in str(exc)
    else:
        raise AssertionError("vLLM rollout must reject R3 router replay.")


def test_verl_sglang_rollout_r3_shape_helper_executes_without_sglang_import():
    tree = ast.parse(SGLANG_ROLLOUT_SERVER.read_text(encoding="utf-8"))
    helper_names = {
        "_compact_sglang_routed_experts_dtype",
        "_routed_experts_shape_from_hf_config",
        "_sglang_routed_experts_digest",
        "_sglang_routed_experts_trace_extra_fields",
        "_shape_sglang_routed_experts",
    }
    helper_defs = [
        _strip_annotations(copy.deepcopy(node))
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(module)

    class FakeScalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeTensor:
        def __init__(self, shape, *, dtype="int64", min_value=0, max_value=1, contiguous=False):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.min_value = min_value
            self.max_value = max_value
            self.contiguous_called = contiguous

        def numel(self):
            out = 1
            for dim in self.shape:
                out *= dim
            return out

        def dim(self):
            return len(self.shape)

        def min(self):
            return FakeScalar(self.min_value)

        def max(self):
            return FakeScalar(self.max_value)

        def reshape(self, *shape):
            inferred = list(shape)
            if -1 in inferred:
                known = 1
                infer_idx = inferred.index(-1)
                for dim in inferred:
                    if dim != -1:
                        known *= dim
                inferred[infer_idx] = self.numel() // known
            return FakeTensor(
                inferred,
                dtype=self.dtype,
                min_value=self.min_value,
                max_value=self.max_value,
            )

        def size(self, dim):
            return self.shape[dim]

        def unbind(self, dim):
            if dim != 1:
                raise AssertionError("FakeTensor only supports unbind(dim=1)")
            return [
                FakeTensor(
                    (self.shape[0], self.shape[2]),
                    dtype=self.dtype,
                    min_value=self.min_value,
                    max_value=self.max_value,
                )
                for _ in range(self.shape[1])
            ]

        def detach(self):
            return self

        def to(self, *args, **kwargs):
            dtype = kwargs.get("dtype", args[0] if args else self.dtype)
            return FakeTensor(
                self.shape,
                dtype=dtype,
                min_value=self.min_value,
                max_value=self.max_value,
            )

        def contiguous(self):
            return FakeTensor(
                self.shape,
                dtype=self.dtype,
                min_value=self.min_value,
                max_value=self.max_value,
                contiguous=True,
            )

        def tolist(self):
            return [self.max_value for _ in range(self.numel())]

    class FakeTorch:
        bool = "bool"
        uint8 = "uint8"
        int16 = "int16"
        int64 = "int64"
        Tensor = FakeTensor

        @staticmethod
        def as_tensor(value):
            if isinstance(value, FakeTensor):
                return value
            if value == "float":
                return FakeTensor((1, 2, 2), dtype="float32")
            return FakeTensor(value)

        @staticmethod
        def is_floating_point(tensor):
            return str(tensor.dtype).startswith("float")

        @staticmethod
        def is_complex(tensor):
            return str(tensor.dtype).startswith("complex")

    namespace = {
        "Any": object,
        "_ROUTED_EXPERTS_DIGEST_VERSION": b"mlite-router-replay-routed-experts-v1",
        "hashlib": __import__("hashlib"),
        "torch": FakeTorch,
    }
    exec(compile(module, str(SGLANG_ROLLOUT_SERVER), "exec"), namespace)
    shape = namespace["_shape_sglang_routed_experts"]
    trace_fields = namespace["_sglang_routed_experts_trace_extra_fields"]
    hf_config = types.SimpleNamespace(num_hidden_layers=3, num_experts_per_tok=2, num_experts=8)

    shaped = shape(FakeTensor((4, 3, 2), max_value=7), hf_config, request_id="ok")
    assert shaped.shape == (4, 3, 2)
    assert shaped.dtype == "uint8"
    assert shaped.contiguous_called is True
    shaped_trace = trace_fields(shaped, sglang_version="0.5.6.post3")
    shaped_digest = shaped_trace.pop("routed_experts_digest")
    assert shaped_digest.startswith("sha256:")
    assert len(shaped_digest) == len("sha256:") + 64
    assert shaped_trace == {
        "rollout_backend": "sglang",
        "sglang_version": "0.5.6.post3",
        "sglang_supports_routed_experts": True,
        "enable_return_routed_experts": True,
        "return_routed_experts": True,
        "enable_rollout_routing_replay": True,
        "responses_with_routed_experts": 1,
        "routed_experts_dtype": "uint8",
        "routed_experts_shape": [4, 3, 2],
        "max_expert_id": 7,
        "routers": 3,
        "topk": 2,
        "valid_tokens": 4,
        "batch_size": 1,
    }

    wide_hf_config = types.SimpleNamespace(num_hidden_layers=3, num_experts_per_tok=2, num_experts=512)
    flattened_by_layer = shape(FakeTensor((12, 2), max_value=256), wide_hf_config, request_id="ok")
    assert flattened_by_layer.shape == (4, 3, 2)
    assert flattened_by_layer.dtype == "int16"
    assert flattened_by_layer.contiguous_called is True
    assert trace_fields(flattened_by_layer, sglang_version="0.5.6.post3")[
        "routed_experts_dtype"
    ] == "int16"

    alt_count_config = types.SimpleNamespace(
        num_hidden_layers=3,
        num_experts_per_tok=2,
        num_experts=None,
        n_routed_experts=8,
    )
    shaped_with_alt_count = shape(FakeTensor((4, 3, 2), max_value=7), alt_count_config, request_id="ok")
    assert shaped_with_alt_count.shape == (4, 3, 2)

    for bad, expected in [
        (None, "SGLang response is missing"),
        (FakeTensor((4, 3, 2), min_value=-1, max_value=7), "expert ids must be non-negative"),
        (FakeTensor((4, 3, 2), max_value=8), "less than hf_config num_experts"),
        (FakeTensor((4, 3, 1)), "last dimension must match"),
        (FakeTensor((4, 2, 2)), "layer dimension must match"),
        (FakeTensor((5, 2)), "element count must be divisible"),
        ("float", "integer expert ids"),
    ]:
        try:
            shape(bad, hf_config, request_id="bad")
        except (TypeError, ValueError) as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"expected bad routed_experts to fail: {bad!r}")

    try:
        shape(
            FakeTensor((4, 3, 2), max_value=32768),
            types.SimpleNamespace(num_hidden_layers=3, num_experts_per_tok=2, num_experts=40000),
            request_id="bad",
        )
    except ValueError as exc:
        assert "compact storage range" in str(exc)
    else:
        raise AssertionError("expert ids outside RouterReplay compact range must fail")

    try:
        shape(FakeTensor((4, 3, 2)), types.SimpleNamespace(num_hidden_layers=3), request_id="bad")
    except AttributeError as exc:
        assert "num_hidden_layers' or 'num_experts_per_tok" in str(exc)
    else:
        raise AssertionError("missing hf_config MoE fields must fail")

    try:
        shape(
            FakeTensor((4, 3, 4), max_value=3),
            types.SimpleNamespace(num_hidden_layers=3, num_experts_per_tok=4, num_experts=3),
            request_id="bad",
        )
    except ValueError as exc:
        assert "num_experts_per_tok <= num_experts" in str(exc)
    else:
        raise AssertionError("topk larger than total expert count must fail")


def test_verl_padding_r3_marks_unpadded_routed_experts_as_true_tokens_static():
    text = PADDING_UTILS.read_text(encoding="utf-8")

    assert "_ROUTER_REPLAY_SCALAR_NON_TENSOR_KEYS" in text
    assert '"routed_experts_digest"' in text
    assert '"router_replay_trace_collection_path"' in text
    assert '"router_replay_trace_source"' in text
    assert '"router_replay_trace_preserved_through_scheduler"' in text
    assert '"router_replay_trace_metadata_conflicts"' in text
    assert "def _collapse_uniform_non_tensor_stack_fields" in text
    assert "must be uniform across a router replay micro-batch" in text
    assert "_ROUTER_REPLAY_SEGMENT_NON_TENSOR_STACK_KEYS" in text
    assert "_ROUTER_REPLAY_COMPACT_MAX_EXPERT_ID = 32767" in text
    assert "def _collapse_router_replay_segment_stack_fields" in text
    assert "batch-level router replay segment data" in text
    assert "def _compact_router_replay_routed_experts(" in text
    assert "routed_experts must contain integer expert ids" in text
    assert "routed_experts expert ids must be non-negative" in text
    assert "RouterReplay compact storage range" in text
    assert "routed_experts = data.get(\"routed_experts\", None)" in text
    assert "routed_experts = _compact_router_replay_routed_experts(routed_experts)" in text
    assert "routed_experts_rmpad = index_first_axis(" in text
    assert "torch.nested.nested_tensor_from_jagged(" in text
    assert "data[\"routed_experts\"] = routed_experts_nested" in text
    assert 'tu.assign_non_tensor_data(data, "router_replay_layout", "true_tokens")' in text
    assert (
        "_collapse_uniform_non_tensor_stack_fields(data, _ROUTER_REPLAY_SCALAR_NON_TENSOR_KEYS)"
        in text
    )
    assert 'elif data.get("routed_experts_segments", None) is not None:' in text
    assert "_collapse_router_replay_segment_stack_fields(data)" in text

    tree = ast.parse(text)
    support_names = {
        "_ROUTER_REPLAY_SCALAR_NON_TENSOR_KEYS",
        "_ROUTER_REPLAY_SEGMENT_NON_TENSOR_STACK_KEYS",
        "_collapse_uniform_non_tensor_stack_fields",
        "_router_replay_values_equal",
        "_collapse_router_replay_segment_stack_fields",
    }
    support_defs = (
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in support_names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in support_names)
    )
    module = ast.Module(
        body=[_strip_annotations(copy.deepcopy(node)) for node in support_defs],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    class FakeTU:
        @staticmethod
        def get(data, key, default=None):
            return data.get(key, default)

        @staticmethod
        def assign_non_tensor_data(data, key, value):
            data[key] = value

    class FakeTensor:
        def __init__(self, value, *, dtype="long", shape=(1,)):
            self.value = value
            self.dtype = dtype
            self.shape = shape

    class FakeTorch:
        Tensor = FakeTensor

        @staticmethod
        def equal(left, right):
            return left.value == right.value

    class FakeNonTensorStack:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    namespace = {"tu": FakeTU, "torch": FakeTorch, "NonTensorStack": FakeNonTensorStack}
    exec(compile(module, str(PADDING_UTILS), "exec"), namespace)
    collapse = namespace["_collapse_uniform_non_tensor_stack_fields"]
    collapse_segments = namespace["_collapse_router_replay_segment_stack_fields"]
    keys = namespace["_ROUTER_REPLAY_SCALAR_NON_TENSOR_KEYS"]

    data = {
        "router_replay_trace_collection_path": ["router_generate_request", "router_generate_request"],
        "router_replay_trace_source": ["routed_experts", "routed_experts"],
        "router_replay_trace_preserved_through_scheduler": [True, True],
        "routed_experts_digest": ["sha256:" + "a" * 64, "sha256:" + "a" * 64],
        "router_replay_trace_metadata_conflicts": [[], []],
    }
    collapse(data, keys)
    assert data["router_replay_trace_collection_path"] == "router_generate_request"
    assert data["router_replay_trace_source"] == "routed_experts"
    assert data["router_replay_trace_preserved_through_scheduler"] is True
    assert data["routed_experts_digest"] == "sha256:" + "a" * 64
    assert data["router_replay_trace_metadata_conflicts"] == []

    bad = {"router_replay_trace_source": ["routed_experts", "routed_experts_segments"]}
    try:
        collapse(bad, keys)
    except ValueError as exc:
        assert "router_replay_trace_source must be uniform" in str(exc)
    else:
        raise AssertionError("mixed router replay metadata must be rejected")

    bad_conflicts = {"router_replay_trace_metadata_conflicts": [[], ["sglang_version"]]}
    try:
        collapse(bad_conflicts, keys)
    except ValueError as exc:
        assert "router_replay_trace_metadata_conflicts must be uniform" in str(exc)
    else:
        raise AssertionError("mixed router replay trace conflict metadata must be rejected")

    segment = FakeTensor("segment", shape=(1, 2, 1, 1))
    seq_lens = FakeTensor("seq_lens", shape=(1,))
    segment_data = {
        "routed_experts_segments": FakeNonTensorStack([[segment], [segment]]),
        "routed_experts_segment_seq_lens": FakeNonTensorStack([[seq_lens], [seq_lens]]),
        "routed_experts_num_routers": FakeNonTensorStack([1, 1]),
    }
    collapse_segments(segment_data)
    assert segment_data["routed_experts_segments"] == [segment]
    assert segment_data["routed_experts_segment_seq_lens"] == [seq_lens]
    assert segment_data["routed_experts_num_routers"] == 1

    bad_segments = {
        "routed_experts_segments": FakeNonTensorStack(
            [[FakeTensor("first")], [FakeTensor("second")]]
        )
    }
    try:
        collapse_segments(bad_segments)
    except ValueError as exc:
        assert "routed_experts_segments must be stored as batch-level" in str(exc)
    else:
        raise AssertionError("mixed per-sample router replay segments must be rejected")
