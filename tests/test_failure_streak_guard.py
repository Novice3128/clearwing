"""Tests for the identical-failure streak guard in the agent runtime.

Session 932ff8ec repeated one failing tool call 293 times (~$249) because
nothing bounded consecutive identical failures. The guard nudges the model
after N failures and halts the turn at 2N.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from clearwing.agent.runtime import NativeAgentGraph
from clearwing.agent.tooling import tool
from clearwing.llm.messages import HumanMessage
from clearwing.providers.binding import AgentLimits


@tool(name="always_fails")
def always_fails(target: str) -> dict:
    """Tool that always returns an error payload."""
    return {"error": f"boom on {target}"}


@tool(name="flaky_then_ok")
def flaky_then_ok(target: str, attempt: dict) -> dict:
    """Fails unless the caller varies the target (streak breaker)."""
    if target == "bad":
        return {"error": "boom on bad"}
    return {"status": "ok", "target": target}


def _call(name: str, args: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(fn_name=name, call_id=call_id, fn_arguments=args)


def _graph(limits: AgentLimits | None = None) -> NativeAgentGraph:
    return NativeAgentGraph(
        llm=object(),
        native_tools=[],
        tools=[always_fails, flaky_then_ok],
        system_prompt_fn=lambda s: "sys",
        model_name="m",
        session_id=None,
        state_updater_fn=lambda *a, **k: {},
        knowledge_graph_populator_fn=None,
        input_guardrail_tool_names=frozenset(),
        output_guardrail_tool_names=frozenset(),
        enable_cost_tracker=False,
        enable_episodic_memory=False,
        enable_audit=False,
        enable_knowledge_graph=False,
        enable_input_guardrail=False,
        enable_output_guardrail=False,
        enable_event_bus=False,
        enable_context_summarizer=False,
        agent_limits=limits,
    )


class TestIdenticalFailureStreak:
    @pytest.mark.asyncio
    async def test_nudge_injected_at_bound(self):
        graph = _graph(AgentLimits(identical_failure_streak=2))
        state = graph._get_or_create_state("t1")

        await graph._arun_tool_calls(
            state, [_call("always_fails", {"target": "x"}, "c1")], resume_decision=...
        )
        assert not isinstance(state["messages"][-1], HumanMessage)

        events, paused, halted = await graph._arun_tool_calls(
            state, [_call("always_fails", {"target": "x"}, "c2")], resume_decision=...
        )
        assert paused is False
        assert halted is False
        assert isinstance(state["messages"][-1], HumanMessage)
        assert "always_fails" in state["messages"][-1].content
        assert events

    @pytest.mark.asyncio
    async def test_halt_at_double_bound(self):
        graph = _graph(AgentLimits(identical_failure_streak=2))
        state = graph._get_or_create_state("t1")
        calls = [_call("always_fails", {"target": "x"}, f"c{i}") for i in range(4)]

        events, paused, halted = await graph._arun_tool_calls(
            state, calls, resume_decision=...
        )
        # Fourth call (2 * bound) halts the batch, but every tool_call in the
        # batch must end up answered — unanswered tool_use 400s the next turn.
        assert halted is True
        assert paused is False
        tool_messages = [
            m for m in state["messages"] if getattr(m, "name", None) == "always_fails"
        ]
        assert len(tool_messages) == len(calls)
        answered_ids = {m.tool_call_id for m in tool_messages}
        assert answered_ids == {c.call_id for c in calls}
        assert isinstance(state["messages"][-1], HumanMessage)
        assert "Automatic stop" in state["messages"][-1].content

    @pytest.mark.asyncio
    async def test_halt_mid_batch_answers_remaining_calls(self):
        graph = _graph(AgentLimits(identical_failure_streak=1))
        state = graph._get_or_create_state("t1")
        # 5 calls, halt triggers on the 2nd (2 * bound with bound=1);
        # calls 3-5 must still receive placeholder results.
        calls = [_call("always_fails", {"target": "x"}, f"c{i}") for i in range(5)]

        _events, _paused, halted = await graph._arun_tool_calls(
            state, calls, resume_decision=...
        )
        assert halted is True
        tool_messages = [
            m for m in state["messages"] if getattr(m, "name", None) == "always_fails"
        ]
        assert {m.tool_call_id for m in tool_messages} == {c.call_id for c in calls}
        skipped = [m for m in tool_messages if "skipped" in m.content]
        assert len(skipped) == 3

    @pytest.mark.asyncio
    async def test_json_error_null_is_not_a_failure(self):
        graph = _graph(AgentLimits(identical_failure_streak=1))
        state = graph._get_or_create_state("t1")

        @tool(name="returns_null_error")
        def returns_null_error(target: str) -> dict:
            """Shape that used to be misclassified as a failure."""
            return {"error": None, "status": "ok", "target": target}

        graph.tools["returns_null_error"] = returns_null_error
        for i in range(5):
            _, _paused, halted = await graph._arun_tool_calls(
                state,
                [_call("returns_null_error", {"target": "x"}, f"c{i}")],
                resume_decision=...,
            )
            assert halted is False
        assert not any(isinstance(m, HumanMessage) for m in state["messages"])

    @pytest.mark.asyncio
    async def test_user_denial_is_not_a_failure(self):
        from clearwing.agent.runtime import _tool_result_failed

        assert not _tool_result_failed('{"success": false, "error": "Exploit denied by user"}')

    @pytest.mark.asyncio
    async def test_max_tool_calls_stop_answers_tool_calls(self):
        """Reaching max_tool_calls must not orphan the pending tool_use."""
        limits = AgentLimits(max_tool_calls=1)
        graph = _graph(limits)
        thread_id = "t1"
        state = graph._get_or_create_state(thread_id)

        class _Msg:
            def __init__(self, call_id):
                self.content = ""
                self.tool_calls = [_call("always_fails", {"target": "x"}, call_id)]

        step = {"n": 0}

        async def fake_step(st):
            step["n"] += 1
            st["messages"].append(_Msg(f"step-{step['n']}"))
            return {}

        graph._aassistant_step = fake_step

        async for _ in graph._arun_loop(thread_id):
            pass

        tool_call_ids, answered_ids = set(), set()
        for m in state["messages"]:
            if getattr(m, "tool_calls", None):
                tool_call_ids.update(tc.call_id for tc in m.tool_calls)
            if getattr(m, "role", "") == "tool":
                answered_ids.add(m.tool_call_id)
        assert tool_call_ids == answered_ids, "orphaned tool_use would 400 the next turn"

    @pytest.mark.asyncio
    async def test_varying_args_breaks_streak(self):
        graph = _graph(AgentLimits(identical_failure_streak=2))
        state = graph._get_or_create_state("t1")

        for i in range(3):
            _, paused, halted = await graph._arun_tool_calls(
                state,
                [_call("always_fails", {"target": f"host-{i}"}, f"c{i}")],
                resume_decision=...,
            )
            assert halted is False
        assert not isinstance(state["messages"][-1], HumanMessage)

    @pytest.mark.asyncio
    async def test_success_resets_streak(self):
        graph = _graph(AgentLimits(identical_failure_streak=2))
        state = graph._get_or_create_state("t1")

        await graph._arun_tool_calls(
            state, [_call("flaky_then_ok", {"target": "bad", "attempt": {}}, "c1")], resume_decision=...
        )
        _, _paused, halted = await graph._arun_tool_calls(
            state,
            [_call("flaky_then_ok", {"target": "good", "attempt": {}}, "c2")],
            resume_decision=...,
        )
        assert halted is False
        await graph._arun_tool_calls(
            state, [_call("flaky_then_ok", {"target": "bad", "attempt": {}}, "c3")], resume_decision=...
        )
        assert not isinstance(state["messages"][-1], HumanMessage)

    @pytest.mark.asyncio
    async def test_env_override_used_when_limits_none(self, monkeypatch):
        monkeypatch.setenv("CLEARWING_IDENTICAL_FAILURE_STREAK", "1")
        graph = _graph(None)
        state = graph._get_or_create_state("t1")

        _, _paused, halted = await graph._arun_tool_calls(
            state, [_call("always_fails", {"target": "x"}, "c1")], resume_decision=...
        )
        assert isinstance(state["messages"][-1], HumanMessage)
        assert "always_fails" in state["messages"][-1].content
        # Halt needs 2 * 1 = 2 consecutive failures.
        _, _paused, halted = await graph._arun_tool_calls(
            state, [_call("always_fails", {"target": "x"}, "c2")], resume_decision=...
        )
        assert halted is True

    @pytest.mark.asyncio
    async def test_zero_disables_guard(self, monkeypatch):
        monkeypatch.setenv("CLEARWING_IDENTICAL_FAILURE_STREAK", "0")
        graph = _graph(None)
        state = graph._get_or_create_state("t1")

        for i in range(10):
            _, _paused, halted = await graph._arun_tool_calls(
                state, [_call("always_fails", {"target": "x"}, f"c{i}")], resume_decision=...
            )
            assert halted is False
        assert not any(isinstance(m, HumanMessage) for m in state["messages"])


class TestStreakValidator:
    def test_zero_and_positive_accepted(self):
        from clearwing.providers.binding import validate_agent_limits

        assert validate_agent_limits("r", {"identical_failure_streak": 0}) == []
        assert validate_agent_limits("r", {"identical_failure_streak": 6}) == []

    def test_bad_types_rejected(self):
        from clearwing.providers.binding import validate_agent_limits

        assert validate_agent_limits("r", {"identical_failure_streak": "6"})
        assert validate_agent_limits("r", {"identical_failure_streak": -1})
        assert validate_agent_limits("r", {"identical_failure_streak": True})


class TestStructuredFailureShapes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            {"error": {"code": "E", "message": "boom"}},  # potentials-style
            {"status": "error", "message": "boom"},  # callback_listener-style
            {"ok": False, "error": {"code": "E"}},
        ],
    )
    async def test_structured_failures_feed_the_guard(self, payload):
        graph = _graph(AgentLimits(identical_failure_streak=2))
        state = graph._get_or_create_state("t1")

        @tool(name="structured_fail")
        def structured_fail(target: str) -> dict:
            """Returns repo-typical structured failure shapes."""
            return dict(payload)

        graph.tools["structured_fail"] = structured_fail
        calls = [
            _call("structured_fail", {"target": "x"}, f"c{i}") for i in range(4)
        ]
        _events, _paused, halted = await graph._arun_tool_calls(
            state, calls, resume_decision=...
        )
        assert halted is True

    @pytest.mark.asyncio
    async def test_batch_trimmed_to_remaining_budget(self):
        """A parallel batch larger than the remaining budget is trimmed,
        and the tail still gets answered (no orphaned tool_use)."""
        limits = AgentLimits(max_tool_calls=2)
        graph = _graph(limits)
        thread_id = "t1"
        state = graph._get_or_create_state(thread_id)

        class _Msg:
            def __init__(self):
                self.content = ""
                self.tool_calls = [
                    _call("always_fails", {"target": "x"}, f"c{i}") for i in range(4)
                ]

        async def fake_step(st):
            st["messages"].append(_Msg())
            return {}

        graph._aassistant_step = fake_step

        async for _ in graph._arun_loop(thread_id):
            pass

        tool_call_ids, answered_ids = set(), set()
        executed = 0
        for m in state["messages"]:
            if getattr(m, "tool_calls", None):
                tool_call_ids.update(tc.call_id for tc in m.tool_calls)
            if getattr(m, "role", "") == "tool":
                answered_ids.add(m.tool_call_id)
                if "skipped" not in (m.content or ""):
                    executed += 1
        assert tool_call_ids == answered_ids
        assert executed == 2  # budget cap honored inside the batch
