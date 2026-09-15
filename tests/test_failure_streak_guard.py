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
        # Fourth call (2 * bound) halts the batch: only 3 tool results recorded,
        # the halt notice is appended, and the loop is told to stop.
        assert halted is True
        assert paused is False
        assert sum(1 for m in state["messages"] if getattr(m, "name", None) == "always_fails") == 3
        assert isinstance(state["messages"][-1], HumanMessage)
        assert "Automatic stop" in state["messages"][-1].content

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
