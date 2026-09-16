"""Regression tests for issue #23: approval resume must not re-grant the
logical turn's budget guards.

`_aresume` restarts `_arun_loop` after executing a pending (approval-gated)
tool batch. Before the fix, `steps`/`tool_calls_total` were loop locals and
every pause -> approve -> resume cycle re-granted the full max_steps /
max_tool_calls budget; repeated approving could bypass both caps. The
counters now live in `NativeAgentGraph._loop_counters` (per-thread) and are
reset only when a NEW user turn starts in `astream`.

Every test here drives the pause through `astream`/`Command(resume=...)`
the way production clients do — calling `_arun_tool_calls` directly would
bypass the counting at the request site and prove nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from clearwing.agent.runtime import Command, NativeAgentGraph
from clearwing.agent.tooling import interrupt, tool
from clearwing.providers.binding import AgentLimits

APPROVAL_PROMPT = "Approve gated op on x?"


@tool(name="gated_tool")
def gated_tool(target: str) -> dict:
    """Approval-gated tool: pauses the turn via interrupt()."""
    if not interrupt(APPROVAL_PROMPT):
        return {"status": "denied", "target": target}
    return {"status": "approved", "target": target}


@tool(name="plain_tool")
def plain_tool(target: str) -> dict:
    """Ungated tool."""
    return {"status": "ok", "target": target}


def _call(name: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(fn_name=name, call_id=call_id, fn_arguments={"target": "x"})


def _graph(limits: AgentLimits | None) -> NativeAgentGraph:
    return NativeAgentGraph(
        llm=object(),
        native_tools=[],
        tools=[gated_tool, plain_tool],
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


def _scripted_step(graph: NativeAgentGraph, scripts: list[list[SimpleNamespace]]) -> dict:
    """Replace _aassistant_step with a script: the n-th step requests the
    n-th entry's tool_calls (the last entry repeats). Returns a counter dict
    so tests can assert how many assistant steps actually ran."""

    class _Msg:
        def __init__(self, calls):
            self.content = ""
            self.tool_calls = list(calls)

    idx = {"n": 0}

    async def fake_step(st):
        i = min(idx["n"], len(scripts) - 1)
        idx["n"] += 1
        st.setdefault("messages", []).append(_Msg(scripts[i]))
        return {}

    graph._aassistant_step = fake_step
    return idx


def _tool_messages(state, name):
    return [m for m in state.get("messages", []) if getattr(m, "name", "") == name]


class TestBudgetCountersAcrossResume:
    @pytest.mark.asyncio
    async def test_resume_does_not_regrant_tool_call_budget(self):
        graph = _graph(AgentLimits(max_tool_calls=1))
        config = {"configurable": {"thread_id": "t1"}}
        state = graph._get_or_create_state("t1")
        _scripted_step(graph, [[_call("gated_tool", "c1")], [_call("plain_tool", "c2")]])

        async for _ in graph.astream({}, config):
            pass
        assert graph._pending.get("t1") is not None, "gated call should pause the turn"

        async for _ in graph.astream(Command(resume=True), config):
            pass

        # The approved gated call ran (counted at request time, before the fix
        # survived only until the loop restart); the post-resume request hits
        # the exhausted budget and must be answered as skipped, never run.
        gated = _tool_messages(state, "gated_tool")
        assert len(gated) == 1 and "approved" in gated[0].content
        plain = _tool_messages(state, "plain_tool")
        assert len(plain) == 1 and "skipped" in plain[0].content
        assert "Automatic stop" in state["messages"][-1].content
        assert graph._loop_counters["t1"]["tool_calls_total"] == 1

    @pytest.mark.asyncio
    async def test_resume_does_not_regrant_steps(self):
        graph = _graph(AgentLimits(max_steps=1))
        config = {"configurable": {"thread_id": "t2"}}
        idx = _scripted_step(graph, [[_call("gated_tool", "c1")], [_call("plain_tool", "c2")]])

        async for _ in graph.astream({}, config):
            pass
        assert idx["n"] == 1 and graph._pending.get("t2") is not None

        async for _ in graph.astream(Command(resume=True), config):
            pass

        # max_steps=1 was consumed by the step that requested the gated call;
        # the resumed loop must break at the top without a second step.
        assert idx["n"] == 1, "resume re-ran an assistant step past max_steps"
        assert graph._loop_counters["t2"]["steps"] == 1
        # The pending batch was still executed and answered (provider
        # invariant: no orphaned tool_use).
        gated = _tool_messages(graph._get_or_create_state("t2"), "gated_tool")
        assert len(gated) == 1 and "approved" in gated[0].content

    @pytest.mark.asyncio
    async def test_new_user_turn_regrants_budget(self):
        graph = _graph(AgentLimits(max_tool_calls=1))
        config = {"configurable": {"thread_id": "t3"}}
        state = graph._get_or_create_state("t3")
        _scripted_step(
            graph,
            [[_call("plain_tool", f"c{i}")] for i in range(4)],
        )

        # Turn 1: first call executes, second request is budget-stopped.
        async for _ in graph.astream({}, config):
            pass
        assert len(_tool_messages(state, "plain_tool")) == 2  # one ok, one skipped
        assert "Automatic stop" in state["messages"][-1].content

        # Turn 2 (new user input): fresh logical turn, fresh budget.
        async for _ in graph.astream({"messages": []}, config):
            pass
        ok = [m for m in _tool_messages(state, "plain_tool") if '"ok"' in m.content]
        assert len(ok) == 2, "a new user turn must re-grant the tool-call budget"

    @pytest.mark.asyncio
    async def test_repeated_declines_share_turn_budget(self):
        """cicd's drive_with_auto_decline resumes with deny repeatedly within
        ONE logical turn; after the fix all declines share the original
        turn's budget instead of each restart re-granting it."""
        graph = _graph(AgentLimits(max_tool_calls=1))
        config = {"configurable": {"thread_id": "t4"}}
        state = graph._get_or_create_state("t4")
        _scripted_step(graph, [[_call("gated_tool", "c1")], [_call("gated_tool", "c2")]])

        async for _ in graph.astream({}, config):
            pass
        async for _ in graph.astream(Command(resume=False), config):
            pass

        # First gated exchange was declined for real; the model's retry
        # request is answered as skipped by the exhausted budget, so only
        # one genuine denial ever executes.
        denied = [
            m
            for m in _tool_messages(state, "gated_tool")
            if "denied" in m.content
        ]
        skipped = [
            m for m in _tool_messages(state, "gated_tool") if "skipped" in m.content
        ]
        assert len(denied) == 1
        assert len(skipped) == 1
        assert "Automatic stop" in state["messages"][-1].content

    @pytest.mark.asyncio
    async def test_operator_stop_preserves_spent_budget(self):
        graph = _graph(AgentLimits(max_tool_calls=1))
        config = {"configurable": {"thread_id": "t5"}}
        state = graph._get_or_create_state("t5")
        _scripted_step(graph, [[_call("gated_tool", "c1")], [_call("plain_tool", "c2")], []])

        async for _ in graph.astream({}, config):
            pass
        assert graph._loop_counters["t5"] == {"steps": 1, "tool_calls_total": 1}

        # Operator stop mid-pause answers the pending batch as skipped but
        # must NOT reset the counters — spent budget stays spent until a new
        # user turn.
        graph.discard_interrupt(config)
        assert graph._loop_counters["t5"] == {"steps": 1, "tool_calls_total": 1}

        # A new turn then re-grants (the plain call executes, no auto-stop).
        async for _ in graph.astream({}, config):
            pass
        plain = _tool_messages(state, "plain_tool")
        assert len(plain) == 1 and '"ok"' in plain[0].content
