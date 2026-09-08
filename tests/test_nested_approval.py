"""Regression tests for nested tool approval.

`AgentTool.ainvoke` used to open a fresh `tool_execution_context()`, which
reset the enclosing resume decision. A nested tool invoked by an
already-approved parent then re-raised `InterruptRequest`, discarding the
approval and pausing again — the client-approved run looped forever on
resume (every approve produced another approval_needed for the nested
gate).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from clearwing.agent.runtime import Command, NativeAgentGraph
from clearwing.agent.tooling import (
    InterruptRequest,
    interrupt,
    tool,
    tool_execution_context,
)
from clearwing.core.events import EventBus, EventType

OUTER_PROMPT = "Approve outer op that delegates to a nested gated tool?"
INNER_PROMPT = "Approve nested op?"


@tool(name="inner_gated")
def inner_gated() -> dict:
    """Nested tool with its own approval gate."""
    if not interrupt(INNER_PROMPT):
        return {"inner": "denied"}
    return {"inner": "approved"}


@tool(name="outer_gated_wrapper")
def outer_gated_wrapper() -> dict:
    """Gated tool that invokes another gated tool while running."""
    if not interrupt(OUTER_PROMPT):
        return {"outer": "denied"}
    nested = inner_gated.invoke()
    return {"outer": "approved", "nested": nested}


def _outer_tool_call() -> SimpleNamespace:
    return SimpleNamespace(fn_name="outer_gated_wrapper", call_id="call-outer", fn_arguments={})


def _graph() -> NativeAgentGraph:
    return NativeAgentGraph(
        llm=object(),
        native_tools=[],
        tools=[outer_gated_wrapper, inner_gated],
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
        enable_event_bus=True,
        enable_context_summarizer=False,
    )


def _stub_assistant_step(graph: NativeAgentGraph) -> None:
    class _Msg:
        content = "done"
        tool_calls: list = []

    async def fake_step(st):
        st["messages"].append(_Msg())
        return {}

    graph._aassistant_step = fake_step


class TestNestedDecisionInheritance:
    @pytest.mark.asyncio
    async def test_ainvoke_inherits_approved_decision(self):
        with tool_execution_context(resume_decision=True):
            assert await inner_gated.ainvoke({}) == {"inner": "approved"}

    @pytest.mark.asyncio
    async def test_ainvoke_inherits_denied_decision(self):
        with tool_execution_context(resume_decision=False):
            assert await inner_gated.ainvoke({}) == {"inner": "denied"}

    @pytest.mark.asyncio
    async def test_ainvoke_outside_tool_context_still_interrupts(self):
        with pytest.raises(InterruptRequest) as excinfo:
            await inner_gated.ainvoke({})
        assert excinfo.value.prompt == INNER_PROMPT

    @pytest.mark.asyncio
    async def test_ainvoke_inside_undecided_context_still_interrupts(self):
        # First pass (no decision yet): the nested gate must still pause
        # rather than silently auto-approve by inheriting "unset".
        with tool_execution_context():
            with pytest.raises(InterruptRequest) as excinfo:
                await inner_gated.ainvoke({})
        assert excinfo.value.prompt == INNER_PROMPT


class TestNestedApprovalAtGraphLevel:
    @pytest.mark.asyncio
    async def test_approve_once_completes_nested_exchange(self):
        graph = _graph()
        config = {"configurable": {"thread_id": "nested-1"}}
        state = graph._get_or_create_state("nested-1")

        emitted: list[dict] = []

        def handler(data):
            emitted.append(data)

        bus = EventBus()
        bus.subscribe(EventType.APPROVAL_NEEDED, handler)
        try:
            _, paused = await graph._arun_tool_calls(
                state, [_outer_tool_call()], resume_decision=...
            )
            assert paused is True
            assert len(emitted) == 1
            assert emitted[0]["prompt"] == OUTER_PROMPT

            _stub_assistant_step(graph)
            async for _ in graph.astream(Command(resume=True), config):
                pass
        finally:
            bus.unsubscribe(EventType.APPROVAL_NEEDED, handler)

        # Exactly one approval for the whole exchange: approving the outer
        # gate must carry through the nested gate instead of resetting it.
        assert len(emitted) == 1
        assert graph._pending.get("nested-1") is None
        assert graph.get_state(config).next == ()
        tool_messages = [
            m for m in state["messages"] if getattr(m, "name", "") == "outer_gated_wrapper"
        ]
        assert len(tool_messages) == 1
        assert "approved" in tool_messages[0].content

    @pytest.mark.asyncio
    async def test_deny_outer_skips_nested(self):
        graph = _graph()
        config = {"configurable": {"thread_id": "nested-2"}}
        state = graph._get_or_create_state("nested-2")

        _, paused = await graph._arun_tool_calls(state, [_outer_tool_call()], resume_decision=...)
        assert paused is True

        _stub_assistant_step(graph)
        async for _ in graph.astream(Command(resume=False), config):
            pass

        assert graph._pending.get("nested-2") is None
        tool_messages = [
            m for m in state["messages"] if getattr(m, "name", "") == "outer_gated_wrapper"
        ]
        assert len(tool_messages) == 1
        assert "denied" in tool_messages[0].content
