"""Regression tests for the approval pause/resume event flow.

docs/web-api.md documents an `approval_needed` event and the web UI's
WebSocket client renders Approve/Deny buttons from it, but unlike the
TUI/CLI the web client has no get_state() polling fallback — a missing
emit leaves the session silently hung at the approval gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from clearwing.agent.runtime import Command, NativeAgentGraph
from clearwing.agent.tooling import interrupt, tool
from clearwing.core.events import EventBus, EventType

APPROVAL_PROMPT = "Approve demo destructive op on demo-target?"


@tool(name="needs_approval")
def needs_approval(target: str) -> dict:
    """Demo tool that gates on human approval."""
    if not interrupt(APPROVAL_PROMPT):
        return {"status": "denied", "target": target}
    return {"status": "approved", "target": target}


def _tool_call() -> SimpleNamespace:
    return SimpleNamespace(
        fn_name="needs_approval", call_id="call-1", fn_arguments={"target": "demo-target"}
    )


def _graph() -> NativeAgentGraph:
    return NativeAgentGraph(
        llm=object(),
        native_tools=[],
        tools=[needs_approval],
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


class TestApprovalNeededEvent:
    @pytest.mark.asyncio
    async def test_pause_emits_approval_needed(self):
        graph = _graph()
        config = {"configurable": {"thread_id": "t1"}}
        state = graph._get_or_create_state("t1")

        received: list[dict] = []

        def handler(data):
            received.append(data)

        bus = EventBus()
        bus.subscribe(EventType.APPROVAL_NEEDED, handler)
        try:
            events, paused = await graph._arun_tool_calls(
                state, [_tool_call()], resume_decision=...
            )
        finally:
            bus.unsubscribe(EventType.APPROVAL_NEEDED, handler)

        assert paused is True
        assert events == []
        assert len(received) == 1
        assert received[0]["prompt"] == APPROVAL_PROMPT
        assert received[0]["tool"] == "needs_approval"

        # In-process clients (TUI/CLI) keep discovering the pause via get_state().
        snapshot = graph.get_state(config)
        assert snapshot.next == ("tools",)
        assert snapshot.tasks[0].interrupts[0].value == APPROVAL_PROMPT

    @pytest.mark.asyncio
    async def test_resume_true_approves_and_completes(self):
        graph = _graph()
        config = {"configurable": {"thread_id": "t2"}}
        state = graph._get_or_create_state("t2")

        emitted: list[dict] = []

        def handler(data):
            emitted.append(data)

        bus = EventBus()
        bus.subscribe(EventType.APPROVAL_NEEDED, handler)
        try:
            _, paused = await graph._arun_tool_calls(state, [_tool_call()], resume_decision=...)
            assert paused is True
            assert len(emitted) == 1

            _stub_assistant_step(graph)
            resumed = [event async for event in graph.astream(Command(resume=True), config)]
        finally:
            bus.unsubscribe(EventType.APPROVAL_NEEDED, handler)

        # The approved decision path must not raise InterruptRequest again,
        # so exactly one approval event fires for the whole exchange.
        assert len(emitted) == 1
        assert resumed
        assert graph._pending.get("t2") is None
        assert graph.get_state(config).next == ()
        tool_messages = [m for m in state["messages"] if getattr(m, "name", "") == "needs_approval"]
        assert len(tool_messages) == 1
        assert "approved" in tool_messages[0].content

    @pytest.mark.asyncio
    async def test_resume_false_denies(self):
        graph = _graph()
        config = {"configurable": {"thread_id": "t3"}}
        state = graph._get_or_create_state("t3")

        _, paused = await graph._arun_tool_calls(state, [_tool_call()], resume_decision=...)
        assert paused is True

        _stub_assistant_step(graph)
        resumed = [event async for event in graph.astream(Command(resume=False), config)]

        assert resumed
        assert graph._pending.get("t3") is None
        tool_messages = [m for m in state["messages"] if getattr(m, "name", "") == "needs_approval"]
        assert len(tool_messages) == 1
        assert "denied" in tool_messages[0].content
