"""Stop/cancellation semantics for the agent runtime (#6).

A webui operator must be able to stop a running turn. Cancelling the
consumer task mid-`astream` used to leave an assistant ``tool_use`` batch
unanswered, permanently 400-ing the session on the next turn. The runtime
now answers every unanswered tool call on cancellation, and exposes
``discard_interrupt`` for the "paused awaiting approve" stop path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from clearwing.agent.runtime import Command, NativeAgentGraph
from clearwing.agent.tooling import interrupt, tool
from clearwing.llm.messages import AIMessage, ToolMessage


@tool(name="slow_tool")
async def slow_tool(x: str) -> dict:
    """Async tool that never finishes within a test's lifetime."""
    await asyncio.sleep(30)
    return {"status": "done"}


@tool(name="gated_tool")
def gated_tool() -> dict:
    """Tool guarded by a human-approval gate."""
    if not interrupt("Approve gated op?"):
        return {"gated": "denied"}
    return {"gated": "approved"}


def _call(name: str, call_id: str, args: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(fn_name=name, call_id=call_id, fn_arguments=args or {})


def _graph(extra_tools: list) -> NativeAgentGraph:
    return NativeAgentGraph(
        llm=object(),
        native_tools=[],
        tools=extra_tools,
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
    )


def _stub_assistant_tool_calls(graph: NativeAgentGraph, calls: list) -> None:
    """Make the next assistant step request *calls* (raw genai ToolCalls)."""

    async def fake_step(st):
        st.setdefault("messages", []).append(
            AIMessage(content="running tools", tool_calls=list(calls))
        )
        return {}

    graph._aassistant_step = fake_step


def _stub_assistant_done(graph: NativeAgentGraph) -> None:
    async def fake_step(st):
        st.setdefault("messages", []).append(AIMessage(content="done"))
        return {}

    graph._aassistant_step = fake_step


class TestCancelDuringToolBatch:
    @pytest.mark.asyncio
    async def test_cancel_answers_dangling_tool_calls(self):
        graph = _graph([slow_tool])
        config = {"configurable": {"thread_id": "stop-cancel-1"}}
        _stub_assistant_tool_calls(graph, [_call("slow_tool", "c1", {"x": "1"})])

        first_event = asyncio.Event()

        async def consume():
            async for _ in graph.astream({"messages": []}, config):
                first_event.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(first_event.wait(), timeout=5)
        await asyncio.sleep(0.1)  # tool batch is now awaiting slow_tool
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        state = graph.get_state(config).values
        messages = state["messages"]
        assert isinstance(messages[-1], ToolMessage)
        assert messages[-1].tool_call_id == "c1"
        assert "stopped by operator" in messages[-1].content

        # The next turn must not 400 on orphaned tool_use: a fresh assistant
        # step completes normally.
        _stub_assistant_done(graph)
        async for _ in graph.astream(
            {"messages": [{"role": "user", "content": "next"}]}, config
        ):
            pass


class TestDiscardInterrupt:
    @pytest.mark.asyncio
    async def test_discard_answers_pending_approval_batch(self):
        graph = _graph([gated_tool])
        config = {"configurable": {"thread_id": "stop-discard-1"}}
        _stub_assistant_tool_calls(graph, [_call("gated_tool", "c1")])

        async for _ in graph.astream({"messages": []}, config):
            pass  # pauses on the approval gate

        assert graph.get_state(config).next == ("tools",)

        assert graph.discard_interrupt(config) is True
        assert graph.get_state(config).next == ()
        messages = graph.get_state(config).values["messages"]
        assert isinstance(messages[-1], ToolMessage)
        assert messages[-1].tool_call_id == "c1"
        assert "stopped by operator" in messages[-1].content

        # Session stays usable afterwards.
        _stub_assistant_done(graph)
        async for _ in graph.astream(
            {"messages": [{"role": "user", "content": "next"}]}, config
        ):
            pass

    @pytest.mark.asyncio
    async def test_discard_on_idle_thread_is_a_no_op(self):
        graph = _graph([gated_tool])
        config = {"configurable": {"thread_id": "stop-discard-2"}}
        assert graph.discard_interrupt(config) is False

    @pytest.mark.asyncio
    async def test_discard_after_explicit_deny_is_no_op(self):
        graph = _graph([gated_tool])
        config = {"configurable": {"thread_id": "stop-discard-3"}}
        _stub_assistant_tool_calls(graph, [_call("gated_tool", "c1")])

        async for _ in graph.astream({"messages": []}, config):
            pass  # pause
        _stub_assistant_done(graph)
        async for _ in graph.astream(Command(resume=False), config):
            pass  # deny resolves the batch

        assert graph.discard_interrupt(config) is False
