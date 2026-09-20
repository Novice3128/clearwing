"""Tests for the one-shot save_report delivery nudge (issue #82).

e2e scenario prompts explicitly require the report as a FILE produced with
the save_report tool, but the model regularly ends the turn with a prose
summary and zero save_report calls — the runtime generates no report on its
own and the model never sees the webui's report_path. The runtime now
injects ONE HumanMessage nudge per logical turn when the turn's user input
mentioned ``save_report`` and no save_report tool call has happened yet.

Latch/turn-boundary semantics mirror the budget guards (issue #23): the
state rides in ``NativeAgentGraph._loop_counters`` (per-thread instance
dict), survives approval resumes, and resets only on new user input.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from clearwing.agent.runtime import Command, NativeAgentGraph
from clearwing.agent.tooling import interrupt, tool
from clearwing.llm.messages import HumanMessage
from clearwing.providers.binding import AgentLimits

ASKS_SAVE_REPORT = "Scan the fixture target, then produce the report file with save_report."
NO_SAVE_REPORT = "Scan the fixture target and summarize the findings."


@tool(name="save_report")
def fake_save_report(filepath: str, format: str, scan_data: dict) -> dict:
    """Stub save_report — no filesystem writes in tests."""
    return {"status": "saved", "path": filepath}


@tool(name="plain_tool")
def plain_tool(target: str) -> dict:
    """Ungated filler tool."""
    return {"status": "ok", "target": target}


@tool(name="gated_tool")
def gated_tool(target: str) -> dict:
    """Approval-gated tool: pauses the turn via interrupt()."""
    if not interrupt("Approve gated op on fixture target?"):
        return {"status": "denied", "target": target}
    return {"status": "approved", "target": target}


def _call(name: str, call_id: str, args: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(fn_name=name, call_id=call_id, fn_arguments=args or {"target": "t"})


def _graph(limits: AgentLimits | None = None) -> NativeAgentGraph:
    return NativeAgentGraph(
        llm=object(),
        native_tools=[],
        tools=[fake_save_report, plain_tool, gated_tool],
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


class _Msg:
    """Assistant step stand-in: text-only when calls is empty."""

    def __init__(self, calls: list[SimpleNamespace] | None = None):
        self.content = "final summary"
        self.tool_calls = list(calls or [])


def _scripted_step(graph: NativeAgentGraph, scripts: list[list[SimpleNamespace]]) -> dict:
    """The n-th assistant step emits the n-th script entry (last repeats).
    An empty entry means a text-only reply (turn would end)."""
    idx = {"n": 0}

    async def fake_step(st):
        i = min(idx["n"], len(scripts) - 1)
        idx["n"] += 1
        st.setdefault("messages", []).append(_Msg(scripts[i]))
        return {}

    graph._aassistant_step = fake_step
    return idx


def _nudges(state) -> list[HumanMessage]:
    return [
        m
        for m in state["messages"]
        if isinstance(m, HumanMessage) and "Delivery reminder" in m.text
    ]


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def _run(graph, config, text: str):
    async for _ in graph.astream({"messages": [HumanMessage(content=text)]}, config):
        pass


class TestSaveReportNudge:
    @pytest.mark.asyncio
    async def test_text_finish_without_call_triggers_one_nudge(self):
        """First text-only finish injects the nudge and runs one more round;
        a second text-only finish ends the turn — exactly one nudge."""
        graph = _graph()
        config = _config("t1")
        state = graph._get_or_create_state("t1")
        idx = _scripted_step(graph, [[], []])

        await _run(graph, config, ASKS_SAVE_REPORT)

        assert idx["n"] == 2, "nudge must make the loop run another round"
        nudges = _nudges(state)
        assert len(nudges) == 1
        assert "save_report" in nudges[0].text
        assert isinstance(state["messages"][-1], _Msg), "turn ends on the second text reply"

    @pytest.mark.asyncio
    async def test_save_report_call_suppresses_nudge(self):
        graph = _graph()
        config = _config("t2")
        state = graph._get_or_create_state("t2")
        idx = _scripted_step(
            graph,
            [
                [
                    _call(
                        "save_report",
                        "c1",
                        {
                            "filepath": "/tmp/clearwing-test-report.md",
                            "format": "markdown",
                            "scan_data": {"target": "t"},
                        },
                    )
                ],
                [],
            ],
        )

        await _run(graph, config, ASKS_SAVE_REPORT)

        assert idx["n"] == 2
        assert _nudges(state) == []
        tool_msgs = [m for m in state["messages"] if getattr(m, "name", "") == "save_report"]
        assert len(tool_msgs) == 1 and '"saved"' in tool_msgs[0].content

    @pytest.mark.asyncio
    async def test_input_without_keyword_never_nudges(self):
        graph = _graph()
        config = _config("t3")
        state = graph._get_or_create_state("t3")
        idx = _scripted_step(graph, [[]])

        await _run(graph, config, NO_SAVE_REPORT)

        assert idx["n"] == 1, "text-only reply must end the turn immediately"
        assert _nudges(state) == []

    @pytest.mark.asyncio
    async def test_steps_budget_exhausted_skips_nudge(self):
        """The loop would break at the top before answering the nudge —
        nudging would only append a dead message to history."""
        graph = _graph(AgentLimits(max_steps=1))
        config = _config("t4")
        state = graph._get_or_create_state("t4")
        idx = _scripted_step(graph, [[]])

        await _run(graph, config, ASKS_SAVE_REPORT)

        assert idx["n"] == 1
        assert _nudges(state) == []

    @pytest.mark.asyncio
    async def test_tool_budget_exhausted_skips_nudge(self):
        """Mirrors the max_tool_calls stop: the nudged save_report call could
        only be answered as skipped, so the nudge is withheld."""
        graph = _graph(AgentLimits(max_tool_calls=1))
        config = _config("t5")
        state = graph._get_or_create_state("t5")
        idx = _scripted_step(graph, [[_call("plain_tool", "c1")], []])

        await _run(graph, config, ASKS_SAVE_REPORT)

        assert idx["n"] == 2
        assert _nudges(state) == []

    @pytest.mark.asyncio
    async def test_latch_survives_approval_resume_within_turn(self):
        """Nudge, then a gated pause, then resume: still the same logical
        turn — the latch must not reset on Command(resume=...)."""
        graph = _graph()
        config = _config("t6")
        state = graph._get_or_create_state("t6")
        _scripted_step(graph, [[], [_call("gated_tool", "c1")], []])

        await _run(graph, config, ASKS_SAVE_REPORT)
        assert len(_nudges(state)) == 1
        assert graph._pending.get("t6") is not None, "gated call should pause the turn"

        async for _ in graph.astream(Command(resume=True), config):
            pass

        assert len(_nudges(state)) == 1, "resume must not re-arm the nudge latch"
        gated = [m for m in state["messages"] if getattr(m, "name", "") == "gated_tool"]
        assert len(gated) == 1 and '"approved"' in gated[0].content

    @pytest.mark.asyncio
    async def test_new_user_turn_resets_latch(self):
        graph = _graph()
        config = _config("t7")
        state = graph._get_or_create_state("t7")
        _scripted_step(graph, [[], []])

        await _run(graph, config, ASKS_SAVE_REPORT)
        assert len(_nudges(state)) == 1

        # New user input: fresh logical turn, fresh one-shot latch.
        await _run(graph, config, ASKS_SAVE_REPORT)
        assert len(_nudges(state)) == 2

    @pytest.mark.asyncio
    async def test_new_turn_without_keyword_does_not_inherit_activation(self):
        """Activation is computed from THIS turn's input only — an earlier
        turn that asked for save_report must not arm later turns."""
        graph = _graph()
        config = _config("t8")
        state = graph._get_or_create_state("t8")
        _scripted_step(graph, [[], []])

        await _run(graph, config, ASKS_SAVE_REPORT)
        assert len(_nudges(state)) == 1

        await _run(graph, config, NO_SAVE_REPORT)
        assert len(_nudges(state)) == 1, "turn 2 never mentioned save_report"

    @pytest.mark.asyncio
    async def test_tool_name_match_is_exact(self):
        """Only the literal registered name counts as delivery: a call with
        different case executed (and errored) but must not suppress the
        nudge."""
        graph = _graph()
        config = _config("t9")
        state = graph._get_or_create_state("t9")
        idx = _scripted_step(graph, [[_call("Save_report", "c1")], []])

        await _run(graph, config, ASKS_SAVE_REPORT)

        # step 1: bad-case call; step 2: text -> nudge; step 3: text again.
        assert idx["n"] == 3
        tool_msgs = [m for m in state["messages"] if getattr(m, "name", "") == "Save_report"]
        assert len(tool_msgs) == 1 and "unknown tool" in tool_msgs[0].content
        assert len(_nudges(state)) == 1

    @pytest.mark.asyncio
    async def test_earlier_turn_save_report_call_does_not_suppress_next_turn(self):
        """The zero-call scan is bounded by the turn start index."""
        graph = _graph()
        config = _config("t10")
        state = graph._get_or_create_state("t10")
        _scripted_step(
            graph,
            [
                [
                    _call(
                        "save_report",
                        "c1",
                        {
                            "filepath": "/tmp/clearwing-test-report.md",
                            "format": "markdown",
                            "scan_data": {"target": "t"},
                        },
                    )
                ],
                [],
            ],
        )

        await _run(graph, config, ASKS_SAVE_REPORT)  # turn 1 delivers, no nudge
        assert _nudges(state) == []

        # Turn 2 asks again and replies in prose: nudged (turn 1's call is
        # outside this turn's boundary).
        await _run(graph, config, ASKS_SAVE_REPORT)
        assert len(_nudges(state)) == 1
