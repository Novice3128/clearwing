"""Regression tests for issue #52: per-thread cost totals.

`NativeAgentGraph._cost_totals` used to be a single graph-level bucket.
One graph serving several thread_ids meant thread A's spend was written
into thread B's state totals (and vice versa) — bidirectional pollution.
The totals now key by the running loop's thread_id (contextvar set in
_arun_loop), mirroring _loop_counters.
"""

from __future__ import annotations

import pytest

from clearwing.agent.runtime import NativeAgentGraph
from clearwing.observability.telemetry import CostTracker


class _FakeUsage:
    def __init__(self, prompt=0, completion=0, total=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _FakeResponse:
    def __init__(self, text="", usage=None):
        self.first_text = text
        self.texts = [text] if text else []
        self.tool_calls = []
        self.usage = usage or _FakeUsage()
        self.provider_model_name = "fake-model"
        self.reasoning_content = None


class _FakeNativeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    async def achat_stream(self, **kwargs):
        return self._responses.pop(0)


def _build_graph(session_id: str) -> NativeAgentGraph:
    # Two scripted turns: the first answers with a big usage block, the
    # second with a small one.
    client = _FakeNativeClient(
        [
            _FakeResponse(text="thread answer", usage=_FakeUsage(1000, 500, 1500)),
            _FakeResponse(text="done", usage=_FakeUsage(10, 5, 15)),
        ]
    )
    graph = NativeAgentGraph(
        llm=client,
        native_tools=[],
        tools=[],
        system_prompt_fn=lambda s: "sys",
        model_name="m",
        session_id=session_id,
        state_updater_fn=lambda *a, **k: {},
        knowledge_graph_populator_fn=None,
        input_guardrail_tool_names=frozenset(),
        output_guardrail_tool_names=frozenset(),
        enable_cost_tracker=True,
        enable_episodic_memory=False,
        enable_audit=False,
        enable_knowledge_graph=False,
        enable_input_guardrail=False,
        enable_output_guardrail=False,
        enable_event_bus=False,
        enable_context_summarizer=False,
        agent_limits=None,
    )
    return graph


class TestCostTotalsPerThread:
    @pytest.mark.asyncio
    async def test_thread_spends_do_not_cross_contaminate(self):
        CostTracker().reset()
        graph = _build_graph("sess-thread-cost")

        cfg_a = {"configurable": {"thread_id": "ws-threadA"}}
        cfg_b = {"configurable": {"thread_id": "ws-threadB"}}

        # Thread A runs one expensive turn first.
        async for _ in graph.astream(
            {"messages": [{"role": "user", "content": "go"}]}, cfg_a
        ):
            pass

        state_a = graph.get_state(cfg_a).values
        assert state_a["total_tokens"] == 1500

        # Thread B on the SAME graph must start from zero, not inherit A.
        async for _ in graph.astream(
            {"messages": [{"role": "user", "content": "go"}]}, cfg_b
        ):
            pass

        state_b = graph.get_state(cfg_b).values
        assert state_b["total_tokens"] == 15
        assert state_b["total_cost_usd"] == CostTracker.estimate_cost(
            10, 5, "fake-model"
        )

        # And A's recorded totals are unchanged by B's turn.
        state_a = graph.get_state(cfg_a).values
        assert state_a["total_tokens"] == 1500

        # The buckets are separate entries on the instance dict.
        assert set(graph._cost_totals) == {"ws-threadA", "ws-threadB"}
        assert graph._cost_totals["ws-threadA"]["input_tokens"] == 1000
        assert graph._cost_totals["ws-threadB"]["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_direct_step_booking_falls_back_to_default_bucket(self):
        """Steps invoked outside a running loop (tests, embedders) share
        the legacy 'default' bucket instead of crashing."""
        CostTracker().reset()
        graph = _build_graph("sess-default-bucket")

        state = {"messages": [{"role": "user", "content": "hi"}]}
        await graph._aassistant_step(state)

        assert graph._cost_totals["default"]["input_tokens"] == 1000
        assert state["total_tokens"] == 1500


class TestThreadLookupMechanics:
    """Codex PR-55 r4: the thread is derived from the state dict in hand —
    no ContextVar, so interleaved loops and direct calls cannot cross-book."""

    def test_thread_lookup_is_identity_based_not_ambient(self):
        """Attribution must come from the state dict in hand (identity), not
        from ambient task state: an ambient ContextVar leaks out of async
        generators and cross-books interleaved loops on one graph."""
        graph = _build_graph("sess-identity")
        state_a = graph._get_or_create_state("ws-A")
        state_b = graph._get_or_create_state("ws-B")

        assert graph._thread_for_state(state_a) == "ws-A"
        assert graph._thread_for_state(state_b) == "ws-B"
        # Equal CONTENT but different identity is not a thread's state.
        assert graph._thread_for_state({"messages": []}) == "default"

        # And the implementation carries no ambient thread id at all.
        import clearwing.agent.runtime as runtime_module

        assert not hasattr(runtime_module, "_current_thread_id")

    def test_assistant_step_keeps_its_tracer_span(self):
        """The cost-bucket helper must not steal the @tracer.chain
        decorator from _aassistant_step (Codex PR-55 r4)."""
        from clearwing.agent.runtime import NativeAgentGraph

        step = NativeAgentGraph._aassistant_step
        # Decorated functions are wrapped; the helper is not.
        assert hasattr(step, "__wrapped__") or "BoundFunctionWrapper" in type(
            step
        ).__name__
        assert NativeAgentGraph._cost_totals_for.__name__ == "_cost_totals_for"

    def test_unregistered_state_uses_the_default_bucket(self):
        graph = _build_graph("sess-unregistered")
        totals = graph._cost_totals_for(
            graph._thread_for_state({"messages": []})
        )
        assert graph._cost_totals["default"] is totals
