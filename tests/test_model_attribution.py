"""Cost/audit model attribution (#21) and episodic-memory attribution (#19).

`NativeAgentGraph.model_name` is the caller-supplied label — the webui start
frame's placeholder. Pricing and audit used it verbatim, so a session routed
to glm/opus was billed at the placeholder's rate. The runtime now prefers the
resolved endpoint model (client `.model_name`, echoed per response as
`provider_model_name`).

Episodes likewise landed with `target='unknown'` and `session_id=''` because
the runtime never threaded its own session_id into EpisodicMemory, and the
webui never put the start-frame target into graph state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from clearwing.agent.runtime import NativeAgentGraph
from clearwing.agent.tooling import tool


@tool(name="echo_tool")
def echo_tool() -> dict:
    """Plain successful tool."""
    return {"ok": True}


def _graph(
    *,
    session_id: str | None = None,
    llm: object = object(),
    enable_episodic_memory: bool = False,
) -> NativeAgentGraph:
    return NativeAgentGraph(
        llm=llm,
        native_tools=[],
        tools=[echo_tool],
        system_prompt_fn=lambda s: "sys",
        model_name="claude-sonnet-4-6",
        session_id=session_id,
        state_updater_fn=lambda *a, **k: {},
        knowledge_graph_populator_fn=None,
        input_guardrail_tool_names=frozenset(),
        output_guardrail_tool_names=frozenset(),
        enable_cost_tracker=False,
        enable_episodic_memory=enable_episodic_memory,
        enable_audit=False,
        enable_knowledge_graph=False,
        enable_input_guardrail=False,
        enable_output_guardrail=False,
        enable_event_bus=False,
        enable_context_summarizer=False,
    )


def _call(name: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(fn_name=name, call_id=call_id, fn_arguments={})


class _Response:
    def __init__(self, provider_model_name):
        self.first_text = "done"
        self.texts = ["done"]
        self.tool_calls = []
        self.provider_model_name = provider_model_name
        self.usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )


class _FakeLLM:
    def __init__(self, client_model_name, provider_model_name):
        self.model_name = client_model_name
        self.provider_name = "zai"
        self._provider_model_name = provider_model_name

    async def achat_stream(self, **kwargs):
        return _Response(self._provider_model_name)


def _wire_recorders(graph: NativeAgentGraph) -> tuple[MagicMock, MagicMock]:
    cost = MagicMock()
    cost.input_tokens = 100
    cost.output_tokens = 50
    cost.total_cost_usd = 0.01
    audit = MagicMock()
    graph.cost_tracker = cost
    graph.audit_logger = audit
    return cost, audit


class TestEffectiveModelAttribution:
    @pytest.mark.asyncio
    async def test_cost_and_audit_use_provider_model(self):
        # Response echoes the provider-served model; the graph label is the
        # webui placeholder.
        graph = _graph(llm=_FakeLLM("glm-5.3", "glm-5.3"))
        cost, audit = _wire_recorders(graph)

        state = graph._get_or_create_state("attr-1")
        await graph._aassistant_step(state)

        cost.record_llm_call.assert_called_once_with(
            100, 50, "glm-5.3", provider="zai"
        )
        assert audit.log_llm_call.call_args.kwargs["model"] == "glm-5.3"
        assert state["messages"][-1].response_metadata["model"] == "glm-5.3"

    @pytest.mark.asyncio
    async def test_falls_back_to_client_model_name(self):
        # Some adapters leave provider_model_name unset; the client's
        # resolved endpoint model is the next best attribution.
        graph = _graph(llm=_FakeLLM("claude-opus-4-7", None))
        cost, audit = _wire_recorders(graph)

        state = graph._get_or_create_state("attr-2")
        await graph._aassistant_step(state)

        assert cost.record_llm_call.call_args.args[2] == "claude-opus-4-7"
        assert audit.log_llm_call.call_args.kwargs["model"] == "claude-opus-4-7"

    @pytest.mark.asyncio
    async def test_last_resort_is_graph_label(self):
        graph = _graph(llm=_FakeLLM(None, None))
        cost, audit = _wire_recorders(graph)

        state = graph._get_or_create_state("attr-3")
        await graph._aassistant_step(state)

        assert cost.record_llm_call.call_args.args[2] == "claude-sonnet-4-6"


class TestEpisodeAttribution:
    @pytest.mark.asyncio
    async def test_episodes_carry_target_and_session_id(self, tmp_path, monkeypatch):
        import clearwing.agent.runtime as runtime_mod
        from clearwing.data.memory.episodic_memory import EpisodicMemory

        constructed = {}

        def factory(session_id=""):
            constructed["session_id"] = session_id
            return EpisodicMemory(
                db_path=str(tmp_path / "memory.db"), session_id=session_id
            )

        monkeypatch.setattr(runtime_mod, "EpisodicMemory", factory)
        # Drive the real constructor path: the runtime must thread its own
        # session_id into EpisodicMemory (it used to construct it bare,
        # landing every episode with session_id='').
        graph = _graph(session_id="sess19", enable_episodic_memory=True)
        assert constructed["session_id"] == "sess19"
        assert graph.episodic_memory.session_id == "sess19"

        state = graph._get_or_create_state("epis-1")
        state["target"] = "10.1.2.3"
        await graph._arun_tool_calls(
            state, [_call("echo_tool", "c1")], resume_decision=...
        )

        episodes = graph.episodic_memory.recall("10.1.2.3")
        assert episodes, "episode for the state's target must be recorded"
        assert episodes[0].session_id == "sess19"
        assert episodes[0].event_type == "tool:echo_tool"
        assert graph.episodic_memory.recall("other-target") == []
