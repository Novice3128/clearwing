"""Tests for the native agent runtime."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clearwing.agent.graph import create_agent
from clearwing.agent.prompts import build_dynamic_context, build_system_prompt
from clearwing.agent.state import AgentState
from clearwing.agent.tooling import tool
from clearwing.agent.tools import get_all_tools
from clearwing.agent.tools.meta.reporting_tools import generate_report
from clearwing.agent.tools.meta.utility_tools import calculate_severity, validate_target
from clearwing.agent.tools.scan.scanner_tools import detect_os, detect_services, scan_ports


class TestAgentState:
    def test_state_instantiation(self):
        state: AgentState = {
            "messages": [],
            "target": "192.168.1.1",
            "open_ports": [],
            "services": [],
            "vulnerabilities": [],
            "exploit_results": [],
            "os_info": None,
            "kali_container_id": None,
            "custom_tool_names": [],
        }
        assert state["target"] == "192.168.1.1"
        assert state["messages"] == []
        assert state["open_ports"] == []

    def test_state_with_data(self):
        state: AgentState = {
            "messages": [],
            "target": "10.0.0.1",
            "open_ports": [{"port": 22, "protocol": "tcp", "state": "open", "service": "SSH"}],
            "services": [{"port": 22, "service": "SSH", "version": "7.4"}],
            "vulnerabilities": [{"cve": "CVE-2018-15473", "cvss": 5.3}],
            "exploit_results": [],
            "os_info": "Linux/Unix",
            "kali_container_id": None,
            "custom_tool_names": ["my_tool"],
        }
        assert len(state["open_ports"]) == 1
        assert state["os_info"] == "Linux/Unix"
        assert "my_tool" in state["custom_tool_names"]


class TestSystemPrompt:
    def test_empty_state(self):
        state = {
            "target": None,
            "open_ports": [],
            "services": [],
            "vulnerabilities": [],
            "exploit_results": [],
            "os_info": None,
            "kali_container_id": None,
            "custom_tool_names": [],
        }
        prompt = build_system_prompt(state)
        # Static prompt: byte-stable across steps for prefix caching (#36);
        # state-derived content moved to the dynamic context note.
        assert "Clearwing Agent" in prompt
        assert "{" not in prompt  # no unfilled template slots
        assert build_dynamic_context(state) == ""

    def test_populated_state(self):
        state = {
            "target": "10.0.0.1",
            "open_ports": [{"port": 80, "protocol": "tcp", "service": "HTTP"}],
            "services": [{"service": "HTTP", "port": 80, "version": "2.4"}],
            "vulnerabilities": [{"cve": "CVE-2017-5638", "cvss": 10.0}],
            "exploit_results": [{"success": True}],
            "os_info": "Linux/Unix",
            "kali_container_id": "abc123def456",
            "custom_tool_names": ["my_scanner"],
        }
        prompt = build_system_prompt(state)
        note = build_dynamic_context(state)
        # The dynamic context note carries all state-derived content...
        assert "## Current Context" in note
        assert "10.0.0.1" in note
        assert "80/tcp" in note
        assert "CVE-2017-5638" in note
        assert "Linux/Unix" in note
        assert "abc123def456" in note
        assert "my_scanner" in note
        # ...while the static prompt stays byte-identical regardless of state.
        empty_note_prompt = build_system_prompt({})
        assert prompt == empty_note_prompt


class TestToolList:
    def test_get_all_tools(self):
        tools = get_all_tools()
        assert len(tools) >= 20
        tool_names = [t.name for t in tools]
        assert "scan_ports" in tool_names
        assert "detect_services" in tool_names
        assert "exploit_vulnerability" in tool_names
        assert "kali_setup" in tool_names
        assert "generate_report" in tool_names
        assert "validate_target" in tool_names
        assert "create_custom_tool" in tool_names


class TestGraphConstruction:
    def test_create_agent(self):
        with patch("clearwing.agent.graph._create_llm") as mock_create_llm:
            mock_llm = MagicMock()
            mock_create_llm.return_value = mock_llm
            graph = create_agent(model_name="claude-sonnet-4-6")
            assert graph is not None
            mock_create_llm.assert_called_once_with(
                "claude-sonnet-4-6",
                base_url=None,
                api_key=None,
                model_explicit=False,
            )
            # The native client is threaded straight through — no bind_tools.
            assert graph.llm is mock_llm
            assert graph.native_tools

    def test_create_agent_with_custom_tools(self):
        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        with patch("clearwing.agent.graph._create_llm") as mock_create_llm:
            mock_llm = MagicMock()
            mock_create_llm.return_value = mock_llm
            graph = create_agent(model_name="claude-sonnet-4-6", custom_tools=[dummy_tool])
            assert graph is not None
            # The custom tool is registered in the runtime's tool map and its
            # NativeToolSpec is built for the LLM call.
            assert "dummy_tool" in graph.tools
            assert any(spec.name == "dummy_tool" for spec in graph.native_tools)

    def test_create_agent_with_custom_endpoint(self):
        with patch("clearwing.agent.graph._create_llm") as mock_create_llm:
            mock_llm = MagicMock()
            mock_create_llm.return_value = mock_llm
            graph = create_agent(
                model_name="my-model",
                base_url="http://localhost:8000/v1",
                api_key="test-key",
            )
            assert graph is not None
            mock_create_llm.assert_called_once_with(
                "my-model",
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model_explicit=False,
            )


class TestScannerToolWrapping:
    @pytest.mark.asyncio
    async def test_scan_ports_wraps_scanner(self):
        mock_result = [{"port": 22, "protocol": "tcp", "state": "open", "service": "SSH"}]

        mock_scanner = MagicMock()
        mock_scanner.scan = AsyncMock(return_value=mock_result)
        mock_class = MagicMock(return_value=mock_scanner)

        with patch("clearwing.scanning.PortScanner", mock_class):
            await scan_ports.ainvoke(
                {
                    "target": "192.168.1.1",
                    "ports": [22],
                    "scan_type": "connect",
                    "threads": 10,
                }
            )
            mock_scanner.scan.assert_called_once_with("192.168.1.1", [22], "connect", 10)

    @pytest.mark.asyncio
    async def test_detect_services_wraps_scanner(self):
        ports = [{"port": 80, "service": "HTTP"}]
        mock_result = [{"port": 80, "service": "HTTP", "banner": "Apache", "version": "2.4"}]

        mock_scanner = MagicMock()
        mock_scanner.detect = AsyncMock(return_value=mock_result)
        mock_class = MagicMock(return_value=mock_scanner)

        with patch("clearwing.scanning.ServiceScanner", mock_class):
            await detect_services.ainvoke(
                {
                    "target": "192.168.1.1",
                    "open_ports": ports,
                }
            )
            mock_scanner.detect.assert_called_once_with("192.168.1.1", ports)

    @pytest.mark.asyncio
    async def test_detect_os_wraps_scanner(self):
        mock_scanner = MagicMock()
        mock_scanner.detect = AsyncMock(return_value="Linux/Unix")
        mock_class = MagicMock(return_value=mock_scanner)

        with patch("clearwing.scanning.OSScanner", mock_class):
            await detect_os.ainvoke({"target": "192.168.1.1"})
            mock_scanner.detect.assert_called_once_with("192.168.1.1")


class TestUtilityTools:
    def test_validate_target_ip(self):
        result = validate_target.invoke({"ip_or_cidr": "192.168.1.1"})
        assert result["valid"] is True
        assert result["is_cidr"] is False
        assert result["ips"] == ["192.168.1.1"]

    def test_validate_target_invalid(self):
        result = validate_target.invoke({"ip_or_cidr": "not-an-ip"})
        assert result["valid"] is False

    def test_validate_target_cidr(self):
        result = validate_target.invoke({"ip_or_cidr": "192.168.1.0/30"})
        assert result["valid"] is True
        assert result["is_cidr"] is True
        assert len(result["ips"]) == 2  # /30 has 2 usable hosts

    def test_calculate_severity(self):
        assert calculate_severity.invoke({"cvss_score": 9.5}) == "CRITICAL"
        assert calculate_severity.invoke({"cvss_score": 7.5}) == "HIGH"
        assert calculate_severity.invoke({"cvss_score": 5.0}) == "MEDIUM"
        assert calculate_severity.invoke({"cvss_score": 2.0}) == "LOW"
        assert calculate_severity.invoke({"cvss_score": 0.0}) == "NONE"


class TestReportingTools:
    def test_generate_report(self):
        scan_data = {
            "target": "192.168.1.1",
            "open_ports": [{"port": 22, "protocol": "tcp", "state": "open", "service": "SSH"}],
            "services": [],
            "vulnerabilities": [],
            "exploits": [],
            "os_info": "Linux/Unix",
        }
        result = generate_report.invoke({"format": "text", "scan_data": scan_data})
        assert "192.168.1.1" in result
        assert "CLEARWING SCAN REPORT" in result


class _FakeUsage:
    def __init__(self, prompt=0, completion=0, total=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _FakeResponse:
    """Minimal stand-in for a genai ChatResponse."""

    def __init__(self, text="", tool_calls=None, usage=None):
        self.first_text = text
        self.texts = [text] if text else []
        self.tool_calls = tool_calls or []
        self.usage = usage or _FakeUsage()
        self.provider_model_name = "fake-model"
        self.reasoning_content = None


class _FakeNativeClient:
    """Records the ChatMessage history it is handed and replays scripted responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # each entry: (messages, system, tools)
        self.calls_kwargs = []  # each entry: cache/prompt-cache kwargs

    async def achat_stream(
        self,
        *,
        messages,
        system=None,
        tools=None,
        on_text_delta=None,
        cache_prefix=False,
        prompt_cache_key=None,
        context_note=None,
    ):
        self.calls.append((list(messages), system, tools))
        self.calls_kwargs.append(
            {
                "cache_prefix": cache_prefix,
                "prompt_cache_key": prompt_cache_key,
                "context_note": context_note,
            }
        )
        resp = self._responses.pop(0)
        if on_text_delta and resp.first_text:
            on_text_delta(resp.first_text)
        return resp

    async def achat(self, *, messages, system=None, tools=None, **kwargs):
        return await self.achat_stream(messages=messages, system=system, tools=tools)


class TestNativeToolLoopRoundTrip:
    """The riskiest path: a multi-turn tool call over the native client.

    Asserts that genai ``ToolCall`` objects are consumed by ``.fn_name`` /
    ``.fn_arguments`` / ``.call_id``, that the assistant + tool result turns
    round-trip into well-formed ChatMessages (assistant carries ``tool_calls``,
    tool result carries ``tool_response_call_id``), and that usage is tracked.
    """

    @pytest.mark.asyncio
    async def test_tool_call_executes_and_round_trips(self):
        import json

        from genai_pyo3 import ToolCall

        from clearwing.agent.graph import build_react_graph

        seen_args = {}

        @tool
        def echo_tool(value: str) -> str:
            """Echo the value."""
            seen_args["value"] = value
            return f"echoed:{value}"

        # Turn 1: model asks to call echo_tool. Turn 2: model responds with text.
        tc = ToolCall("call-1", "echo_tool", json.dumps({"value": "hello"}))
        client = _FakeNativeClient(
            [
                _FakeResponse(text="", tool_calls=[tc], usage=_FakeUsage(10, 5, 15)),
                _FakeResponse(text="all done", usage=_FakeUsage(3, 2, 5)),
            ]
        )

        graph = build_react_graph(
            llm_with_tools=client,
            tools=[echo_tool],
            system_prompt_fn=lambda state: "sys",
            model_name="fake-model",
            session_id=None,
            enable_knowledge_graph=False,
            enable_audit=False,
            enable_episodic_memory=False,
            enable_context_summarizer=False,
        )

        config = {"configurable": {"thread_id": "t1"}}
        events = []
        async for event in graph.astream(
            {"messages": [{"role": "user", "content": "please echo hello"}]}, config
        ):
            events.append(event)

        # The tool actually ran with the decoded arguments.
        assert seen_args == {"value": "hello"}

        # Two LLM turns were made.
        assert len(client.calls) == 2

        # Prompt-cache wiring (#36): every assistant step asks for a
        # cacheable prefix routed by the session id.
        for kwargs in client.calls_kwargs:
            assert kwargs["cache_prefix"] is True

        # On the SECOND call, the history sent to the model must contain the
        # assistant tool-call turn and the paired tool-result turn.
        second_messages = client.calls[1][0]
        roles = [m.role for m in second_messages]
        assert "assistant" in roles
        assert "tool" in roles

        assistant_msg = next(m for m in second_messages if m.role == "assistant")
        assert assistant_msg.tool_calls  # carries the tool_calls
        assert assistant_msg.tool_calls[0].call_id == "call-1"

        tool_msg = next(m for m in second_messages if m.role == "tool")
        assert tool_msg.tool_response_call_id == "call-1"
        assert "echoed:hello" in tool_msg.content

        # Final state carries the assistant's closing text and usage.
        final = graph.get_state(config).values
        last = final["messages"][-1]
        assert last.type == "ai"
        assert last.text == "all done"
        assert final.get("total_tokens", 0) > 0

    @pytest.mark.asyncio
    async def test_cache_kwargs_and_context_note_flow_through_graph(self):
        import json

        from genai_pyo3 import ToolCall

        from clearwing.agent.graph import build_react_graph
        from clearwing.agent.prompts import build_dynamic_context

        @tool
        def noop_tool(value: str) -> str:
            """Noop."""
            return value

        tc = ToolCall("call-1", "noop_tool", json.dumps({"value": "x"}))
        client = _FakeNativeClient(
            [
                _FakeResponse(text="", tool_calls=[tc], usage=_FakeUsage(10, 5, 15)),
                _FakeResponse(text="done", usage=_FakeUsage(3, 2, 5)),
            ]
        )
        graph = build_react_graph(
            llm_with_tools=client,
            tools=[noop_tool],
            system_prompt_fn=lambda state: "sys",
            model_name="fake-model",
            session_id="sess-cache-1",
            dynamic_context_fn=build_dynamic_context,
            enable_knowledge_graph=False,
            enable_audit=False,
            enable_episodic_memory=False,
            enable_context_summarizer=False,
        )
        config = {"configurable": {"thread_id": "t1"}}
        async for _ in graph.astream(
            {"messages": [{"role": "user", "content": "go"}], "target": "10.9.9.9"},
            config,
        ):
            pass

        assert len(client.calls_kwargs) == 2
        for kwargs in client.calls_kwargs:
            assert kwargs["cache_prefix"] is True
            assert kwargs["prompt_cache_key"] == "sess-cache-1"
            # The dynamic context note carries the state-derived target...
            assert "## Current Context" in (kwargs["context_note"] or "")
            assert "10.9.9.9" in (kwargs["context_note"] or "")
        # ...while the system prompt stays byte-static across steps.
        assert client.calls[0][1] == client.calls[1][1]


class _SummarizingFakeClient(_FakeNativeClient):
    """Fake client that also answers the summarizer's aask_text calls."""

    # 0.8 * 1000 = 800 tokens ≈ 3200 chars crosses the threshold.
    context_budget_tokens = 1000

    def __init__(self, responses):
        super().__init__(responses)
        self.summarize_calls = []

    async def aask_text(
        self, *, system, user, cache_prefix=False, prompt_cache_key=None, **kwargs
    ):
        self.summarize_calls.append(
            {
                "system": system,
                "user": user,
                "cache_prefix": cache_prefix,
                "prompt_cache_key": prompt_cache_key,
            }
        )
        return _FakeResponse(text="compacted session summary", usage=_FakeUsage(1, 1, 2))


class TestContextSummarizerCachePrefix:
    """Issue #38: the summarizer must not destroy the prompt-cache prefix.

    Past the 80% threshold the old runtime re-ran the LLM summary on every
    step and put a different summary at the front of the request (and into
    the system prompt), so every step re-paid full input price plus one
    extra summarization call.
    """

    def _build(self, client):
        from clearwing.agent.graph import build_react_graph

        @tool
        def noop_tool(value: str) -> str:
            """Noop."""
            return value

        return build_react_graph(
            llm_with_tools=client,
            tools=[noop_tool],
            system_prompt_fn=lambda state: "sys",
            model_name="fake-model",
            session_id="sess-sum-1",
            enable_knowledge_graph=False,
            enable_audit=False,
            enable_episodic_memory=False,
            enable_cost_tracker=False,
            enable_event_bus=False,
            enable_context_summarizer=True,
        )

    @pytest.mark.asyncio
    async def test_summary_reused_and_prefix_byte_identical(self):
        import json

        from genai_pyo3 import ToolCall

        tc1 = ToolCall("call-1", "noop_tool", json.dumps({"value": "x"}))
        tc2 = ToolCall("call-2", "noop_tool", json.dumps({"value": "y"}))
        tc3 = ToolCall("call-3", "noop_tool", json.dumps({"value": "z"}))
        # 0.8 * 1000 tokens ≈ 3200 chars: one 2000-char message stays under,
        # two of them cross the threshold so the oldest coverable ones get
        # compacted and the surviving view drops back under the threshold.
        m1 = "A" * 2000
        m2 = "C" * 2000
        client = _SummarizingFakeClient(
            [
                # Turn 1 (under threshold): tool call then text.
                _FakeResponse(text="", tool_calls=[tc1], usage=_FakeUsage(10, 5, 15)),
                _FakeResponse(text="done", usage=_FakeUsage(3, 2, 5)),
                # Turn 2 (still under threshold after compaction math).
                _FakeResponse(text="", tool_calls=[tc2], usage=_FakeUsage(10, 5, 15)),
                _FakeResponse(text="done2", usage=_FakeUsage(3, 2, 5)),
                # Turn 3: m2 crosses the threshold → compaction fires once;
                # then two steps inside the new epoch.
                _FakeResponse(text="", tool_calls=[tc3], usage=_FakeUsage(10, 5, 15)),
                _FakeResponse(text="done3", usage=_FakeUsage(3, 2, 5)),
                # Turn 4: small follow-up — must reuse the summary, not
                # regenerate it.
                _FakeResponse(text="done4", usage=_FakeUsage(3, 2, 5)),
            ]
        )
        graph = self._build(client)
        config = {"configurable": {"thread_id": "t1"}}

        async for _ in graph.astream({"messages": [{"role": "user", "content": "go"}]}, config):
            pass
        async for _ in graph.astream({"messages": [{"role": "user", "content": m1}]}, config):
            pass
        # steps 5 and 6 (indices 4, 5) are within the post-compaction epoch
        async for _ in graph.astream({"messages": [{"role": "user", "content": m2}]}, config):
            pass
        async for _ in graph.astream(
            {"messages": [{"role": "user", "content": "and then?"}]}, config
        ):
            pass

        # The summary LLM ran exactly once — not once per step past the
        # threshold, and not again on the follow-up turn.
        assert len(client.summarize_calls) == 1
        assert client.summarize_calls[0]["cache_prefix"] is True
        assert client.summarize_calls[0]["prompt_cache_key"] == "sess-sum-1"

        # Steps within the same epoch send a byte-identical prefix: step 6's
        # message list starts with exactly step 5's list (role + content).
        step5 = [(m.role, m.content) for m in client.calls[4][0]]
        step6 = [(m.role, m.content) for m in client.calls[5][0]]
        assert step6[: len(step5)] == step5

        # The summary rides after the cache breakpoint in the context note —
        # never inside the message history and never in the system prompt.
        for index in (4, 5, 6):
            assert "Session Summary" in (client.calls_kwargs[index]["context_note"] or "")
            for m in client.calls[index][0]:
                assert "compacted session summary" not in (m.content or "")
            assert client.calls[index][1] == "sys"

        # The compacted view is committed to state: the first coverable
        # messages are gone and the summary state persists with its
        # coverage count.
        final = graph.get_state(config).values
        assert final["context_summary"]["covered_count"] >= 1
        assert not any(
            "go" == str(getattr(m, "content", "") or "") for m in final["messages"]
        )
        assert not any(
            m1 in str(getattr(m, "content", "") or "") for m in final["messages"]
        )


class TestSummaryUsageAccounting:
    """The summarizer's LLM call is real spend and must be booked.

    Three-lens review F2: ``summarize()``'s usage used to ride on a
    discarded response object — invisible to the CostTracker, the graph's
    instance totals, and therefore to cost limits and result fields.
    """

    def _build(self, client, session_id):
        from clearwing.agent.graph import build_react_graph

        @tool
        def noop_tool(value: str) -> str:
            """Noop."""
            return value

        return build_react_graph(
            llm_with_tools=client,
            tools=[noop_tool],
            system_prompt_fn=lambda state: "sys",
            model_name="fake-model",
            session_id=session_id,
            enable_knowledge_graph=False,
            enable_audit=False,
            enable_episodic_memory=False,
            enable_cost_tracker=True,
            enable_event_bus=False,
            enable_context_summarizer=True,
        )

    @pytest.mark.asyncio
    async def test_summary_usage_recorded_in_tracker_and_totals(self):
        from clearwing.observability.telemetry import CostTracker

        # Same flow shape as TestContextSummarizerCachePrefix: compaction
        # fires exactly once (turn 3), the summary usage is (1, 1).
        client = _SummarizingFakeClient(
            [
                _FakeResponse(text="done", usage=_FakeUsage(10, 5, 15)),
                _FakeResponse(text="done2", usage=_FakeUsage(3, 2, 5)),
                _FakeResponse(text="done3", usage=_FakeUsage(3, 2, 5)),
            ]
        )
        # Prime the history past the threshold in one turn: two 2000-char
        # user messages with an intermediate model response.
        m1 = "A" * 2000
        m2 = "C" * 2000
        graph = self._build(client, "sess-usage-1")
        config = {"configurable": {"thread_id": "t1"}}
        async for _ in graph.astream({"messages": [{"role": "user", "content": "go"}]}, config):
            pass
        async for _ in graph.astream({"messages": [{"role": "user", "content": m1}]}, config):
            pass
        async for _ in graph.astream({"messages": [{"role": "user", "content": m2}]}, config):
            pass

        assert len(client.summarize_calls) == 1

        # The summary call's tokens landed in the instance totals on top of
        # the graph's own assistant steps, and in the session-scoped tracker
        # record (fake-model bills at the Sonnet fallback tier). Totals are
        # keyed by the running loop's thread (issue #52).
        expected_in = 10 + 3 + 3 + 1  # assistant steps + summary
        expected_out = 5 + 2 + 2 + 1
        totals = graph._cost_totals_for("t1")
        assert totals["input_tokens"] == expected_in
        assert totals["output_tokens"] == expected_out
        expected_cost = CostTracker.estimate_cost(
            expected_in, expected_out, "fake-model"
        )
        assert totals["cost_usd"] == pytest.approx(expected_cost)

        final = graph.get_state(config).values
        assert final["total_tokens"] == expected_in + expected_out
        assert final["total_cost_usd"] == pytest.approx(expected_cost)
        assert CostTracker().session_total("sess-usage-1") == pytest.approx(expected_cost)


class TestContextSummarizedEventGuard:
    """F6: the "context summarized" event fires only on ACTUAL compaction."""

    def _build(self, client, session_id="sess-ev-1"):
        from clearwing.agent.graph import build_react_graph

        return build_react_graph(
            llm_with_tools=client,
            tools=[],
            system_prompt_fn=lambda state: "sys",
            model_name="fake-model",
            session_id=session_id,
            enable_knowledge_graph=False,
            enable_audit=False,
            enable_episodic_memory=False,
            enable_cost_tracker=False,
            enable_event_bus=True,
            enable_context_summarizer=True,
        )

    @pytest.mark.asyncio
    async def test_no_event_when_view_unchanged(self):
        """should_summarize fires but nothing is newly coverable → silence.

        The old code announced "context summarized" whenever the summarize
        call ran, even when the returned view was byte-identical —
        misleading operators and spamming the transcript every step past
        the threshold.
        """
        from clearwing.core.events import EventBus, EventType

        messages_seen: list[dict] = []

        def handler(data):
            if isinstance(data, dict) and "context summarized" in str(data.get("content", "")):
                messages_seen.append(data)

        bus = EventBus()
        bus.subscribe(EventType.MESSAGE, handler)
        try:
            client = _FakeNativeClient([_FakeResponse(text="done", usage=_FakeUsage(1, 1, 2))])
            graph = self._build(client)
            config = {"configurable": {"thread_id": "t1"}}

            # Force the summarize path with an unchanged (equal-length) view.
            async def _unchanged_summary(msgs, llm, **kwargs):
                return {
                    "text": "prior summary",
                    "covered_count": 3,
                    "view": list(msgs),  # nothing removed
                    "usage": None,
                }

            graph.context_summarizer.summarize = _unchanged_summary
            graph.context_summarizer.should_summarize = lambda msgs, max_tokens=None: True

            async for _ in graph.astream(
                {"messages": [{"role": "user", "content": "hi"}]}, config
            ):
                pass
        finally:
            bus.unsubscribe(EventType.MESSAGE, handler)

        assert messages_seen == []
        # The summary state itself still updates (coverage bookkeeping) —
        # only the announcement is suppressed.
        final = graph.get_state(config).values
        assert final["context_summary"]["covered_count"] == 3

    @pytest.mark.asyncio
    async def test_event_fires_on_actual_compaction(self):
        from clearwing.core.events import EventBus, EventType

        messages_seen: list[dict] = []

        def handler(data):
            if isinstance(data, dict) and "context summarized" in str(data.get("content", "")):
                messages_seen.append(data)

        bus = EventBus()
        bus.subscribe(EventType.MESSAGE, handler)
        try:
            client = _SummarizingFakeClient([_FakeResponse(text="done", usage=_FakeUsage(1, 1, 2))])
            graph = self._build(client, "sess-ev-2")
            config = {"configurable": {"thread_id": "t1"}}

            async def _compacting_summary(msgs, llm, **kwargs):
                return {
                    "text": "summary",
                    "covered_count": 1,
                    "view": list(msgs)[1:],  # one message actually removed
                    "usage": None,
                }

            graph.context_summarizer.summarize = _compacting_summary
            graph.context_summarizer.should_summarize = lambda msgs, max_tokens=None: True

            async for _ in graph.astream(
                {"messages": [{"role": "user", "content": "hi"}]}, config
            ):
                pass
        finally:
            bus.unsubscribe(EventType.MESSAGE, handler)

        assert len(messages_seen) == 1
        assert "history compacted" in messages_seen[0]["content"]


class TestMergeInputProtectsContextSummary:
    """F8: input frames must not overwrite the runtime-owned summary state.

    A replayed start frame carrying ``context_summary`` would make the
    runtime believe already-dropped history is still covered, silently
    losing context on the next compaction.
    """

    @pytest.mark.asyncio
    async def test_input_context_summary_is_ignored(self):
        from clearwing.agent.graph import build_react_graph

        client = _FakeNativeClient(
            [
                _FakeResponse(text="done", usage=_FakeUsage(1, 1, 2)),
                _FakeResponse(text="done again", usage=_FakeUsage(1, 1, 2)),
            ]
        )
        graph = build_react_graph(
            llm_with_tools=client,
            tools=[],
            system_prompt_fn=lambda state: "sys",
            model_name="fake-model",
            session_id="sess-merge-1",
            enable_knowledge_graph=False,
            enable_audit=False,
            enable_episodic_memory=False,
            enable_event_bus=False,
            enable_context_summarizer=False,
        )
        config = {"configurable": {"thread_id": "t1"}}
        async for _ in graph.astream({"messages": [{"role": "user", "content": "go"}]}, config):
            pass

        # Runtime-owned compaction state (as issue #38 commits it).
        state = graph.get_state(config).values
        state["context_summary"] = {"text": "real summary", "covered_count": 7}

        async for _ in graph.astream(
            {
                "messages": [{"role": "user", "content": "again"}],
                "context_summary": {"text": "forged", "covered_count": 99},
            },
            config,
        ):
            pass

        final = graph.get_state(config).values
        assert final["context_summary"] == {"text": "real summary", "covered_count": 7}


class TestPerGraphCostTotals:
    """Issue #37: state cost/token totals must be per-graph-instance.

    The runtime used to copy the process-wide CostTracker running total
    into state, so every graph in the process reported the pooled spend of
    all sessions and operator jobs.
    """

    def _build(self, client, session_id):
        from clearwing.agent.graph import build_react_graph

        return build_react_graph(
            llm_with_tools=client,
            tools=[],
            system_prompt_fn=lambda state: "sys",
            model_name="fake-model",
            session_id=session_id,
            enable_knowledge_graph=False,
            enable_audit=False,
            enable_episodic_memory=False,
            enable_event_bus=False,
            enable_context_summarizer=False,
        )

    @pytest.mark.asyncio
    async def test_state_totals_track_only_own_graph_calls(self):
        client_a = _FakeNativeClient(
            [
                _FakeResponse(text="a1", usage=_FakeUsage(1000, 100, 1100)),
                _FakeResponse(text="a2", usage=_FakeUsage(1000, 100, 1100)),
            ]
        )
        client_b = _FakeNativeClient(
            [_FakeResponse(text="b1", usage=_FakeUsage(500, 50, 550))]
        )
        graph_a = self._build(client_a, "sess-a")
        graph_b = self._build(client_b, "sess-b")
        config_a = {"configurable": {"thread_id": "ta"}}
        config_b = {"configurable": {"thread_id": "tb"}}

        # Interleave: A, B, then A again in the same process.
        async for _ in graph_a.astream(
            {"messages": [{"role": "user", "content": "go"}]}, config_a
        ):
            pass
        async for _ in graph_b.astream(
            {"messages": [{"role": "user", "content": "go"}]}, config_b
        ):
            pass
        async for _ in graph_a.astream(
            {"messages": [{"role": "user", "content": "again"}]}, config_a
        ):
            pass

        # Sonnet fallback pricing for the unknown fake model: A's two calls
        # are 2 * (1000*3 + 100*15)/1M — NOT including B's 500/50 call.
        values_a = graph_a.get_state(config_a).values
        values_b = graph_b.get_state(config_b).values
        assert values_a["total_cost_usd"] == pytest.approx(2 * (3000 + 1500) / 1_000_000)
        assert values_a["total_tokens"] == 2 * 1100
        assert values_b["total_cost_usd"] == pytest.approx((500 * 3 + 50 * 15) / 1_000_000)
        assert values_b["total_tokens"] == 550
