"""Tests for the Operator agent."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _astream_of(events):
    """Return an async-generator function yielding the given events."""

    async def _gen(*args, **kwargs):
        for event in events:
            yield event

    return _gen


def _astream_raises(exc):
    async def _gen(*args, **kwargs):
        raise exc
        yield  # pragma: no cover

    return _gen


def _run(coro):
    return asyncio.run(coro)


from clearwing.agent.operator import (
    _OPERATOR_SYSTEM_PROMPT,
    OperatorAgent,
    OperatorConfig,
    OperatorResult,
)


class TestOperatorConfig:
    def test_defaults(self):
        cfg = OperatorConfig(goals=["scan ports"], target="10.0.0.1")
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.max_turns == 100
        assert cfg.timeout_minutes == 60
        assert cfg.auto_approve_scans is True
        assert cfg.auto_approve_exploits is False
        assert cfg.cost_limit == 0.0
        assert cfg.lhost == "host.docker.internal"
        assert cfg.lport == 9999

    def test_custom_values(self):
        cfg = OperatorConfig(
            goals=["a", "b"],
            target="192.168.1.1",
            model="gpt-4o",
            max_turns=50,
            timeout_minutes=30,
            cost_limit=5.0,
            auto_approve_exploits=True,
            base_url="http://localhost:8000/v1",
            api_key="test-key",
        )
        assert cfg.goals == ["a", "b"]
        assert cfg.model == "gpt-4o"
        assert cfg.cost_limit == 5.0
        assert cfg.base_url == "http://localhost:8000/v1"

    def test_operator_model_defaults_empty(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        assert cfg.operator_model == ""


class TestOperatorResult:
    def test_fields(self):
        r = OperatorResult(
            goals=["scan ports"],
            target="10.0.0.1",
            status="completed",
            turns=5,
            cost_usd=0.05,
        )
        assert r.status == "completed"
        assert r.turns == 5
        assert r.findings == []
        assert r.flags_found == []
        assert r.escalation_question == ""

    def test_escalated_result(self):
        r = OperatorResult(
            goals=["scan"],
            target="10.0.0.1",
            status="escalated",
            escalation_question="What credentials to use?",
        )
        assert r.status == "escalated"
        assert "credentials" in r.escalation_question


class TestOperatorAgentInit:
    def test_init(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)
        assert op._turns == 0
        assert op._progress == []
        assert op._escalated is False


class TestFormatGoals:
    def test_single_goal(self):
        cfg = OperatorConfig(goals=["Scan for open ports"], target="10.0.0.1")
        op = OperatorAgent(cfg)
        text = op._format_goals()
        assert "10.0.0.1" in text
        assert "Scan for open ports" in text
        assert "1." in text
        assert "start_callback_listener" in text
        assert "records the request body" in text

    def test_multiple_goals(self):
        cfg = OperatorConfig(
            goals=["Scan ports", "Find vulnerabilities", "Generate report"],
            target="192.168.1.1",
        )
        op = OperatorAgent(cfg)
        text = op._format_goals()
        assert "1." in text
        assert "2." in text
        assert "3." in text
        assert "192.168.1.1" in text


class TestOperatorSystemPrompt:
    def test_prompt_contains_placeholders(self):
        assert "{goals}" in _OPERATOR_SYSTEM_PROMPT
        assert "{target}" in _OPERATOR_SYSTEM_PROMPT
        assert "{progress}" in _OPERATOR_SYSTEM_PROMPT

    def test_prompt_formatting(self):
        result = _OPERATOR_SYSTEM_PROMPT.format(
            goals="1. Scan ports",
            target="10.0.0.1",
            progress="No progress yet.",
        )
        assert "10.0.0.1" in result
        assert "Scan ports" in result
        assert "ESCALATE" in result
        assert "GOALS_COMPLETE" in result


class TestRunInnerTurn:
    def test_extracts_ai_content(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_ai = MagicMock()
        mock_ai.type = "ai"
        mock_ai.content = "I found port 22 open."

        mock_graph = MagicMock()
        mock_graph.astream = _astream_of([{"messages": [mock_ai]}])

        result = _run(op._arun_inner_turn(mock_graph, {}, {"messages": []}))
        assert "port 22" in result

    def test_handles_list_content(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_ai = MagicMock()
        mock_ai.type = "ai"
        mock_ai.content = [{"type": "text", "text": "Found SSH"}]

        mock_graph = MagicMock()
        mock_graph.astream = _astream_of([{"messages": [mock_ai]}])

        result = _run(op._arun_inner_turn(mock_graph, {}, {"messages": []}))
        assert "SSH" in result

    def test_handles_exception(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_graph = MagicMock()
        mock_graph.astream = _astream_raises(RuntimeError("connection lost"))

        result = _run(op._arun_inner_turn(mock_graph, {}, {"messages": []}))
        assert "error" in result.lower()


class TestDecideNext:
    def test_goals_complete(self):
        cfg = OperatorConfig(goals=["scan ports"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.first_text = "GOALS_COMPLETE"
        mock_llm.aask_text.return_value = mock_response

        decision = _run(op._adecide_next(mock_llm, "All ports scanned, report generated."))
        assert decision.startswith("GOALS_COMPLETE")

    def test_escalate(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.first_text = "ESCALATE: What are the login credentials?"
        mock_llm.aask_text.return_value = mock_response

        decision = _run(op._adecide_next(mock_llm, "I need credentials to log in."))
        assert decision.startswith("ESCALATE:")
        assert "credentials" in decision

    def test_next_instruction(self):
        cfg = OperatorConfig(goals=["scan", "exploit"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.first_text = "Now scan for vulnerabilities on the open ports."
        mock_llm.aask_text.return_value = mock_response

        decision = _run(op._adecide_next(mock_llm, "Found ports 22, 80, 443 open."))
        assert "scan" in decision.lower() or "vulnerabilities" in decision.lower()

    def test_handles_llm_error(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_llm = AsyncMock()
        mock_llm.aask_text.side_effect = RuntimeError("API error")

        decision = _run(op._adecide_next(mock_llm, "Agent output"))
        assert "Continue" in decision


class TestBuildResult:
    def test_completed_result(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)
        op._turns = 5

        mock_state = MagicMock()
        mock_state.values = {
            "vulnerabilities": [{"cve": "CVE-2024-1234", "severity": "high"}],
            "exploit_results": [],
            "flags_found": [{"flag": "flag{test}", "pattern": "flag\\{.*\\}"}],
            "total_cost_usd": 0.12,
            "total_tokens": 5000,
        }
        mock_graph = MagicMock()
        mock_graph.get_state.return_value = mock_state

        result = op._build_result(mock_graph, {}, time.time() - 10, "completed")
        assert result.status == "completed"
        assert result.turns == 5
        assert len(result.findings) == 1
        assert len(result.flags_found) == 1
        assert result.cost_usd == 0.12

    def test_escalated_result(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_state = MagicMock()
        mock_state.values = {}
        mock_graph = MagicMock()
        mock_graph.get_state.return_value = mock_state

        result = op._build_result(
            mock_graph,
            {},
            time.time(),
            "escalated",
            escalation_question="Need credentials",
        )
        assert result.status == "escalated"
        assert result.escalation_question == "Need credentials"

    def test_exploit_results_added_to_findings(self):
        cfg = OperatorConfig(goals=["exploit"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_state = MagicMock()
        mock_state.values = {
            "vulnerabilities": [],
            "exploit_results": [
                {"success": True, "vulnerability": "RCE"},
                {"success": False, "vulnerability": "SQLi"},
            ],
            "flags_found": [],
            "total_cost_usd": 0.0,
            "total_tokens": 0,
        }
        mock_graph = MagicMock()
        mock_graph.get_state.return_value = mock_state

        result = op._build_result(mock_graph, {}, time.time(), "completed")
        # Only successful exploits become findings
        assert len(result.findings) == 1
        assert "RCE" in result.findings[0]["description"]


class TestEmit:
    def test_calls_callback(self):
        calls = []
        cfg = OperatorConfig(
            goals=["scan"],
            target="10.0.0.1",
            on_message=lambda role, content: calls.append((role, content)),
        )
        op = OperatorAgent(cfg)
        op._emit("agent", "hello")
        assert len(calls) == 1
        assert calls[0] == ("agent", "hello")

    def test_no_callback(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)
        op._emit("agent", "hello")  # Should not raise

    def test_callback_exception_ignored(self):
        def bad_callback(role, content):
            raise ValueError("boom")

        cfg = OperatorConfig(
            goals=["scan"],
            target="10.0.0.1",
            on_message=bad_callback,
        )
        op = OperatorAgent(cfg)
        op._emit("agent", "hello")  # Should not raise


class TestHandleInterrupt:
    def test_auto_approve_scan(self):
        cfg = OperatorConfig(
            goals=["scan"],
            target="10.0.0.1",
            auto_approve_scans=True,
        )
        op = OperatorAgent(cfg)

        mock_interrupt = MagicMock()
        mock_interrupt.value = "Approve scan of port 80?"
        mock_task = MagicMock()
        mock_task.interrupts = [mock_interrupt]

        mock_state = MagicMock()
        mock_state.tasks = [mock_task]

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()

        result = _run(op._ahandle_interrupt(mock_state, mock_graph, {}))
        assert result is True

    def test_exploit_not_auto_approved(self):
        cfg = OperatorConfig(
            goals=["exploit"],
            target="10.0.0.1",
            auto_approve_scans=True,
            auto_approve_exploits=False,
        )
        op = OperatorAgent(cfg)

        mock_interrupt = MagicMock()
        mock_interrupt.value = "Approve RCE exploit?"
        mock_task = MagicMock()
        mock_task.interrupts = [mock_interrupt]

        mock_state = MagicMock()
        mock_state.tasks = [mock_task]

        mock_graph = MagicMock()

        result = _run(op._ahandle_interrupt(mock_state, mock_graph, {}))
        assert result is False

    def test_exploit_auto_approved_when_enabled(self):
        cfg = OperatorConfig(
            goals=["exploit"],
            target="10.0.0.1",
            auto_approve_exploits=True,
        )
        op = OperatorAgent(cfg)

        mock_interrupt = MagicMock()
        mock_interrupt.value = "Approve RCE exploit?"
        mock_task = MagicMock()
        mock_task.interrupts = [mock_interrupt]

        mock_state = MagicMock()
        mock_state.tasks = [mock_task]

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()

        result = _run(op._ahandle_interrupt(mock_state, mock_graph, {}))
        assert result is True

    def test_no_tasks(self):
        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)

        mock_state = MagicMock()
        mock_state.tasks = None

        result = _run(op._ahandle_interrupt(mock_state, MagicMock(), {}))
        assert result is True


class TestOperatorRun:
    """Integration-level tests for the full run() loop using mocks."""

    @staticmethod
    def _make_mock_graph(responses: list[str], state_values: dict = None):
        """Create a mock graph that returns the given responses in sequence."""
        mock_graph = MagicMock()

        def make_events(resp_text):
            mock_ai = MagicMock()
            mock_ai.type = "ai"
            mock_ai.content = resp_text
            return [{"messages": [mock_ai]}]

        event_sequences = [make_events(r) for r in responses]
        empty = [{"messages": []}]
        event_sequences.extend([empty] * 200)
        sequence_iter = iter(event_sequences)

        async def _astream(*args, **kwargs):
            try:
                events = next(sequence_iter)
            except StopIteration:
                events = empty
            for event in events:
                yield event

        mock_graph.astream = _astream
        mock_graph.ainvoke = AsyncMock()

        sv = state_values or {
            "vulnerabilities": [],
            "exploit_results": [],
            "flags_found": [],
            "total_cost_usd": 0.01,
            "total_tokens": 100,
        }
        mock_state = MagicMock()
        mock_state.values = sv
        mock_state.next = None  # no interrupts
        mock_state.tasks = []
        mock_graph.get_state.return_value = mock_state

        return mock_graph

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_completes_when_goals_met(self, mock_create, mock_create_llm, mock_decide):
        mock_graph = self._make_mock_graph(["Scanning ports...", "All done."])
        mock_create.return_value = mock_graph
        mock_create_llm.return_value = MagicMock()

        # First call: continue, second call: goals complete
        mock_decide.side_effect = ["Now find vulnerabilities.", "GOALS_COMPLETE"]

        cfg = OperatorConfig(goals=["scan ports"], target="10.0.0.1")
        op = OperatorAgent(cfg)
        result = op.run()

        assert result.status == "completed"
        assert result.turns == 2

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_escalates_on_unknown_question(self, mock_create, mock_create_llm, mock_decide):
        mock_graph = self._make_mock_graph(["What credentials should I use?"])
        mock_create.return_value = mock_graph
        mock_create_llm.return_value = MagicMock()

        mock_decide.return_value = "ESCALATE: What are the SSH credentials?"

        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)
        result = op.run()

        assert result.status == "escalated"
        assert "SSH credentials" in result.escalation_question

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_escalate_with_callback(self, mock_create, mock_create_llm, mock_decide):
        mock_graph = self._make_mock_graph(
            [
                "What credentials?",
                "Logged in successfully.",
            ]
        )
        mock_create.return_value = mock_graph
        mock_create_llm.return_value = MagicMock()

        mock_decide.side_effect = [
            "ESCALATE: What is the SSH password?",
            "GOALS_COMPLETE",
        ]

        cfg = OperatorConfig(
            goals=["scan"],
            target="10.0.0.1",
            on_escalate=lambda q: "password123",
        )
        op = OperatorAgent(cfg)
        result = op.run()

        assert result.status == "completed"

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_max_turns_stops(self, mock_create, mock_create_llm, mock_decide):
        mock_graph = self._make_mock_graph(["still scanning..."] * 5)
        mock_create.return_value = mock_graph
        mock_create_llm.return_value = MagicMock()
        mock_decide.return_value = "Continue scanning."

        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1", max_turns=3)
        op = OperatorAgent(cfg)
        result = op.run()

        assert result.turns == 3
        assert result.status == "max_turns"
        assert "max turns" in result.error.lower()

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_on_complete_callback(self, mock_create, mock_create_llm, mock_decide):
        mock_graph = self._make_mock_graph(["done"])
        mock_create.return_value = mock_graph
        mock_create_llm.return_value = MagicMock()
        mock_decide.return_value = "GOALS_COMPLETE"

        results = []
        cfg = OperatorConfig(
            goals=["scan"],
            target="10.0.0.1",
            on_complete=lambda r: results.append(r),
        )
        op = OperatorAgent(cfg)
        op.run()

        assert len(results) == 1
        assert results[0].status == "completed"

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_on_message_callback(self, mock_create, mock_create_llm, mock_decide):
        mock_graph = self._make_mock_graph(["scanning..."])
        mock_create.return_value = mock_graph
        mock_create_llm.return_value = MagicMock()
        mock_decide.return_value = "GOALS_COMPLETE"

        messages = []
        cfg = OperatorConfig(
            goals=["scan"],
            target="10.0.0.1",
            on_message=lambda role, content: messages.append((role, content)),
        )
        op = OperatorAgent(cfg)
        op.run()

        # Should have at least the agent message
        agent_msgs = [m for m in messages if m[0] == "agent"]
        assert len(agent_msgs) >= 1

    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_empty_response_ends_loop(self, mock_create, mock_create_llm):
        mock_graph = self._make_mock_graph([])  # no real responses
        mock_create.return_value = mock_graph
        mock_create_llm.return_value = MagicMock()

        cfg = OperatorConfig(goals=["scan"], target="10.0.0.1")
        op = OperatorAgent(cfg)
        result = op.run()

        # Should exit cleanly after first turn yields empty
        assert result.turns <= 1


class TestOperatorCostIsolation:
    """Issue #37: operator jobs must not read each other's spend.

    The inner graphs used to write the process-wide CostTracker running
    total into graph state, so a second job in the same process reported
    (and budget-checked against) every earlier job's accumulated cost.
    """

    @staticmethod
    def _real_graph(client):
        """Build a REAL NativeAgentGraph around a fake LLM client."""
        from clearwing.agent.graph import build_react_graph

        def factory(**kwargs):
            return build_react_graph(
                llm_with_tools=client,
                tools=[],
                system_prompt_fn=lambda state: "sys",
                model_name="fake-model",
                session_id=kwargs.get("session_id"),
                enable_knowledge_graph=False,
                enable_audit=False,
                enable_episodic_memory=False,
                enable_event_bus=False,
                enable_context_summarizer=False,
            )

        return factory

    @staticmethod
    def _client(usages):
        """Fake client whose achat_stream replays text responses with usage."""
        class _Resp:
            def __init__(self, prompt, completion):
                self.first_text = "working"
                self.texts = ["working"]
                self.tool_calls = []
                self.provider_model_name = "fake-model"
                self.reasoning_content = None

                class _Usage:
                    prompt_tokens = prompt
                    completion_tokens = completion
                    total_tokens = prompt + completion

                self.usage = _Usage()

        class _Client:
            model_name = "fake-model"

            def __init__(self, responses):
                self._responses = list(responses)

            async def achat_stream(self, **kwargs):
                return self._responses.pop(0)

        return _Client([_Resp(p, c) for p, c in usages])

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_second_job_reports_only_its_own_spend(
        self, mock_create, mock_create_llm, mock_decide
    ):
        mock_decide.return_value = "GOALS_COMPLETE"
        mock_create_llm.return_value = MagicMock()

        job1 = self._real_graph(self._client([(3000, 500)]))
        job2 = self._real_graph(self._client([(100, 20)]))
        mock_create.side_effect = [job1(), job2()]

        r1 = OperatorAgent(OperatorConfig(goals=["g"], target="10.0.0.1")).run()
        r2 = OperatorAgent(OperatorConfig(goals=["g"], target="10.0.0.2")).run()

        # Sonnet fallback pricing (fake-model): job2 must report only its
        # own call — not job1's 0.0165 on top of it.
        assert r2.cost_usd == pytest.approx((100 * 3 + 20 * 15) / 1_000_000)
        assert r1.cost_usd == pytest.approx((3000 * 3 + 500 * 15) / 1_000_000)
        assert r2.tokens_used == 120

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_cost_limit_of_second_job_unpolluted(
        self, mock_create, mock_create_llm, mock_decide
    ):
        mock_create_llm.return_value = MagicMock()
        # Job 1 spends 0.0165 per turn. Job 2 spends 0.0006 per turn; its
        # limit of 0.001 must be evaluated against ITS OWN spend only — a
        # second turn has to fit under the limit and the run must complete.
        # (The mocked _adecide_next is shared across both jobs: each one
        # consumes one "continue" + one "GOALS_COMPLETE".)
        mock_decide.side_effect = ["continue", "GOALS_COMPLETE"] * 2

        job1 = self._real_graph(self._client([(3000, 500), (3000, 500)]))
        job2 = self._real_graph(self._client([(100, 20), (100, 20)]))
        mock_create.side_effect = [job1(), job2()]

        r1 = OperatorAgent(OperatorConfig(goals=["g"], target="10.0.0.1")).run()
        assert r1.status == "completed"

        r2 = OperatorAgent(
            OperatorConfig(goals=["g"], target="10.0.0.2", cost_limit=0.001)
        ).run()
        assert r2.status == "completed"
        assert r2.turns == 2

    @staticmethod
    def _hunt_spending_graph(captured, usages, hunt_tokens):
        """Real graph whose client also books hunt-style spend per call.

        The first achat_stream books an extra CostTracker call attributed to
        the job's session id — exactly what a sourcehunt hunt spawned inside
        the job does (issue #41 ambient attribution) — so tests can assert
        the operator's limit/result see hunt spend.
        """

        def factory(**kwargs):
            from clearwing.agent.graph import build_react_graph
            from clearwing.observability.telemetry import CostTracker

            session_id = kwargs.get("session_id")
            captured["session_id"] = session_id

            class _Resp:
                def __init__(self, prompt, completion):
                    self.first_text = "working"
                    self.texts = ["working"]
                    self.tool_calls = []
                    self.provider_model_name = "fake-model"
                    self.reasoning_content = None

                    class _Usage:
                        prompt_tokens = prompt
                        completion_tokens = completion
                        total_tokens = prompt + completion

                    self.usage = _Usage()

            class _Client:
                model_name = "fake-model"

                def __init__(self):
                    self._responses = [_Resp(p, c) for p, c in usages]

                async def achat_stream(self, **kwargs_):
                    if captured.get("hunts_fired") is None and session_id:
                        captured["hunts_fired"] = True
                        CostTracker().record_llm_call(
                            hunt_tokens,
                            0,
                            "claude-sonnet-4-6",
                            session_id=session_id,
                        )
                    return self._responses.pop(0)

            return build_react_graph(
                llm_with_tools=_Client(),
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

        return factory

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_cost_limit_counts_hunt_spend(self, mock_create, mock_create_llm, mock_decide):
        """A job's cost_limit must fire on hunt spend attributed to the job.

        The limit used to read the graph's state total, which only covers
        the inner agent's own calls — hunts (usually the pipeline's largest
        share) were invisible, so a limited job ran them unbounded.
        """
        mock_create_llm.return_value = MagicMock()
        # Turn 1 books the hunt spend; the loop's SECOND limit check (before
        # turn 2) must trip on it.
        mock_decide.side_effect = ["continue", "GOALS_COMPLETE"]

        captured: dict = {}
        # Inner agent call: (100 in, 20 out) at Sonnet fallback = $0.0006.
        # Hunt: 1M input tokens at $3/M = $3.0, recorded under the job's
        # session id during turn 1. NB side_effect IS the factory (not its
        # result) so it receives create_agent's real kwargs — the job's
        # session id.
        mock_create.side_effect = self._hunt_spending_graph(
            captured, [(100, 20)], 1_000_000
        )

        result = OperatorAgent(
            OperatorConfig(goals=["g"], target="10.0.0.1", cost_limit=1.0)
        ).run()

        assert result.status == "cost_limit"
        # The limit tripped only because the hunt's $3.0 counted.
        assert result.cost_usd == pytest.approx(3.0 + 0.0006)

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_result_cost_includes_hunt_spend(self, mock_create, mock_create_llm, mock_decide):
        mock_create_llm.return_value = MagicMock()
        mock_decide.return_value = "GOALS_COMPLETE"

        captured: dict = {}
        mock_create.side_effect = self._hunt_spending_graph(captured, [(100, 20)], 200_000)

        result = OperatorAgent(OperatorConfig(goals=["g"], target="10.0.0.1")).run()

        assert result.status == "completed"
        # $0.0006 (inner call) + 200k * $3/M = $0.6 (hunt) — the result
        # field reports the job's whole spend, not just the graph's.
        assert result.cost_usd == pytest.approx(0.0006 + 0.6)

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_result_tokens_include_hunt_spend(self, mock_create, mock_create_llm, mock_decide):
        """PR #44 review P2: tokens_used mirrors cost attribution.

        Hunt tokens are recorded against the job's session id (the cost
        field already picks them up), but tokens_used used to read only the
        inner graph's state — the job reported full cost next to a token
        count missing the pipeline's main workload.
        """
        mock_create_llm.return_value = MagicMock()
        mock_decide.return_value = "GOALS_COMPLETE"

        captured: dict = {}
        mock_create.side_effect = self._hunt_spending_graph(captured, [(100, 20)], 200_000)

        result = OperatorAgent(OperatorConfig(goals=["g"], target="10.0.0.1")).run()

        assert result.status == "completed"
        # 200k hunt input tokens + the inner call's (100, 20) — not the
        # graph-state 120 alone.
        assert result.tokens_used == 200_000 + 120

    @patch(
        "clearwing.agent.operator.OperatorAgent._adecide_next",
        new_callable=AsyncMock,
    )
    @patch("clearwing.agent.graph._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_job_session_entry_forgotten_after_run(
        self, mock_create, mock_create_llm, mock_decide
    ):
        """PR #44 review P2: the job's session entry must be retired.

        Operator ids are uuid4().hex[:8]; in a long-lived process a
        colliding id would inherit this job's spend. arun() now forgets
        the entry on completion — the RESULT still reports the spend.
        """
        from clearwing.observability.telemetry import CostTracker

        mock_create_llm.return_value = MagicMock()
        mock_decide.return_value = "GOALS_COMPLETE"

        captured: dict = {}
        mock_create.side_effect = self._hunt_spending_graph(captured, [(100, 20)], 200_000)

        result = OperatorAgent(OperatorConfig(goals=["g"], target="10.0.0.1")).run()

        assert result.status == "completed"
        assert result.cost_usd == pytest.approx(0.0006 + 0.6)
        # But the tracker no longer holds an entry for the job's id.
        job_session_id = captured["session_id"]
        assert CostTracker().session_total(job_session_id) == 0.0
        assert CostTracker().session_tokens(job_session_id) == (0, 0)


class TestSupervisorUsageAccounting:
    """PR #44 review P1: the operator LLM's own calls must be booked.

    ``_adecide_next`` calls ``operator_llm.aask_text()`` every turn, but
    ``session_scope`` only attributes calls the runtime books — the
    supervisor's usage was invisible to the session total, so the cost
    limit and result ignored a potentially expensive operator_model.
    """

    @staticmethod
    def _fake_operator_llm(first_text: str, prompt_tokens: int, completion_tokens: int):
        from types import SimpleNamespace

        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=None,
        )

        class _Client:
            model_name = "claude-opus-4-6"
            provider_name = "anthropic"

            async def aask_text(self, **kwargs):
                del kwargs
                return SimpleNamespace(
                    first_text=first_text, texts=[first_text], usage=usage
                )

        return _Client()

    @patch("clearwing.agent.operator._create_llm")
    @patch("clearwing.agent.operator.create_agent")
    def test_supervisor_spend_counts_toward_cost_limit(self, mock_create, mock_create_llm):
        # NB: patch operator's OWN _create_llm binding — the module imports
        # the name from graph, so patching graph._create_llm would leave the
        # real client (and this host's config.yaml endpoint) in play.
        mock_graph = TestOperatorRun._make_mock_graph(["scanning..."])
        mock_create.return_value = mock_graph
        # Opus supervisor: 1M input + 100 output = $15 + $0.0075 = $15.0075.
        mock_create_llm.return_value = self._fake_operator_llm(
            "Continue with the next goal.", 1_000_000, 100
        )

        result = OperatorAgent(
            OperatorConfig(goals=["g"], target="10.0.0.1", cost_limit=5.0)
        ).run()

        # Turn 1 runs; the supervisor decision is booked; the loop's SECOND
        # limit check must trip on the supervisor spend alone.
        assert result.status == "cost_limit"
        assert result.cost_usd == pytest.approx(15.0075)  # beats state's 0.01
        assert result.tokens_used == 1_000_100  # beats state's 100
