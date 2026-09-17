"""Issue #61: audit coverage must stay in lockstep with cost accounting.

``CostTracker.record_llm_call`` fires a COST_UPDATE frame from five call
sites, but only the runtime's main step also wrote an ``AuditLogger``
row — live reconciliation of session 61279304 showed a 0.16% gap (one
430-in/722-out summarizer call priced but never audited). The single-entry
``book_llm_call`` helper now moves both halves together, and the operator
supervisor + sourcehunt hunter carry runtime-gated AuditLoggers of their
own.

The reconciliation contract under test: for a session with the tracker
enabled, ``sum(llm_call row cost_usd in ~/.clearwing/audit/<sid>/audit.jsonl)``
equals ``CostTracker().session_total(sid)``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest
from genai_pyo3 import ChatResponse, Usage

from clearwing.agent.operator import OperatorAgent, OperatorConfig
from clearwing.agent.runtime import NativeAgentGraph
from clearwing.agent.tools.hunt import HunterContext
from clearwing.observability.bookkeeping import (
    book_llm_call,
    init_session_audit_logger,
)
from clearwing.observability.telemetry import CostTracker
from clearwing.safety.audit import AuditLogger
from clearwing.sourcehunt.hunter import NativeHunter

_FIXTURE_C = Path(__file__).parent / "fixtures" / "vuln_samples" / "c_propagation"


def _audit_rows(base_dir, session_id: str) -> list[dict]:
    path = base_dir / session_id / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _llm_cost_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("event_type") == "llm_call"]


class _FakeUsage:
    def __init__(self, prompt=0, completion=0, total=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total
        self.prompt_tokens_details = None


class _FakeResponse:
    def __init__(self, text="done", usage=None):
        self.first_text = text
        self.texts = [text] if text else []
        self.tool_calls = []
        self.usage = usage or _FakeUsage()
        self.provider_model_name = "served-model-x"
        self.reasoning_content = None


class TestBookLlmCallHelper:
    def _patch_audit_home(self, monkeypatch, tmp_path):
        monkeypatch.setattr(AuditLogger, "BASE_DIR", tmp_path / "audit-home")

    def test_record_and_audit_move_together(self, monkeypatch, tmp_path):
        self._patch_audit_home(monkeypatch, tmp_path)
        sid = f"bk-{uuid.uuid4().hex[:8]}"
        tracker = CostTracker()

        cost = book_llm_call(
            430,
            722,
            tracker=tracker,
            model="claude-sonnet-4-6",
            provider="anthropic",
            session_id=sid,
            audit_logger=init_session_audit_logger(sid),
            agent="summarizer",
        )

        assert cost == pytest.approx(
            CostTracker.estimate_cost(430, 722, "claude-sonnet-4-6")
        )
        rows = _llm_cost_rows(_audit_rows(tmp_path / "audit-home", sid))
        assert len(rows) == 1
        details = rows[0]["details"]
        assert details["input_tokens"] == 430
        assert details["output_tokens"] == 722
        assert details["cost_usd"] == pytest.approx(cost)
        assert rows[0]["agent"] == "summarizer"
        # The reconciliation contract: audit sum == tracker session total.
        assert sum(r["details"]["cost_usd"] for r in rows) == pytest.approx(
            tracker.session_total(sid)
        )

    def test_none_audit_logger_still_records(self, monkeypatch, tmp_path):
        self._patch_audit_home(monkeypatch, tmp_path)
        sid = f"bk-{uuid.uuid4().hex[:8]}"
        cost = book_llm_call(10, 5, tracker=CostTracker(), model="claude-sonnet-4-6", session_id=sid)
        assert cost > 0.0
        assert CostTracker().session_total(sid) == pytest.approx(cost)
        # The audit home must be PINNED for this assertion to mean anything:
        # an unpinned run read the real ~/.clearwing tree — a path no
        # in-test write could ever land in — making "no rows" vacuously
        # true rather than evidence that audit_logger=None writes nothing.
        assert _audit_rows(tmp_path / "audit-home", sid) == []

    def test_pricing_and_audit_model_split(self, monkeypatch, tmp_path):
        self._patch_audit_home(monkeypatch, tmp_path)
        sid = f"bk-{uuid.uuid4().hex[:8]}"
        audit = init_session_audit_logger(sid)

        cost = book_llm_call(
            100,
            50,
            tracker=CostTracker(),
            model="claude-opus-4-7",  # priced under the configured name
            audit_model="claude-opus-4-7-20260901",  # audited under the served echo
            session_id=sid,
            audit_logger=audit,
        )

        assert cost == pytest.approx(
            CostTracker.estimate_cost(100, 50, "claude-opus-4-7")
        )
        rows = _llm_cost_rows(_audit_rows(tmp_path / "audit-home", sid))
        assert rows[0]["details"]["model"] == "claude-opus-4-7-20260901"

    def test_tracker_none_still_audits_with_estimated_cost(self, monkeypatch, tmp_path):
        self._patch_audit_home(monkeypatch, tmp_path)
        sid = f"bk-{uuid.uuid4().hex[:8]}"
        audit = init_session_audit_logger(sid)

        cost = book_llm_call(
            20, 10, tracker=None, model="claude-sonnet-4-6", session_id=sid, audit_logger=audit
        )

        assert cost == pytest.approx(
            CostTracker.estimate_cost(20, 10, "claude-sonnet-4-6")
        )
        rows = _llm_cost_rows(_audit_rows(tmp_path / "audit-home", sid))
        assert len(rows) == 1
        assert rows[0]["details"]["cost_usd"] == pytest.approx(cost)

    def test_both_none_is_a_noop(self):
        assert book_llm_call(100, 100, tracker=None, model="m") == 0.0

    def test_audit_write_failure_never_breaks_accounting(self, monkeypatch):
        class _ExplodingAudit:
            def log_llm_call(self, **kwargs):
                raise OSError("disk full")

        sid = f"bk-{uuid.uuid4().hex[:8]}"
        cost = book_llm_call(
            10,
            5,
            tracker=CostTracker(),
            model="claude-sonnet-4-6",
            session_id=sid,
            audit_logger=_ExplodingAudit(),
        )
        assert cost > 0.0
        assert CostTracker().session_total(sid) == pytest.approx(cost)

    def test_init_gating_requires_session_and_capability(self, monkeypatch, tmp_path):
        self._patch_audit_home(monkeypatch, tmp_path)
        assert init_session_audit_logger(None) is None
        assert init_session_audit_logger("") is None
        built = init_session_audit_logger("gate-check")
        assert isinstance(built, AuditLogger)
        # A capability-less process must degrade to None, not raise.
        import clearwing.capabilities as caps

        class _NoAudit:
            def has(self, name):
                return name != "audit"

        original = caps.capabilities
        monkeypatch.setattr(caps, "capabilities", _NoAudit())
        try:
            assert init_session_audit_logger("gate-check-2") is None
        finally:
            monkeypatch.setattr(caps, "capabilities", original)


class _SummarizerAwareLLM:
    """Drives the runtime's main step (achat_stream) and the context
    summarizer (aask_text) with distinct usages so both booking paths run."""

    model_name = "stub-model"
    provider_name = "stub"
    context_budget_tokens = 1000  # tiny window → the 10-msg history trips it

    async def achat_stream(self, **kwargs):
        return _FakeResponse("assistant answer", usage=_FakeUsage(1000, 200, 1200))

    async def aask_text(self, **kwargs):
        return _FakeResponse("summary text", usage=_FakeUsage(430, 722, 1152))


class TestRuntimeSummarizerAndMainStep:
    @pytest.mark.asyncio
    async def test_summarizer_and_main_step_both_audited_and_reconciled(
        self, monkeypatch, tmp_path
    ):
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        sid = f"rt-{uuid.uuid4().hex[:8]}"

        graph = NativeAgentGraph(
            llm=_SummarizerAwareLLM(),
            native_tools=[],
            tools=[],
            system_prompt_fn=lambda s: "sys",
            model_name="m",
            session_id=sid,
            state_updater_fn=lambda *a, **k: {},
            knowledge_graph_populator_fn=None,
            input_guardrail_tool_names=frozenset(),
            output_guardrail_tool_names=frozenset(),
            enable_cost_tracker=True,
            enable_episodic_memory=False,
            enable_audit=True,
            enable_knowledge_graph=False,
            enable_input_guardrail=False,
            enable_output_guardrail=False,
            enable_event_bus=False,
            enable_context_summarizer=True,
            agent_limits=None,
        )

        big = "x" * 2000  # ~500 tokens each; 10 messages ≈ 5000 > 800 threshold
        history = [{"role": "user", "content": f"note {i} {big}"} for i in range(10)]
        cfg = {"configurable": {"thread_id": "ws-audit"}}
        async for _ in graph.astream({"messages": history}, cfg):
            pass

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        agents = [r["agent"] for r in rows]
        # The session 61279304 gap class: the summarizer call (430/722 in
        # the live incident) MUST now leave an audit row too.
        assert "summarizer" in agents
        assert "main" in agents
        summarizer_row = next(r for r in rows if r["agent"] == "summarizer")
        assert summarizer_row["details"]["input_tokens"] == 430
        assert summarizer_row["details"]["output_tokens"] == 722
        # Audit model keeps the served echo; pricing normalized upstream.
        main_row = next(r for r in rows if r["agent"] == "main")
        assert main_row["details"]["model"] == "served-model-x"
        # Reconciliation: audit sum == tracker session total.
        assert sum(r["details"]["cost_usd"] for r in rows) == pytest.approx(
            CostTracker().session_total(sid)
        )


class TestOperatorSupervisorAudit:
    def test_supervisor_call_is_audited_and_reconciled(self, monkeypatch, tmp_path):
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        sid = f"op-{uuid.uuid4().hex[:8]}"

        operator = OperatorAgent(OperatorConfig(goals=["scan"], target="127.0.0.1"))
        operator._session_id = sid
        operator._audit_logger = init_session_audit_logger(sid)

        class _OpLLM:
            model_name = "operator-model"
            provider_name = "openai"

            async def aask_text(self, **kwargs):
                return _FakeResponse("Continue with the next goal.", usage=_FakeUsage(210, 90, 300))

        decision = asyncio.run(operator._adecide_next(_OpLLM(), "found ports 22, 80"))
        assert decision.startswith("Continue")

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 1
        assert rows[0]["agent"] == "operator"
        assert rows[0]["details"]["input_tokens"] == 210
        assert rows[0]["details"]["output_tokens"] == 90
        assert sum(r["details"]["cost_usd"] for r in rows) == pytest.approx(
            CostTracker().session_total(sid)
        )


class TestHunterAudit:
    def test_hunter_main_call_is_audited_and_reconciled(self, monkeypatch, tmp_path):
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        sid = f"sh-{uuid.uuid4().hex[:8]}"

        class _HunterLLM:
            model_name = "hunter-model"
            provider_name = "stub"

            async def achat(self, **kwargs):
                return ChatResponse(
                    content=[{"text": "No findings."}],
                    usage=Usage(prompt_tokens=300, completion_tokens=120, total_tokens=420),
                    provider_model_name="hunter-served",
                )

        ctx = HunterContext(
            repo_path=str(_FIXTURE_C),
            findings=[],
            file_path="src/codec_a.c",
            session_id=sid,
            specialist="general",
        )
        hunter = NativeHunter(
            llm=_HunterLLM(),
            prompt="system prompt",
            tools=[],
            ctx=ctx,
            max_steps=1,
        )

        result = asyncio.run(hunter.arun())
        assert result.findings == []

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 1
        assert rows[0]["agent"] == "hunter"
        # Audit prefers the served model echo (runtime effective_model parity).
        assert rows[0]["details"]["model"] == "hunter-served"
        assert rows[0]["details"]["input_tokens"] == 300
        assert rows[0]["details"]["output_tokens"] == 120
        assert sum(r["details"]["cost_usd"] for r in rows) == pytest.approx(
            CostTracker().session_total(sid)
        )


class TestEndToEndReconciliation:
    def test_main_summarizer_and_operator_sum_to_session_total(
        self, monkeypatch, tmp_path
    ):
        """One session, three spend sources (main step + summarizer +
        operator supervisor): audit rows and CostTracker agree exactly."""

        async def _drive_graph(graph, cfg) -> None:
            big = "y" * 2000
            history = [{"role": "user", "content": f"note {i} {big}"} for i in range(10)]
            async for _ in graph.astream({"messages": history}, cfg):
                pass

        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        sid = f"e2e-{uuid.uuid4().hex[:8]}"

        graph = NativeAgentGraph(
            llm=_SummarizerAwareLLM(),
            native_tools=[],
            tools=[],
            system_prompt_fn=lambda s: "sys",
            model_name="m",
            session_id=sid,
            state_updater_fn=lambda *a, **k: {},
            knowledge_graph_populator_fn=None,
            input_guardrail_tool_names=frozenset(),
            output_guardrail_tool_names=frozenset(),
            enable_cost_tracker=True,
            enable_episodic_memory=False,
            enable_audit=True,
            enable_knowledge_graph=False,
            enable_input_guardrail=False,
            enable_output_guardrail=False,
            enable_event_bus=False,
            enable_context_summarizer=True,
            agent_limits=None,
        )
        asyncio.run(_drive_graph(graph, {"configurable": {"thread_id": "ws-e2e"}}))

        operator = OperatorAgent(OperatorConfig(goals=["scan"], target="127.0.0.1"))
        operator._session_id = sid
        operator._audit_logger = init_session_audit_logger(sid)

        class _OpLLM:
            model_name = "operator-model"
            provider_name = "openai"

            async def aask_text(self, **kwargs):
                return _FakeResponse("Continue.", usage=_FakeUsage(210, 90, 300))

        asyncio.run(operator._adecide_next(_OpLLM(), "progress"))

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert {r["agent"] for r in rows} == {"main", "summarizer", "operator"}
        assert sum(r["details"]["cost_usd"] for r in rows) == pytest.approx(
            CostTracker().session_total(sid)
        )
