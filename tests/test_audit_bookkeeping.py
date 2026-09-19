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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from genai_pyo3 import ChatResponse, Usage

from clearwing.agent.operator import OperatorAgent, OperatorConfig
from clearwing.agent.runtime import NativeAgentGraph
from clearwing.agent.tools.hunt import HunterContext
from clearwing.bench.crash_classifier import CrashClassifier
from clearwing.bench.ossfuzz import BenchmarkTarget, OssFuzzBenchmark
from clearwing.bench.results import TargetResult
from clearwing.data.memory.summarizer import ContextSummarizer
from clearwing.llm.native import AsyncLLMClient
from clearwing.observability.bookkeeping import (
    book_llm_call,
    init_session_audit_logger,
)
from clearwing.observability.telemetry import CostTracker
from clearwing.providers.env import EndpointPricing
from clearwing.safety.audit import AuditLogger
from clearwing.sourcehunt.hunter import NativeHunter
from clearwing.sourcehunt.runner import SourceHuntRunner, _specialist_book_role

_FIXTURE_C = Path(__file__).parent / "fixtures" / "vuln_samples" / "c_propagation"


def _audit_rows(base_dir, session_id: str) -> list[dict]:
    path = base_dir / session_id / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _llm_cost_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("event_type") == "llm_call"]


def _audit_session_dirs(base_dir, prefix: str) -> list[str]:
    """Audit session dir names equal to *prefix* or freshly suffixed
    (``{prefix}-{uuid8}`` — the standalone hunter's per-execution ids)."""
    if not base_dir.exists():
        return []
    return sorted(
        d.name
        for d in base_dir.iterdir()
        if d.name == prefix or d.name.startswith(f"{prefix}-")
    )


class _FakeUsage:
    def __init__(self, prompt=0, completion=0, total=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total
        self.prompt_tokens_details = None


class _FakeResponse:
    def __init__(self, text="done", usage=None, provider_model_name="served-model-x"):
        self.first_text = text
        self.texts = [text] if text else []
        self.tool_calls = []
        self.usage = usage or _FakeUsage()
        self.provider_model_name = provider_model_name
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

    def test_endpoint_pricing_overrides_table_for_both_halves(self, monkeypatch, tmp_path):
        """Codex PR-69 r1 (P2): the booked path must price with the
        client's authoritative ``EndpointPricing`` when it has one — the
        tracker table's Sonnet fallback would diverge from the spend
        ledger on custom endpoints whose model has no PRICING entry."""
        self._patch_audit_home(monkeypatch, tmp_path)
        sid = f"bk-{uuid.uuid4().hex[:8]}"
        tracker = CostTracker()

        cost = book_llm_call(
            1000,
            500,
            cached_tokens=400,
            tracker=tracker,
            model="custom-endpoint-model",  # no PRICING entry
            session_id=sid,
            audit_logger=init_session_audit_logger(sid),
            pricing=EndpointPricing(
                input_per_mtok=2.0, output_per_mtok=6.0, cached_per_mtok=0.2
            ),
        )

        # 600 uncached * 2.0 + 400 cached * 0.2 + 500 out * 6.0, per 1M.
        expected = (600 * 2.0 + 400 * 0.2 + 500 * 6.0) / 1_000_000
        assert cost == pytest.approx(expected)
        # NOT the Sonnet fallback tier the table would use.
        fallback = CostTracker.estimate_cost(1000, 500, "custom-endpoint-model", 400)
        assert expected != pytest.approx(fallback)
        # BOTH halves carry the endpoint price: audit row + tracker bucket.
        rows = _llm_cost_rows(_audit_rows(tmp_path / "audit-home", sid))
        assert rows[0]["details"]["cost_usd"] == pytest.approx(expected)
        assert tracker.session_total(sid) == pytest.approx(expected)

    def test_invalid_endpoint_pricing_falls_back_to_table(self, monkeypatch, tmp_path):
        """Never-raise doctrine: a malformed pricing object degrades to the
        table lookup instead of raising into the bookkeeping path."""
        self._patch_audit_home(monkeypatch, tmp_path)
        sid = f"bk-{uuid.uuid4().hex[:8]}"
        for bad in (
            EndpointPricing(input_per_mtok=-1.0, output_per_mtok=6.0),  # negative
            object(),  # no pricing fields at all
        ):
            cost = book_llm_call(
                10,
                5,
                tracker=CostTracker(),
                model="claude-sonnet-4-6",
                session_id=sid,
                pricing=bad,
            )
            assert cost == pytest.approx(
                CostTracker.estimate_cost(10, 5, "claude-sonnet-4-6")
            )

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

    @pytest.mark.asyncio
    async def test_summarizer_audit_model_keeps_provider_echo(self, monkeypatch, tmp_path):
        """Codex PR-63 r2: the summarizer's audit row must record the model
        the PROVIDER echoed on the SUMMARY response (``served_model`` from
        ``ContextSummarizer.summarize``), while pricing stays on the
        configured member key — the audit_model split the main loop already
        had, which the summary path used to miss entirely (the row carried
        the pricing key, hiding which model actually served the summary)."""
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        sid = f"rt-{uuid.uuid4().hex[:8]}"

        class _EchoingSummaryLLM(_SummarizerAwareLLM):
            model_name = "claude-sonnet-4-6"  # the configured pricing key

            async def aask_text(self, **kwargs):
                return _FakeResponse(
                    "summary text",
                    usage=_FakeUsage(430, 722, 1152),
                    provider_model_name="claude-opus-4-7-20260901",
                )

        graph = NativeAgentGraph(
            llm=_EchoingSummaryLLM(),
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

        big = "x" * 2000
        history = [{"role": "user", "content": f"note {i} {big}"} for i in range(10)]
        cfg = {"configurable": {"thread_id": "ws-audit-echo"}}
        async for _ in graph.astream({"messages": history}, cfg):
            pass

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        summarizer_row = next(r for r in rows if r["agent"] == "summarizer")
        # The audit row shows the provider's echo, not the configured key.
        assert summarizer_row["details"]["model"] == "claude-opus-4-7-20260901"
        # Pricing stayed on the configured member key — the two tiers
        # differ, so the cost pins WHICH key priced the call.
        configured = CostTracker.estimate_cost(430, 722, "claude-sonnet-4-6")
        echoed = CostTracker.estimate_cost(430, 722, "claude-opus-4-7-20260901")
        assert configured != echoed  # guard: the split must be observable
        assert summarizer_row["details"]["cost_usd"] == pytest.approx(configured)
        # Reconciliation survives the split: audit sum == tracker total.
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

    def test_supervisor_audit_model_keeps_provider_echo(self, monkeypatch, tmp_path):
        """Codex PR-63 r1: the operator books PRICING under the member
        (configured/served) key while the audit row records the model the
        PROVIDER echoed — the runtime/hunter audit_model split, so a
        versioned echo never rewrites the pricing key or vice versa."""
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        sid = f"op-{uuid.uuid4().hex[:8]}"

        operator = OperatorAgent(OperatorConfig(goals=["scan"], target="127.0.0.1"))
        operator._session_id = sid
        operator._audit_logger = init_session_audit_logger(sid)

        class _OpLLM:
            model_name = "claude-sonnet-4-6"  # the configured member key
            provider_name = "anthropic"

            async def aask_text(self, **kwargs):
                return _FakeResponse(
                    "Continue with the next goal.",
                    usage=_FakeUsage(210, 90, 300),
                    provider_model_name="claude-opus-4-7-20260901",
                )

        decision = asyncio.run(operator._adecide_next(_OpLLM(), "progress"))
        assert decision.startswith("Continue")

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 1
        # The audit row shows the provider's echo, not the configured key.
        assert rows[0]["details"]["model"] == "claude-opus-4-7-20260901"
        # Pricing stayed on the configured member key — the two tiers
        # differ, so the cost pins WHICH key priced the call.
        configured = CostTracker.estimate_cost(210, 90, "claude-sonnet-4-6")
        echoed = CostTracker.estimate_cost(210, 90, "claude-opus-4-7-20260901")
        assert configured != echoed  # guard: the split must be observable
        assert rows[0]["details"]["cost_usd"] == pytest.approx(configured)
        assert CostTracker().session_total(sid) == pytest.approx(configured)


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

        # Codex PR-63 r1: a standalone hunt books under a FRESH suffixed
        # id (ctx.session_id-<uuid8>) — find the execution's own trail.
        session_dirs = _audit_session_dirs(audit_home, sid)
        assert len(session_dirs) == 1
        book_id = session_dirs[0]
        assert book_id.startswith(f"{sid}-")

        rows = _llm_cost_rows(_audit_rows(audit_home, book_id))
        assert len(rows) == 1
        assert rows[0]["agent"] == "hunter"
        # Audit prefers the served model echo (runtime effective_model parity).
        assert rows[0]["details"]["model"] == "hunter-served"
        assert rows[0]["details"]["input_tokens"] == 300
        assert rows[0]["details"]["output_tokens"] == 120
        # Codex PR-63 r3: reconciliation holds within the session's
        # LIFETIME; arun now forgets the minted standalone bucket on exit
        # (it would otherwise leak one tracker entry per hunt forever),
        # so post-hoc the DISK audit trail is the source of truth and the
        # tracker side reads zero (bookkeeping.py doctrine).
        assert rows[0]["details"]["cost_usd"] == pytest.approx(
            CostTracker.estimate_cost(300, 120, "hunter-model")
        )
        assert CostTracker().session_total(book_id) == 0.0

    def test_hunter_summarizer_row_keeps_provider_echo(self, monkeypatch, tmp_path):
        """Codex PR-63 r2: the hunter's CONTEXT-SUMMARIZER book call must
        carry audit_model too — the provider echo from the summary response
        (``served_model``), with pricing on the configured member key. The
        main-call split existed (r1); the summary path wrote only the
        pricing key into the audit row."""

        class _HunterSummaryLLM:
            model_name = "hunter-model"  # the configured pricing key
            provider_name = "stub"

            async def achat(self, **kwargs):
                return ChatResponse(
                    content=[{"text": "No findings."}],
                    usage=Usage(prompt_tokens=300, completion_tokens=120, total_tokens=420),
                    provider_model_name="hunter-served",
                )

            async def aask_text(self, **kwargs):
                return ChatResponse(
                    content=[{"text": "compacted summary"}],
                    usage=Usage(prompt_tokens=430, completion_tokens=722, total_tokens=1152),
                    provider_model_name="hunter-summary-served",
                )

        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        sid = f"sh-{uuid.uuid4().hex[:8]}"

        ctx = HunterContext(
            repo_path=str(_FIXTURE_C),
            findings=[],
            file_path="src/codec_a.c",
            session_id=sid,
            specialist="general",
        )
        hunter = NativeHunter(
            llm=_HunterSummaryLLM(),
            prompt="system prompt",
            tools=[],
            ctx=ctx,
            max_steps=1,
            summarizer=ContextSummarizer(),
            # ~125k estimated tokens > 80% of the summarizer's default
            # 150k window, with a coverable old segment — trips the
            # summary path on the hunt's first model step.
            initial_user_message="x " * 250_000,
        )

        result = asyncio.run(hunter.arun())
        assert result.findings == []

        session_dirs = _audit_session_dirs(audit_home, sid)
        assert len(session_dirs) == 1
        rows = _llm_cost_rows(_audit_rows(audit_home, session_dirs[0]))
        agents = [r["agent"] for r in rows]
        assert "summarizer" in agents
        assert "hunter" in agents

        summarizer_row = next(r for r in rows if r["agent"] == "summarizer")
        # The audit row records the model that served the SUMMARY call,
        # while pricing stayed on the configured member key.
        assert summarizer_row["details"]["model"] == "hunter-summary-served"
        assert summarizer_row["details"]["input_tokens"] == 430
        assert summarizer_row["details"]["output_tokens"] == 722
        assert summarizer_row["details"]["cost_usd"] == pytest.approx(
            CostTracker.estimate_cost(430, 722, "hunter-model")
        )
        # The main call keeps its own (r1) echo split.
        hunter_row = next(r for r in rows if r["agent"] == "hunter")
        assert hunter_row["details"]["model"] == "hunter-served"
        # Codex PR-63 r3: the minted standalone bucket is forgotten when
        # arun exits — post-hoc reconciliation reads the DISK trail (the
        # tracker side is zero by design), per the bookkeeping doctrine.
        assert CostTracker().session_total(session_dirs[0]) == 0.0


class TestHunterAuditSessionId:
    """Codex PR-63 r1: the hunter's bookkeeping id.

    Standalone hunts (no ambient session) get a FRESH ``{ctx.session_id}-
    {uuid8}`` id per execution — deterministic ctx ids (exploit-{finding},
    elaborate-{finding}) are REUSED across runs/restarts, and appending a
    new run onto the stale audit.jsonl made executions unreconcilable.
    An ambient session (operator job / webui turn that spawned the hunt)
    still keeps ONE shared trail per session.
    """

    def _make_hunter(self, session_id: str) -> NativeHunter:
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
            session_id=session_id,
            specialist="general",
        )
        return NativeHunter(
            llm=_HunterLLM(),
            prompt="system prompt",
            tools=[],
            ctx=ctx,
            max_steps=1,
        )

    def test_standalone_runs_each_get_a_fresh_suffixed_trail(self, monkeypatch, tmp_path):
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        ctx_sid = f"sh-{uuid.uuid4().hex[:8]}"

        asyncio.run(self._make_hunter(ctx_sid).arun())
        asyncio.run(self._make_hunter(ctx_sid).arun())

        # Two executions of the SAME deterministic ctx id → two distinct
        # audit trails, both prefixed with the ctx id (+ "-" + 8 hex).
        session_dirs = _audit_session_dirs(audit_home, ctx_sid)
        assert len(session_dirs) == 2
        assert session_dirs[0] != session_dirs[1]
        for name in session_dirs:
            suffix = name[len(ctx_sid) + 1 :]
            assert len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix)
            rows = _llm_cost_rows(_audit_rows(audit_home, name))
            assert len(rows) == 1  # this run's call, not the stale one's
            assert rows[0]["agent"] == "hunter"

    def test_ambient_session_keeps_one_shared_trail(self, monkeypatch, tmp_path):
        from clearwing.agent.tooling import session_scope

        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        ctx_sid = f"sh-{uuid.uuid4().hex[:8]}"
        parent_sid = f"job-{uuid.uuid4().hex[:8]}"

        with session_scope(parent_sid):
            asyncio.run(self._make_hunter(ctx_sid).arun())
            asyncio.run(self._make_hunter(ctx_sid).arun())

        # No per-execution trail was created: both hunts appended to the
        # spawning session's ONE shared audit file.
        assert _audit_session_dirs(audit_home, ctx_sid) == []
        rows = _llm_cost_rows(_audit_rows(audit_home, parent_sid))
        assert len(rows) == 2
        assert all(r["agent"] == "hunter" for r in rows)
        assert sum(r["details"]["cost_usd"] for r in rows) == pytest.approx(
            CostTracker().session_total(parent_sid)
        )


class TestStandaloneCostBucketReclamation:
    """Codex PR-63 r3 (P2): the standalone hunter's minted cost bucket.

    ``arun`` books standalone executions under a fresh
    ``{ctx.session_id}-{uuid8}`` id in the process-global CostTracker;
    without reclamation, every hunt in a long-lived process leaked one
    ``_session_totals``/``_session_tokens`` entry forever. The bucket is
    now dropped on EVERY exit path — but only for ids WE minted: an
    ambient session id (operator job / webui turn) belongs to its parent
    and must survive the hunt. The disk audit trail and the COST_UPDATE
    frames (which carry their values at emit time) are unaffected by the
    forget; ``HunterRunResult`` totals come from local counters.
    """

    def _make_hunter(self, session_id: str) -> NativeHunter:
        class _HunterLLM:
            model_name = "claude-sonnet-4-6"  # priced row → bucket total > 0
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
            session_id=session_id,
            specialist="general",
        )
        return NativeHunter(
            llm=_HunterLLM(),
            prompt="system prompt",
            tools=[],
            ctx=ctx,
            max_steps=1,
        )

    def _spy_on_forget(self, monkeypatch) -> list[str]:
        forgotten: list[str] = []
        original_forget = CostTracker.forget_session

        def _forget_spy(tracker_self, session_id):
            forgotten.append(session_id)
            original_forget(tracker_self, session_id)

        monkeypatch.setattr(CostTracker, "forget_session", _forget_spy)
        return forgotten

    def test_standalone_bucket_reclaimed_after_run(self, monkeypatch, tmp_path):
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        ctx_sid = f"sh-{uuid.uuid4().hex[:8]}"
        forgotten = self._spy_on_forget(monkeypatch)

        result = asyncio.run(self._make_hunter(ctx_sid).arun())
        assert result.findings == []

        session_dirs = _audit_session_dirs(audit_home, ctx_sid)
        assert len(session_dirs) == 1
        book_id = session_dirs[0]
        rows = _llm_cost_rows(_audit_rows(audit_home, book_id))
        # The audit trail is disk-persistent and unaffected by the forget.
        assert len(rows) == 1
        assert rows[0]["details"]["cost_usd"] > 0.0
        # The minted bucket WAS reclaimed: the id reached forget_session
        # and neither tracker dict still holds it.
        assert book_id in forgotten
        assert CostTracker().session_total(book_id) == 0.0
        assert CostTracker().session_tokens(book_id) == (0, 0)

    def test_ambient_bucket_survives_standalone_reclaim(self, monkeypatch, tmp_path):
        from clearwing.agent.tooling import session_scope

        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        ctx_sid = f"sh-{uuid.uuid4().hex[:8]}"
        parent_sid = f"job-{uuid.uuid4().hex[:8]}"
        forgotten = self._spy_on_forget(monkeypatch)

        with session_scope(parent_sid):
            asyncio.run(self._make_hunter(ctx_sid).arun())

        rows = _llm_cost_rows(_audit_rows(audit_home, parent_sid))
        assert len(rows) == 1
        total = sum(r["details"]["cost_usd"] for r in rows)
        # The ambient bucket's lifecycle belongs to the parent session —
        # the hunt must NOT forget an id it did not mint.
        assert parent_sid not in forgotten
        assert total > 0.0
        assert CostTracker().session_total(parent_sid) == pytest.approx(total)
        assert CostTracker().session_tokens(parent_sid) == (300, 120)

    def test_failed_reclaim_never_breaks_the_run(self, monkeypatch, tmp_path):
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        ctx_sid = f"sh-{uuid.uuid4().hex[:8]}"

        def _exploding_forget(tracker_self, session_id):
            raise RuntimeError("tracker lock poisoned")

        monkeypatch.setattr(CostTracker, "forget_session", _exploding_forget)

        # Cleanup must never mask the hunt's real result: the run
        # completes and the audit trail still lands on disk.
        result = asyncio.run(self._make_hunter(ctx_sid).arun())
        assert result.findings == []
        session_dirs = _audit_session_dirs(audit_home, ctx_sid)
        assert len(session_dirs) == 1
        assert len(_llm_cost_rows(_audit_rows(audit_home, session_dirs[0]))) == 1


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


class TestSpecialistAudit:
    """Issue #64: sourcehunt specialist LLM calls were completely unmetered.

    The runner now attaches a ``with_bookkeeping`` view to every REAL
    AsyncLLMClient it hands out, so every successful ``achat`` lands BOTH
    halves (CostTracker bucket + audit row) under the stage's role tag,
    with ambient-or-runner session attribution. AsyncMock/MagicMock test
    seams pass through the isinstance gate untouched.
    """

    # (task, budget_stage, expected agent role) — the production call
    # shapes for every specialist that used to run unmetered.
    _STAGES = [
        ("sourcehunt_exploit", "auto_patch", "patcher"),
        ("sourcehunt_exploit", "exploit", "exploiter"),
        ("verifier", "variant_loop", "variant"),
        ("hunter", "harness", "harness"),
        ("verifier", "stability", "stability"),
        ("verifier", "mechanism_extraction", "mechanism"),
        ("proof_local", "proof_local", "proof"),
        # Review round: production verifier sites pass stage "verify"
        # (the _verify_finding client + _preflight_budget_clients) — the
        # old ("verifier","verifier") tuple pinned a phantom stage that
        # never reached a real client.
        ("verifier", "verify", "verifier"),
        # Review round: hunter stages keep the historical "hunter" audit
        # tag (analytics continuity with standalone-hunt rows).
        ("hunter", "hunt", "hunter"),
    ]

    def _make_booked_runner(
        self,
        monkeypatch,
        tmp_path,
        *,
        parent_session_id: str | None = None,
        usage: Usage | None = None,
    ) -> tuple[SourceHuntRunner, Path]:
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        monkeypatch.setattr(AsyncLLMClient, "_build_client", lambda self, cls: object())

        effective_usage = usage or Usage(
            prompt_tokens=210, completion_tokens=90, total_tokens=300
        )

        async def fake_policy(self, client_obj, request, options):
            return ChatResponse(
                content=[{"text": "done"}],
                usage=effective_usage,
                provider_model_name="specialist-served",
            )

        monkeypatch.setattr(AsyncLLMClient, "_achat_with_provider_policy", fake_policy)
        runner = SourceHuntRunner(
            repo_url=str(tmp_path),
            local_path=str(tmp_path),
            depth="quick",
            output_dir=str(tmp_path / "out"),
            parent_session_id=parent_session_id,
            enable_knowledge_graph=False,
            enable_mechanism_memory=False,
        )
        return runner, audit_home

    @staticmethod
    def _real_client() -> AsyncLLMClient:
        return AsyncLLMClient(
            model_name="claude-sonnet-4-6", provider_name="anthropic", api_key="test"
        )

    def test_role_map_aliases_and_verbatim_fallback(self):
        assert _specialist_book_role("auto_patch") == "patcher"
        assert _specialist_book_role("proof_frontier") == "proof"
        # Any unlisted proof_* route still maps to the proof role.
        assert _specialist_book_role("proof_some_new_route") == "proof"
        assert _specialist_book_role("dynamic_verification") == "verifier"
        # Review round: production verifier sites pass stage "verify".
        assert _specialist_book_role("verify") == "verifier"
        # Review round: hunter stages keep the historical "hunter" audit
        # tag for analytics continuity with standalone-hunt rows.
        assert _specialist_book_role("hunt") == "hunter"
        assert _specialist_book_role("subsystem_hunt") == "hunter"
        assert _specialist_book_role("elaboration") == "hunter"
        # Unknown stages fall back to the stage string verbatim.
        assert _specialist_book_role("rank") == "rank"
        assert _specialist_book_role("some_new_stage") == "some_new_stage"

    @pytest.mark.parametrize("task,stage,role", _STAGES)
    def test_each_stage_books_both_halves(self, monkeypatch, tmp_path, task, stage, role):
        runner, audit_home = self._make_booked_runner(monkeypatch, tmp_path)
        view = runner._get_native_client(task, self._real_client(), budget_stage=stage)
        assert isinstance(view, AsyncLLMClient)

        asyncio.run(view.aask_text(system="s", user="u"))

        sid = runner._session_id
        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 1
        assert rows[0]["agent"] == role
        assert rows[0]["details"]["input_tokens"] == 210
        assert rows[0]["details"]["output_tokens"] == 90
        # Audit keeps the served model echo; pricing stays on the
        # configured key (the hunter/runtime audit_model split).
        assert rows[0]["details"]["model"] == "specialist-served"
        total = sum(r["details"]["cost_usd"] for r in rows)
        assert total == pytest.approx(
            CostTracker.estimate_cost(210, 90, "claude-sonnet-4-6")
        )
        assert total > 0.0
        # BOTH halves: the tracker bucket holds the same total.
        assert CostTracker().session_total(sid) == pytest.approx(total)
        runner._reclaim_minted_cost_bucket()

    def test_booked_view_prices_with_endpoint_pricing(self, monkeypatch, tmp_path):
        """Codex PR-69 r1 (P2): the with_bookkeeping view prices each call
        with the client's authoritative ``EndpointPricing`` (the same
        ``self.pricing`` the spend ledger reads) — the tracker's table
        lookup would bill unknown custom models at Sonnet rates and make
        audit/COST_UPDATE totals diverge from the ledger."""
        runner, audit_home = self._make_booked_runner(monkeypatch, tmp_path)
        client = AsyncLLMClient(
            model_name="custom-endpoint-model",  # no PRICING entry
            provider_name="openai",
            api_key="test",
            pricing=EndpointPricing(input_per_mtok=1.0, output_per_mtok=3.0),
        )
        view = runner._get_native_client("hunter", client, budget_stage="hunt")
        asyncio.run(view.aask_text(system="s", user="u"))

        sid = runner._session_id
        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 1
        # 210 in * 1.0 + 90 out * 3.0 per 1M — the endpoint's rates.
        expected = (210 * 1.0 + 90 * 3.0) / 1_000_000
        assert rows[0]["details"]["cost_usd"] == pytest.approx(expected)
        assert CostTracker().session_total(sid) == pytest.approx(expected)
        # Guard: the table would have billed the Sonnet fallback tier.
        assert not CostTracker.has_pricing("custom-endpoint-model")
        assert expected != pytest.approx(
            CostTracker.estimate_cost(210, 90, "custom-endpoint-model")
        )
        runner._reclaim_minted_cost_bucket()

    def test_mock_seams_pass_through_untouched(self, monkeypatch, tmp_path):
        runner, audit_home = self._make_booked_runner(monkeypatch, tmp_path)
        seam = AsyncMock()
        out = runner._get_native_client("verifier", seam, budget_stage="verify")
        assert out is seam  # no with_bookkeeping on test doubles
        assert not _audit_rows(audit_home, runner._session_id)

    def test_booked_override_still_binds_spend_ledger(self, monkeypatch, tmp_path):
        """Codex PR-69 r2 (P1): the double-wrap guard must not bypass the
        spend ledger. An override that already carries a with_bookkeeping
        view must keep its booking attrs, but under budget_usd the runner
        still attaches its spend ledger — the old guard returned the
        override before with_spend_ledger, so a booked override ran with
        ZERO reservations. with_spend_ledger is a copy.copy view that only
        sets _spend_ledger/_spend_stage: the booking survives unchanged
        and the call settles AND books exactly once."""
        runner, _ = self._make_booked_runner(
            monkeypatch,
            tmp_path,
            usage=Usage(prompt_tokens=210, completion_tokens=1, total_tokens=211),
        )
        runner.budget_usd = 1.0
        runner.input_price_per_million = 0.0
        runner.output_price_per_million = 1_000_000.0  # 1 output token = $1
        ledger = runner._ensure_spend_ledger()
        assert ledger.enforcing

        booked = self._real_client().with_bookkeeping(
            agent="harness",
            session_id="sh-external",
            tracker=CostTracker(),
            audit_logger=None,
        )
        out = runner._get_native_client("hunter", booked, budget_stage="hunt")

        # A NEW view — the ledger landed on the booked override.
        assert out is not booked
        assert out.spend_ledger is ledger
        assert out._spend_stage == "hunt"
        # The booking attrs carried over via the shallow copy, unchanged.
        assert out._book_agent == "harness"
        assert out._book_session_id == "sh-external"

        asyncio.run(out.aask_text(system="s", user="u"))

        # The ledger half settled — the reservation the old guard skipped.
        assert ledger.spent_usd == pytest.approx(1.0)
        # The bookkeeping half ran EXACTLY once (a second stacked layer
        # would double it): one call, one charge in the override's own
        # bucket (no ambient session → the view's fallback id).
        assert CostTracker().session_total("sh-external") == pytest.approx(
            CostTracker.estimate_cost(210, 1, "claude-sonnet-4-6")
        )
        CostTracker().forget_session("sh-external")
        runner._reclaim_minted_cost_bucket()

    def test_booked_override_without_ledger_returns_unchanged(self, monkeypatch, tmp_path):
        """No ledger → nothing to attach: the booked override passes
        through as-is (the original double-wrap-guard contract)."""
        runner, _ = self._make_booked_runner(monkeypatch, tmp_path)
        assert runner._spend_ledger is None
        booked = self._real_client().with_bookkeeping(
            agent="harness",
            session_id="sh-external",
            tracker=CostTracker(),
            audit_logger=None,
        )
        out = runner._get_native_client("hunter", booked, budget_stage="hunt")
        assert out is booked
        assert out.spend_ledger is None

    def test_no_ledger_runner_still_books(self, monkeypatch, tmp_path):
        """Metering is independent of budget enforcement (issue #64)."""
        runner, audit_home = self._make_booked_runner(monkeypatch, tmp_path)
        assert runner._spend_ledger is None  # _ensure_spend_ledger never ran
        view = runner._get_native_client(
            "sourcehunt_exploit", self._real_client(), budget_stage="auto_patch"
        )
        asyncio.run(view.aask_text(system="s", user="u"))
        rows = _llm_cost_rows(_audit_rows(audit_home, runner._session_id))
        assert len(rows) == 1
        assert rows[0]["agent"] == "patcher"
        runner._reclaim_minted_cost_bucket()

    def test_ledger_runner_composes_settle_and_book(self, monkeypatch, tmp_path):
        """A ledger run gets ONE view that both settles the reservation
        AND books the audit/tracker pair — not two competing clients."""
        runner, audit_home = self._make_booked_runner(
            monkeypatch,
            tmp_path,
            usage=Usage(prompt_tokens=210, completion_tokens=1, total_tokens=211),
        )
        runner.budget_usd = 1.0
        runner.input_price_per_million = 0.0
        runner.output_price_per_million = 1_000_000.0  # 1 output token = $1
        ledger = runner._ensure_spend_ledger()
        assert ledger.enforcing

        view = runner._get_native_client("hunter", self._real_client(), budget_stage="hunt")
        asyncio.run(view.aask_text(system="s", user="u"))

        # The ledger half settled.
        assert ledger.spent_usd == pytest.approx(1.0)
        # The bookkeeping half landed too, on the SAME view.
        rows = _llm_cost_rows(_audit_rows(audit_home, runner._session_id))
        assert len(rows) == 1
        assert rows[0]["agent"] == "hunter"  # hunt stage → historical tag
        assert sum(r["details"]["cost_usd"] for r in rows) == pytest.approx(
            CostTracker().session_total(runner._session_id)
        )
        runner._reclaim_minted_cost_bucket()

    def test_ambient_session_wins_over_runner_fallback(self, monkeypatch, tmp_path):
        """Review round: the reconciliation contract holds for BOTH halves
        under the SAME resolved id. The tracker bucket always followed the
        ambient webui session, but the audit row used to ride the runner's
        single AuditLogger (its own fallback id) — breaking
        ``sum(audit rows) == tracker.session_total(sid)`` for BOTH ids.
        The audit logger must resolve per resolved id, exactly like the
        bucket."""
        from clearwing.agent.tooling import session_scope

        runner, audit_home = self._make_booked_runner(monkeypatch, tmp_path)
        view = runner._get_native_client("hunter", self._real_client(), budget_stage="hunt")
        ambient = f"webui-{uuid.uuid4().hex[:8]}"

        with session_scope(ambient):
            asyncio.run(view.aask_text(system="s", user="u"))

        # BOTH halves follow the ambient id: the audit rows land in
        # audit/<ambient>/audit.jsonl AND the ambient bucket equals their
        # sum — the reconciliation contract, same id, both halves.
        rows = _llm_cost_rows(_audit_rows(audit_home, ambient))
        assert len(rows) == 1
        assert rows[0]["agent"] == "hunter"
        total = sum(r["details"]["cost_usd"] for r in rows)
        assert total > 0.0
        assert CostTracker().session_total(ambient) == pytest.approx(total)
        # No rows for the runner id — nothing landed in the fallback trail
        # (its empty dir may exist from the runner's eager logger, but no
        # audit.jsonl row is ever written there for this call).
        assert not _audit_rows(audit_home, runner._session_id)
        assert CostTracker().session_total(runner._session_id) == 0.0
        # The ambient bucket is process-global (CostTracker is a singleton)
        # — reclaim it so the test does not leak the entry.
        CostTracker().forget_session(ambient)
        runner._reclaim_minted_cost_bucket()

    def test_none_usage_fields_skip_booking_and_call_succeeds(self, monkeypatch, tmp_path):
        runner, audit_home = self._make_booked_runner(monkeypatch, tmp_path)
        # Older genai-pyo3 responses and lightweight doubles may expose
        # None token fields — the documented hole: skip booking silently.
        async def none_usage_policy(self, client_obj, request, options):
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=None,
                    completion_tokens=None,
                    prompt_tokens_details=None,
                ),
                tool_calls=[],
            )

        monkeypatch.setattr(
            AsyncLLMClient, "_achat_with_provider_policy", none_usage_policy
        )
        view = runner._get_native_client("hunter", self._real_client(), budget_stage="hunt")

        response = asyncio.run(view.aask_text(system="s", user="u"))

        assert response is not None  # the call itself succeeded
        assert not _audit_rows(audit_home, runner._session_id)
        assert CostTracker().session_total(runner._session_id) == 0.0

    def test_hunter_with_booked_view_books_once_under_stage_role(
        self, monkeypatch, tmp_path
    ):
        """Regression guard for the double-booking blocker: when the runner
        hands NativeHunter a booked view, the VIEW books the call (role =
        the stage tag; "hunt" maps to the historical "hunter") and the
        hunter's own booking stands down — exactly one audit row per call,
        never two."""
        runner, audit_home = self._make_booked_runner(monkeypatch, tmp_path)
        view = runner._get_native_client("hunter", self._real_client(), budget_stage="hunt")

        ctx_sid = f"sh-{uuid.uuid4().hex[:8]}"
        ctx = HunterContext(
            repo_path=str(_FIXTURE_C),
            findings=[],
            file_path="src/codec_a.c",
            session_id=ctx_sid,
            specialist="general",
        )
        hunter = NativeHunter(
            llm=view,
            prompt="system prompt",
            tools=[],
            ctx=ctx,
            max_steps=1,
        )
        result = asyncio.run(hunter.arun())
        assert result.findings == []

        # The view booked under the RUNNER's session (its fallback), once.
        rows = _llm_cost_rows(_audit_rows(audit_home, runner._session_id))
        assert len(rows) == 1
        assert rows[0]["agent"] == "hunter"  # stage role = historical tag
        # Review round: under a booked view the hunter builds NO per-
        # execution AuditLogger at all (it would only mkdir an empty
        # audit/<ctx-id>-<8hex>/ shell) — no minted trail exists, so the
        # hunter's own site could not have added a duplicate row either.
        minted_dirs = _audit_session_dirs(audit_home, ctx_sid)
        assert minted_dirs == []
        hunter_rows = [
            r
            for name in minted_dirs
            for r in _llm_cost_rows(_audit_rows(audit_home, name))
        ]
        assert hunter_rows == []
        # Exactly one row exists anywhere for this call — the count (not
        # the tag: view and standalone site now share "hunter") is what
        # pins the no-double-booking invariant.
        assert len(rows) + len(hunter_rows) == 1
        runner._reclaim_minted_cost_bucket()

    def test_hunter_with_booked_view_summarizer_books_once_under_stage_role(
        self, monkeypatch, tmp_path
    ):
        """Review round: the summarizer booking guard had ZERO coverage —
        the view test above runs max_steps=1 with no compaction, so the
        guarded site at hunter.py (agent="summarizer") was never exercised
        under a booked view. Force a summarizer run (same seam as
        TestHunterAudit: a ~125k-token initial message trips the 150k
        window's 80% threshold) and assert each LLM call — the summary
        AND the main call — booked EXACTLY once under the stage role tag,
        with no "summarizer"-tagged duplicate from the hunter's own site.

        Runs under session_scope(runner id) so a broken guard is caught
        by reconciliation: the hunter's site would charge the SAME id's
        tracker bucket with no matching audit row (its audit logger is
        skipped under a booked view), breaking the sum==total equation."""
        from clearwing.agent.tooling import session_scope

        runner, audit_home = self._make_booked_runner(monkeypatch, tmp_path)
        view = runner._get_native_client("hunter", self._real_client(), budget_stage="hunt")

        ctx_sid = f"sh-{uuid.uuid4().hex[:8]}"
        ctx = HunterContext(
            repo_path=str(_FIXTURE_C),
            findings=[],
            file_path="src/codec_a.c",
            session_id=ctx_sid,
            specialist="general",
        )
        hunter = NativeHunter(
            llm=view,
            prompt="system prompt",
            tools=[],
            ctx=ctx,
            max_steps=1,
            summarizer=ContextSummarizer(),
            # ~125k estimated tokens > 80% of the summarizer's default
            # 150k window, with a coverable old segment — trips the
            # summary path on the hunt's first model step.
            initial_user_message="x " * 250_000,
        )
        with session_scope(runner._session_id):
            result = asyncio.run(hunter.arun())
        assert result.findings == []

        # Two LLM calls ran (summary + main); each booked EXACTLY once via
        # the view — both rows carry the stage role tag "hunter", and no
        # standalone-site "summarizer" duplicate exists anywhere.
        rows = _llm_cost_rows(_audit_rows(audit_home, runner._session_id))
        assert len(rows) == 2
        assert all(r["agent"] == "hunter" for r in rows)
        assert "summarizer" not in [r["agent"] for r in rows]
        # Reconciliation on the runner's id, both halves: a broken guard
        # at either hunter site would inflate the tracker side only.
        total = sum(r["details"]["cost_usd"] for r in rows)
        assert total > 0.0
        assert CostTracker().session_total(runner._session_id) == pytest.approx(total)
        # Under a booked view no per-execution minted trail exists.
        assert _audit_session_dirs(audit_home, ctx_sid) == []
        runner._reclaim_minted_cost_bucket()

    def test_minted_bucket_forgotten_parent_never(self, monkeypatch, tmp_path):
        forgotten: list[str] = []
        original_forget = CostTracker.forget_session

        def _forget_spy(tracker_self, session_id):
            forgotten.append(session_id)
            original_forget(tracker_self, session_id)

        monkeypatch.setattr(CostTracker, "forget_session", _forget_spy)

        def _explode():
            raise RuntimeError("pipeline exploded")

        # Minted sh-* id: arun's finally reclaims it on the failure path.
        runner, _ = self._make_booked_runner(monkeypatch, tmp_path)
        monkeypatch.setattr(runner, "_preprocess", _explode)
        with pytest.raises(RuntimeError, match="pipeline exploded"):
            asyncio.run(runner.arun())
        assert runner._session_id in forgotten
        assert runner._session_id.startswith("sh-")

        # Parent id: NEVER forgotten — its bucket belongs to the parent.
        parent = f"webui-{uuid.uuid4().hex[:8]}"
        runner2, _ = self._make_booked_runner(monkeypatch, tmp_path, parent_session_id=parent)
        monkeypatch.setattr(runner2, "_preprocess", _explode)
        with pytest.raises(RuntimeError, match="pipeline exploded"):
            asyncio.run(runner2.arun())
        assert parent not in forgotten
        assert runner2._session_id == parent

        # The proof flow's exit path reclaims a minted id too.
        runner3, _ = self._make_booked_runner(monkeypatch, tmp_path)
        runner3._flow = "proof"

        async def _proof_explode(self):
            raise RuntimeError("proof flow exploded")

        monkeypatch.setattr(SourceHuntRunner, "_arun_proof_flow", _proof_explode)
        with pytest.raises(RuntimeError, match="proof flow exploded"):
            asyncio.run(runner3.arun())
        assert runner3._session_id in forgotten

    def test_run_belt_and_braces_reclaim(self, monkeypatch, tmp_path):
        """run() adds an idempotent reclaim after arun — even a stubbed
        arun that bypasses the internal finally blocks gets cleaned up."""
        forgotten: list[str] = []
        original_forget = CostTracker.forget_session

        def _forget_spy(tracker_self, session_id):
            forgotten.append(session_id)
            original_forget(tracker_self, session_id)

        monkeypatch.setattr(CostTracker, "forget_session", _forget_spy)
        runner, _ = self._make_booked_runner(monkeypatch, tmp_path)

        async def _ok_arun(self):
            return "stub-result"

        monkeypatch.setattr(SourceHuntRunner, "arun", _ok_arun)
        assert runner.run() == "stub-result"
        assert runner._session_id in forgotten


_ASAN_STDERR = (
    "==1== ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000010\n"
    "READ of size 4 at 0x602000000010 thread T0"
)


class _BenchLLM:
    model_name = "claude-sonnet-4-6"
    provider_name = "anthropic"

    async def aask_text(self, **kwargs):
        return ChatResponse(
            content=[{"text": '{"tier": 2, "rationale": "no attacker control"}'}],
            usage=Usage(prompt_tokens=120, completion_tokens=45, total_tokens=165),
            provider_model_name="bench-served",
        )


class TestBenchAudit:
    """Issue #64 bench half: the crash classifier's aask_text was priced
    from a dead ``cost_usd`` attribute read (native responses never carry
    it) — every bench classification was free by construction. It now
    books for real under one minted bench-* bucket per classifier, which
    the sweep forgets at the end."""

    def _pin(self, monkeypatch, tmp_path) -> Path:
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        return audit_home

    @pytest.mark.asyncio
    async def test_classifier_books_each_call_into_one_bucket(self, monkeypatch, tmp_path):
        audit_home = self._pin(monkeypatch, tmp_path)
        classifier = CrashClassifier(llm=_BenchLLM())

        await classifier.aclassify(exit_code=1, stdout="", stderr=_ASAN_STDERR, poc="")
        bucket = classifier._bench_bucket
        assert bucket is not None and bucket.startswith("bench-")
        await classifier.aclassify(exit_code=1, stdout="", stderr=_ASAN_STDERR, poc="")

        # ONE bucket per classifier instance: both calls landed together.
        rows = _llm_cost_rows(_audit_rows(audit_home, bucket))
        assert len(rows) == 2
        assert all(r["agent"] == "bench" for r in rows)
        assert rows[0]["details"]["model"] == "bench-served"
        total = sum(r["details"]["cost_usd"] for r in rows)
        assert total == pytest.approx(
            CostTracker.estimate_cost(120, 45, "claude-sonnet-4-6") * 2
        )
        # Reconciliation holds while the bucket is alive.
        assert CostTracker().session_total(bucket) == pytest.approx(total)

        classifier.forget_bench_bucket()
        assert CostTracker().session_total(bucket) == 0.0
        # Disk trail survives the forget (bookkeeping doctrine).
        assert len(_llm_cost_rows(_audit_rows(audit_home, bucket))) == 2

    @pytest.mark.asyncio
    async def test_second_sweep_on_same_classifier_mints_fresh_bucket(
        self, monkeypatch, tmp_path
    ):
        """Codex PR-69 r1 (P2): forget_bench_bucket must also CLEAR the
        retained bucket id / audit logger. A second sweep on the same
        classifier instance reused them, so the stale audit file kept
        accumulating rows from BOTH sweeps while ``session_total()``
        held only the current one — a reconciliation break."""
        audit_home = self._pin(monkeypatch, tmp_path)
        classifier = CrashClassifier(llm=_BenchLLM())

        await classifier.aclassify(exit_code=1, stdout="", stderr=_ASAN_STDERR, poc="")
        bucket1 = classifier._bench_bucket
        assert bucket1 is not None
        rows1 = _llm_cost_rows(_audit_rows(audit_home, bucket1))
        assert len(rows1) == 1
        # Reconciliation holds while sweep 1's bucket is alive.
        assert CostTracker().session_total(bucket1) == pytest.approx(
            rows1[0]["details"]["cost_usd"]
        )

        classifier.forget_bench_bucket()
        # The retained identity is cleared — the next sweep starts fresh.
        assert classifier._bench_bucket is None
        assert classifier._bench_audit_logger is None

        await classifier.aclassify(exit_code=1, stdout="", stderr=_ASAN_STDERR, poc="")
        bucket2 = classifier._bench_bucket
        assert bucket2 is not None and bucket2 != bucket1
        rows2 = _llm_cost_rows(_audit_rows(audit_home, bucket2))
        # Sweep 2 wrote to ITS OWN trail: one row here, still exactly one
        # row in sweep 1's file (the old bug appended both to bucket1).
        assert len(rows2) == 1
        assert len(_llm_cost_rows(_audit_rows(audit_home, bucket1))) == 1
        assert CostTracker().session_total(bucket2) == pytest.approx(
            rows2[0]["details"]["cost_usd"]
        )

    @pytest.mark.asyncio
    async def test_classifier_prices_with_endpoint_pricing(self, monkeypatch, tmp_path):
        """Codex PR-69 r1 (P2): the classifier books with its client's
        authoritative endpoint pricing when present, not the table."""
        self._pin(monkeypatch, tmp_path)

        class _PricedBenchLLM(_BenchLLM):
            model_name = "custom-bench-model"  # no PRICING entry
            pricing = EndpointPricing(input_per_mtok=1.0, output_per_mtok=4.0)

        classifier = CrashClassifier(llm=_PricedBenchLLM())
        result = await classifier.aclassify(
            exit_code=1, stdout="", stderr=_ASAN_STDERR, poc=""
        )
        # 120 in * 1.0 + 45 out * 4.0 per 1M — the endpoint's rates.
        assert result.cost_usd == pytest.approx((120 * 1.0 + 45 * 4.0) / 1_000_000)
        classifier.forget_bench_bucket()

    @pytest.mark.asyncio
    async def test_magicmock_usage_skips_booking_and_never_mints(
        self, monkeypatch, tmp_path
    ):
        self._pin(monkeypatch, tmp_path)
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.first_text = '{"tier": 2, "rationale": "none"}'
        mock_llm.aask_text = AsyncMock(return_value=mock_response)
        classifier = CrashClassifier(llm=mock_llm)

        result = await classifier.aclassify(
            exit_code=1, stdout="", stderr=_ASAN_STDERR, poc=""
        )

        # No crash, no cost, no minted bucket: non-int usage skips booking.
        assert result.tier == 2
        assert result.cost_usd == 0.0
        assert classifier._bench_bucket is None
        classifier.forget_bench_bucket()  # no-op, must not raise

    def test_sweep_forgets_bucket_at_end(self, monkeypatch, tmp_path):
        audit_home = self._pin(monkeypatch, tmp_path)
        bench = OssFuzzBenchmark(
            llm=_BenchLLM(),
            mode="standard",
            output_dir=str(tmp_path / "bench-out"),
            llm_classify=True,
        )

        async def fake_run_target(self, target):
            classification = await self._classifier.aclassify(
                exit_code=1, stdout="", stderr=_ASAN_STDERR, poc=""
            )
            return TargetResult(
                project_name=target.project_name,
                entry_point=target.entry_point,
                tier=classification.tier,
                cost_usd=classification.cost_usd,
            )

        monkeypatch.setattr(OssFuzzBenchmark, "_run_target", fake_run_target)
        target = BenchmarkTarget(
            project_name="proj", repo_path="/tmp/proj", entry_point="fuzz.c", language="c"
        )
        result = asyncio.run(bench.arun([target]))

        # The sweep's finally called forget_bench_bucket — Codex PR-69 r1:
        # the retained id is now CLEARED there too, so recover the run's
        # bucket from the surviving disk trail instead of the attribute.
        assert bench._classifier._bench_bucket is None
        bench_dirs = sorted(
            d.name for d in audit_home.iterdir() if d.name.startswith("bench-")
        )
        assert len(bench_dirs) == 1
        bucket = bench_dirs[0]
        rows = _llm_cost_rows(_audit_rows(audit_home, bucket))
        assert len(rows) == 1
        assert rows[0]["agent"] == "bench"
        # The booked per-call cost is what the sweep totals charged.
        assert rows[0]["details"]["cost_usd"] == pytest.approx(result.total_cost_usd)
        # Sweep end reclaimed the minted bucket; the disk trail survives.
        assert CostTracker().session_total(bucket) == 0.0
