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
from clearwing.data.memory.summarizer import ContextSummarizer
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
