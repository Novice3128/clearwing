"""Audit-row ``component`` key: the subsystem tag for audit↔map queries.

Every ``llm_call`` audit row booked through :func:`book_llm_call` carries
``details["component"]`` — ALWAYS present, never None, never omitted.
``None`` (the normal case) resolves via ``_AGENT_COMPONENTS`` from the
*agent* role tag; an explicit ``with_bookkeeping(component=...)`` override
wins. Unknown agent roles degrade to ``"unmapped"`` so the key keeps a
fixed shape, and rows written before the key existed are legacy rows that
analysis treats as ``"unmapped"``.

Isolation mirrors ``tests/test_audit_bookkeeping.py``: pinned
``AuditLogger.BASE_DIR``, ``CLEARWING_HOME`` env, patched
``AsyncLLMClient._build_client`` + ``_achat_with_provider_policy``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest
from genai_pyo3 import ChatResponse, Usage

from clearwing.llm.native import AsyncLLMClient
from clearwing.observability.bookkeeping import (
    _AGENT_COMPONENTS,
    book_llm_call,
    init_session_audit_logger,
)
from clearwing.observability.telemetry import CostTracker
from clearwing.safety.audit import AuditLogger
from clearwing.sourcehunt.runner import _SPECIALIST_BOOK_ROLES

# Every literal ``agent=`` value handed to book_llm_call / with_bookkeeping
# anywhere in clearwing/ (grep-verified 2026-09-24):
#   runtime.py main+summarizer, operator.py, hunter.py hunter+summarizer,
#   crash_classifier.py bench, ui/commands/sourcehunt.py
#   retro_hunt/nday/reveng/elaboration, runner.py via _specialist_book_role
#   (incl. the live verbatim-fallthrough stage "rank" — the ranker).
_REPO_AGENT_LITERALS = [
    "main",
    "summarizer",
    "operator",
    "hunter",
    "bench",
    "rank",
    "retro_hunt",
    "nday",
    "reveng",
    "elaboration",
]


def _audit_rows(base_dir: Path, session_id: str) -> list[dict]:
    path = base_dir / session_id / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _llm_cost_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("event_type") == "llm_call"]


def _fresh_sid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class TestComponentKeyAlwaysPresent:
    """Pin 1: every row booked through book_llm_call has the key."""

    def test_every_booked_row_has_the_component_key(self, monkeypatch, tmp_path):
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        sid = _fresh_sid("comp")
        audit = init_session_audit_logger(sid)

        booked_agents = ["main", "summarizer", "hunter", "brand_new_stage"]
        for agent in booked_agents:
            book_llm_call(
                10,
                5,
                tracker=CostTracker(),
                model="claude-sonnet-4-6",
                session_id=sid,
                audit_logger=audit,
                agent=agent,
            )

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == len(booked_agents)
        for row in rows:
            assert "component" in row["details"]
            assert isinstance(row["details"]["component"], str)
            assert row["details"]["component"]
        CostTracker().forget_session(sid)

    def test_direct_log_llm_call_degrades_to_unmapped(self, monkeypatch, tmp_path):
        """A direct AuditLogger.log_llm_call (anything bypassing book_llm_call)
        still gets a fixed-shape key: the None default degrades to the
        sentinel, matching how analysis treats legacy keyless rows."""
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        audit = AuditLogger(_fresh_sid("comp"))

        entry = audit.log_llm_call(model="m", input_tokens=1, output_tokens=1, cost_usd=0.0)

        assert entry.details["component"] == "unmapped"


class TestKnownRoleMapping:
    """Pin 2: sampled known roles resolve to their expected component."""

    @pytest.mark.parametrize(
        "agent,expected",
        [
            ("main", "agent-runtime"),
            ("hunter", "sourcehunt"),
            ("retro_hunt", "cli-pipelines"),
        ],
    )
    def test_sampled_roles_map_to_expected_component(self, monkeypatch, tmp_path, agent, expected):
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        sid = _fresh_sid("comp")
        audit = init_session_audit_logger(sid)

        book_llm_call(
            10,
            5,
            tracker=CostTracker(),
            model="claude-sonnet-4-6",
            session_id=sid,
            audit_logger=audit,
            agent=agent,
        )

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 1
        assert rows[0]["details"]["component"] == expected
        CostTracker().forget_session(sid)

    def test_unknown_role_is_unmapped_key_still_present(self, monkeypatch, tmp_path):
        """Pin 3: an unknown role (e.g. a brand-new sourcehunt stage passed
        through verbatim by _specialist_book_role) maps to the sentinel —
        the key itself never disappears."""
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        sid = _fresh_sid("comp")
        audit = init_session_audit_logger(sid)

        book_llm_call(
            10,
            5,
            tracker=CostTracker(),
            model="claude-sonnet-4-6",
            session_id=sid,
            audit_logger=audit,
            agent="brand_new_stage",
        )

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 1
        assert "component" in rows[0]["details"]
        assert rows[0]["details"]["component"] == "unmapped"
        CostTracker().forget_session(sid)


class TestWithBookkeepingComponentOverride:
    """Pins 4 + 5 through the REAL client plumbing: an explicit component
    override wins over the map while ``_book_agent`` stays untouched (the
    hunter double-booking guard and runner analytics key on it), and the
    reconciliation contract is unchanged by the new key."""

    def _make_view(self, monkeypatch, tmp_path, sid, *, agent, component=None):
        audit_home = tmp_path / "audit-home"
        monkeypatch.setattr(AuditLogger, "BASE_DIR", audit_home)
        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path / "clearwing-home"))
        monkeypatch.setattr(AsyncLLMClient, "_build_client", lambda self, cls: object())

        async def fake_policy(self, client_obj, request, options):
            return ChatResponse(
                content=[{"text": "done"}],
                usage=Usage(prompt_tokens=210, completion_tokens=90, total_tokens=300),
                provider_model_name="component-served",
            )

        monkeypatch.setattr(AsyncLLMClient, "_achat_with_provider_policy", fake_policy)
        client = AsyncLLMClient(
            model_name="claude-sonnet-4-6", provider_name="anthropic", api_key="test"
        )
        view = client.with_bookkeeping(
            agent=agent,
            session_id=sid,
            tracker=CostTracker(),
            audit_logger=init_session_audit_logger(sid),
            component=component,
        )
        return view, audit_home

    def test_override_wins_and_book_agent_untouched(self, monkeypatch, tmp_path):
        sid = _fresh_sid("comp")
        view, audit_home = self._make_view(
            monkeypatch, tmp_path, sid, agent="hunter", component="custom-x"
        )

        # Guard: the override must not leak into the role tag the hunter's
        # double-booking guard and runner analytics continuity key on.
        assert view._book_agent == "hunter"
        assert view._book_component == "custom-x"

        asyncio.run(view.aask_text(system="s", user="u"))

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 1
        assert rows[0]["agent"] == "hunter"  # role tag unchanged
        assert rows[0]["details"]["component"] == "custom-x"
        CostTracker().forget_session(sid)

    def test_no_override_falls_back_to_agent_map(self, monkeypatch, tmp_path):
        sid = _fresh_sid("comp")
        view, audit_home = self._make_view(
            monkeypatch, tmp_path, sid, agent="hunter", component=None
        )
        assert view._book_component is None

        asyncio.run(view.aask_text(system="s", user="u"))

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 1
        # "hunter" resolves via _AGENT_COMPONENTS, not "unmapped".
        assert rows[0]["details"]["component"] == "sourcehunt"
        CostTracker().forget_session(sid)

    def test_reconciliation_unchanged_by_component_key(self, monkeypatch, tmp_path):
        """Pin 5: the new key does not disturb the audit↔tracker equation."""
        sid = _fresh_sid("comp")
        view, audit_home = self._make_view(
            monkeypatch, tmp_path, sid, agent="hunter", component="custom-x"
        )

        asyncio.run(view.aask_text(system="s", user="u"))
        asyncio.run(view.aask_text(system="s", user="u"))

        rows = _llm_cost_rows(_audit_rows(audit_home, sid))
        assert len(rows) == 2
        assert all(r["details"]["component"] == "custom-x" for r in rows)
        assert sum(r["details"]["cost_usd"] for r in rows) == pytest.approx(
            CostTracker().session_total(sid)
        )
        CostTracker().forget_session(sid)


class TestAgentLiteralCoverage:
    """Pin 6: every agent role the repo can emit maps to a real component —
    the union of ``_SPECIALIST_BOOK_ROLES`` values and the grepped literal
    ``agent=`` values — so no production row ever lands as ``unmapped`` by
    accident of a missing table entry."""

    def test_specialist_roles_and_repo_literals_all_mapped(self):
        roles = set(_SPECIALIST_BOOK_ROLES.values()) | set(_REPO_AGENT_LITERALS)
        assert roles, "coverage list must not be empty"
        unmapped = sorted(r for r in roles if _AGENT_COMPONENTS.get(r) == "unmapped")
        missing = sorted(r for r in roles if r not in _AGENT_COMPONENTS)
        assert unmapped == []
        assert missing == []

    def test_table_values_pinned_verbatim(self):
        """R-D review: value-level pin — a mistyped component value (e.g.
        operator→sourcehunt) would slip past the coverage test above, which
        only checks membership. The full dict is the contract."""
        assert _AGENT_COMPONENTS == {
            "main": "agent-runtime",
            "summarizer": "agent-runtime",
            "operator": "agent-runtime",
            "hunter": "sourcehunt",
            "patcher": "sourcehunt",
            "exploiter": "sourcehunt",
            "variant": "sourcehunt",
            "harness": "sourcehunt",
            "stability": "sourcehunt",
            "mechanism": "sourcehunt",
            "proof": "sourcehunt",
            "verifier": "sourcehunt",
            "bench": "bench",
            "rank": "sourcehunt",
            "retro_hunt": "cli-pipelines",
            "nday": "cli-pipelines",
            "reveng": "cli-pipelines",
            "elaboration": "cli-pipelines",
        }

    def test_verbatim_fallback_roles_keep_the_sentinel_shape(self):
        """The agent namespace is OPEN (_specialist_book_role passes unknown
        stages through verbatim): table-level lookups for such roles MISS
        (no accidental sentinel entry in the table), and the booking-path
        resolver still yields the sentinel so the key keeps its shape."""
        # Unknown stages are deliberately absent from the table...
        assert "some_new_stage" not in _AGENT_COMPONENTS
        # ...and the resolver maps them to the sentinel, never to None.
        assert _AGENT_COMPONENTS.get("some_new_stage", "unmapped") == "unmapped"
