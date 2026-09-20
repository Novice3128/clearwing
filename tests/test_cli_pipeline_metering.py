"""Issue #76: the five runner-external CLI pipelines must meter their LLM spend.

Retro-hunt, n-day, reveng, and both elaborate modes used to grab
``provider_manager.get_native_client("default")`` raw — zero CostTracker
bucket, zero audit row. The CLI now routes them through
``_booked_cli_client`` (a ``with_bookkeeping`` copy view, same doctrine as
the runner's ``_get_native_client``) under a CLI-minted
``sh-<kind>-<uuid8>`` id — or an ``elaborate-{hitl,auto}-<session>`` id for
the elaborate modes — and reclaims the minted bucket via
``_reclaim_cli_cost_bucket`` on every exit path.

The reconciliation contract under test (both halves, same id):
``sum(llm_call row cost_usd in audit/<sid>/audit.jsonl)`` equals
``CostTracker().session_total(sid)`` for the bucket's lifetime, and reads
zero after ``forget_session``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from genai_pyo3 import ChatResponse, Usage

from clearwing.llm import ChatMessage
from clearwing.llm.native import AsyncLLMClient
from clearwing.observability.telemetry import CostTracker
from clearwing.safety.audit import AuditLogger
from clearwing.ui.commands.sourcehunt import (
    _booked_cli_client,
    _mint_cli_book_id,
    _reclaim_cli_cost_bucket,
)

# The role tag each CLI branch passes (ui/commands/sourcehunt.py).
_CLI_ROLES = ["retro_hunt", "nday", "reveng", "elaboration"]


def _audit_rows(base_dir: Path, session_id: str) -> list[dict]:
    path = base_dir / session_id / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _llm_cost_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("event_type") == "llm_call"]


def _patch_transport(monkeypatch, tmp_path, usage: Usage | None = None) -> Path:
    """The test_audit_bookkeeping four-piece setup: pinned audit home,
    isolated CLEARWING_HOME, and a fake transport returning real Usage."""
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
            provider_model_name="cli-served",
        )

    monkeypatch.setattr(AsyncLLMClient, "_achat_with_provider_policy", fake_policy)
    return audit_home


def _real_client() -> AsyncLLMClient:
    return AsyncLLMClient(
        model_name="claude-sonnet-4-6", provider_name="anthropic", api_key="test"
    )


def _assert_reconciled(audit_home: Path, sid: str, agent: str) -> float:
    """Both halves under ONE id: exactly one audit row with the role tag,
    and the tracker bucket equals the audit sum."""
    rows = _llm_cost_rows(_audit_rows(audit_home, sid))
    assert len(rows) == 1
    assert rows[0]["agent"] == agent
    assert rows[0]["details"]["input_tokens"] == 210
    assert rows[0]["details"]["output_tokens"] == 90
    total = rows[0]["details"]["cost_usd"]
    assert total > 0.0
    assert CostTracker().session_total(sid) == pytest.approx(total)
    return total


class TestCliPipelineMetering:
    def test_minted_book_id_shape(self):
        sid = _mint_cli_book_id("retro")
        assert sid.startswith("sh-retro-")
        suffix = sid[len("sh-retro-") :]
        assert len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix)
        # Distinct mints never collide in shape or value.
        assert sid != _mint_cli_book_id("retro")

    def test_retro_hunt_sync_chat_books_both_halves(self, monkeypatch, tmp_path):
        """The retro-hunter calls the SYNC ``chat()`` wrapper
        (retro_hunt.py ``self.llm.chat(...)`` shape) — booking rides the
        achat return inside the sync wrapper, so metering is safe there."""
        audit_home = _patch_transport(monkeypatch, tmp_path)
        sid = _mint_cli_book_id("retro")
        view = _booked_cli_client(_real_client(), agent="retro_hunt", session_id=sid)

        response = view.chat(
            messages=[ChatMessage("user", "CVE-2026-1234 patch diff...")],
            system="generate a retro-hunt rule",
        )
        assert response is not None

        total = _assert_reconciled(audit_home, sid, "retro_hunt")
        _reclaim_cli_cost_bucket(sid)
        # The minted bucket is gone; the disk audit trail survives.
        assert CostTracker().session_total(sid) == 0.0
        assert len(_llm_cost_rows(_audit_rows(audit_home, sid))) == 1
        assert total > 0.0

    @pytest.mark.parametrize("agent", _CLI_ROLES)
    def test_async_aask_text_books_both_halves(self, monkeypatch, tmp_path, agent):
        """The n-day filter calls ``aask_text`` (nday_filter.py shape); the
        same booking view covers every CLI role tag."""
        audit_home = _patch_transport(monkeypatch, tmp_path)
        kind = {"retro_hunt": "retro"}.get(agent, agent)
        sid = _mint_cli_book_id(kind)
        view = _booked_cli_client(_real_client(), agent=agent, session_id=sid)

        asyncio.run(view.aask_text(system="filter these CVEs", user="CVE-2026-0001 ..."))

        _assert_reconciled(audit_home, sid, agent)
        _reclaim_cli_cost_bucket(sid)
        assert CostTracker().session_total(sid) == 0.0

    def test_elaborate_book_id_shapes_and_forget(self, monkeypatch, tmp_path):
        """Both elaborate modes book under their HunterContext-shaped ids
        (``elaborate-hitl-<session>`` / ``elaborate-auto-<session>``) and
        reclaim via the same never-raise helper."""
        audit_home = _patch_transport(monkeypatch, tmp_path)
        session_id = "sh-deadbeef"
        for book_id in (
            f"elaborate-hitl-{session_id}",
            f"elaborate-auto-{session_id}",
        ):
            view = _booked_cli_client(
                _real_client(), agent="elaboration", session_id=book_id
            )
            asyncio.run(view.aask_text(system="elaborate", user="upgrade the exploit"))
            _assert_reconciled(audit_home, book_id, "elaboration")

        # Distinct ids kept distinct buckets; each reclaim zeroes its own.
        _reclaim_cli_cost_bucket(f"elaborate-hitl-{session_id}")
        assert CostTracker().session_total(f"elaborate-hitl-{session_id}") == 0.0
        assert CostTracker().session_total(f"elaborate-auto-{session_id}") > 0.0
        _reclaim_cli_cost_bucket(f"elaborate-auto-{session_id}")
        assert CostTracker().session_total(f"elaborate-auto-{session_id}") == 0.0

    def test_reclaim_survives_tracker_failure(self, monkeypatch, tmp_path):
        """The reclaim is never-raise cleanup — a poisoned tracker lock must
        not break the CLI path that calls it in a finally."""
        _patch_transport(monkeypatch, tmp_path)

        def _exploding_forget(tracker_self, session_id):
            raise RuntimeError("tracker lock poisoned")

        monkeypatch.setattr(CostTracker, "forget_session", _exploding_forget)
        _reclaim_cli_cost_bucket("sh-retro-00000000")  # must not raise

    def test_duck_typed_client_passthrough(self):
        """Test doubles / injected stubs come back untouched — the same
        isinstance seam the runner's ``_get_native_client`` keeps, so
        stub-based tests keep working. AsyncMock matters most: mocks
        synthesize a ``with_bookkeeping`` attribute, so a hasattr gate
        would wrap them in a phantom booking view."""

        class _StubLLM:
            model_name = "stub-model"
            provider_name = "stub"

            async def achat(self, **kwargs):
                return None

        stub = _StubLLM()
        assert (
            _booked_cli_client(stub, agent="reveng", session_id="sh-reveng-00000000")
            is stub
        )
        mock = AsyncMock()
        out = _booked_cli_client(mock, agent="nday", session_id="sh-nday-00000000")
        assert out is mock
        bare = object()
        assert _booked_cli_client(bare, agent="retro_hunt", session_id="sh-retro-0") is bare
