"""Unit tests for the interactive @tool wrapper around sourcehunt."""

from __future__ import annotations

from pathlib import Path

from clearwing.agent.prompts import SYSTEM_PROMPT_TEMPLATE
from clearwing.agent.tooling import session_scope
from clearwing.agent.tools import get_all_tools
from clearwing.agent.tools.meta.sourcehunt_tools import (
    _RECENT_SESSIONS,
    get_sourcehunt_tools,
    hunt_source_code,
    list_sourcehunt_findings,
)

FIXTURE_PY_SQLI = Path(__file__).parent / "fixtures" / "vuln_samples" / "py_sqli"


def test_hunt_source_code_is_discoverable_via_get_all_tools():
    tools = get_all_tools()
    names = {t.name for t in tools}
    assert "hunt_source_code" in names
    assert "list_sourcehunt_findings" in names


def test_get_sourcehunt_tools_returns_two_tools():
    tools = get_sourcehunt_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"hunt_source_code", "list_sourcehunt_findings"}


def test_hunt_source_code_runs_quick_against_local_fixture(tmp_path):
    """The @tool wrapper should run the runner end-to-end on a local path."""
    summary = hunt_source_code.invoke(
        {
            "repo_url_or_path": str(FIXTURE_PY_SQLI),
            "depth": "quick",
            "output_dir": str(tmp_path),
        }
    )
    assert isinstance(summary, str)
    assert "Source hunt complete" in summary
    assert "Files ranked: 1" in summary
    # The py_sqli fixture has an f-string SQL injection that SourceAnalyzer
    # catches, so the summary should include at least one finding line
    assert "CRITICAL" in summary or "Top findings" in summary


def test_hunt_source_code_returns_summary_for_empty_repo(tmp_path):
    """An empty directory still produces a valid summary, not an error."""
    empty = tmp_path / "empty_repo"
    empty.mkdir()
    summary = hunt_source_code.invoke(
        {
            "repo_url_or_path": str(empty),
            "depth": "quick",
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert "Source hunt complete" in summary
    # Files ranked = 0 because there are no source files
    assert "Files ranked: 0" in summary


def test_two_hunts_mint_distinct_session_ids_under_one_ambient_session(tmp_path):
    """Codex PR-69 r1 (P1): run identity must stay separate from billing.

    The tool must NOT pass the ambient (stable, long-lived) webui/operator
    session id down as the runner's ``parent_session_id``: two hunts from
    the same session would then share one output/checkpoint directory,
    auto-load each other's checkpoints, and replace each other in
    ``_RECENT_SESSIONS``. Each invocation mints its own ``sh-`` execution
    id instead — billing attribution to the parent is covered by the
    booked views' ambient resolution, not by hijacking run identity."""
    _RECENT_SESSIONS.clear()
    ambient = "webui-abc12345"

    with session_scope(ambient):
        hunt_source_code.invoke(
            {
                "repo_url_or_path": str(FIXTURE_PY_SQLI),
                "depth": "quick",
                "output_dir": str(tmp_path),
            }
        )
        hunt_source_code.invoke(
            {
                "repo_url_or_path": str(FIXTURE_PY_SQLI),
                "depth": "quick",
                "output_dir": str(tmp_path),
            }
        )

    session_ids = list(_RECENT_SESSIONS.keys())
    assert len(session_ids) == 2
    assert session_ids[0] != session_ids[1]
    for sid in session_ids:
        assert sid.startswith("sh-")
        assert sid != ambient  # the ambient id never became the run id
    # Distinct run identity means distinct session directories on disk.
    out_dirs = [tmp_path / sid for sid in session_ids]
    assert all(d.is_dir() for d in out_dirs)
    assert out_dirs[0] != out_dirs[1]


def test_list_sourcehunt_findings_recalls_recent_run(tmp_path):
    """After hunt_source_code runs, list_sourcehunt_findings returns the cache."""
    # Clear the cache to start fresh
    _RECENT_SESSIONS.clear()

    hunt_source_code.invoke(
        {
            "repo_url_or_path": str(FIXTURE_PY_SQLI),
            "depth": "quick",
            "output_dir": str(tmp_path),
        }
    )

    findings = list_sourcehunt_findings.invoke({})
    assert isinstance(findings, list)
    assert len(findings) >= 1
    # Each finding has a file and line_number
    for f in findings:
        assert "file" in f
        assert "line_number" in f


def test_list_sourcehunt_findings_with_unknown_session_id():
    _RECENT_SESSIONS.clear()
    out = list_sourcehunt_findings.invoke({"session_id": "missing"})
    assert isinstance(out, list)
    assert "no source hunts" in out[0]["error"]


def test_prompt_template_mentions_hunt_source_code():
    """The interactive agent's system prompt should advertise the new tool."""
    assert "hunt_source_code" in SYSTEM_PROMPT_TEMPLATE
    assert "source-hunt" in SYSTEM_PROMPT_TEMPLATE
