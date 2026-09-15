"""Robustness tests for generate_report / save_report tools.

Session e70ba957 reached the report stage and then crashed with
`string indices must be integers, not 'str'` because the LLM passed
scan_data in a shape the formatter didn't expect (string lists, single-key
wrappers, JSON strings). These tests pin the tolerant behavior.
"""

from __future__ import annotations

import json

import pytest

from clearwing.agent.tools.meta.reporting_tools import (
    _normalize_scan_data,
    generate_report,
    save_report,
)

GOOD_SCAN_DATA = {
    "target": "10.0.0.1",
    "open_ports": [{"port": 80, "protocol": "tcp", "service": "http", "state": "open"}],
    "services": [{"port": 80, "service": "http"}],
    "vulnerabilities": [{"cve": "CVE-2020-1472", "description": "Zerologon"}],
    "exploits": [],
    "os_info": "Windows",
}


class TestNormalizeScanData:
    def test_plain_dict_passthrough(self):
        normalized = _normalize_scan_data(GOOD_SCAN_DATA)
        assert normalized["target"] == "10.0.0.1"
        assert normalized["open_ports"][0]["port"] == 80

    def test_string_list_open_ports(self):
        normalized = _normalize_scan_data(
            {"target": "h", "open_ports": ["80/tcp http", "443"]}
        )
        ports = normalized["open_ports"]
        assert ports[0]["port"] == "80"
        assert ports[0]["protocol"] == "tcp"
        assert ports[0]["service"] == "http"
        assert ports[1]["port"] == "443"

    def test_single_key_wrapper_unwrapped(self):
        normalized = _normalize_scan_data({"item": dict(GOOD_SCAN_DATA)})
        assert normalized["target"] == "10.0.0.1"

    def test_json_string_parsed(self):
        normalized = _normalize_scan_data(json.dumps(GOOD_SCAN_DATA))
        assert normalized["target"] == "10.0.0.1"

    def test_garbage_string_becomes_empty(self):
        normalized = _normalize_scan_data("total garbage not json")
        assert normalized.get("target") == "unknown"
        assert normalized.get("open_ports") == []

    def test_none_and_missing_fields(self):
        normalized = _normalize_scan_data({"target": "h"})
        for field in ("open_ports", "services", "vulnerabilities", "exploits"):
            assert normalized[field] == []

    def test_vulnerability_string_gets_cve_extracted(self):
        normalized = _normalize_scan_data(
            {"target": "h", "vulnerabilities": ["CVE-2017-0144: ms17-010 smb"]},
        )
        vuln = normalized["vulnerabilities"][0]
        assert vuln["cve"] == "CVE-2017-0144"
        assert "ms17-010" in vuln["description"]


class TestGenerateReportTolerant:
    @pytest.mark.parametrize("bad_scan_data", [
        {"target": "h", "open_ports": ["80/tcp http"]},
        {"item": dict(GOOD_SCAN_DATA)},
        json.dumps(GOOD_SCAN_DATA),
        {},
        "garbage",
        None,
    ])
    def test_generate_report_never_crashes(self, bad_scan_data):
        report = generate_report.func(format="markdown", scan_data=bad_scan_data)
        assert isinstance(report, str)
        assert "Clearwing" in report or report

    def test_generate_report_markdown_content(self):
        report = generate_report.func(format="markdown", scan_data=GOOD_SCAN_DATA)
        assert "10.0.0.1" in report
        assert "CVE-2020-1472" in report


class TestSaveReportTolerant:
    def test_save_report_with_weird_shape(self, tmp_path):
        out = tmp_path / "report.md"
        result = save_report.func(
            filepath=str(out),
            format="markdown",
            scan_data={"open_ports": "80/tcp", "vulnerabilities": ["CVE-2020-1472 x"]},
        )
        assert result["status"] == "saved"
        content = out.read_text(encoding="utf-8")
        assert "CVE-2020-1472" in content


class TestSingletonListWrapper:
    def test_singleton_list_wrapper_unwrapped(self):
        normalized = _normalize_scan_data({"item": [dict(GOOD_SCAN_DATA)]})
        assert normalized["target"] == "10.0.0.1"
        assert normalized["open_ports"][0]["port"] == 80

    def test_generate_report_singleton_list(self):
        report = generate_report.func(
            format="markdown", scan_data={"item": [dict(GOOD_SCAN_DATA)]}
        )
        assert "10.0.0.1" in report


class TestSaveReportVerification:
    """Issue #9: a "saved" claim must be verifiable on disk, with the
    resolved absolute path (a /tmp path inside a container, or one wiped
    by reboot, used to be indistinguishable from a real failure)."""

    def test_returns_resolved_absolute_path_and_size(self, tmp_path):
        out = tmp_path / "sub" / "report.md"
        result = save_report.func(
            filepath=str(out), format="markdown", scan_data=dict(GOOD_SCAN_DATA)
        )
        assert result["status"] == "saved"
        assert result["path"] == str(out.resolve())
        assert result["bytes"] == out.stat().st_size

    def test_unwritable_target_is_an_error_not_fake_success(self, tmp_path):
        result = save_report.func(
            filepath=str(tmp_path),  # a directory cannot be written as a file
            format="markdown",
            scan_data=dict(GOOD_SCAN_DATA),
        )
        assert "error" in result
        assert "Failed to save report" in result["error"]

    def test_empty_filepath_is_rejected(self):
        result = save_report.func(filepath="  ", format="text", scan_data={})
        assert "error" in result


class TestMissingArgumentValidation:
    """Issue #9: `save_report() missing 3 required positional arguments`
    must become a field-naming error the model can act on."""

    def test_missing_fields_are_named(self):
        missing = save_report.missing_required_arguments({})
        assert set(missing) == {"filepath", "format", "scan_data"}

    def test_partial_arguments_name_only_the_gap(self):
        missing = save_report.missing_required_arguments({"filepath": "/tmp/x.md"})
        assert missing == ["format", "scan_data"]

    def test_defaults_are_not_required(self):
        from clearwing.agent.tools.data.cve_tools import cve_db_update

        assert cve_db_update.missing_required_arguments({}) == []

    @pytest.mark.asyncio
    async def test_runtime_rejects_missing_arguments_with_field_names(self):
        from types import SimpleNamespace

        from clearwing.agent.runtime import NativeAgentGraph

        graph = NativeAgentGraph(
            llm=object(),
            native_tools=[],
            tools=[save_report],
            system_prompt_fn=lambda s: "sys",
            model_name="m",
            session_id=None,
            state_updater_fn=lambda *a, **k: {},
            knowledge_graph_populator_fn=None,
            input_guardrail_tool_names=frozenset(),
            output_guardrail_tool_names=frozenset(),
            enable_cost_tracker=False,
            enable_episodic_memory=False,
            enable_audit=False,
            enable_knowledge_graph=False,
            enable_input_guardrail=False,
            enable_output_guardrail=False,
            enable_event_bus=False,
            enable_context_summarizer=False,
        )
        state = graph._get_or_create_state("arg-validation")
        call = SimpleNamespace(fn_name="save_report", call_id="c1", fn_arguments={})
        events, paused, halted = await graph._arun_tool_calls(
            state, [call], resume_decision=...
        )
        assert paused is False and halted is False
        assert events
        tool_result = json.loads(state["messages"][-1].content)
        assert "missing required argument(s)" in tool_result["error"]
        assert "filepath" in tool_result["error"]

    def test_tilde_path_writes_where_it_reports(self, tmp_path, monkeypatch):
        """The write and the verification must target the same file: a
        bare `~` used to write ./~ while the check looked in $HOME."""
        monkeypatch.setenv("HOME", str(tmp_path))
        out = tmp_path / "report.md"
        result = save_report.func(
            filepath="~/report.md", format="markdown", scan_data=dict(GOOD_SCAN_DATA)
        )
        assert result["status"] == "saved"
        assert result["path"] == str(out.resolve())
        assert out.is_file()
