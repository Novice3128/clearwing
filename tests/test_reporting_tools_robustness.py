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
        message = save_report.func(
            filepath=str(out),
            format="markdown",
            scan_data={"open_ports": "80/tcp", "vulnerabilities": ["CVE-2020-1472 x"]},
        )
        assert "saved" in message.lower()
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
