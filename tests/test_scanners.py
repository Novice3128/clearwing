import errno
import logging
import os
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clearwing.scanning import OSScanner, PortScanner, ServiceScanner, VulnerabilityScanner


class TestPortScanner:
    """Tests for PortScanner module."""

    @pytest.fixture
    def scanner(self):
        return PortScanner()

    @pytest.mark.asyncio
    async def test_syn_scan(self, scanner):
        """Test SYN scan on localhost."""
        # This test requires a running service on localhost
        result = await scanner.scan("127.0.0.1", [22, 80], "syn")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_connect_scan(self, scanner):
        """Test TCP connect scan on localhost."""
        result = await scanner.scan("127.0.0.1", [22, 80], "connect")
        assert isinstance(result, list)

    def test_scan_sync(self, scanner):
        """Test synchronous scan method."""
        result = scanner.scan_sync("127.0.0.1", [22, 80])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_syn_scan_warns_when_unprivileged(self, scanner, monkeypatch, caplog):
        """PR #20 regression: `scan_type='syn'` without root used to fail
        silently and return 0 ports. The scanner now emits a WARNING so
        the user can either re-run with sudo or switch to 'connect'.
        """
        # Pretend we're a regular (non-root) user. geteuid is guaranteed
        # to exist on Linux, where this test runs; on Windows the
        # privilege check skips entirely, so skip the test there.
        if not hasattr(os, "geteuid"):
            pytest.skip("raw-socket privilege check is Unix-only")
        monkeypatch.setattr(os, "geteuid", lambda: 1000)

        with caplog.at_level(logging.WARNING, logger="clearwing.scanning.port_scanner"):
            # Scan a single closed port on localhost so we exit fast
            # regardless of whether libpnet_pyo3 actually fires a packet.
            await scanner.scan("127.0.0.1", [1], scan_type="syn")

        warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "raw-socket" in r.message
        ]
        assert warnings, f"expected raw-socket WARNING, got {[r.message for r in caplog.records]}"
        assert "syn" in warnings[0].message

    @pytest.mark.asyncio
    async def test_connect_scan_does_not_warn_when_unprivileged(self, scanner, monkeypatch, caplog):
        """The raw-socket warning must not fire for the default
        `scan_type='connect'`, which doesn't need root."""
        if not hasattr(os, "geteuid"):
            pytest.skip("raw-socket privilege check is Unix-only")
        monkeypatch.setattr(os, "geteuid", lambda: 1000)

        with caplog.at_level(logging.WARNING, logger="clearwing.scanning.port_scanner"):
            await scanner.scan("127.0.0.1", [1], scan_type="connect")

        assert not [
            r for r in caplog.records if r.levelno == logging.WARNING and "raw-socket" in r.message
        ]


class TestServiceScanner:
    """Tests for ServiceScanner module."""

    @pytest.fixture
    def scanner(self):
        return ServiceScanner()

    @pytest.mark.asyncio
    async def test_banner_grabbing(self, scanner):
        """Test banner grabbing from open ports."""
        open_ports = [{"port": 80, "service": "HTTP"}]
        result = await scanner.detect("127.0.0.1", open_ports)
        assert isinstance(result, list)

    def test_detect_sync(self, scanner):
        """Test synchronous detect method."""
        open_ports = [{"port": 80, "service": "HTTP"}]
        result = scanner.detect_sync("127.0.0.1", open_ports)
        assert isinstance(result, list)


class TestVulnerabilityScanner:
    """Tests for VulnerabilityScanner module."""

    @pytest.fixture
    def scanner(self):
        return VulnerabilityScanner()

    @pytest.mark.asyncio
    async def test_vulnerability_scan(self, scanner):
        """Test vulnerability scanning."""
        services = [{"port": 80, "service": "HTTP", "version": "2.4.41"}]
        result = await scanner.scan("127.0.0.1", services)
        assert isinstance(result, list)

    def test_local_db_lookup(self, scanner):
        """Test local vulnerability database lookup.

        Issue #15: a bare service name ("FTP") no longer returns every
        FTP-ish CVE — ProFTPD/vsftpd entries require a product identity,
        and version verification needs the banner's version.
        """
        # Product + version verified via banner → finding.
        vulns = scanner._check_local_db("FTP", version="", banner="220 ProFTPD 1.3.5 Server")
        assert [v["cve"] for v in vulns] == ["CVE-2015-3306"]
        assert vulns[0]["match_quality"] == "version-verified"
        assert vulns[0]["product"] == "proftpd"
        assert vulns[0]["version"] == "1.3.5"

        # Version outside the affected range → dropped (not affected).
        vulns = scanner._check_local_db("FTP", version="", banner="220 ProFTPD 1.3.8 Server")
        assert vulns == []

        # Product recognized, version unknown → heuristic, honestly labeled.
        vulns = scanner._check_local_db("FTP", version="", banner="220 ProFTPD Server ready")
        assert [v["cve"] for v in vulns] == ["CVE-2015-3306"]
        assert vulns[0]["match_quality"] == "service-heuristic"

        # Bare service name without any banner → no product identity.
        assert scanner._check_local_db("FTP") == []

        # Protocol-level advisory (no product binding) stays as heuristic.
        vulns = scanner._check_local_db("RDP")
        assert [v["cve"] for v in vulns] == ["CVE-2019-0708"]
        assert vulns[0]["match_quality"] == "service-heuristic"

    def test_version_range_comparator(self, scanner):
        """Issue #15: dotted versions with patch suffixes compare sanely."""
        from clearwing.scanning.vulnerability_scanner import _version_in_ranges

        lt_78 = [{"end_excluding": "7.8"}]
        assert _version_in_ranges("7.1", lt_78)
        assert _version_in_ranges("7.7p1", lt_78)
        assert not _version_in_ranges("7.8", lt_78)
        assert not _version_in_ranges("8.0", lt_78)

        lt_71p2 = [{"end_excluding": "7.1p2"}]
        assert _version_in_ranges("7.1", lt_71p2)
        assert _version_in_ranges("7.1p1", lt_71p2)
        assert not _version_in_ranges("7.1p2", lt_71p2)

        exact = [{"start_including": "2.4.49", "end_including": "2.4.50"}]
        assert _version_in_ranges("2.4.49", exact)
        assert _version_in_ranges("2.4.50", exact)
        assert not _version_in_ranges("2.4.51", exact)
        assert not _version_in_ranges("2.4.48", exact)

        # Unbounded range matches everything; empty version matches nothing.
        assert _version_in_ranges("1.0", [{}])
        assert not _version_in_ranges("", [{}])
        assert not _version_in_ranges("1.0", None)

    def test_resolve_identity_from_banner(self, scanner):
        """Issue #15: banners carry the product truth, not the service name."""
        cases = [
            ("FTP", "220 ProFTPD 1.3.5 Server", "proftpd", "1.3.5"),
            ("FTP", "220 (vsFTPd 3.0.3)", "vsftpd", "3.0.3"),
            ("SSH", "SSH-2.0-OpenSSH_7.4", "openssh", "7.4"),
            ("SSH", "SSH-2.0-OpenSSH_8.2p1 Debian", "openssh", "8.2p1"),
            ("HTTP", "Server: Apache/2.4.49 (Unix)", "apache", "2.4.49"),
            ("HTTP", "Server: nginx/1.18.0", "nginx", "1.18.0"),
            ("MYSQL", "mysql  Ver 8.0.31", "mysql", "8.0.31"),
        ]
        for service, banner, product, version in cases:
            identity = scanner._resolve_identity(service, "", banner)
            assert identity is not None, f"no identity from {banner!r}"
            assert identity.product == product, banner
            assert identity.version == version, banner

        # No product signal anywhere → no identity.
        assert scanner._resolve_identity("HTTP", "", "") is None
        assert scanner._resolve_identity("HTTP", "2.4.41", "") is None

    def test_cvss_extraction(self, scanner):
        """Test CVSS score extraction."""
        metrics = {"cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}]}
        score = scanner._extract_cvss(metrics)
        assert score == 9.8

    @pytest.mark.asyncio
    async def test_close_session(self, scanner):
        """Test closing aiohttp session."""
        await scanner.close()
        assert scanner.session is None

    @pytest.mark.asyncio
    async def test_nvd_query_retries_transient_failures(self, scanner, monkeypatch):
        """Issue #20: one NVD timeout must not drop the query outright."""
        import asyncio as aio

        calls: list[str] = []

        class _FakeResponse:
            @property
            def status(self):
                return 200

            async def json(self):
                return {"vulnerabilities": []}

        class _FakeGet:
            def __init__(self, exc):
                self._exc = exc

            async def __aenter__(self):
                if self._exc is not None:
                    raise self._exc
                return _FakeResponse()

            async def __aexit__(self, *args):
                return False

        class _FakeSession:
            def get(self, url, timeout=None):
                calls.append(url)
                return _FakeGet(aio.TimeoutError() if len(calls) < 3 else None)

        async def _no_sleep(_delay):
            return None

        scanner.session = _FakeSession()
        monkeypatch.setattr(aio, "sleep", _no_sleep)

        result = await scanner._query_nvd("http")
        assert result == []
        assert len(calls) == 3  # two timeouts retried, third attempt succeeded

    @pytest.mark.asyncio
    async def test_nvd_query_gives_up_after_retries(self, scanner, monkeypatch, caplog):
        import asyncio as aio

        calls: list[str] = []

        class _FakeGet:
            async def __aenter__(self):
                raise aio.TimeoutError()

            async def __aexit__(self, *args):
                return False

        class _FakeSession:
            def get(self, url, timeout=None):
                calls.append(url)
                return _FakeGet()

        async def _no_sleep(_delay):
            return None

        scanner.session = _FakeSession()
        monkeypatch.setattr(aio, "sleep", _no_sleep)

        with caplog.at_level("WARNING", logger="clearwing.scanning.vulnerability_scanner"):
            result = await scanner._query_nvd("http")

        assert result == []
        assert len(calls) == 3
        assert any("after 3 attempts" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_nvd_retries_rate_limit_statuses(self, scanner, monkeypatch):
        """NVD rate-limits arrive as HTTP 429 without an exception; they
        must enter the retry path instead of silently returning empty."""
        import asyncio as aio

        calls: list[str] = []

        class _StatusResponse:
            def __init__(self, status):
                self._status = status

            @property
            def status(self):
                return self._status

            async def json(self):
                return {"vulnerabilities": []}

        class _FakeGet:
            def __init__(self, status):
                self._status = status

            async def __aenter__(self):
                return _StatusResponse(self._status)

            async def __aexit__(self, *args):
                return False

        class _FakeSession:
            def get(self, url, timeout=None):
                calls.append(url)
                return _FakeGet(429 if len(calls) < 3 else 200)

        async def _no_sleep(_delay):
            return None

        scanner.session = _FakeSession()
        monkeypatch.setattr(aio, "sleep", _no_sleep)

        result = await scanner._query_nvd("http")
        assert result == []
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_scan_dedups_same_cve_across_ports(self, scanner):
        """Issue #15: the same CVE on N ports is ONE entry with a ports list."""
        services = [
            {"port": 21, "service": "FTP", "banner": "220 ProFTPD 1.3.5 Server", "version": ""},
            {"port": 2121, "service": "FTP", "banner": "220 ProFTPD 1.3.5 Server", "version": ""},
        ]

        async def _fake_query(service, identity=None):
            return []

        scanner._query_nvd = _fake_query
        result = await scanner.scan("127.0.0.1", services)

        assert len(result) == 1
        assert result[0]["cve"] == "CVE-2015-3306"
        assert result[0]["ports"] == [21, 2121]
        assert result[0]["match_quality"] == "version-verified"

    @pytest.mark.asyncio
    async def test_scan_false_positive_inflation_is_gone(self, scanner, monkeypatch):
        """Issue #15 offline replay: same service on many ports used to
        multiply keyword hits into dozens of unverified entries; findings
        are now unique verified CVEs."""
        services = [
            # Banners unreadable → no product identity → keyword path.
            {"port": p, "service": "HTTP", "banner": "", "version": None}
            for p in (80, 81, 8080, 8081, 8000)
        ]

        async def _fake_query(service, identity=None):
            # NVD keyword noise: one irrelevant CVE per port query.
            return [
                {
                    "cve": "CVE-2023-99999",
                    "description": "some other product entirely",
                    "cvss": 5.0,
                    "references": [],
                    "match_quality": "keyword-candidate",
                }
            ]

        scanner._query_nvd = _fake_query
        result = await scanner.scan("127.0.0.1", services)

        # 2.4.41 is outside every local Apache range → no local findings.
        # The keyword noise dedups to one entry, labeled candidate.
        assert len(result) == 1
        assert result[0]["cve"] == "CVE-2023-99999"
        assert result[0]["match_quality"] == "keyword-candidate"
        assert result[0]["ports"] == [80, 81, 8080, 8081, 8000]

    @pytest.mark.asyncio
    async def test_nvd_cpe_query_verifies_version_ranges(self, scanner, monkeypatch):
        """With a product identity the NVD query uses cpeName and drops
        CVEs whose CPE criteria ranges exclude the detected version."""
        urls: list[str] = []

        def _cve_item(cve_id, start, end):
            return {
                "cve": {
                    "id": cve_id,
                    "descriptions": [{"value": f"{cve_id} desc"}],
                    "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 7.5}}]},
                    "configurations": [
                        {
                            "nodes": [
                                {
                                    "cpeMatch": [
                                        {
                                            "cpe23Uri": f"cpe:2.3:a:apache:http_server:{start or '*'}:*:*:*:*:*:*:*",
                                            "versionStartIncluding": start,
                                            "versionEndIncluding": end,
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            }

        class _FakeResponse:
            @property
            def status(self):
                return 200

            async def json(self):
                return {
                    "vulnerabilities": [
                        _cve_item("CVE-2021-41773", "2.4.49", "2.4.49"),
                        _cve_item("CVE-2019-0215", "2.4.17", "2.4.38"),
                    ]
                }

        class _FakeGet:
            async def __aenter__(self):
                return _FakeResponse()

            async def __aexit__(self, *args):
                return False

        class _FakeSession:
            def get(self, url, timeout=None):
                urls.append(url)
                return _FakeGet()

        scanner.session = _FakeSession()

        identity = scanner._resolve_identity("HTTP", "", "Server: Apache/2.4.49 (Unix)")
        assert identity is not None and identity.version == "2.4.49"
        result = await scanner._query_nvd("HTTP", identity)

        # NVD's current CPE product name for httpd is "http_server".
        assert "cpeName=cpe%3A2.3%3Aa%3Aapache%3Ahttp_server%3A2.4.49" in urls[0]
        assert [v["cve"] for v in result] == ["CVE-2021-41773"]
        assert result[0]["match_quality"] == "version-verified"
        assert result[0]["version"] == "2.4.49"

    @pytest.mark.asyncio
    async def test_nvd_verification_accepts_legacy_cpe_product_names(self, scanner):
        """Pre-migration NVD records still carry cpe:...:httpd:... — the
        alias must verify like the current http_server spelling."""

        class _FakeResponse:
            @property
            def status(self):
                return 200

            async def json(self):
                return {
                    "vulnerabilities": [
                        {
                            "cve": {
                                "id": "CVE-2017-9788",
                                "descriptions": [{"value": "mod_http2"}],
                                "metrics": {},
                                "configurations": [
                                    {
                                        "nodes": [
                                            {
                                                "cpeMatch": [
                                                    {
                                                        "cpe23Uri": "cpe:2.3:a:apache:httpd:2.4.49",
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ],
                            }
                        }
                    ]
                }

        class _FakeGet:
            async def __aenter__(self):
                return _FakeResponse()

            async def __aexit__(self, *args):
                return False

        class _FakeSession:
            def get(self, url, timeout=None):
                return _FakeGet()

        scanner.session = _FakeSession()
        identity = scanner._resolve_identity("HTTP", "", "Apache/2.4.49")
        result = await scanner._query_nvd("HTTP", identity)
        assert [v["cve"] for v in result] == ["CVE-2017-9788"]
        assert result[0]["match_quality"] == "version-verified"

    @pytest.mark.asyncio
    async def test_versionless_identity_queries_by_keyword(self, scanner):
        """NVD rejects versionless cpeName with a permanent 404, so a
        versionless product identity must fall back to keyword search."""

        class _FakeResponse:
            @property
            def status(self):
                return 200

            async def json(self):
                return {"vulnerabilities": []}

        class _FakeGet:
            async def __aenter__(self):
                return _FakeResponse()

            async def __aexit__(self, *args):
                return False

        urls: list[str] = []

        class _FakeSession:
            def get(self, url, timeout=None):
                urls.append(url)
                return _FakeGet()

        scanner.session = _FakeSession()
        identity = scanner._resolve_identity("FTP", "", "220 ProFTPD Server ready")
        assert identity is not None and identity.version is None
        await scanner._query_nvd("FTP", identity)
        assert "keywordSearch=proftpd" in urls[0]
        assert "cpeName" not in urls[0]

    @pytest.mark.asyncio
    async def test_versionless_keyword_hits_require_product_cpe(self, scanner):
        """A keyword hit whose CPE criteria never name the product is a
        description-text coincidence — it must not count as a finding."""

        def _item(cve_id, cpe_product):
            uri = f"cpe:2.3:a:someone:{cpe_product}:1.0" if cpe_product else None
            criterion = {"cpe23Uri": uri} if uri else {}
            return {
                "cve": {
                    "id": cve_id,
                    "descriptions": [{"value": "mentions proftpd in prose"}],
                    "metrics": {},
                    "configurations": [{"nodes": [{"cpeMatch": [criterion]}]}],
                }
            }

        payload = {
            "vulnerabilities": [
                _item("CVE-2022-11111", "otherproduct"),
                _item("CVE-2022-22222", None),
                _item("CVE-2022-33333", "proftpd"),
            ]
        }

        class _FakeResponse:
            @property
            def status(self):
                return 200

            async def json(self):
                return payload

        class _FakeGet:
            async def __aenter__(self):
                return _FakeResponse()

            async def __aexit__(self, *args):
                return False

        class _FakeSession:
            def get(self, url, timeout=None):
                return _FakeGet()

        scanner.session = _FakeSession()
        identity = scanner._resolve_identity("FTP", "", "220 ProFTPD Server ready")
        result = await scanner._query_nvd("FTP", identity)

        by_cve = {v["cve"]: v for v in result}
        assert by_cve["CVE-2022-11111"]["match_quality"] == "keyword-candidate"
        assert by_cve["CVE-2022-22222"]["match_quality"] == "keyword-candidate"
        assert by_cve["CVE-2022-33333"]["match_quality"] == "service-heuristic"

    @pytest.mark.asyncio
    async def test_nvd_keyword_hits_without_identity_are_candidates(self, scanner):
        """No product identity → keyword search runs but hits are labeled
        candidates, never findings (issue #15)."""

        class _FakeResponse:
            @property
            def status(self):
                return 200

            async def json(self):
                return {
                    "vulnerabilities": [
                        {
                            "cve": {
                                "id": "CVE-2020-1234",
                                "descriptions": [{"value": "keyword hit"}],
                                "metrics": {},
                                "configurations": [],
                            }
                        }
                    ]
                }

        class _FakeGet:
            async def __aenter__(self):
                return _FakeResponse()

            async def __aexit__(self, *args):
                return False

        class _FakeSession:
            def get(self, url, timeout=None):
                assert "keywordSearch=" in url
                return _FakeGet()

        scanner.session = _FakeSession()
        result = await scanner._query_nvd("HTTP", identity=None)

        assert len(result) == 1
        assert result[0]["match_quality"] == "keyword-candidate"

    def test_report_separates_findings_from_candidates(self):
        """The rendered report counts only verified/heuristic entries as
        findings and lists keyword candidates separately."""
        from clearwing.core.config import ScanConfig  # noqa: F401  (import parity)
        from clearwing.core.engine import ScanResult
        from clearwing.reporting.report_generator import ReportGenerator

        result = ScanResult(target="127.0.0.1")
        result.vulnerabilities = [
            {
                "cve": "CVE-2015-3306",
                "description": "ProFTPD mod_copy RCE",
                "cvss": 9.8,
                "port": 21,
                "ports": [21],
                "service": "FTP",
                "match_quality": "version-verified",
                "product": "proftpd",
                "version": "1.3.5",
            },
            {
                "cve": "CVE-2020-9999",
                "description": "keyword noise",
                "cvss": 5.0,
                "port": 80,
                "ports": [80],
                "service": "HTTP",
                "match_quality": "keyword-candidate",
            },
        ]
        report = ReportGenerator().generate(result, "text")
        assert "Findings: 1 (plus 1 unverified keyword candidates)" in report
        assert "Unverified keyword candidates" in report
        assert "Match: version-verified (proftpd 1.3.5)" in report

    @pytest.mark.asyncio
    async def test_engine_closes_scanner_even_when_scan_raises(self):
        """PR #20 regression: `CoreEngine._vulnerability_scan` wraps the
        scanner in `try/finally: await scanner.close()` so the
        lazily-allocated aiohttp ClientSession is reliably cleaned up
        even when `scanner.scan()` raises. Without the finally block,
        aiohttp emits an `Unclosed client session` warning at interpreter
        teardown.
        """
        from clearwing.core.config import ScanConfig
        from clearwing.core.engine import CoreEngine, ScanResult

        engine = CoreEngine()
        engine.scan_result = ScanResult(target="127.0.0.1")
        engine.scan_result.services = [{"port": 80, "service": "HTTP", "version": "2.4.41"}]

        fake_scanner = AsyncMock()
        fake_scanner.scan = AsyncMock(side_effect=RuntimeError("simulated NVD failure"))
        fake_scanner.close = AsyncMock()

        with (
            patch("clearwing.core.engine.VulnerabilityScanner", return_value=fake_scanner),
            pytest.raises(RuntimeError, match="simulated NVD failure"),
        ):
            await engine._vulnerability_scan("127.0.0.1", ScanConfig(target="127.0.0.1"))

        fake_scanner.close.assert_awaited_once()


class TestOSScanner:
    """Tests for OSScanner module."""

    @pytest.fixture
    def scanner(self):
        return OSScanner()

    @pytest.mark.asyncio
    async def test_os_detection(self, scanner):
        """Test OS detection."""
        result = await scanner.detect("127.0.0.1")
        assert isinstance(result, str)

    def test_ttl_guessing(self, scanner):
        """Test OS guessing by TTL."""
        assert scanner._guess_os_by_ttl(64) == "Linux/Unix"
        assert scanner._guess_os_by_ttl(128) == "Windows"
        assert scanner._guess_os_by_ttl(255) == "Network Device"

    def test_detect_sync(self, scanner):
        """Test synchronous detect method."""
        result = scanner.detect_sync("127.0.0.1")
        assert isinstance(result, str)


class TestRawSocketCapabilityFallback:
    """Issue #34: syn (default) without CAP_NET_RAW silently returned []."""

    @pytest.mark.asyncio
    async def test_syn_scan_falls_back_to_connect(self, monkeypatch, caplog):
        from clearwing.scanning import port_scanner

        monkeypatch.setattr(port_scanner, "_has_raw_socket_privilege", lambda: False)
        monkeypatch.setattr(
            port_scanner.PortScanner,
            "_connect_scan",
            AsyncMock(return_value=True),
        )
        syn_mock = AsyncMock(side_effect=AssertionError("syn must not run unprivileged"))
        monkeypatch.setattr(port_scanner.PortScanner, "_syn_scan", syn_mock)

        with caplog.at_level(logging.WARNING, logger="clearwing.scanning.port_scanner"):
            result = await port_scanner.PortScanner().scan(
                "127.0.0.1", [80], scan_type="syn"
            )

        syn_mock.assert_not_called()
        assert len(result) == 1
        assert result[0]["port"] == 80
        assert result[0]["scan_type_used"] == "connect"
        assert result[0]["scan_type_fallback"] is True
        assert any("Falling back" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_capable_process_keeps_syn(self, monkeypatch):
        from clearwing.scanning import port_scanner

        monkeypatch.setattr(port_scanner, "_has_raw_socket_privilege", lambda: True)
        syn_mock = AsyncMock(return_value=True)
        monkeypatch.setattr(port_scanner.PortScanner, "_syn_scan", syn_mock)

        result = await port_scanner.PortScanner().scan("127.0.0.1", [80], scan_type="syn")

        syn_mock.assert_called_once()
        assert result[0]["port"] == 80
        assert "scan_type_fallback" not in result[0]

    def test_probe_reports_permission_denied(self, monkeypatch):
        from clearwing.scanning import port_scanner

        def refused(*args, **kwargs):
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr(port_scanner, "_raw_socket_probe_result", None)
        monkeypatch.setattr(port_scanner.socket, "socket", refused)
        try:
            assert port_scanner._probe_raw_socket_support() is False
            assert port_scanner._has_raw_socket_privilege() is False
        finally:
            port_scanner._raw_socket_probe_result = None

    def test_probe_success_is_cached(self, monkeypatch):
        from clearwing.scanning import port_scanner

        class _FakeRawSocket:
            def close(self):
                pass

        calls = []

        def fake_socket(*args, **kwargs):
            calls.append(1)
            return _FakeRawSocket()

        monkeypatch.setattr(port_scanner, "_raw_socket_probe_result", None)
        monkeypatch.setattr(port_scanner.socket, "socket", fake_socket)
        try:
            assert port_scanner._probe_raw_socket_support() is True
            port_scanner._probe_raw_socket_support()
            assert len(calls) == 1  # cached after the first probe
        finally:
            port_scanner._raw_socket_probe_result = None


class TestPortScanFailureSurfacing:
    """Issue #14: probe failures must not masquerade as "no open ports"."""

    @pytest.mark.asyncio
    async def test_all_probes_failed_raises_descriptive_error(self, monkeypatch, caplog):
        from clearwing.scanning import port_scanner

        async def unreachable(self, target, port):
            raise OSError("network unreachable")

        monkeypatch.setattr(port_scanner.PortScanner, "_connect_scan", unreachable)

        with caplog.at_level(logging.WARNING, logger="clearwing.scanning.port_scanner"):
            with pytest.raises(RuntimeError, match="network unreachable"):
                await port_scanner.PortScanner().scan("10.255.255.1", [22, 80], "connect")

        assert any("probes failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_partial_failure_still_returns_open_ports(self, monkeypatch, caplog):
        from clearwing.scanning import port_scanner

        async def flaky(self, target, port):
            if port == 22:
                raise OSError("probe glitch")
            return True

        monkeypatch.setattr(port_scanner.PortScanner, "_connect_scan", flaky)

        with caplog.at_level(logging.WARNING, logger="clearwing.scanning.port_scanner"):
            result = await port_scanner.PortScanner().scan("127.0.0.1", [22, 80], "connect")

        assert [r["port"] for r in result] == [80]
        assert any("probes failed" in r.message for r in caplog.records)


class TestConnectScanErrorPropagation:
    """PR #44 review P1: the connect fallback must not swallow target-level
    errors. Without raw-socket privileges every raw scan type resolves to a
    connect scan (issue #34) — and ``_connect_scan`` used to return False
    for ANY OSError, so DNS failures and unreachable networks produced a
    "scan ran, nothing open" result for targets that were never scanned."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            socket.gaierror(socket.EAI_NONAME, "Name or service not known"),
            OSError(errno.ENETUNREACH, "Network is unreachable"),
            OSError(errno.EHOSTUNREACH, "No route to host"),
        ],
    )
    async def test_target_level_errors_reach_failure_ledger(self, monkeypatch, exc):
        import asyncio

        from clearwing.scanning import port_scanner

        async def failing_open_connection(target, port, **kwargs):
            raise exc

        monkeypatch.setattr(asyncio, "open_connection", failing_open_connection)

        # All probes fail at the target level → the aggregate error fires
        # instead of an empty "no open ports" list.
        with pytest.raises(RuntimeError, match="probed ports"):
            await port_scanner.PortScanner().scan("invalid.invalid", [22, 80], "connect")

    @pytest.mark.asyncio
    async def test_target_level_error_on_some_ports_keeps_open_findings(
        self, monkeypatch, caplog
    ):
        import asyncio

        from clearwing.scanning import port_scanner

        async def half_failing_open_connection(target, port, **kwargs):
            if port == 22:
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            # Simulate an open port: hand back a minimal writer.
            writer = MagicMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()
            return (MagicMock(), writer)

        monkeypatch.setattr(asyncio, "open_connection", half_failing_open_connection)

        with caplog.at_level(logging.WARNING, logger="clearwing.scanning.port_scanner"):
            result = await port_scanner.PortScanner().scan("invalid.invalid", [22, 80], "connect")

        # The open port is still reported, and the failed probe lands in
        # the failure ledger — not silently as "closed".
        assert [r["port"] for r in result] == [80]
        assert any("gaierror" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_refused_and_timeout_stay_closed(self, monkeypatch):
        import asyncio

        from clearwing.scanning import port_scanner

        scanner = port_scanner.PortScanner()

        async def refused(target, port, **kwargs):
            raise ConnectionRefusedError()

        monkeypatch.setattr(asyncio, "open_connection", refused)
        assert await scanner._connect_scan("127.0.0.1", 1) is False
        # A refused-only scan is a clean empty result, not an error.
        assert await scanner.scan("127.0.0.1", [1, 2], "connect") == []

        async def stalled(target, port, **kwargs):
            await asyncio.sleep(10)

        monkeypatch.setattr(asyncio, "open_connection", stalled)
        assert await scanner._connect_scan("127.0.0.1", 1) is False


class TestScannerToolErrorEvents:
    """Issue #14 tool layer: failures emit ERROR-level events and return
    explicit error dicts instead of silently empty lists."""

    @pytest.mark.asyncio
    async def test_scan_ports_failure_emits_error_event(self, monkeypatch):
        import clearwing.scanning as scanning_pkg
        from clearwing.agent.tools.scan.scanner_tools import scan_ports
        from clearwing.core.events import EventBus, EventType

        async def boom(self, target, ports, scan_type, threads):
            raise RuntimeError("all probes failed")

        monkeypatch.setattr(scanning_pkg.PortScanner, "scan", boom)

        events: list[dict] = []
        bus = EventBus()
        bus.subscribe(EventType.ERROR, events.append)
        try:
            result = await scan_ports.ainvoke({"target": "127.0.0.1"})
        finally:
            bus.unsubscribe(EventType.ERROR, events.append)

        assert result["error"].startswith("port scan failed:")
        assert "all probes failed" in result["error"]
        assert events and events[0]["tool"] == "scan_ports"
        assert "scan_ports" in events[0]["message"]

    @pytest.mark.asyncio
    async def test_scan_ports_fallback_empty_result_is_annotated(self, monkeypatch):
        import clearwing.scanning as scanning_pkg
        from clearwing.agent.tools.scan.scanner_tools import scan_ports

        async def empty(self, target, ports, scan_type, threads):
            return []

        monkeypatch.setattr(scanning_pkg.PortScanner, "scan", empty)
        monkeypatch.setattr(
            scanning_pkg, "resolve_scan_type", lambda scan_type: ("connect", True)
        )

        result = await scan_ports.ainvoke({"target": "127.0.0.1", "scan_type": "syn"})

        assert result["open_ports"] == []
        assert result["scan_type_fallback"] is True
        assert result["scan_type_used"] == "connect"
        assert "CAP_NET_RAW" in result["note"]

    @pytest.mark.asyncio
    async def test_clean_empty_scan_stays_a_plain_list(self, monkeypatch):
        import clearwing.scanning as scanning_pkg
        from clearwing.agent.tools.scan.scanner_tools import scan_ports

        async def empty(self, target, ports, scan_type, threads):
            return []

        monkeypatch.setattr(scanning_pkg.PortScanner, "scan", empty)
        monkeypatch.setattr(
            scanning_pkg, "resolve_scan_type", lambda scan_type: (scan_type, False)
        )

        result = await scan_ports.ainvoke({"target": "127.0.0.1", "scan_type": "connect"})
        assert result == []
