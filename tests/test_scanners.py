import logging
import os
from unittest.mock import AsyncMock, patch

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
        """Test local vulnerability database lookup."""
        vulns = scanner._check_local_db("FTP")
        assert isinstance(vulns, list)
        assert len(vulns) > 0

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
