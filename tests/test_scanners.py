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
