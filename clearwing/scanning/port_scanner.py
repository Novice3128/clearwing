import asyncio
import logging
import os
import socket
from typing import Any

import libpnet_pyo3

logger = logging.getLogger(__name__)

# Scan types that need raw sockets / packet-crafting privileges.
_RAW_SCAN_TYPES = frozenset({"syn", "fin", "ack", "xmas", "null"})

# Process-wide cache for the raw-socket capability probe (issue #34): the
# probe opens a socket, so run it once per process instead of once per scan.
_raw_socket_probe_result: bool | None = None


def _probe_raw_socket_support() -> bool | None:
    """Actually try to open a raw TCP socket.

    Returns True/False when the probe is conclusive; None when the platform
    can't be probed this way (caller falls back to the euid heuristic).
    """
    global _raw_socket_probe_result
    if _raw_socket_probe_result is not None:
        return _raw_socket_probe_result
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
    except PermissionError:
        _raw_socket_probe_result = False
        return False
    except OSError:
        # Unsupported family / platform quirk — not a privilege verdict.
        return None
    try:
        probe.close()
    except OSError:
        pass
    _raw_socket_probe_result = True
    return True


def _has_raw_socket_privilege() -> bool:
    """Best-effort check that this process can open raw sockets / BPF.

    A conclusive raw-socket probe wins (issue #34): it covers both the
    non-root-without-CAP_NET_RAW webui case (probe refused → False) and a
    capability-granted non-root process (probe succeeds → True). When the
    probe is inconclusive, fall back to the euid heuristic: root on Unix
    (covers macOS and the common Linux case). On Windows / platforms
    without ``os.geteuid`` we return True and let libpnet_pyo3 surface the
    privilege error itself.
    """
    probe = _probe_raw_socket_support()
    if probe is not None:
        return probe
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return True
    return geteuid() == 0


def resolve_scan_type(scan_type: str) -> tuple[str, bool]:
    """Resolve *scan_type* against the process's raw-socket capability.

    Raw-socket scan types without the capability resolve to a userland TCP
    connect scan (issue #34): without the fallback every raw probe raises
    PermissionError and the report silently shows "no open ports".
    Returns ``(effective_scan_type, fell_back)``.
    """
    if scan_type in _RAW_SCAN_TYPES and not _has_raw_socket_privilege():
        return "connect", True
    return scan_type, False


COMMON_PORTS = [
    21,
    22,
    23,
    25,
    53,
    80,
    110,
    111,
    135,
    139,
    143,
    443,
    445,
    993,
    995,
    1723,
    3306,
    3389,
    5900,
    8080,
    8443,
    2222,
    2323,
    2525,
    3333,
    4444,
    5555,
    6666,
    7777,
    8888,
    9999,
    10000,
    12345,
    20000,
    30000,
    40000,
    50000,
]

SERVICE_NAMES = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    111: "RPCbind",
    135: "MSRPC",
    139: "NetBIOS",
    143: "IMAP",
    443: "HTTPS",
    445: "SMB",
    993: "IMAPS",
    995: "POP3S",
    1723: "PPTP",
    3306: "MySQL",
    3389: "RDP",
    5900: "VNC",
    8080: "HTTP-Proxy",
    8443: "HTTPS-Alt",
}


class PortScanner:
    """Port scanning module with multiple scan techniques."""

    def __init__(self):
        self.timeout = 1

    async def scan(
        self, target: str, ports: list[int] = None, scan_type: str = "syn", threads: int = 100
    ) -> list[dict[str, Any]]:
        """
        Scan target for open ports.

        Args:
            target: Target IP address
            ports: List of ports to scan (defaults to COMMON_PORTS)
            scan_type: Type of scan ('syn', 'connect', 'fin', 'ack', 'xmas', 'null')
            threads: Number of concurrent threads

        Returns:
            List of dictionaries containing port information
        """
        ports = ports or COMMON_PORTS
        open_ports: list[dict[str, Any]] = []
        # Probe failures must not masquerade as "closed" ports (issue #14):
        # collect them and surface a summary after the scan.
        failures: list[str] = []

        # Issue #34: raw-socket scan types without the capability silently
        # answered PermissionError per port and the report showed "no open
        # ports". Resolve the capability once and fall back to a userland
        # TCP connect scan, annotating the results so downstream consumers
        # know which scan actually ran.
        effective_scan_type, fell_back = resolve_scan_type(scan_type)
        if fell_back:
            logger.warning(
                "scan_type=%r needs raw-socket privileges (CAP_NET_RAW); "
                "this process cannot open raw sockets, so every probe "
                "would fail and the report would silently show no open "
                "ports. Falling back to scan_type='connect' (issue #34).",
                scan_type,
            )

        # Create semaphore for limiting concurrent connections
        semaphore = asyncio.Semaphore(threads)

        async def scan_port(port: int) -> dict[str, Any]:
            async with semaphore:
                try:
                    if effective_scan_type == "syn":
                        result = await self._syn_scan(target, port)
                    elif effective_scan_type == "connect":
                        result = await self._connect_scan(target, port)
                    else:
                        result = await self._syn_scan(target, port)

                    if result:
                        entry = {
                            "port": port,
                            "protocol": "tcp",
                            "state": "open",
                            "service": SERVICE_NAMES.get(port, "Unknown"),
                        }
                        if fell_back:
                            entry["scan_type_used"] = effective_scan_type
                            entry["scan_type_fallback"] = True
                        open_ports.append(entry)
                except Exception as exc:
                    failures.append(f"{target}:{port}: {type(exc).__name__}: {exc}")
            return None

        # Run all scans concurrently
        tasks = [scan_port(port) for port in ports]
        await asyncio.gather(*tasks)

        if failures:
            logger.warning(
                "port scan of %s: %d/%d probes failed (first failure: %s)",
                target,
                len(failures),
                len(ports),
                failures[0],
            )
            if len(failures) == len(ports) and not open_ports:
                # Every probe died abnormally (permission, unreachable
                # network, ...): an empty result here would read as "no
                # open ports" — raise so the caller gets a real error
                # (issue #14).
                raise RuntimeError(
                    f"port scan failed for all {len(ports)} probed ports of "
                    f"{target} (first failure: {failures[0]})"
                )

        return sorted(open_ports, key=lambda x: x["port"])

    async def _syn_scan(self, target: str, port: int) -> bool:
        """Perform SYN scan on a single port.

        Transport/probe errors propagate to the caller's failure ledger —
        a PermissionError or dead libpnet must not read as "closed".
        """
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: libpnet_pyo3.tcp_sr1(
                dst=target, dport=port, flags="S", timeout=self.timeout
            ),
        )
        if resp is not None and resp.is_synack():
            # Send RST to close the connection (fire-and-forget).
            try:
                await loop.run_in_executor(
                    None,
                    lambda: libpnet_pyo3.tcp_send(dst=target, dport=port, flags="R"),
                )
            except Exception:
                logger.debug("RST send failed for %s:%d", target, port, exc_info=True)
            return True
        return False

    async def _connect_scan(self, target: str, port: int) -> bool:
        """Perform TCP connect scan on a single port.

        Only the expected negative outcomes of a connect scan read as "port
        closed": connection refused (RST) and timeout (filtered). Target or
        network level errors — DNS resolution failure (``gaierror``),
        unreachable network (``ENETUNREACH``), ... — propagate to the
        caller's failure ledger (issue #14), so an unscannable target is
        distinguishable from a clean "no open ports" result instead of
        silently producing one (PR #44 review P1).
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target, port), timeout=self.timeout
            )
        except (asyncio.TimeoutError, ConnectionRefusedError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            # The connect succeeded, so the port IS open even when the
            # teardown handshake hiccups — never demote it to a failure.
            logger.debug("close() failed for %s:%d", target, port, exc_info=True)
        return True

    def scan_sync(
        self, target: str, ports: list[int] = None, scan_type: str = "syn"
    ) -> list[dict[str, Any]]:
        """Synchronous version of scan for backward compatibility."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.scan(target, ports, scan_type))
        finally:
            loop.close()
