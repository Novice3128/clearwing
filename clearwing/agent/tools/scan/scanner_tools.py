import logging

import clearwing.scanning as scanning
from clearwing.agent.tooling import tool
from clearwing.core.events import EventBus, EventType

logger = logging.getLogger(__name__)


def _emit_scan_error(tool_name: str, exc: Exception) -> str:
    """Log + emit an ERROR-level event for a failed scan (issue #14).

    Scan failures used to surface as silently empty results with at most a
    debug log — the audit trail saw zero error events and the agent read
    the empty list as "nothing found". Returns the descriptive error text
    so the tool result carries it too.
    """
    detail = str(exc) or type(exc).__name__
    # Subprocess-driven helpers carry the exit status on the exception —
    # include it plus a stderr excerpt when available.
    returncode = getattr(exc, "returncode", None)
    if returncode is not None:
        detail = f"exit={returncode}: {detail}"
    stderr = getattr(exc, "stderr", None)
    if stderr:
        stderr_text = stderr.decode("utf-8", "ignore") if isinstance(stderr, bytes) else str(stderr)
        detail = f"{detail}; stderr: {stderr_text[:200]}"
    logger.warning("%s failed: %s", tool_name, detail)
    try:
        EventBus().emit(
            EventType.ERROR,
            {"message": f"{tool_name}: {detail}", "tool": tool_name},
        )
    except Exception:
        logger.debug("failed to emit scan error event", exc_info=True)
    return detail


@tool
async def scan_ports(
    target: str,
    ports: list[int] | None = None,
    scan_type: str = "syn",
    threads: int = 100,
) -> list[dict] | dict:
    """Scan a target for open ports.

    Args:
        target: Target IP address.
        ports: List of ports to scan. Defaults to common ports if not provided.
        scan_type: Type of scan - 'syn' or 'connect'. Raw-socket types
            ('syn'/'fin'/'ack'/'xmas'/'null') automatically fall back to a
            userland 'connect' scan when the process lacks CAP_NET_RAW;
            results then carry scan_type_used/scan_type_fallback fields.
        threads: Number of concurrent threads.

    Returns:
        List of open port info dicts with keys: port, protocol, state,
        service (plus scan_type_used/scan_type_fallback after a capability
        fallback). On total scan failure: an error dict. After a fallback
        scan that finds nothing: a dict with open_ports/scan_type_fallback
        so the empty result is not misread as "no open ports".
    """
    scanner = scanning.PortScanner()
    try:
        results = await scanner.scan(target, ports or [], scan_type, threads)
    except Exception as e:
        return {"error": f"port scan failed: {_emit_scan_error('scan_ports', e)}"}
    if results:
        return results
    # Empty result: distinguish a clean "no open ports" from a capability
    # fallback (issue #34) — the agent must never read a fallback scan as
    # proof the target is closed.
    _, fell_back = scanning.resolve_scan_type(scan_type)
    if fell_back:
        return {
            "open_ports": [],
            "scan_type_used": "connect",
            "scan_type_fallback": True,
            "note": (
                f"scan_type={scan_type!r} needs raw sockets (CAP_NET_RAW); "
                "ran a userland TCP connect scan instead and found no open "
                "ports in the probed set"
            ),
        }
    return results


@tool
async def detect_services(target: str, open_ports: list[dict]) -> list[dict] | dict:
    """Detect services running on open ports via banner grabbing.

    Args:
        target: Target IP address.
        open_ports: List of open port dicts from scan_ports (must have 'port' key).

    Returns:
        List of service info dicts with keys: port, service, banner, version,
        protocol. On failure: an error dict.
    """
    scanner = scanning.ServiceScanner()
    try:
        return await scanner.detect(target, open_ports)
    except Exception as e:
        return {"error": f"service detection failed: {_emit_scan_error('detect_services', e)}"}


@tool
async def scan_vulnerabilities(target: str, services: list[dict]) -> list[dict] | dict:
    """Scan detected services for known vulnerabilities using local DB and NVD.

    Args:
        target: Target IP address.
        services: List of service dicts from detect_services.

    Returns:
        List of vulnerability dicts with keys: cve, description, cvss, port,
        service. On failure: an error dict.
    """
    scanner = scanning.VulnerabilityScanner()
    try:
        return await scanner.scan(target, services)
    finally:
        await scanner.close()


@tool
async def detect_os(target: str) -> str | dict:
    """Detect the operating system of the target using TTL and TCP fingerprinting.

    Args:
        target: Target IP address.

    Returns:
        Detected OS string (e.g. 'Linux/Unix', 'Windows') or 'Unknown'.
        On failure: an error dict.
    """
    scanner = scanning.OSScanner()
    try:
        return await scanner.detect(target)
    except Exception as e:
        return {"error": f"OS detection failed: {_emit_scan_error('detect_os', e)}"}
