import json
import re
from pathlib import Path
from typing import Any

from clearwing.agent.tooling import tool
from clearwing.core.engine import ScanResult
from clearwing.data.database import Database
from clearwing.reporting import ReportGenerator

_SCAN_DATA_KEYS = {"target", "open_ports", "services", "vulnerabilities", "exploits", "os_info"}
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)


def _as_dict(item: Any, field: str) -> dict:
    """Best-effort conversion of a scalar list item into a formatter dict.

    LLMs regularly pass ``open_ports: ["80/tcp http"]`` or bare integers
    where the report formatter expects a list of dicts.
    """
    text = str(item).strip()
    if field == "open_ports":
        entry: dict[str, Any] = {"port": "", "protocol": "tcp", "service": "", "state": "open"}
        tokens = text.split()
        consumed: set[str] = set()
        for tok in tokens:
            if "/" in tok and tok.split("/", 1)[1].isalpha():
                port, _, proto = tok.partition("/")
                entry.update(port=port, protocol=proto or "tcp")
                consumed.add(tok)
                break
        if entry["port"] == "":
            for tok in tokens:
                if tok.isdigit():
                    entry["port"] = tok
                    consumed.add(tok)
                    break
        entry["service"] = next((t for t in tokens if t not in consumed), "")
        return entry
    if field == "vulnerabilities":
        cve = _CVE_RE.search(text)
        return {
            "cve": cve.group(0).upper() if cve else "",
            "description": text,
            "service": "",
            "port": "",
            "cvss": "",
        }
    if field == "services":
        port = next((t.strip("/:") for t in text.split() if t.strip("/:").isdigit()), "")
        return {"port": port, "service": text, "version": "", "banner": ""}
    if field == "exploits":
        return {"exploit_name": text, "success": False, "cve": "", "message": text}
    return {"value": text}


def _normalize_scan_data(scan_data: Any) -> dict:
    """Coerce LLM-produced ``scan_data`` into the shape ReportGenerator expects.

    Observed failure mode (session e70ba957): the model passed a JSON
    string or a single-key wrapper (``{"item": [...]}``) around the real
    payload, which crashed ``generate_report`` with
    ``string indices must be integers, not 'str'``. Normalize defensively;
    anything unparseable becomes an empty placeholder instead of a crash.
    """
    if isinstance(scan_data, str):
        try:
            scan_data = json.loads(scan_data)
        except json.JSONDecodeError:
            scan_data = {}
    if not isinstance(scan_data, dict):
        scan_data = {}
    if len(scan_data) == 1:
        only_key = next(iter(scan_data))
        inner = scan_data[only_key]
        if only_key not in _SCAN_DATA_KEYS:
            if isinstance(inner, dict):
                scan_data = inner
            elif (
                isinstance(inner, list)
                and len(inner) == 1
                and isinstance(inner[0], dict)
            ):
                # {"item": [{...scan data...}]} — same wrapper as the dict
                # form, observed from the same session.
                scan_data = inner[0]

    normalized = dict(scan_data)
    for field in ("open_ports", "services", "vulnerabilities", "exploits"):
        value = normalized.get(field)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            normalized[field] = []
            continue
        normalized[field] = [
            item if isinstance(item, dict) else _as_dict(item, field) for item in value
        ]
    if not isinstance(normalized.get("target"), str) or not normalized.get("target"):
        normalized["target"] = str(normalized.get("target") or "unknown")
    return normalized


@tool
def generate_report(format: str, scan_data: dict) -> str:
    """Generate a report from scan data collected by the agent.

    Args:
        format: Report format - 'text', 'json', 'html', or 'markdown'.
        scan_data: Dict with keys: target, open_ports, services, vulnerabilities,
                   exploits, os_info.

    Returns:
        Formatted report string.
    """
    scan_data = _normalize_scan_data(scan_data)
    result = ScanResult(target=scan_data.get("target", "unknown"))
    result.open_ports = scan_data.get("open_ports", [])
    result.services = scan_data.get("services", [])
    result.vulnerabilities = scan_data.get("vulnerabilities", [])
    result.exploits = scan_data.get("exploits", [])
    result.os_info = scan_data.get("os_info")

    generator = ReportGenerator()
    return generator.generate(result, format)


@tool
def save_report(filepath: str, format: str, scan_data: dict) -> dict:
    """Save a report to a file.

    Args:
        filepath: Path to save the report file.
        format: Report format - 'text', 'json', 'html', or 'markdown'.
        scan_data: Dict with keys: target, open_ports, services, vulnerabilities,
                   exploits, os_info.

    Returns:
        Dict with the resolved absolute path and file size, or an error.
        The write is verified after the fact and the path is resolved —
        a success the operator cannot find on disk (issue #9: writes to
        /tmp inside a container / wiped by reboot were indistinguishable
        from real failures) must never be reported.
    """
    if not isinstance(filepath, str) or not filepath.strip():
        return {"error": "filepath must be a non-empty string"}
    scan_data = _normalize_scan_data(scan_data)
    result = ScanResult(target=scan_data.get("target", "unknown"))
    result.open_ports = scan_data.get("open_ports", [])
    result.services = scan_data.get("services", [])
    result.vulnerabilities = scan_data.get("vulnerabilities", [])
    result.exploits = scan_data.get("exploits", [])
    result.os_info = scan_data.get("os_info")

    generator = ReportGenerator()
    try:
        generator.save(result, filepath, format)
    except Exception as exc:
        return {"error": f"Failed to save report to {filepath}: {exc}"}
    written = Path(filepath).expanduser().resolve()
    if not written.is_file():
        return {"error": f"Report write verification failed: {written} does not exist"}
    return {"status": "saved", "path": str(written), "bytes": written.stat().st_size}


@tool
def query_scan_history(target: str | None = None) -> list[dict]:
    """Query scan history from the database.

    Args:
        target: Optional target IP to filter. Returns all targets if not provided.

    Returns:
        List of scan history records.
    """
    db = Database()
    if target:
        return db.get_target_history(target)
    return db.get_all_targets()


@tool
def search_cves(pattern: str) -> list[dict]:
    """Search for CVEs in the scan database by pattern.

    Args:
        pattern: SQL LIKE pattern for CVE IDs (e.g. 'CVE-2017%').

    Returns:
        List of matching vulnerability records.
    """
    db = Database()
    return db.search_vulnerabilities(pattern)
