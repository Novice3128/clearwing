from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any

from clearwing.agent.graph import create_agent
from clearwing.agent.runtime import Command
from clearwing.observability.telemetry import CostTracker

from .sarif import SARIFGenerator

logger = logging.getLogger(__name__)


async def drive_with_auto_decline(
    graph: Any,
    initial_state: dict[str, Any],
    config: dict,
    *,
    limits_exceeded: Any,
    max_declines: int = 5,
) -> None:
    """Drive *graph* to completion, auto-declining every approval gate.

    An unattended run has nobody to answer an approval prompt, and an
    unanswered one used to abort the whole task on its first gated tool
    (issue #20 — session 9526f073 died on cve_db_update's download
    confirm). Declining is the conservative outcome and keeps the run
    going. A model that keeps re-requesting gated tools is bounded by
    *max_declines* before the run is abandoned.
    """
    first = True
    declines = 0
    # session_scope: cost attribution and kali container scoping
    # (current_session_id) must see this run's session id, not fall back
    # to the shared adhoc scope (parallel CI runs would share a container).
    from clearwing.agent.tooling import session_scope

    with session_scope(getattr(graph, "session_id", None)):
        while True:
            payload = initial_state if first else Command(resume=False)
            first = False
            async for _event in graph.astream(payload, config, stream_mode="values"):
                if limits_exceeded():
                    return
            if limits_exceeded():
                return
            snapshot = graph.get_state(config)
            if not getattr(snapshot, "next", ()):
                return
            declines += 1
            if declines > max_declines:
                logger.warning(
                    "CI/CD run abandoned after %d auto-declined approval gates",
                    max_declines,
                )
                return
            logger.info(
                "CI/CD run auto-declined a pending approval gate (%d/%d)",
                declines,
                max_declines,
            )


@dataclass
class CICDResult:
    """Result of a CI/CD penetration testing run."""

    exit_code: int  # 0=clean, 1=medium findings, 2=critical/high findings
    target: str
    depth: str
    findings: list[dict[str, Any]]  # {description, severity, cve, details}
    duration_seconds: float
    cost_usd: float
    tokens_used: int
    output_path: str | None


class CICDRunner:
    """Non-interactive headless penetration testing runner for CI/CD pipelines."""

    DEPTH_GOALS: dict[str, str] = {
        "quick": (
            "Perform a quick vulnerability scan of the target. "
            "Do a fast port scan of the top 100 most common ports, "
            "identify running services, and check for any critical or "
            "high-severity known vulnerabilities. Focus on speed over "
            "thoroughness. Summarize all findings."
        ),
        "standard": (
            "Perform a standard vulnerability assessment of the target. "
            "Scan the top 1000 ports, enumerate services and versions, "
            "detect the operating system, and run vulnerability checks "
            "against all discovered services. Check for default credentials "
            "and common misconfigurations. Provide a detailed summary of "
            "all findings with severity ratings."
        ),
        "deep": (
            "Perform a deep and comprehensive penetration test of the target. "
            "Scan all 65535 ports, thoroughly enumerate all services and versions, "
            "detect the operating system, and run exhaustive vulnerability checks. "
            "Attempt exploitation of discovered vulnerabilities to verify they are "
            "real. Check for default credentials, misconfigurations, and privilege "
            "escalation paths. Use the Kali container for advanced testing. "
            "Provide an exhaustive report with all findings, evidence, and "
            "remediation recommendations."
        ),
    }

    def __init__(
        self,
        target: str,
        depth: str = "quick",
        model: str = "claude-sonnet-4-6",
        output_format: str = "json",
        output_path: str | None = None,
        cost_limit: float | None = None,
        timeout_minutes: int = 30,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.target = target
        self.depth = depth
        self.model = model
        self.output_format = output_format
        self.output_path = output_path
        self.cost_limit = cost_limit
        self.timeout_minutes = timeout_minutes
        self.base_url = base_url
        self.api_key = api_key

    def run(self) -> CICDResult:
        """Execute the scan and return results.

        Steps:
            1. Validate target
            2. Create agent with session_id
            3. Build goal message based on depth (quick/standard/deep)
            4. Run agent loop with timeout
            5. Collect findings
            6. Generate output (JSON or SARIF)
            7. Return CICDResult with exit code
        """
        start_time = time.monotonic()

        # 1. Validate target
        self._validate_target()

        session_id = uuid.uuid4().hex[:8]
        graph = create_agent(
            model_name=self.model,
            session_id=session_id,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        config = {"configurable": {"thread_id": f"cicd-{session_id}"}}

        # Set up cost tracking
        cost_tracker = CostTracker()
        cost_tracker.reset()
        if self.cost_limit is not None:
            cost_tracker.cost_limit = self.cost_limit

        # 3. Build goal message
        goal = self._build_goal()

        initial_state = {
            "messages": [{"role": "user", "content": goal}],
            "target": self.target,
            "open_ports": [],
            "services": [],
            "vulnerabilities": [],
            "exploit_results": [],
            "os_info": None,
            "kali_container_id": None,
            "custom_tool_names": [],
            "session_id": session_id,
            "flags_found": [],
            "loaded_skills": [],
            "paused": False,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
        }

        # 4. Run agent loop with timeout
        timeout_seconds = self.timeout_minutes * 60
        deadline = start_time + timeout_seconds

        def _limits_exceeded() -> bool:
            return time.monotonic() > deadline or (
                cost_tracker is not None and cost_tracker.is_over_limit()
            )

        try:
            asyncio.run(
                drive_with_auto_decline(
                    graph, initial_state, config, limits_exceeded=_limits_exceeded
                )
            )
        except Exception:
            logger.warning("CI/CD agent loop failed", exc_info=True)

        # 5. Collect findings
        final_state = graph.get_state(config)
        state_values = final_state.values if hasattr(final_state, "values") else {}

        vulnerabilities = state_values.get("vulnerabilities", [])
        exploit_results = state_values.get("exploit_results", [])

        findings = self._collect_findings(vulnerabilities, exploit_results)

        total_cost = state_values.get("total_cost_usd", 0.0)
        total_tokens = state_values.get("total_tokens", 0)

        # 6. Generate output
        exit_code = self._determine_exit_code(findings)
        duration = time.monotonic() - start_time

        result = CICDResult(
            exit_code=exit_code,
            target=self.target,
            depth=self.depth,
            findings=findings,
            duration_seconds=round(duration, 2),
            cost_usd=total_cost,
            tokens_used=total_tokens,
            output_path=self.output_path,
        )

        if self.output_path:
            self._write_output(findings)

        return result

    def _validate_target(self) -> None:
        """Validate that the target is a resolvable host or valid IP."""
        try:
            socket.getaddrinfo(self.target, None)
        except socket.gaierror as exc:
            raise ValueError(
                f"Target '{self.target}' is not a valid IP or resolvable hostname."
            ) from exc

    def _build_goal(self) -> str:
        """Build the goal message based on scan depth."""
        base = self.DEPTH_GOALS.get(self.depth)
        if base is None:
            raise ValueError(f"Unknown depth '{self.depth}'. Must be one of: quick, standard, deep")
        return (
            f"TARGET: {self.target}\n\n"
            f"{base}\n\n"
            "IMPORTANT: This is a non-interactive CI/CD run. Do not ask questions or "
            "wait for user input. Complete the scan autonomously and report all findings "
            "with severity ratings (critical, high, medium, low, info)."
        )

    def _determine_exit_code(self, findings: list[dict[str, Any]]) -> int:
        """Determine exit code based on finding severities.

        Returns:
            0 if no findings of medium severity or above (clean).
            1 if medium-severity findings exist but no critical/high.
            2 if critical or high severity findings exist.
        """
        severities = {(f.get("severity") or "info").lower() for f in findings}

        if severities & {"critical", "high"}:
            return 2
        if "medium" in severities:
            return 1
        return 0

    def _collect_findings(
        self,
        vulnerabilities: list[dict[str, Any]],
        exploit_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize vulnerabilities and exploit results into finding dicts."""
        findings: list[dict[str, Any]] = []

        for vuln in vulnerabilities:
            # Issue #15: keyword-candidate entries are unverified NVD
            # description-text hits — promoting them to actionable findings
            # failed CI on unrelated high-CVSS noise (Codex PR-55 P1).
            if vuln.get("match_quality") == "keyword-candidate":
                continue
            finding: dict[str, Any] = {
                "description": vuln.get("description", vuln.get("cve", "Unknown vulnerability")),
                "severity": vuln.get("severity", self._cvss_to_severity(vuln.get("cvss", 0.0))),
                "cve": vuln.get("cve"),
                "details": vuln.get("details", ""),
            }
            findings.append(finding)

        for exploit in exploit_results:
            if exploit.get("success"):
                finding = {
                    "description": f"Exploitable: {exploit.get('vulnerability', 'unknown')}",
                    "severity": "critical",
                    "cve": exploit.get("cve"),
                    "details": exploit.get("details", "Exploitation was successful"),
                }
                findings.append(finding)

        return findings

    @staticmethod
    def _cvss_to_severity(cvss: float) -> str:
        """Map a CVSS score to a severity label."""
        if cvss >= 9.0:
            return "critical"
        if cvss >= 7.0:
            return "high"
        if cvss >= 4.0:
            return "medium"
        if cvss > 0.0:
            return "low"
        return "info"

    def _write_output(self, findings: list[dict[str, Any]]) -> None:
        """Write scan results to the configured output path."""
        if self.output_format == "sarif":
            generator = SARIFGenerator()
            output = generator.generate(findings, self.target)
        else:
            output = {
                "target": self.target,
                "depth": self.depth,
                "findings": findings,
            }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
