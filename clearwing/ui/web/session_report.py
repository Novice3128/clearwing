"""Deterministic per-session markdown reports for WebUI agent sessions.

Every chat session gets a ``report.md`` under
``<results>/sessions/<session_id>/`` (see
:func:`clearwing.core.config.default_results_dir`), updated after each turn
and on disconnect. Unlike the model's final message, this artifact always
exists — even when the client disconnects mid-run.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from clearwing.core.config import default_results_dir
from clearwing.reporting.safety import redact_text

logger = logging.getLogger(__name__)


def coerce_text(value: Any) -> str:
    """Normalize a WS message payload into report-safe plain text.

    WS clients can send ``content: null`` / list / object; the transcript
    render path assumes strings and used to crash on anything else, which
    permanently broke report writing for the session (#25). Anthropic-style
    content blocks are joined; other shapes fall back to JSON so no
    information is silently dropped.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [
            part["text"]
            for part in value
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        ]
        if parts:
            return "\n".join(parts)
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)

# The shared redact_text misses secret shapes that chat transcripts commonly
# carry (operator-pasted LLM keys, the webui key itself, `password: ...`
# prose). Applied on top of redact_text for THIS artifact only — the shared
# patterns must stay conservative because sourcehunt reports legitimately
# contain 32-hex hashes as findings.
_EXTRA_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
    # Quote-tolerant so JSON-fallen dict/list content (see coerce_text) is
    # covered too: `"token": "x"` has a quote between key and colon.
    re.compile(
        r"(?i)\b(?:password|passwd|pwd|token|secret|api[_-]?key)['\"]?\s*[:=]\s*\S+"
    ),
]


def _redact(text: str) -> str:
    for pattern in _EXTRA_SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return redact_text(text)


def session_report_path(session_id: str) -> Path:
    """Path of the markdown report for a WebUI chat session."""
    return Path(default_results_dir("sessions")) / session_id / "report.md"


def _md_cell(value: Any) -> str:
    text = _redact(str(value if value is not None else ""))
    return text.replace("|", "\\|").replace("\n", " ").strip()


class SessionTranscript:
    """Accumulates one chat session's turns for deterministic reporting."""

    def __init__(self, session_id: str, target: str = "", model: str = "") -> None:
        self.session_id = session_id
        self.target = target
        self.model = model
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.updated_at = self.started_at
        self.user_messages: list[str] = []
        self.agent_messages: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.cost_usd: float | None = None
        self.tokens: int | None = None

    def add_user(self, content: Any) -> None:
        self.user_messages.append(coerce_text(content))

    def add_agent(self, content: Any) -> None:
        self.agent_messages.append(coerce_text(content))

    def add_tool(self, name: str, args: Any = None, content_length: Any = None) -> None:
        self.tool_calls.append({"name": name, "args": args, "content_length": content_length})

    def add_error(self, message: Any) -> None:
        self.errors.append(coerce_text(message))

    def set_cost(self, cost_usd: Any, tokens: Any) -> None:
        if isinstance(cost_usd, int | float):
            self.cost_usd = float(cost_usd)
        if isinstance(tokens, int):
            self.tokens = tokens

    def report_path(self) -> Path:
        return session_report_path(self.session_id)

    def render(self) -> str:
        lines = [
            "# Clearwing Session Report",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Session | {_md_cell(self.session_id)} |",
            f"| Target | {_md_cell(self.target) or '(none)'} |",
            f"| Model | {_md_cell(self.model)} |",
            f"| Started | {_md_cell(self.started_at)} |",
            f"| Updated | {_md_cell(self.updated_at)} |",
            f"| User requests | {len(self.user_messages)} |",
            f"| Agent responses | {len(self.agent_messages)} |",
            f"| Tool calls | {len(self.tool_calls)} |",
            f"| Errors | {len(self.errors)} |",
            f"| Cost | {f'${self.cost_usd:.4f}' if self.cost_usd is not None else 'n/a'} |",
            f"| Tokens | {self.tokens if self.tokens is not None else 'n/a'} |",
            "",
        ]

        if self.user_messages:
            lines.append("## User Requests")
            lines.append("")
            for i, msg in enumerate(self.user_messages, 1):
                lines.append(f"### Request {i}")
                lines.append("")
                lines.append(_redact(msg).strip() or "(empty)")
                lines.append("")

        if self.agent_messages:
            lines.append("## Agent Responses")
            lines.append("")
            for i, msg in enumerate(self.agent_messages, 1):
                lines.append(f"### Response {i}")
                lines.append("")
                lines.append(_redact(msg).strip() or "(empty)")
                lines.append("")

        if self.tool_calls:
            lines.append("## Tool Activity")
            lines.append("")
            lines.append("| # | Tool | Arguments | Result size |")
            lines.append("|---|------|-----------|-------------|")
            for i, call in enumerate(self.tool_calls, 1):
                args = call.get("args")
                args_text = _md_cell(args) if args else ""
                if len(args_text) > 120:
                    args_text = args_text[:117] + "..."
                lines.append(
                    f"| {i} | {_md_cell(call.get('name', 'tool'))} "
                    f"| {args_text} | {_md_cell(call.get('content_length', ''))} |"
                )
            lines.append("")

        if self.errors:
            lines.append("## Errors")
            lines.append("")
            for err in self.errors:
                lines.append(f"- {_redact(err).strip()}")
            lines.append("")

        lines.append(
            "Note: conversation-derived content is generated by an LLM agent; "
            "verify claims against the tool activity above before acting on them."
        )
        lines.append("")
        return "\n".join(lines)

    def _render_fallback(self, reason: str) -> str:
        """Minimal header-only report used when the full render fails.

        The transcript can be poisoned by a shape render() cannot handle;
        the artifact itself must still land (#25), with an honest note.
        """
        return "\n".join(
            [
                "# Clearwing Session Report",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Session | {_md_cell(self.session_id)} |",
                f"| Target | {_md_cell(self.target) or '(none)'} |",
                f"| Model | {_md_cell(self.model)} |",
                f"| Started | {_md_cell(self.started_at)} |",
                f"| Updated | {_md_cell(self.updated_at)} |",
                f"| User requests | {len(self.user_messages)} |",
                f"| Agent responses | {len(self.agent_messages)} |",
                f"| Tool calls | {len(self.tool_calls)} |",
                f"| Errors | {len(self.errors)} |",
                "",
                f"Note: full transcript render failed ({_md_cell(reason)}); "
                "this report contains the session summary only.",
                "",
            ]
        )

    def write(self) -> Path:
        """Render and atomically write ``report.md``; returns its path."""
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            body = self.render()
        except Exception as exc:
            logger.warning(
                "Session report render failed; writing fallback report", exc_info=True
            )
            body = self._render_fallback(f"{type(exc).__name__}: {exc}")
        path = self.report_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(body, encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        # The transcript can carry target credentials; keep the artifact
        # owner-readable only regardless of umask.
        os.chmod(path, 0o600)
        logger.info("Session report updated: %s", path)
        return path
