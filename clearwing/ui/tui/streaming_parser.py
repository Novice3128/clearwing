"""Streaming LLM output parser with flag detection."""

from __future__ import annotations

from clearwing.agent.runtime import FLAG_PATTERNS
from clearwing.core.events import EventBus

# Flag detection patterns: shared verbatim with the agent runtime
# (clearwing/agent/runtime.py) — the 32-hex lookaround especially must not
# drift between streamed output scanning and tool-result scanning (issue
# #35). No import cycle: agent.runtime never imports clearwing.ui.


class StreamingParser:
    """Parse streaming LLM output for tool calls and flags."""

    def __init__(self):
        self._buffer = ""
        self._bus = EventBus()

    def feed(self, chunk: str) -> None:
        """Process a chunk of streaming output."""
        self._buffer += chunk
        self._check_flags(chunk)

    def _check_flags(self, text: str) -> None:
        for pattern in FLAG_PATTERNS:
            for match in pattern.finditer(text):
                self._bus.emit_flag(match.group(), "Detected in output")

    def reset(self) -> None:
        self._buffer = ""
