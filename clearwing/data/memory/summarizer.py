"""Context window summarizer for long-running penetration testing sessions."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FLAG_PATTERNS = re.compile(
    r"(flag\{[^}]*\}|FLAG\{[^}]*\}|HTB\{[^}]*\}|CTF\{[^}]*\})", re.IGNORECASE
)

_SUMMARIZE_PROMPT = (
    "Summarize these penetration testing findings concisely, preserving: "
    "discovered ports, services, vulnerabilities, exploit results, and any flags found."
)


class ContextSummarizer:
    """Compresses message history, keeping tool calls and flags verbatim.

    Issue #38: ``summarize()`` used to return ``[summary, *preserved,
    *recent]`` with the summary as the first message — and the runtime never
    committed the compacted list, so every step past the 80% threshold
    re-ran the LLM summary and sent a different front-of-history (or a
    different system prompt, since the summary rode as a ``system`` message)
    — destroying the prompt-cache prefix each step.

    The compaction is now the caller's to keep: ``summarize()`` returns the
    compacted ``view`` (covered messages removed, tool calls / tool results /
    flag-bearing messages preserved verbatim) plus a summary ``state``
    (``{"text": str, "covered_count": int}``). Callers commit the view to
    their history and re-inject the summary text AFTER the prompt-cache
    breakpoint (context-note tail), so between two steps without
    re-summarization the sent prefix is byte-identical, and the summary LLM
    is only called when there is a newly coverable segment.
    """

    @staticmethod
    def _message_role(message: Any) -> str:
        """Role label for either message shape (object or dict)."""
        if isinstance(message, dict):
            return str(message.get("role", "msg"))
        return str(getattr(message, "role", "msg"))

    @staticmethod
    def _message_text(message: Any) -> str:
        """Content of a message as text, for either message shape.

        Dict-shaped (legacy LangChain-style) messages carry ``content`` as a
        mapping key, not an attribute — ``getattr`` on them yields None,
        which used to silently drop their text from the summary input while
        the compaction still removed them from history (PR #44 review P1).
        Non-string content (block lists, ...) is stringified so whatever the
        provider sent still reaches the summarizer verbatim.
        """
        if isinstance(message, dict):
            content = message.get("content", None)
        else:
            content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _estimate_tokens(messages: list) -> int:
        total_chars = 0
        for msg in messages:
            content = ContextSummarizer._message_text(msg) or str(msg)
            total_chars += len(content)
            for tc in getattr(msg, "tool_calls", None) or []:
                total_chars += len(getattr(tc, "fn_arguments_json", None) or "")
        return total_chars // 4

    @staticmethod
    def _is_coverable(message: Any) -> bool:
        """True when a message may be replaced by the running summary.

        Tool calls and tool results must stay verbatim — providers 400 a
        history with an orphaned ``tool_use``/``tool_result`` pair — and
        flag-bearing messages are kept verbatim by policy.
        """
        if isinstance(message, dict):
            # Dict-shaped (legacy LangChain-style) messages: a tool result
            # (``role: tool`` / ``tool_call_id``) or an assistant turn
            # carrying ``tool_calls`` must never be summarized away —
            # covering either orphans the paired tool_use (provider 400).
            # No in-repo caller emits this shape today; defensive.
            if (
                message.get("role") == "tool"
                or message.get("tool_call_id")
                or message.get("tool_calls")
            ):
                return False
        content = ContextSummarizer._message_text(message)
        if _FLAG_PATTERNS.search(content):
            return False
        if getattr(message, "tool_calls", None):
            return False
        # genai ChatMessage carries ``tool_response_call_id``; the runtime's
        # internal ToolMessage carries ``tool_call_id``. Both mark a message
        # that must stay verbatim to keep tool_use/tool_result pairs intact.
        if getattr(message, "tool_response_call_id", None) or getattr(
            message, "tool_call_id", None
        ):
            return False
        return True

    def should_summarize(self, messages: list, max_tokens: int = 150_000) -> bool:
        return self._estimate_tokens(messages) > int(max_tokens * 0.8)

    @staticmethod
    def summary_note(text: str) -> str:
        """Format the summary for injection after the cache breakpoint."""
        return f"[Session Summary]\n{text}"

    async def summarize(
        self,
        messages: list,
        llm: Any,
        *,
        prior: dict[str, Any] | None = None,
        prompt_cache_key: str | None = None,
        cache_prefix: bool = True,
    ) -> dict[str, Any]:
        """(Re)generate the running summary over the oldest 70% of *messages*.

        Returns ``{"text": str, "covered_count": int, "view": list,
        "usage": dict | None, "served_model": str | None}``. The four
        #38-contract keys keep their meaning: ``view`` is *messages* with
        the covered segment removed — the caller replaces its history with
        it, which is what keeps the threshold from re-firing (and
        re-billing a summary LLM call) on every step. ``covered_count`` is
        the cumulative number of summarized messages across epochs.
        ``usage`` carries the summary call's token counts
        (input/output/cached) when the provider reported them, so callers
        can book the summarizer's spend; None on every path that skipped
        the LLM. ``served_model`` (additive, Codex PR-63 r2) is the
        provider's model echo from the summary response so callers audit
        the call under the model that actually served it — None wherever
        no LLM was called or the response carried no concrete string echo
        (defensive getattr + isinstance: test doubles may not expose the
        attribute, and mocks auto-create non-string junk). When the old
        segment holds nothing newly coverable the prior state is returned
        unchanged and the LLM is NOT called (coverage unchanged -> reuse;
        issue #38).

        *prior* is the previous ``{"text", "covered_count"}`` state; its text
        is fed to the summarization prompt so knowledge accumulates across
        epochs instead of being dropped at each compaction.
        """
        prior = prior or {}
        if not messages:
            return {
                "text": prior.get("text", ""),
                "covered_count": prior.get("covered_count", 0),
                "view": messages,
                "usage": None,
                "served_model": None,
            }

        total = len(messages)
        split_idx = int(total * 0.7)
        old_messages = messages[:split_idx]
        to_summarize = [m for m in old_messages if self._is_coverable(m)]
        if not to_summarize:
            # Nothing newly coverable (e.g. the old segment is all tool
            # traffic): reuse the prior summary verbatim so the request
            # payload stays byte-identical — no LLM call, no prefix churn.
            return {
                "text": prior.get("text", ""),
                "covered_count": prior.get("covered_count", 0),
                "view": list(messages),
                "usage": None,
                "served_model": None,
            }

        prior_text = prior.get("text") or ""
        blocks: list[str] = []
        if prior_text:
            blocks.append(f"[prior session summary]: {prior_text}")
        # PR #44 review P1: every covered message's content MUST reach the
        # summary input. Dict-shaped messages used to fall through
        # ``getattr`` here (None content) — they were selected as coverable
        # and removed from the committed view, permanently losing their
        # text (early user goals on long sessions).
        blocks.extend(
            f"[{self._message_role(m)}]: {self._message_text(m)}"
            for m in to_summarize
            if self._message_text(m)
        )
        text_block = "\n\n".join(blocks)

        from clearwing.llm.native import response_text

        # Prompt-cache wiring mirrors the main agent loop (see hunter.py):
        # a transport/billing hint only, routed by a stable per-session key.
        # Inert on providers without caching.
        summary_response = await llm.aask_text(
            system=_SUMMARIZE_PROMPT,
            user=text_block,
            cache_prefix=cache_prefix,
            prompt_cache_key=prompt_cache_key,
        )
        summary_text = response_text(summary_response)

        # Token usage of the summary call, when the provider reported it.
        # Defensive getattr throughout: fake clients and older genai builds
        # may not expose usage (or its detail fields) at all — anything that
        # is not a concrete int token count yields None so callers never
        # book a hallucinated figure.
        usage_obj = getattr(summary_response, "usage", None)
        usage_input = getattr(usage_obj, "prompt_tokens", None)
        usage_output = getattr(usage_obj, "completion_tokens", None)
        usage = None
        if isinstance(usage_input, int) and isinstance(usage_output, int):
            details = getattr(usage_obj, "prompt_tokens_details", None)
            usage_cached = getattr(details, "cached_tokens", None) if details else None
            usage = {
                "input_tokens": usage_input,
                "output_tokens": usage_output,
                "cached_tokens": usage_cached if isinstance(usage_cached, int) else 0,
            }

        # The provider's model echo for the summary call (Codex PR-63 r2):
        # callers price under the configured member key but must AUDIT the
        # model that actually served — same split as the main loop's
        # effective_model. Defensive getattr + isinstance, matching the
        # usage handling above: fake clients (unittest mocks auto-create
        # attributes) and older builds yield None, never a junk object.
        served_model = getattr(summary_response, "provider_model_name", None)
        if not isinstance(served_model, str):
            served_model = None

        if not summary_text.strip():
            # Empty/whitespace summary: committing it would REPLACE the
            # covered messages with nothing — silently destroying history
            # (and any knowledge the prior summary held). Keep the prior
            # state and the original view; the usage is still returned so
            # the billed call is not hidden from the caller's accounting.
            return {
                "text": prior.get("text", ""),
                "covered_count": prior.get("covered_count", 0),
                "view": list(messages),
                "usage": usage,
                "served_model": served_model,
            }

        covered_ids = {id(m) for m in to_summarize}
        view = [m for m in messages if id(m) not in covered_ids]
        covered_count = prior.get("covered_count", 0) + len(to_summarize)

        logger.info(
            "context summarizer: %d msgs → %d (newly covered=%d, total covered=%d)",
            total,
            len(view),
            len(to_summarize),
            covered_count,
        )
        return {
            "text": summary_text,
            "covered_count": covered_count,
            "view": view,
            "usage": usage,
            "served_model": served_model,
        }
