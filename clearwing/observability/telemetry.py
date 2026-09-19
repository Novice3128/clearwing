"""Cost tracking and telemetry for Clearwing LLM usage."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from clearwing.core.events import EventBus, EventType

logger = logging.getLogger(__name__)


@dataclass
class ToolUsage:
    """Tracks usage statistics for a single tool."""

    name: str
    calls: int = 0
    total_duration_ms: int = 0


@dataclass
class CostSummary:
    """Snapshot of current cost and usage metrics."""

    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    tool_calls: int
    by_tool: dict[str, ToolUsage]


class CostTracker:
    """Singleton that tracks LLM token usage, costs, and tool call metrics.

    Thread-safe via an internal lock.
    """

    _instance: CostTracker | None = None
    _lock_cls = threading.Lock  # used only for singleton creation

    # USD per 1M tokens. Adjust non-Claude rows to match your provider's
    # billing (glm-5.2 below is a self-hosted/gateway estimate).
    # cached_input is the cache-read rate; Anthropic bills cache reads at
    # 10% of the input price (official multiplier). OpenAI-family cached
    # rates vary by model generation — left unset until confirmed.
    PRICING: dict[str, dict[str, float]] = {
        "claude-sonnet-4-6": {"input": 3.0, "cached_input": 0.30, "output": 15.0},
        "claude-opus-4-7": {"input": 15.0, "cached_input": 1.50, "output": 75.0},
        "claude-opus-4-6": {"input": 15.0, "cached_input": 1.50, "output": 75.0},
        "claude-haiku-4-5": {"input": 0.80, "cached_input": 0.08, "output": 4.0},
        # Self-hosted / air-gapped inference has no real per-token API cost,
        # but it must stay nonzero: HunterPool's tier dispatch gate
        # (pool.py _submit_next: `spent >= budget`) uses accumulated cost as
        # a proxy for work done to decide when to stop feeding it new files.
        # A price of exactly 0.0 makes `spent` permanently 0, which disables
        # that gate regardless of --budget and lets a hunt run against every
        # file in the repo. This is nominal — ~1/1000th of Haiku pricing —
        # so it still tracks token volume without applying real API rates to
        # free local generations.
        "local-model": {"input": 0.001, "output": 0.003},
        # Fireworks "Standard" serving path. cached_input applies to the subset
        # of input tokens served from the provider's prompt cache.
        "glm-5.2": {"input": 1.40, "cached_input": 0.14, "output": 4.40},
        # z.ai GLM-5.3 via the OpenAI-compatible endpoint. Rates cross-checked
        # against actual v2-round billing (2026-09-16, issue #10 live
        # evidence: product fallback pricing overstated spend ~2.17x).
        "glm-5.3": {"input": 1.40, "cached_input": 0.14, "output": 4.40},
        "UnCut": {"input": 1.40, "cached_input": 0.14, "output": 4.40},
        "gpt-5.4": {"input": 2.50, "output": 15.0},
        "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    }

    _DEFAULT_MODEL = "claude-sonnet-4-6"

    # Unknown models that have already triggered the fallback warning.
    _warned_pricing_models: set[str] = set()

    @classmethod
    def _resolve_pricing(cls, model: str | None) -> dict[str, float] | None:
        """Pricing row for *model*, or None.

        Case-insensitive: exact key first, then basename (``org/model``
        endpoint-style names), then ``key + "-"`` prefix with the longest
        key winning — providers echo versioned model names (e.g.
        ``claude-sonnet-4-6-20260901``) and mixed-case gateway aliases
        (``UnCut``) that would otherwise silently bill at the default tier.
        """
        if not isinstance(model, str) or not model.strip():
            return None
        name = model.strip().lower()
        base = name.rsplit("/", 1)[-1]
        best: tuple[int, dict[str, float]] | None = None
        for key, row in cls.PRICING.items():
            lowered = key.lower()
            if lowered == base:
                return row
            if base.startswith(lowered + "-"):
                if best is None or len(lowered) > best[0]:
                    best = (len(lowered), row)
        return best[1] if best else None

    @classmethod
    def _warn_pricing_fallback(cls, model: str | None) -> None:
        if not isinstance(model, str):
            return
        key = model.strip().lower()
        if not key or key in cls._warned_pricing_models:
            return
        cls._warned_pricing_models.add(key)
        logger.warning(
            "No PRICING entry for model %r; estimating at the %s reference "
            "tier — reported totals will misstate this model's spend until "
            "an entry is added",
            model,
            cls._DEFAULT_MODEL,
        )

    @classmethod
    def has_pricing(cls, model: str | None) -> bool:
        """True when *model* resolves to an explicit pricing entry (no fallback)."""
        return cls._resolve_pricing(model) is not None

    @classmethod
    def estimate_cost(
        cls,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
        *,
        pricing: dict[str, float] | None = None,
    ) -> float:
        """USD cost for one call. Prices are per 1M tokens.

        ``cached_tokens`` (a subset of ``input_tokens``) bills at the model's
        ``cached_input`` rate when defined, else at the full input rate.
        Versioned echoes of known models (prefix match) bill at their tier;
        unknown models fall back to the default (Sonnet) pricing with a
        one-time-per-model warning.

        ``pricing`` (optional, keyword-only) is an authoritative per-call
        row (``{"input", "output"[, "cached_input"]}``, USD per 1M tokens)
        from the endpoint's configured ``EndpointPricing``. When supplied
        it REPLACES the table lookup entirely — no fallback and no warning,
        the caller priced the call on purpose — so the tracker, audit rows,
        and COST_UPDATE totals agree with the spend ledger on custom
        endpoints whose model names have no PRICING entry.
        """
        if pricing is not None:
            row = pricing
        else:
            row = cls._resolve_pricing(model)
            if row is None:
                cls._warn_pricing_fallback(model)
                row = cls.PRICING[cls._DEFAULT_MODEL]
        cached_rate = row.get("cached_input", row["input"])
        uncached = max(input_tokens - cached_tokens, 0)
        return (
            uncached * row["input"]
            + cached_tokens * cached_rate
            + output_tokens * row["output"]
        ) / 1_000_000

    def __new__(cls) -> CostTracker:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._lock = threading.Lock()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.tool_calls: int = 0
        self.by_tool: dict[str, ToolUsage] = {}
        self.cost_limit: float | None = None
        # Per-session cost accumulation (operator cost_limit follow-up to
        # issue #41): the global totals pool every session in the process,
        # but an operator job's limit check and result cost must see only
        # ITS spend — the inner graph's calls plus hunts attributed to the
        # job's session id. Keyed by the session_id passed to
        # record_llm_call; updated under the same lock as the global
        # counters (single-key dict add is what the GIL makes safe anyway).
        self._session_totals: dict[str, float] = {}
        # Parallel per-session token totals, (input, output). Kept in a
        # parallel dict so session_total()'s float contract stays
        # unchanged; read via session_tokens() so callers (operator result
        # fields) can report attributed token volume next to attributed
        # cost (PR #44 review).
        self._session_tokens: dict[str, tuple[int, int]] = {}
        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
        *,
        elapsed_ms: float | None = None,
        provider: str | None = None,
        session_id: str | None = None,
        pricing: dict[str, float] | None = None,
    ) -> float:
        """Record token usage for a single LLM call and update the running cost.

        If *model* is not present in the pricing table the default Sonnet
        pricing is used.  ``cached_tokens`` bills at the model's cached rate.
        When an ``EventBus`` is available a ``COST_UPDATE`` event is emitted
        after updating counters.

        ``elapsed_ms`` (wall-clock latency of the call), ``provider`` and
        ``session_id`` are optional; when supplied they ride along in the
        ``COST_UPDATE`` payload for UI and metrics consumers — ``session_id``
        lets scoped consumers attribute the call to a session, and also
        accumulates a per-session cost total queryable via
        :meth:`session_total`. Keyword-only to keep call sites explicit and
        future additions non-breaking. ``pricing`` (an authoritative
        per-call row from the endpoint's ``EndpointPricing``) likewise
        overrides the table lookup — see :meth:`estimate_cost`.
        """
        cost = self.estimate_cost(
            input_tokens, output_tokens, model, cached_tokens, pricing=pricing
        )

        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_cost_usd += cost
            if session_id:
                self._session_totals[session_id] = (
                    self._session_totals.get(session_id, 0.0) + cost
                )
                prev_in, prev_out = self._session_tokens.get(session_id, (0, 0))
                self._session_tokens[session_id] = (
                    prev_in + input_tokens,
                    prev_out + output_tokens,
                )

        try:
            EventBus().emit(
                EventType.COST_UPDATE,
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "cost": cost,
                    "total_cost_usd": self.total_cost_usd,
                    "model": model,
                    "provider": provider or "unknown",
                    "session_id": session_id,
                    "elapsed_ms": elapsed_ms or 0,
                },
            )
        except Exception:
            pass  # telemetry should never break the caller
        return cost

    def record_tool_call(self, tool_name: str, duration_ms: int) -> None:
        """Record a tool invocation and its wall-clock duration."""
        with self._lock:
            self.tool_calls += 1
            if tool_name not in self.by_tool:
                self.by_tool[tool_name] = ToolUsage(name=tool_name)
            usage = self.by_tool[tool_name]
            usage.calls += 1
            usage.total_duration_ms += duration_ms

    def session_total(self, session_id: str | None) -> float:
        """Cumulative USD recorded for *session_id* (0.0 when unattributed).

        Every ``record_llm_call`` carrying a ``session_id`` accumulates here:
        the runtime attributes each assistant step to its graph's session,
        and operator jobs wrap their whole loop in a ``session_scope`` so
        spawned hunts land on the job's id too. Operators use this (not the
        graph's state total) for cost limits and result fields, so hunt
        spend is enforced/reported instead of invisible.
        """
        if not session_id:
            return 0.0
        with self._lock:
            return self._session_totals.get(session_id, 0.0)

    def session_tokens(self, session_id: str | None) -> tuple[int, int]:
        """``(input, output)`` token totals recorded for *session_id*.

        Same attribution set as :meth:`session_total`: every
        ``record_llm_call`` carrying the id accumulates here, so callers
        that report session-scoped cost (operator result fields) can report
        the matching token volume — including spend booked outside the
        graph state (hunts, supervisor calls).
        """
        if not session_id:
            return (0, 0)
        with self._lock:
            return self._session_tokens.get(session_id, (0, 0))

    def session_snapshot(self, session_id: str | None) -> tuple[float, int, int]:
        """``(cost_usd, input_tokens, output_tokens)`` for *session_id*,
        read under ONE lock acquisition.

        Frames that report session-scoped totals must show cost and tokens
        from the same point in time: reading :meth:`session_total` and
        :meth:`session_tokens` separately lets a concurrent booking land
        between the two locked reads, producing a frame whose token total
        includes a call its cost total does not.
        """
        if not session_id:
            return (0.0, 0, 0)
        with self._lock:
            cost = self._session_totals.get(session_id, 0.0)
            in_tokens, out_tokens = self._session_tokens.get(session_id, (0, 0))
            return (cost, in_tokens, out_tokens)

    def forget_session(self, session_id: str | None) -> None:
        """Drop *session_id*'s per-session cost/token entries.

        Webui and operator session ids are 8-hex UUID prefixes; in a
        long-lived process a colliding id would inherit the earlier
        session's spend (bogus cost limits and inflated result cost).
        Owners call this on session teardown. Unknown ids are a no-op;
        global counters are untouched (process-wide totals keep every
        call).
        """
        if not session_id:
            return
        with self._lock:
            self._session_totals.pop(session_id, None)
            self._session_tokens.pop(session_id, None)

    def get_summary(self) -> CostSummary:
        """Return a point-in-time snapshot of all tracked metrics."""
        with self._lock:
            return CostSummary(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                total_cost_usd=self.total_cost_usd,
                tool_calls=self.tool_calls,
                by_tool=dict(self.by_tool),
            )

    def is_over_limit(self) -> bool:
        """Return ``True`` if a cost limit is set and the current spend exceeds it."""
        with self._lock:
            if self.cost_limit is None:
                return False
            return self.total_cost_usd > self.cost_limit

    def reset(self) -> None:
        """Reset all counters to their initial state."""
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0
            self.total_cost_usd = 0.0
            self.tool_calls = 0
            self.by_tool = {}
            self._session_totals = {}
            self._session_tokens = {}
