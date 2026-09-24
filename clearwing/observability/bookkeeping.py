"""Single-entry LLM call bookkeeping: cost tracking + audit in one call (#61).

Every LLM call that emits a ``COST_UPDATE`` frame must also land an audit
row — live reconciliation of session 61279304 showed a 0.16% gap (one
430-in/722-out summarizer call that was priced but never audited) because
``CostTracker.record_llm_call`` had five call sites while only one of them
also wrote ``AuditLogger.log_llm_call``. Centralizing the pair here makes
the audit/tracker divergence structurally impossible: a caller that books
a call books BOTH halves or neither.

The audit/tracker reconciliation contract: for a session with the tracker
enabled, ``sum(row.details["cost_usd"] for llm_call rows in
~/.clearwing/audit/<session_id>/audit.jsonl)`` equals
``CostTracker().session_total(session_id)`` — audit rows carry the
PER-CALL cost returned by ``record_llm_call``, never a running total.

The equation holds within the session's LIFETIME (before
``CostTracker.forget_session``): operator job completion and webui socket
teardown both forget the session, after which the tracker side reads zero
while the audit file keeps every row. Post-hoc reconciliation must treat
the audit file as the source of truth, not ``session_total``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from clearwing.observability.telemetry import CostTracker

if TYPE_CHECKING:  # pragma: no cover
    # Duck-typed at runtime: anything exposing log_llm_call(model=...,
    # input_tokens=..., output_tokens=..., cost_usd=..., cached_tokens=...,
    # agent=..., component=...) works — keeps this module import-cycle-free
    # from clearwing.safety.audit (which lazily imports clearwing.core.config).
    # A duck-typed logger missing the component kwarg would TypeError inside
    # the try/except and silently drop the audit half — keep this signature
    # in sync with AuditLogger.log_llm_call.
    from clearwing.safety.audit import AuditLogger

logger = logging.getLogger(__name__)


def _endpoint_pricing_row(pricing: Any) -> dict[str, float] | None:
    """Convert an endpoint pricing object to a PRICING row, or None.

    Duck-typed like the ``AuditLogger`` argument (keeps this module
    import-cycle-free): callers pass the client's ``EndpointPricing`` (USD
    per 1M tokens; ``cached_per_mtok`` optional and falling back to the
    input rate, matching the ledger's ``_pricing_value`` doctrine).
    Missing fields or non-finite/negative values degrade to ``None`` so
    booking falls back to the pricing table instead of raising — the
    ledger's ``BudgetConfigurationError`` contract has no place in a
    never-raise bookkeeping path.
    """
    try:
        row_input = float(pricing.input_per_mtok)
        row_output = float(pricing.output_per_mtok)
        raw_cached = getattr(pricing, "cached_per_mtok", None)
        row_cached = row_input if raw_cached is None else float(raw_cached)
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) and v >= 0 for v in (row_input, row_output, row_cached)):
        return None
    return {"input": row_input, "output": row_output, "cached_input": row_cached}


# component = the subsystem an LLM call belongs to, for audit↔map queries.
# Single source of truth; cross-referenced tables that must NOT drift:
#   - clearwing/sourcehunt/runner.py::_SPECIALIST_BOOK_ROLES (stage→agent tag)
#   - clearwing/providers/roles.py::TASK_ROLES (different key space: nday_filter/reveng_analysis)
# The agent-role namespace is OPEN (_specialist_book_role passes unknown
# stages through verbatim) — unknown roles map to "unmapped" so the
# component key is ALWAYS present with a fixed shape.
_AGENT_COMPONENTS = {
    "main": "agent-runtime",
    "summarizer": "agent-runtime",  # dual-homed: sourcehunt hunter path emits this too — disambiguate by session_id (sh-*/<sid>-<8hex> = sourcehunt)
    "operator": "agent-runtime",
    "hunter": "sourcehunt",
    "patcher": "sourcehunt",
    "exploiter": "sourcehunt",
    "variant": "sourcehunt",
    "harness": "sourcehunt",
    "stability": "sourcehunt",
    "mechanism": "sourcehunt",
    "proof": "sourcehunt",
    "verifier": "sourcehunt",
    "bench": "bench",
    "retro_hunt": "cli-pipelines",
    "nday": "cli-pipelines",
    "reveng": "cli-pipelines",
    "elaboration": "cli-pipelines",
    # Live verbatim-fallthrough stage (not a _SPECIALIST_BOOK_ROLES key):
    # the ranker books under stage "rank" (runner.py _get_native_client
    # budget_stage="rank" → _specialist_book_role passes it through).
    "rank": "sourcehunt",
}


def book_llm_call(
    input_tokens: int,
    output_tokens: int,
    *,
    model: str,
    tracker: CostTracker | None,
    audit_logger: AuditLogger | None = None,
    audit_model: str | None = None,
    cached_tokens: int = 0,
    provider: str | None = None,
    session_id: str | None = None,
    agent: str = "main",
    component: str | None = None,
    pricing: Any = None,
) -> float:
    """Record one LLM call in the cost tracker AND the audit log.

    Returns the PER-CALL USD cost (what ``record_llm_call`` billed; callers
    feed this into per-thread/state totals — never the running total).

    - ``tracker=None, audit_logger=None``: nothing to book, return 0.0.
    - ``audit_logger=None``: cost tracking only.
    - ``tracker=None`` (audit-only installs): the row still carries a
      meaningful per-call cost via the pure ``CostTracker.estimate_cost``
      so audit forensics survive a disabled tracker.
    - ``model`` is the PRICING/attribution key; ``audit_model`` overrides
      the model name written to the audit row (the runtime prices versioned
      provider echoes under the configured name while auditing the served
      one). Defaults to ``model``.
    - ``pricing`` (Codex PR-69 r1) is the client's authoritative endpoint
      price (``EndpointPricing``-like, USD per 1M tokens). When valid it
      replaces the tracker's pricing-table lookup for BOTH the tracker
      record and the audit row — on custom endpoints whose model names
      have no PRICING entry the table would silently bill Sonnet rates
      and diverge from the spend ledger. ``None``/invalid falls back to
      the table (with the usual one-time warning).
    - ``component`` overrides the ``details["component"]`` subsystem tag
      on the audit row. When ``None`` (the normal case) it resolves via
      ``_AGENT_COMPONENTS`` from *agent*, falling back to ``"unmapped"``
      — the key is ALWAYS present, never None, never omitted. The audit
      row's ``details`` dict is otherwise free-form; the reconciliation
      contract only reads ``details["cost_usd"]``, and audit rows from
      BEFORE this key existed are legacy rows (no key) that analysis
      tooling treats as ``"unmapped"``.
    - Audit writes must never break cost tracking; failures are logged and
      swallowed (same doctrine as telemetry's EventBus emit).
    """
    if tracker is None and audit_logger is None:
        return 0.0
    row = _endpoint_pricing_row(pricing) if pricing is not None else None
    if tracker is not None:
        cost = tracker.record_llm_call(
            input_tokens,
            output_tokens,
            model,
            cached_tokens=cached_tokens,
            provider=provider,
            session_id=session_id,
            pricing=row,
        )
    else:
        cost = CostTracker.estimate_cost(
            input_tokens, output_tokens, model, cached_tokens, pricing=row
        )
    if audit_logger is not None:
        try:
            audit_logger.log_llm_call(
                model=audit_model or model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                cached_tokens=cached_tokens,
                agent=agent,
                component=component or _AGENT_COMPONENTS.get(agent, "unmapped"),
            )
        except Exception:
            logger.warning("audit llm_call write failed", exc_info=True)
    return cost


def init_session_audit_logger(session_id: str | None):
    """Runtime-style gated :class:`AuditLogger` for *session_id*, or None.

    Same gating as ``NativeAgentGraph`` init: no session id or no audit
    capability (stripped install) → None; a construction failure degrades
    to None with a warning instead of killing the caller. Operator jobs
    and sourcehunt hunters use this so their LLM spend lands in the SAME
    ``~/.clearwing/audit/<session_id>/audit.jsonl`` as the runtime's.
    """
    if not session_id:
        return None
    from clearwing.capabilities import capabilities

    if not capabilities.has("audit"):
        return None
    from clearwing.safety.audit import AuditLogger

    try:
        return AuditLogger(session_id)
    except Exception:
        logger.warning("Failed to initialize AuditLogger", exc_info=True)
        return None
