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
from typing import TYPE_CHECKING

from clearwing.observability.telemetry import CostTracker

if TYPE_CHECKING:  # pragma: no cover
    # Duck-typed at runtime: anything exposing log_llm_call(model=...,
    # input_tokens=..., output_tokens=..., cost_usd=..., cached_tokens=...,
    # agent=...) works — keeps this module import-cycle-free from
    # clearwing.safety.audit (which lazily imports clearwing.core.config).
    from clearwing.safety.audit import AuditLogger

logger = logging.getLogger(__name__)


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
    - Audit writes must never break cost tracking; failures are logged and
      swallowed (same doctrine as telemetry's EventBus emit).
    """
    if tracker is None and audit_logger is None:
        return 0.0
    if tracker is not None:
        cost = tracker.record_llm_call(
            input_tokens,
            output_tokens,
            model,
            cached_tokens=cached_tokens,
            provider=provider,
            session_id=session_id,
        )
    else:
        cost = CostTracker.estimate_cost(input_tokens, output_tokens, model, cached_tokens)
    if audit_logger is not None:
        try:
            audit_logger.log_llm_call(
                model=audit_model or model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                cached_tokens=cached_tokens,
                agent=agent,
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
