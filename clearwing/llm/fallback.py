"""Ordered provider fallback chain (issue #17).

A single LLM endpoint failing hard — retries exhausted, a persistent 5xx,
a rejected credential — used to kill the whole task (the 1,083s zombie
hang). The chain wraps a primary :class:`AsyncLLMClient` with an ordered
list of fallback clients resolved from ``config.yaml``:

.. code-block:: yaml

    provider:
      base_url: https://primary.example.com/v1
      model: glm-5.3
      api_key: ${PRIMARY_KEY}
      fallbacks:
        - base_url: https://backup.example.com/v1
          model: gpt-5.4-mini
          api_key: ${BACKUP_KEY}

Semantics:

- Each client keeps its OWN retry policy (rate-limit backoff, timeout
  caps, per-attempt spend reservations). The chain only intervenes after
  a client's retries are exhausted or it failed non-retryably.
- Cooldown/stickiness (issue #57): a member whose dispatch failed at chain
  level is skipped for 60s × 2^(consecutive-cooldowns − 1) (max 600s; a
  parsed Retry-After extends the window, capped at 300s). Cooldown expiry
  restores eligibility (half-open); a success resets the streak; when
  every member is cooling the chain fails open to the normal order
  instead of erroring with no candidates.
- Cancellation is never failover: ``asyncio.CancelledError`` /
  ``KeyboardInterrupt`` (BaseException) propagate immediately (and never
  start a cooldown).
- Spend-ledger safety: a chain is inert while any member client runs an
  ENFORCING spend ledger — a fallback dispatch would bill a reservation
  the ledger never sees (its pricing was validated for the primary
  model), so the fallbacks are dropped at construction.
- Cost attribution keeps working because every serving client's response
  echoes ``provider_model_name``; the chain additionally records which
  member served the last call (``served_model_name`` /
  ``served_provider_name``) for callers that attribute by client.

Distinct from the ``model_roles`` binding fallback (docs/model-roles.md):
that one is SELECTION-time — it picks the first provider whose API key
is available before a run starts. This chain is RUNTIME failover — it
switches after a serving provider fails mid-session.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from clearwing.llm.native import AsyncLLMClient

logger = logging.getLogger(__name__)

# Imported lazily-by-name at module load: fallback.py lives beside native.py
# and the client surface it mirrors (aask_text/aask_json post-processing)
# comes from the same module.
from genai_pyo3 import ChatMessage  # noqa: E402

from clearwing.llm.native import (  # noqa: E402
    _is_root_model_type,
    _validate_schema_response,
    extract_json_array,
    extract_json_object,
    parse_retry_after_seconds,
    response_text,
)

# --- Member cooldown (issue #57) ---------------------------------------------
#
# Design calibrated against known-good agent stacks: openai/codex keeps a
# session-scoped failure flag so a failed provider stops being re-tried
# first on every request (codex-rs/core/src/responses_retry.rs), and
# LiteLLM cools a failing deployment over DEFAULT_COOLDOWN_TIME_SECONDS
# (= 60s) with exponential growth (litellm/router_utils/cooldown_handler).
# Without it, a persistently-5xx primary re-charges its FULL retry budget
# on every turn before the chain switches — exactly the stall the chain
# exists to prevent.

_COOLDOWN_BASE_SECONDS: float = 60.0
_COOLDOWN_MAX_SECONDS: float = 600.0
# A Retry-After hint parsed from the failing member's exception may extend
# the window beyond the exponential schedule, but the shared parser caps
# parsed hints at 300s (native._RETRY_AFTER_MAX_SECONDS).
_COOLDOWN_RETRY_AFTER_MAX_SECONDS: float = 300.0

# Marker attribute distinguishing a CONSUMER-side exception (the caller's
# on_text_delta raising through the chain's counting wrapper mid-stream)
# from a provider failure: the wrapper re-raises it tagged, and the
# dispatch handler re-raises it verbatim without cooling the member or
# failing over (Codex PR-63 r1). Attribute tag, not a wrapper class, so
# the caller still receives its own exception type/instance.
_CONSUMER_ERROR_ATTR = "_clearwing_consumer_error"


def _mark_consumer_error(exc: BaseException) -> None:
    try:
        setattr(exc, _CONSUMER_ERROR_ATTR, True)
    except Exception:  # pragma: no cover - exotic exception types
        pass


def _is_consumer_error(exc: BaseException) -> bool:
    return getattr(exc, _CONSUMER_ERROR_ATTR, False) is True


class FallbackChain:
    """Duck-typed stand-in for :class:`AsyncLLMClient` (issue #17).

    Exposes the surface the agent runtime consumes — ``model_name``,
    ``provider_name``, ``achat``, ``achat_stream`` — so the runtime needs
    no branching: a session without configured fallbacks gets a bare
    client, a session with them gets this chain.
    """

    def __init__(self, primary: AsyncLLMClient, fallbacks: Sequence[AsyncLLMClient] = ()):
        self.primary = primary
        usable: list[AsyncLLMClient] = []
        for client in fallbacks:
            ledger = getattr(client, "spend_ledger", None)
            if ledger is not None and ledger.enforcing:
                logger.warning(
                    "Dropping fallback client %s/%s: an enforcing spend ledger "
                    "cannot account cross-provider dispatches",
                    getattr(client, "provider_name", "?"),
                    getattr(client, "model_name", "?"),
                )
                continue
            usable.append(client)
        primary_ledger = getattr(primary, "spend_ledger", None)
        if primary_ledger is not None and primary_ledger.enforcing and usable:
            logger.warning(
                "Fallback chain disabled: the primary client runs an enforcing "
                "spend ledger and fallback dispatches would bill unaccounted"
            )
            usable = []
        self.fallbacks = usable
        # Populated after each call by _record_served(); read by callers
        # that attribute cost/provider by client instead of by response.
        self.served_model_name: str | None = None
        self.served_provider_name: str | None = None
        # True when the PRIMARY member served the last call. Provider-name
        # comparison cannot detect failover when both members share an
        # adapter label (e.g. two openai_compat endpoints) — this flag can.
        self.served_by_primary: bool = True
        self._retry_notice: Callable[[str], None] | None = None
        # Per-member cooldown state (issue #57), keyed by id(client): the
        # chain holds strong references to every member for its whole
        # lifetime, so ids cannot be recycled while the state is live.
        # ``_cooldown_until`` is a time.monotonic() deadline; the streak
        # counts CONSECUTIVE cooldowns and only a success resets it.
        self._cooldown_until: dict[int, float] = {}
        self._cooldown_streak: dict[int, int] = {}
        # Fail-open notices fire once per all-cooling EPISODE (review r1):
        # every dispatch while all members cool would otherwise repeat
        # the same announcement; reset by any dispatch that has an
        # eligible member again.
        self._fail_open_notified = False

    # -- AsyncLLMClient-compatible surface --------------------------------

    @property
    def model_name(self) -> str:
        return self.primary.model_name

    @property
    def provider_name(self) -> str:
        return self.primary.provider_name

    @property
    def clients(self) -> list[AsyncLLMClient]:
        return [self.primary, *self.fallbacks]

    @property
    def context_budget_tokens(self) -> int | None:
        """The SMALLEST context budget among chain members (Codex PR-55 r4).

        The runtime picks its summarizer threshold from this attribute: a
        primary with a large window would let history grow past a fallback's
        smaller one, so the fallback would reject both the summary and the
        next assistant request as oversized — failover defeating itself
        exactly in long sessions.
        """
        budgets = [
            budget
            for budget in (
                getattr(client, "context_budget_tokens", None)
                for client in self.clients
            )
            if budget
        ]
        return min(budgets) if budgets else None

    @property
    def pricing(self):
        """The primary's endpoint pricing, if any."""
        return getattr(self.primary, "pricing", None)

    def set_retry_notice(self, callback: Callable[[str], None] | None) -> None:
        """Route per-client retry notices (and chain switch notices) to *callback*."""
        self._retry_notice = callback
        for client in self.clients:
            setter = getattr(client, "set_retry_notice", None)
            if setter is not None:
                setter(callback)

    def _notify(self, text: str) -> None:
        if self._retry_notice is None:
            return
        try:
            self._retry_notice(text)
        except Exception:
            logger.debug("retry-notice callback failed", exc_info=True)

    def _record_served(self, client: AsyncLLMClient) -> None:
        self.served_model_name = getattr(client, "model_name", None)
        self.served_provider_name = getattr(client, "provider_name", None)
        self.served_by_primary = client is self.primary
        # A success proves the member healthy again — its cooldown streak
        # resets to zero (issue #57 half-open recovery).
        self._clear_cooldown(client)

    # -- Cooldown bookkeeping (issue #57) ---------------------------------

    def _cooldown_remaining(self, client: AsyncLLMClient) -> float:
        """Seconds left in *client*'s cooldown (0.0 when eligible).

        Expiry alone restores eligibility (half-open): the next request
        retries the member naturally, and only a success clears the streak
        that sizes the next window.
        """
        until = self._cooldown_until.get(id(client))
        if until is None:
            return 0.0
        remaining = until - time.monotonic()
        return remaining if remaining > 0.0 else 0.0

    def _start_cooldown(
        self, client: AsyncLLMClient, exc: Exception, *, extend_only: bool = False
    ) -> None:
        """Cool *client* after a chain-level failure (its dispatch raised).

        Window = 60s × 2^(streak-1), capped at 600s. A Retry-After hint
        parsed from the member's exception (shared parser, capped at 300s)
        EXTENDS the window when larger — the provider knows its own load.
        Consecutive cooldowns accumulate: recovering, failing again doubles
        the window; only a success resets the streak.

        ``extend_only`` (fail-open tours): the dispatch that failed started
        while EVERY member was already cooling, so the failure adds no new
        information — refresh the window from the EXISTING streak (base
        60s when the member has none) WITHOUT incrementing, so a total
        outage cannot grow every window per call toward the 600s cap and
        mislead the logs about how long the provider has been down.
        """
        if len(self.clients) < 2:
            # Single-member chains (spend-ledger collapse, dropped
            # fallbacks) have nothing to skip: the chain does not act.
            return
        streak = self._cooldown_streak.get(id(client), 0)
        if not extend_only:
            streak += 1
            self._cooldown_streak[id(client)] = streak
        if streak >= 1:
            # Clamp the exponent before exponentiating (Codex PR-63 r1):
            # a provider down for weeks of capped half-open probes keeps
            # growing its streak, and 60.0 * 2**1024 overflows the float
            # BEFORE min() can apply the 600s cap — turning a provider
            # failure into an OverflowError. 2**10 × 60s already exceeds
            # the cap, so anything past ten doublings is indistinguishable.
            exponent = min(streak - 1, 10)
            window = min(_COOLDOWN_BASE_SECONDS * (2**exponent), _COOLDOWN_MAX_SECONDS)
        else:
            # extend_only with no prior streak: defensive — a cooling
            # member always has one — fall back to the base window.
            window = _COOLDOWN_BASE_SECONDS
        retry_after = parse_retry_after_seconds(
            exc, max_seconds=_COOLDOWN_RETRY_AFTER_MAX_SECONDS
        )
        if retry_after is not None:
            window = max(window, retry_after)
        self._cooldown_until[id(client)] = time.monotonic() + window
        logger.info(
            "LLM provider %s/%s entering cooldown for %.0fs (consecutive cooldowns: %d)",
            getattr(client, "provider_name", "?"),
            getattr(client, "model_name", "?"),
            window,
            streak,
        )

    def _clear_cooldown(self, client: AsyncLLMClient) -> None:
        self._cooldown_until.pop(id(client), None)
        self._cooldown_streak.pop(id(client), None)

    def _dispatch_order(
        self,
    ) -> tuple[list[AsyncLLMClient], list[AsyncLLMClient], bool, dict[int, float]]:
        """Members to dispatch over, in order, plus the cooling ones skipped.

        Returns ``(order, skipped, fail_open, remaining)`` where *remaining*
        maps ``id(client)`` → cooldown seconds left AT DECISION TIME.
        Cooling members are skipped so a persistently-failing primary stops
        re-charging its retry budget on every call — unless EVERY member is
        cooling, in which case the chain fails open to the normal order
        (never an empty-candidates error).

        The remaining seconds are read in a SINGLE pass (review r1): two
        separate comprehensions call ``_cooldown_remaining`` twice per
        member, and time advances between them — a member whose cooldown
        expires exactly between the scans lands in NEITHER list, and the
        skip notices would report a different count than the split acted
        on. Callers (``_notify_skips``) must reuse this map.
        """
        clients = self.clients
        remaining = {id(client): self._cooldown_remaining(client) for client in clients}
        order = [client for client in clients if remaining[id(client)] <= 0.0]
        skipped = [client for client in clients if remaining[id(client)] > 0.0]
        if order:
            # A dispatch with at least one eligible member ends the
            # all-cooling episode — a later one must announce itself again.
            self._fail_open_notified = False
            return order, skipped, False, remaining
        if not self._fail_open_notified:
            longest = max(remaining.values(), default=0.0)
            self._notify(
                f"all llm fallback members cooling (longest {longest:.0f}s remaining); "
                "failing open to the normal dispatch order"
            )
            self._fail_open_notified = True
        return clients, [], True, remaining

    def _notify_skips(
        self,
        skipped: Sequence[AsyncLLMClient],
        first: AsyncLLMClient,
        remaining: dict[int, float],
    ) -> None:
        if first is None:  # pragma: no cover - order always has the primary
            return
        for client in skipped:
            self._notify(
                f"llm provider {getattr(client, 'provider_name', '?')}/"
                f"{getattr(client, 'model_name', '?')} skipping "
                f"(cooldown {remaining[id(client)]:.0f}s remaining), "
                f"dispatching to {getattr(first, 'provider_name', '?')}/"
                f"{getattr(first, 'model_name', '?')}"
            )

    @property
    def spend_ledger(self):
        """The primary client's run-scoped ledger, if one is bound."""
        return getattr(self.primary, "spend_ledger", None)

    def with_spend_ledger(self, ledger, *, stage: str) -> FallbackChain:
        """Bind a spend ledger — and collapse to the primary client alone.

        Budgeted runs keep resilience through the primary's own retries
        only: a cross-provider dispatch would bill reservations the ledger
        cannot price (its models were validated per member), so the
        fallbacks are dropped whenever a ledger binds, enforcing or not.
        """
        bound_primary = self.primary.with_spend_ledger(ledger, stage=stage)
        return FallbackChain(bound_primary, [])

    async def achat_stream(self, **kwargs: Any):
        """Stream through the primary, then each fallback in order.

        Delta policy on failover (Codex PR-55 r4): deltas cannot be
        retracted once a live consumer printed them, and buffering every
        stream would destroy incremental display for the common (healthy)
        case. So the first member streams live; if it dies mid-stream, the
        user gets an explicit restart notice and later members' deltas are
        SUPPRESSED — their text still arrives authoritatively via the
        returned response (and therefore graph state) instead of being
        concatenated onto the abandoned partial output.
        """
        # Cooldown-aware dispatch order (issue #57): cooling members are
        # skipped (with a notice); all-cooling fails open to normal order.
        clients, skipped, fail_open, remaining = self._dispatch_order()
        if skipped:
            self._notify_skips(skipped, clients[0], remaining)
        last_exc: Exception | None = None
        deltas_emitted = False
        original_callback = kwargs.get("on_text_delta")

        def _counting_callback(text: str) -> None:
            nonlocal deltas_emitted
            deltas_emitted = True
            if original_callback is not None:
                try:
                    original_callback(text)
                except Exception as exc:
                    # Consumer-side failure: abort the member's stream by
                    # re-raising, but TAGGED so the dispatch handler below
                    # returns it to the caller instead of mistaking it for
                    # a provider failure (the member did not fail).
                    _mark_consumer_error(exc)
                    raise

        if original_callback is not None:
            kwargs["on_text_delta"] = _counting_callback

        for index, client in enumerate(clients):
            suppressed_this_member = index > 0 and deltas_emitted
            if suppressed_this_member:
                # Abandoned partial output already reached the consumer —
                # never interleave a second answer into it.
                kwargs["on_text_delta"] = None
            # ONLY a provider failure counts (Codex PR-63 r1): consumer-side
            # exceptions — the caller's on_text_delta raising through the
            # counting callback (tagged there), or the suppressed-response
            # re-emission below (outside this try) — must neither cool the
            # member nor move the chain to the next one: the member did not
            # fail, and re-dispatching would double-serve a call whose
            # partial output the consumer already saw.
            try:
                response = await client.achat_stream(**kwargs)
            except Exception as exc:
                # Consumer-side: propagate verbatim — no cooldown, no
                # failover (the marker is set by _counting_callback).
                if _is_consumer_error(exc):
                    raise
                # Cancellation must never fail over (BaseException is not
                # caught); only genuine provider failures move the chain.
                last_exc = exc
                # The member exhausted its OWN retry policy and failed at
                # chain level — cool it (issue #57) so the next call skips
                # it instead of re-paying that budget. This includes the
                # final member whose failure re-raises: otherwise a total
                # outage could never reach the all-cooling state that the
                # fail-open path below resolves. Cancellation never reaches
                # here (BaseException). In a fail-open tour (every member
                # was already cooling) the cooldown is extend-only: the
                # streak does not grow per call.
                self._start_cooldown(client, exc, extend_only=fail_open)
                if index + 1 >= len(clients):
                    raise
                notice = (
                    f"llm provider {getattr(client, 'provider_name', '?')}/"
                    f"{getattr(client, 'model_name', '?')} failed "
                    f"({type(exc).__name__}); falling back to "
                    f"{getattr(clients[index + 1], 'provider_name', '?')}/"
                    f"{getattr(clients[index + 1], 'model_name', '?')}"
                )
                if deltas_emitted:
                    notice += " — partial output above is abandoned; the full answer follows"
                self._notify(notice)
                logger.warning(
                    "LLM provider %s/%s failed (%s); falling back to %s/%s",
                    getattr(client, "provider_name", "?"),
                    getattr(client, "model_name", "?"),
                    self._brief(exc),
                    getattr(clients[index + 1], "provider_name", "?"),
                    getattr(clients[index + 1], "model_name", "?"),
                )
                continue
            # Success path — OUTSIDE the provider-failure handler (Codex
            # PR-63 r1): a raising consumer callback propagates to the
            # caller instead of cooling the member that served the call.
            self._record_served(client)
            if suppressed_this_member and original_callback is not None:
                # The legacy interactive CLI prints ONLY what the delta
                # callback delivered (it discards the returned events),
                # so suppressed failover text would leave the user with
                # nothing but the abandoned fragment — emit the complete
                # response once (Codex PR-55 r5). Event/state consumers
                # keep the authoritative response object either way.
                try:
                    text = response_text(response)
                except Exception:
                    text = ""
                if text:
                    original_callback(text)
            return response
        raise last_exc  # pragma: no cover - loop always returns or raises

    async def achat(self, **kwargs: Any):
        """Non-streaming dispatch through the chain."""
        # Cooldown-aware dispatch order (issue #57), same as achat_stream.
        clients, skipped, fail_open, remaining = self._dispatch_order()
        if skipped:
            self._notify_skips(skipped, clients[0], remaining)
        last_exc: Exception | None = None
        for index, client in enumerate(clients):
            try:
                response = await client.achat(**kwargs)
                self._record_served(client)
                return response
            except Exception as exc:
                last_exc = exc
                # Chain-level failure → cooldown (issue #57), including the
                # final re-raise (mirrors achat_stream's rule). Fail-open
                # tours extend the window without growing the streak.
                self._start_cooldown(client, exc, extend_only=fail_open)
                if index + 1 >= len(clients):
                    raise
                self._notify(
                    f"llm provider {getattr(client, 'provider_name', '?')}/"
                    f"{getattr(client, 'model_name', '?')} failed "
                    f"({type(exc).__name__}); falling back to "
                    f"{getattr(clients[index + 1], 'provider_name', '?')}/"
                    f"{getattr(clients[index + 1], 'model_name', '?')}"
                )
        raise last_exc  # pragma: no cover

    async def aask_text(self, **kwargs: Any):
        """Non-streaming single-prompt call, with failover (Codex PR-55 r2).

        OperatorAgent supervision and context summarization both call this
        surface — without it a configured chain broke them outright.
        """
        return await self.achat(
            messages=[ChatMessage("user", kwargs.pop("user", ""))],
            **kwargs,
        )

    async def aask_json(self, **kwargs: Any):
        """Single-prompt JSON call, with failover; mirrors the client's
        post-processing so callers get identical parse semantics."""
        expect = kwargs.pop("expect", "object")
        schema_model = kwargs.pop("schema_model", None)
        schema_name = kwargs.pop("schema_name", None)
        schema_description = kwargs.pop("schema_description", None)
        response = await self.achat(
            messages=[ChatMessage("user", kwargs.pop("user", ""))],
            response_schema=schema_model,
            response_schema_name=schema_name,
            response_schema_description=schema_description,
            **kwargs,
        )
        text = response_text(response)
        if schema_model is not None:
            parsed_model = _validate_schema_response(schema_model, text)
            if _is_root_model_type(schema_model):
                return parsed_model.root, response
            return parsed_model.model_dump(), response
        if expect == "array":
            return extract_json_array(text), response
        return extract_json_object(text), response

    @staticmethod
    def _brief(exc: BaseException) -> str:
        # Provider error text can embed credentialed URLs or key fragments
        # echoed by relays; sanitize before it lands in shared logs.
        try:
            from clearwing.reporting.safety import redact_text
        except Exception:
            redact_text = None  # type: ignore[assignment]
        text = " ".join(str(exc).split())
        if redact_text is not None:
            try:
                text = redact_text(text)
            except Exception:
                pass
        return text[:160] or type(exc).__name__
