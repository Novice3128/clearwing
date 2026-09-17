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
- Cancellation is never failover: ``asyncio.CancelledError`` /
  ``KeyboardInterrupt`` (BaseException) propagate immediately.
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
    response_text,
)


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
        """The primary's context budget (Codex PR-55 r2): the runtime picks
        its summarizer threshold from this attribute, so a chain without it
        silently fell back to the built-in default."""
        return getattr(self.primary, "context_budget_tokens", None)

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
        """Stream through the primary, then each fallback in order."""
        clients = self.clients
        last_exc: Exception | None = None
        for index, client in enumerate(clients):
            try:
                response = await client.achat_stream(**kwargs)
                self._record_served(client)
                return response
            except Exception as exc:
                # Cancellation must never fail over (BaseException is not
                # caught); only genuine provider failures move the chain.
                last_exc = exc
                if index + 1 >= len(clients):
                    raise
                self._notify(
                    f"llm provider {getattr(client, 'provider_name', '?')}/"
                    f"{getattr(client, 'model_name', '?')} failed "
                    f"({type(exc).__name__}); falling back to "
                    f"{getattr(clients[index + 1], 'provider_name', '?')}/"
                    f"{getattr(clients[index + 1], 'model_name', '?')}"
                )
                logger.warning(
                    "LLM provider %s/%s failed (%s); falling back to %s/%s",
                    getattr(client, "provider_name", "?"),
                    getattr(client, "model_name", "?"),
                    self._brief(exc),
                    getattr(clients[index + 1], "provider_name", "?"),
                    getattr(clients[index + 1], "model_name", "?"),
                )
        raise last_exc  # pragma: no cover - loop always returns or raises

    async def achat(self, **kwargs: Any):
        """Non-streaming dispatch through the chain."""
        clients = self.clients
        last_exc: Exception | None = None
        for index, client in enumerate(clients):
            try:
                response = await client.achat(**kwargs)
                self._record_served(client)
                return response
            except Exception as exc:
                last_exc = exc
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
