"""Tests for issue #17: 5xx retry classification and the provider fallback chain.

- `_is_server_error` matches STRUCTURED status attributes and status-ANCHORED
  text markers; bare digits inside response bodies never classify.
- 5xx failures are retryable under the conservative timeout cap and are
  refused under an enforcing spend ledger (billable-ambiguous, symmetric
  with read timeouts).
- `FallbackChain` switches providers after the primary's retries are
  exhausted, records who served the call, and is inert under enforcing
  ledgers or per-request credentials.
- `resolve_fallback_endpoints` reads complete endpoint blocks with the same
  credential scoping as the primary resolution.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from clearwing.llm.fallback import FallbackChain
from clearwing.llm.native import AsyncLLMClient
from clearwing.providers.env import LLMEndpoint, resolve_fallback_endpoints


def _client(**overrides) -> AsyncLLMClient:
    kwargs = dict(
        model_name="test-model",
        provider_name="openai",
        api_key="sk-test",
        base_url="https://example.test/v1",
    )
    kwargs.update(overrides)
    return AsyncLLMClient(**kwargs)


async def _no_sleep(_delay):
    return None


class _StatusError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        if status is not None:
            self.status = status


class TestServerErrorClassification:
    def test_structured_status_attribute(self):
        assert _client()._is_server_error(_StatusError("upstream failed", status=503))

    def test_structured_status_code_attribute(self):
        exc = _StatusError("nope")
        exc.status_code = 502
        assert _client()._is_server_error(exc)

    def test_string_status_attribute(self):
        exc = _StatusError("nope")
        exc.status_code = "500"
        assert _client()._is_server_error(exc)

    def test_bare_code_attribute_is_not_an_http_status(self):
        # Application-level `code` attributes (gRPC codes, library
        # internals) are not HTTP statuses — a 500-599 value there must
        # not classify as a retryable server error.
        exc = _StatusError("app-level failure")
        exc.code = "502"
        assert not _client()._is_server_error(exc)

    def test_genai_pyo3_quoted_status_wording(self):
        # genai-pyo3's status template quotes the digits.
        assert _client()._is_server_error(
            RuntimeError("Request failed with status code '502'")
        )

    def test_anthropic_overload_wording(self):
        client = _client()
        assert client._is_server_error(RuntimeError("Overloaded"))
        assert client._is_server_error(
            RuntimeError("Error code: 529 - {'type': 'error', 'error': "
                         "{'type': 'overloaded_error'}}")
        )

    def test_status_in_nested_cause(self):
        cause = _StatusError("gateway exploded", status=504)
        wrapped = RuntimeError("Web call failed")
        wrapped.__cause__ = cause
        assert _client()._is_server_error(wrapped)

    def test_anchored_text_markers(self):
        client = _client()
        assert client._is_server_error(RuntimeError("status code 502 from upstream"))
        assert client._is_server_error(RuntimeError("HTTP 500"))
        assert client._is_server_error(RuntimeError("error: 503 service issue"))

    def test_known_5xx_phrases(self):
        client = _client()
        assert client._is_server_error(RuntimeError("502 Bad Gateway"))
        assert client._is_server_error(RuntimeError("service unavailable"))
        assert client._is_server_error(RuntimeError("internal server error"))

    def test_bare_digits_in_body_never_classify(self):
        client = _client()
        assert not client._is_server_error(RuntimeError("see ticket 502 for details"))
        assert not client._is_server_error(RuntimeError("model gpt-500x rejected"))
        assert not client._is_server_error(RuntimeError("502 bytes received"))

    def test_non_5xx_statuses_do_not_classify(self):
        client = _client()
        assert not client._is_server_error(_StatusError("bad request", status=400))
        assert not client._is_server_error(RuntimeError("status code 404"))
        assert not client._is_server_error(RuntimeError("HTTP 200 OK"))


def _enforcing_ledger(tmp_path):
    from clearwing.llm.budget import SpendLedger

    return SpendLedger(
        limit_usd=10.0,
        session_id="fallback-test",
        repo_url="/tmp/repo",
        output_dir=tmp_path,
        input_price_per_million=0.0,
        output_price_per_million=1.0,
    )


class TestServerErrorRetries:
    def test_5xx_retries_then_succeeds(self):
        client = _client(timeout_max_retries=2)
        calls = 0

        async def flaky():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise _StatusError("upstream exploded", status=503)
            return "ok"

        with patch("clearwing.llm.native.asyncio.sleep", new=_no_sleep):
            assert asyncio.run(client._with_retries(flaky)) == "ok"
        assert calls == 3

    def test_5xx_shares_the_conservative_timeout_cap(self):
        client = _client(rate_limit_max_retries=6, timeout_max_retries=1)
        calls = 0

        async def always_503():
            nonlocal calls
            calls += 1
            raise _StatusError("upstream exploded", status=503)

        with patch("clearwing.llm.native.asyncio.sleep", new=_no_sleep):
            with pytest.raises(_StatusError) as exc_info:
                asyncio.run(client._with_retries(always_503))
        assert calls == 2  # initial + 1 retry, NOT the rate-limit budget of 6
        assert exc_info.value._clearwing_attempts == 1

    def test_enforcing_ledger_refuses_5xx_retry(self, tmp_path):
        ledger = _enforcing_ledger(tmp_path)
        client = _client().with_spend_ledger(ledger, stage="test")
        calls = 0

        async def always_503():
            nonlocal calls
            calls += 1
            raise _StatusError("upstream exploded", status=503)

        with patch("clearwing.llm.native.asyncio.sleep", new=_no_sleep):
            with pytest.raises(_StatusError):
                asyncio.run(client._with_retries(always_503))
        assert calls == 1

    def test_retry_notice_emitted(self):
        client = _client(timeout_max_retries=2)
        notices: list[str] = []
        client.set_retry_notice(notices.append)
        calls = 0

        async def flaky():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise _StatusError("upstream exploded", status=503)
            return "ok"

        with patch("clearwing.llm.native.asyncio.sleep", new=_no_sleep):
            asyncio.run(client._with_retries(flaky))
        assert len(notices) == 1
        assert "server error (5xx)" in notices[0]
        assert "retry 1/2" in notices[0]


class _FakeClient:
    """Duck-typed AsyncLLMClient for chain tests."""

    def __init__(self, model_name, provider_name="openai", exc=None, response=None):
        self.model_name = model_name
        self.provider_name = provider_name
        self._exc = exc
        self._response = response
        self.calls = 0
        self.spend_ledger = None

    async def achat_stream(self, **kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._response

    async def achat(self, **kwargs):
        return await self.achat_stream(**kwargs)


class TestFallbackChain:
    def test_primary_failure_falls_over(self):
        primary = _FakeClient("primary-model", exc=RuntimeError("status code 503"))
        backup = _FakeClient("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])
        notices: list[str] = []
        chain.set_retry_notice(notices.append)

        result = asyncio.run(chain.achat_stream(messages=[]))
        assert result == "saved"
        assert chain.served_model_name == "backup-model"
        assert chain.served_provider_name == "openai"
        assert primary.calls == 1 and backup.calls == 1
        assert any("falling back" in n for n in notices)

    def test_primary_success_never_touches_fallback(self):
        primary = _FakeClient("primary-model", response="fine")
        backup = _FakeClient("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        result = asyncio.run(chain.achat_stream(messages=[]))
        assert result == "fine"
        assert backup.calls == 0
        assert chain.served_model_name == "primary-model"

    def test_all_members_failing_raises_last_error(self):
        primary = _FakeClient("primary-model", exc=RuntimeError("primary down"))
        backup = _FakeClient("backup-model", exc=RuntimeError("backup down"))
        chain = FallbackChain(primary, [backup])

        with pytest.raises(RuntimeError, match="backup down"):
            asyncio.run(chain.achat_stream(messages=[]))

    def test_cancellation_never_fails_over(self):
        primary = _FakeClient("primary-model", exc=asyncio.CancelledError())
        backup = _FakeClient("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(chain.achat_stream(messages=[]))
        assert backup.calls == 0

    def test_chain_surface_matches_primary_labels(self):
        primary = _FakeClient("primary-model", "anthropic")
        chain = FallbackChain(primary, [])
        assert chain.model_name == "primary-model"
        assert chain.provider_name == "anthropic"
        assert chain.clients[0] is primary

    def test_enforcing_fallback_client_is_dropped(self, tmp_path):
        primary = _FakeClient("primary-model", response="fine")
        enforcing_client = _FakeClient("enforcing-model", response="x")
        enforcing_client.spend_ledger = _enforcing_ledger(tmp_path)
        plain_backup = _FakeClient("backup-model", response="saved")
        chain = FallbackChain(primary, [enforcing_client, plain_backup])
        assert chain.fallbacks == [plain_backup]

    def test_enforcing_primary_disables_the_chain(self, tmp_path):
        primary = _FakeClient("primary-model")
        primary.spend_ledger = _enforcing_ledger(tmp_path)
        backup = _FakeClient("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])
        assert chain.fallbacks == []


class TestResolveFallbackEndpoints:
    def _config(self, fallbacks):
        return {
            "base_url": "https://primary.test/v1",
            "model": "primary-model",
            "fallbacks": fallbacks,
        }

    def test_keyless_entries_do_not_inherit_the_primary_env_key(self, monkeypatch):
        """CLEARWING_API_KEY authenticates the PRIMARY endpoint's host; a
        fallback entry that omitted its own key must never receive it
        (credential crossing, same class PR #43 banned)."""
        monkeypatch.setenv("CLEARWING_API_KEY", "sk-primary-secret")
        endpoints = resolve_fallback_endpoints(
            self._config([{"base_url": "https://relay.example/v1"}])
        )
        assert len(endpoints) == 1
        assert endpoints[0].api_key != "sk-primary-secret"

    def test_full_entries_resolve(self, monkeypatch):
        monkeypatch.setenv("BACKUP_KEY", "sk-backup")
        endpoints = resolve_fallback_endpoints(
            self._config(
                [
                    {
                        "base_url": "https://backup.test/v1",
                        "model": "backup-model",
                        "api_key": "${BACKUP_KEY}",
                    },
                    {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:32b"},
                ]
            )
        )
        assert len(endpoints) == 2
        assert endpoints[0].base_url == "https://backup.test/v1"
        assert endpoints[0].api_key == "sk-backup"  # ${ENV} expansion
        assert endpoints[1].provider == "openai_compat"

    def test_each_entry_keeps_its_own_credential_scoping(self, monkeypatch):
        monkeypatch.setenv("BACKUP_KEY", "sk-backup")
        endpoints = resolve_fallback_endpoints(
            self._config(
                [{"base_url": "https://backup.test/v1", "api_key": "${BACKUP_KEY}"}]
            )
        )
        # The fallback credential belongs to the fallback base_url only —
        # it never rides along to the primary (whose api_key stays unset
        # here because the primary block is not part of this resolution).
        assert endpoints[0].api_key == "sk-backup"
        assert endpoints[0].base_url == "https://backup.test/v1"

    def test_anthropic_compat_fallback_routes_to_anthropic(self):
        endpoints = resolve_fallback_endpoints(
            self._config([{"base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-6"}])
        )
        assert endpoints[0].provider == "anthropic"

    def test_no_fallbacks_section_is_empty(self):
        assert resolve_fallback_endpoints({"base_url": "https://primary.test/v1"}) == []
        assert resolve_fallback_endpoints({}) == []

    def test_malformed_sections_are_ignored_not_raised(self):
        assert resolve_fallback_endpoints({"fallbacks": "nope"}) == []
        assert resolve_fallback_endpoints({"fallbacks": [42, {"model": None}]}) == []


class TestChainWiringGate:
    def test_per_request_credentials_disable_the_chain(self, monkeypatch):
        from clearwing.agent import graph as graph_module

        monkeypatch.setattr(
            graph_module,
            "resolve_fallback_endpoints",
            lambda config_provider=None: [
                LLMEndpoint(provider="openai_compat", model="backup", base_url="https://b.test")
            ],
        )
        primary = _FakeClient("primary-model")
        result = graph_module._maybe_wrap_fallback_chain(
            primary, cli_base_url="https://explicit.test/v1", cli_api_key=None
        )
        assert result is primary

        result = graph_module._maybe_wrap_fallback_chain(
            primary, cli_base_url=None, cli_api_key="sk-explicit"
        )
        assert result is primary

    def test_env_tier_deployment_gets_the_chain(self, monkeypatch):
        from clearwing.agent import graph as graph_module

        backup = _FakeClient("backup-model")
        captured: dict = {}

        class _FakeManager:
            @staticmethod
            def for_endpoint(endpoint):
                captured["endpoint"] = endpoint
                return type("M", (), {"get_native_client": staticmethod(lambda task: backup)})()

        monkeypatch.setattr(graph_module, "ProviderManager", _FakeManager)
        monkeypatch.setattr(
            graph_module,
            "resolve_fallback_endpoints",
            lambda config_provider=None: [
                LLMEndpoint(
                    provider="openai_compat",
                    model="backup-model",
                    base_url="https://backup.test/v1",
                ),
            ],
        )
        primary = _FakeClient("primary-model")
        chain = graph_module._maybe_wrap_fallback_chain(
            primary, cli_base_url=None, cli_api_key=None
        )
        assert isinstance(chain, FallbackChain)
        assert chain.primary is primary
        assert chain.fallbacks == [backup]
        assert captured["endpoint"].base_url == "https://backup.test/v1"

    def test_no_configured_fallbacks_returns_primary(self, monkeypatch):
        from clearwing.agent import graph as graph_module

        monkeypatch.setattr(graph_module, "resolve_fallback_endpoints", lambda config_provider=None: [])
        primary = _FakeClient("primary-model")
        assert (
            graph_module._maybe_wrap_fallback_chain(
                primary, cli_base_url=None, cli_api_key=None
            )
            is primary
        )
