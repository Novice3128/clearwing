"""Tests for issue #4: transport-timeout resilience and retry visibility.

- timeout_max_retries default raised 1 -> 2 with an env override entry
  (CLEARWING_LLM_TIMEOUT_RETRIES), so one flaky read-timeout no longer
  kills a ~4-minute turn with zero second chances.
- aiohttp's close-before-response wording ("Server disconnected") joins
  the transient-transport markers (chaos round P1: the only failure class
  that killed the task outright).
- The openai-http fallback no longer reroutes read-timeouts (the request
  was sent; a second transport per retry doubles billed round-trips).
- The final exception carries _clearwing_attempts so error frames can say
  how many retries were consumed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from clearwing.llm.native import AsyncLLMClient, _non_negative_int_env


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


class TestTimeoutRetryDefaults:
    def test_default_is_two(self):
        assert _client().timeout_max_retries == 2

    def test_explicit_value_still_wins(self):
        assert _client(timeout_max_retries=5).timeout_max_retries == 5

    def test_env_helper_parses_and_guards(self, monkeypatch):
        monkeypatch.setenv("CLEARWING_TEST_RETRIES", "7")
        assert _non_negative_int_env("CLEARWING_TEST_RETRIES", 2) == 7
        monkeypatch.setenv("CLEARWING_TEST_RETRIES", "junk")
        assert _non_negative_int_env("CLEARWING_TEST_RETRIES", 2) == 2
        monkeypatch.setenv("CLEARWING_TEST_RETRIES", "-1")
        assert _non_negative_int_env("CLEARWING_TEST_RETRIES", 2) == 2
        monkeypatch.delenv("CLEARWING_TEST_RETRIES")
        assert _non_negative_int_env("CLEARWING_TEST_RETRIES", 2) == 2


class TestServerDisconnectedMarker:
    def test_aiohttp_close_wording_is_transient(self):
        # Chaos round P1: accept-then-close killed the task in 17ms with
        # zero retries because no marker matched aiohttp's wording.
        assert _client()._is_transient_transport_error(RuntimeError("Server disconnected"))

    def test_server_disconnected_retries_then_raises(self):
        # Non-timeout transport class: the cap is rate_limit_max_retries.
        client = _client(rate_limit_max_retries=1)
        calls = 0

        async def always_disconnected():
            nonlocal calls
            calls += 1
            raise RuntimeError("Server disconnected")

        with patch("clearwing.llm.native.asyncio.sleep", new=_no_sleep):
            with pytest.raises(RuntimeError, match="Server disconnected"):
                asyncio.run(client._with_retries(always_disconnected))
        assert calls == 2


class TestFallbackTimeoutExclusion:
    def test_read_timeout_still_matches_the_fallback_predicate(self):
        # The predicate is a pure marker matcher; the billing guard lives
        # at the per-retry dispatch call site.
        assert _client()._should_try_openai_http_fallback(
            RuntimeError("Web stream error: read message timeout")
        )

    def test_preresponse_and_plain_transports_match(self):
        assert _client()._should_try_openai_http_fallback(
            RuntimeError("Web call failed: error sending request ... connection refused")
        )
        assert _client()._should_try_openai_http_fallback(
            RuntimeError("Web stream error: connection reset by peer")
        )

    def test_dispatch_does_not_reroute_read_timeouts_per_retry(self):
        # The achat dispatch fallback runs INSIDE _with_retries, i.e. once
        # per attempt: a sent-request timeout must not be re-dispatched
        # through a second transport on every retry.
        client = _client()
        rerouted = []

        async def _record_fallback(*_a, **_k):
            rerouted.append(1)
            raise AssertionError("should not be reached")

        class _TimingOutClient:
            async def achat(self, *_a, **_k):
                raise RuntimeError("Web call failed: read message timed out")

        with (
            patch.object(client, "_openai_chat_http_fallback", _record_fallback),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                asyncio.run(client._achat_provider_dispatch(_TimingOutClient(), None, None))
        assert rerouted == []

    def test_dispatch_does_not_reroute_send_marker_timeout_combo(self):
        # "error sending request ... timed out" pairs the generic send-phase
        # marker with a timeout: the upload may have completed, so it is
        # billable-ambiguous and must NOT reroute per retry (r2 semantics).
        client = _client()
        rerouted = []

        async def _record_fallback(*_a, **_k):
            rerouted.append(1)
            raise AssertionError("should not be reached")

        class _AmbiguousClient:
            async def achat(self, *_a, **_k):
                raise RuntimeError("Web call failed: error sending request ... timed out")

        with patch.object(client, "_openai_chat_http_fallback", _record_fallback):
            with pytest.raises(RuntimeError, match="timed out"):
                asyncio.run(client._achat_provider_dispatch(_AmbiguousClient(), None, None))
        assert rerouted == []


class TestAttemptsAttribute:
    def test_attempts_attached_after_exhausted_retries(self):
        client = _client(rate_limit_max_retries=6, timeout_max_retries=1)

        async def always_times_out():
            raise RuntimeError("request timeout")

        with patch("clearwing.llm.native.asyncio.sleep", new=_no_sleep):
            with pytest.raises(RuntimeError, match="request timeout") as exc_info:
                asyncio.run(client._with_retries(always_times_out))

        assert exc_info.value._clearwing_attempts == 1

    def test_no_attribute_on_fast_fail(self):
        client = _client()

        async def fails_fast():
            raise RuntimeError("HTTP 400: invalid request")

        with pytest.raises(RuntimeError, match="400"):
            asyncio.run(client._with_retries(fails_fast))


class TestEnvWiring:
    def test_env_var_reaches_ctor_default(self):
        # The default binds at import time, so exercise the real chain in a
        # subprocess: a typo in the env var name would silently keep 2.
        import subprocess
        import sys

        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from clearwing.llm.native import AsyncLLMClient; "
                "print(AsyncLLMClient(model_name='m', provider_name='openai', "
                "api_key='k').timeout_max_retries)",
            ],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "CLEARWING_LLM_TIMEOUT_RETRIES": "5"},
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "5"


def _nested_exc(top: str, cause: str) -> RuntimeError:
    outer = RuntimeError(top)
    outer.__cause__ = RuntimeError(cause)
    return outer


class TestExceptionChainClassification:
    """Codex PR-40 P1s: genai nests the transport detail under a terse
    top-level 'Web call failed' message, and 'Server disconnected' carries
    the same billing ambiguity as a read timeout."""

    def test_nested_timeout_is_classified(self):
        exc = _nested_exc(
            "Web call failed for model test-model",
            "Reqwest error: error sending request: operation timed out",
        )
        assert _client()._is_timeout_error(exc)

    def test_server_disconnected_shares_the_timeout_cap(self):
        client = _client(rate_limit_max_retries=6, timeout_max_retries=2)
        calls = 0

        async def always_disconnected():
            nonlocal calls
            calls += 1
            raise RuntimeError("Server disconnected")

        with patch("clearwing.llm.native.asyncio.sleep", new=_no_sleep):
            with pytest.raises(RuntimeError, match="Server disconnected") as exc_info:
                asyncio.run(client._with_retries(always_disconnected))
        # 1 attempt + timeout_max_retries(2), NOT the rate-limit budget of 6.
        assert calls == 3
        assert exc_info.value._clearwing_attempts == 2

    def test_nested_preresponse_marker_counts_as_unbilled(self):
        exc = _nested_exc(
            "Web call failed for model test-model",
            "error sending request: connection refused",
        )
        assert _client()._is_definitely_unbilled_transport_error(exc)

    def test_dispatch_guard_blocks_nested_timeout_reroute(self):
        client = _client()
        rerouted = []

        async def _record_fallback(*_a, **_k):
            rerouted.append(1)
            raise AssertionError("should not be reached")

        class _NestedTimeoutClient:
            async def achat(self, *_a, **_k):
                raise _nested_exc(
                    "Web call failed for model test-model", "read message timed out"
                )

        with patch.object(client, "_openai_chat_http_fallback", _record_fallback):
            with pytest.raises(RuntimeError, match="Web call failed"):
                asyncio.run(
                    client._achat_provider_dispatch(_NestedTimeoutClient(), None, None)
                )
        assert rerouted == []

    def test_dispatch_guard_allows_nested_preresponse_reroute(self):
        client = _client()
        rerouted = []

        async def _fallback(*_a, **_k):
            rerouted.append(1)
            raise RuntimeError("OpenAI-compatible fallback failed with HTTP 500")

        class _NestedRefusedClient:
            async def achat(self, *_a, **_k):
                raise _nested_exc(
                    "Web call failed for model test-model",
                    "error sending request: connection refused",
                )

        with patch.object(client, "_openai_chat_http_fallback", _fallback):
            with pytest.raises(RuntimeError, match="fallback failed"):
                asyncio.run(
                    client._achat_provider_dispatch(_NestedRefusedClient(), None, None)
                )
        assert rerouted == [1]


class TestRoundTwoAmbiguityGuards:
    """Codex PR-40 r2: send-phase-marker + timeout combos are billable
    ambiguous, and an enforcing spend ledger refuses ambiguous resends."""

    def test_send_marker_with_timeout_is_not_definitely_unbilled(self):
        exc = _nested_exc(
            "Web call failed for model test-model",
            "error sending request: operation timed out",
        )
        assert not _client()._is_definitely_unbilled_transport_error(exc)

    def test_pure_preresponse_stays_definitely_unbilled(self):
        exc = _nested_exc(
            "Web call failed for model test-model",
            "error sending request: connection refused",
        )
        assert _client()._is_definitely_unbilled_transport_error(exc)

    def test_enforcing_ledger_refuses_ambiguous_disconnect_retries(self):
        from types import SimpleNamespace

        client = _client(rate_limit_max_retries=6, timeout_max_retries=2)
        client._spend_ledger = SimpleNamespace(enforcing=True)
        calls = 0

        async def always_disconnected():
            nonlocal calls
            calls += 1
            raise RuntimeError("Server disconnected")

        with pytest.raises(RuntimeError, match="Server disconnected"):
            asyncio.run(client._with_retries(always_disconnected))
        assert calls == 1

    def test_non_enforcing_caller_keeps_disconnect_retries(self):
        from types import SimpleNamespace

        client = _client(rate_limit_max_retries=6, timeout_max_retries=2)
        client._spend_ledger = SimpleNamespace(enforcing=False)
        calls = 0

        async def always_disconnected():
            nonlocal calls
            calls += 1
            raise RuntimeError("Server disconnected")

        with patch("clearwing.llm.native.asyncio.sleep", new=_no_sleep):
            with pytest.raises(RuntimeError, match="Server disconnected"):
                asyncio.run(client._with_retries(always_disconnected))
        assert calls == 3
