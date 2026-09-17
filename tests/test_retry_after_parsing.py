"""Issue #59: Retry-After parsing for the LLM retry path.

- ``parse_retry_after_seconds`` walks the FULL __cause__/__context__ chain
  (a bare ``str(exc)`` misses genai-pyo3's nested transport detail and the
  aiohttp fallback's embedded header hint).
- Precedence mirrors opencode's retry.ts: ``Retry-After-Ms`` → numeric
  ``Retry-After`` → HTTP-date; legacy body phrasings keep working with an
  explicit unit so ms values are never read as seconds (the retry-after-ms
  unit bug: 500 ms must be 0.5 s, never 500 s).
- Every parsed value is capped at 300 s.
- The aiohttp transport fallback embeds the header hint into its raised
  RuntimeError text so the retry backoff can honor it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
from genai_pyo3 import ChatMessage, ChatOptions, ChatRequest

from clearwing.llm.native import (
    AsyncLLMClient,
    _retry_after_hint_from_headers,
    _retry_after_suffix,
    parse_retry_after_seconds,
)


def _client(**overrides) -> AsyncLLMClient:
    kwargs = dict(
        model_name="test-model",
        provider_name="openai",
        api_key="sk-test",
        base_url="https://example.test/v1",
    )
    kwargs.update(overrides)
    return AsyncLLMClient(**kwargs)


class TestHeaderTextParsing:
    def test_seconds_header(self):
        assert parse_retry_after_seconds("HTTP 503: slowed down (retry-after: 30)") == 30.0

    def test_ms_header_takes_precedence_over_seconds(self):
        # opencode retry.ts order: milliseconds first.
        assert parse_retry_after_seconds("retry-after-ms: 1500; retry-after: 30") == 1.5

    def test_ms_unit_suffix(self):
        assert parse_retry_after_seconds("retry-after: 500 ms") == 0.5

    def test_retry_after_ms_unit_bug_regression(self):
        # The ms-suffixed header must NEVER be read as seconds.
        assert parse_retry_after_seconds("Retry-After-Ms: 500") == 0.5
        assert parse_retry_after_seconds("retry-after-ms=800") == 0.8

    def test_header_separators(self):
        assert parse_retry_after_seconds("Retry-After:12") == 12.0
        assert parse_retry_after_seconds("retry_after: 7") == 7.0

    def test_cap_300s(self):
        assert parse_retry_after_seconds("retry-after: 9999") == 300.0
        assert parse_retry_after_seconds("retry-after-ms: 999999") == 300.0

    def test_no_hint_is_none(self):
        assert parse_retry_after_seconds("upstream exploded with status 503") is None
        assert parse_retry_after_seconds("") is None

    def test_legacy_body_phrasings(self):
        assert parse_retry_after_seconds("try again in 32s") == 32.0
        assert parse_retry_after_seconds("please wait 5s") == 5.0
        # ms-aware legacy forms (previously unparseable / unit-ambiguous).
        assert parse_retry_after_seconds("try again in 1200ms") == 1.2
        assert parse_retry_after_seconds("wait 500ms") == 0.5


class TestChainParsing:
    def test_hint_in_nested_cause(self):
        # genai-pyo3 shape: a terse wrapper whose real detail is nested.
        cause = RuntimeError("HTTP 503 from relay: retry-after: 12")
        wrapped = RuntimeError("Web call failed for model test-model")
        wrapped.__cause__ = cause
        assert parse_retry_after_seconds(wrapped) == 12.0

    def test_hint_in_context_chain(self):
        # __context__ wiring (implicit chaining): raising a new error
        # while handling the rate-limit error chains them together.
        try:
            try:
                raise RuntimeError("rate limited; retry-after-ms: 2500")
            except RuntimeError:
                raise RuntimeError("Web call failed") from None
        except RuntimeError as exc:
            # ``from None`` clears __cause__ but __context__ still chains.
            assert parse_retry_after_seconds(exc) == 2.5

    def test_plain_exception_without_hint(self):
        assert parse_retry_after_seconds(RuntimeError("connection reset")) is None


class TestHttpDateParsing:
    def test_future_date(self):
        when = datetime.now(timezone.utc) + timedelta(seconds=600)
        text = f"retry-after: {format_datetime(when, usegmt=True)}"
        # Capped at 300s.
        assert parse_retry_after_seconds(text) == 300.0

    def test_small_future_date(self):
        when = datetime.now(timezone.utc) + timedelta(seconds=45)
        text = f"Retry-After: {format_datetime(when, usegmt=True)}"
        assert parse_retry_after_seconds(text) == pytest.approx(45.0, abs=2.0)

    def test_past_date_is_ignored(self):
        when = datetime.now(timezone.utc) - timedelta(seconds=600)
        text = f"retry-after: {format_datetime(when, usegmt=True)}"
        assert parse_retry_after_seconds(text) is None

    def test_date_survives_lowercased_chain(self):
        # The parser matches the date on the original-case chain text even
        # though marker scans use the lowered chain.
        when = datetime.now(timezone.utc) + timedelta(seconds=90)
        exc = RuntimeError(f"Retry-After: {format_datetime(when, usegmt=True)}")
        exc.__cause__ = RuntimeError("Web call failed for model m")
        assert parse_retry_after_seconds(exc) == pytest.approx(90.0, abs=2.0)


class TestHeaderExtraction:
    def test_header_precedence_ms_then_seconds_then_date(self):
        both = {"Retry-After-Ms": "1500", "Retry-After": "30"}
        assert _retry_after_hint_from_headers(both) == 1.5
        assert _retry_after_hint_from_headers({"Retry-After": "30"}) == 30.0

    def test_case_insensitive_lookup(self):
        assert _retry_after_hint_from_headers({"RETRY-AFTER": "5"}) == 5.0
        assert _retry_after_hint_from_headers({"retry-after-ms": "250"}) == 0.25

    def test_http_date_header(self):
        when = datetime.now(timezone.utc) + timedelta(seconds=120)
        headers = {"Retry-After": format_datetime(when, usegmt=True)}
        assert _retry_after_hint_from_headers(headers) == pytest.approx(120.0, abs=2.0)

    def test_past_http_date_header_is_none(self):
        when = datetime.now(timezone.utc) - timedelta(seconds=120)
        headers = {"Retry-After": format_datetime(when, usegmt=True)}
        assert _retry_after_hint_from_headers(headers) is None

    def test_garbage_values_are_none(self):
        assert _retry_after_hint_from_headers({"Retry-After": "soon"}) is None
        assert _retry_after_hint_from_headers({}) is None
        assert _retry_after_hint_from_headers(None) is None

    def test_suffix_renders_the_text_protocol(self):
        assert _retry_after_suffix({"Retry-After-Ms": "2500"}) == " (retry-after: 2.5s)"
        assert _retry_after_suffix({"Retry-After": "30"}) == " (retry-after: 30s)"
        assert _retry_after_suffix({}) == ""


class _FakeResponse:
    def __init__(self, status, headers, body="upstream exploded"):
        self.status = status
        self.headers = headers
        self._body = body

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    response: _FakeResponse = _FakeResponse(503, {})

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return _FakeSession.response


class TestAiohttpFallbackEmbedsHeader:
    def _request(self):
        return ChatRequest(messages=[ChatMessage("user", "hi")])

    def test_raise_carries_retry_after_ms(self, monkeypatch):
        import clearwing.llm.native as native_module

        client = _client()
        _FakeSession.response = _FakeResponse(503, {"Retry-After-Ms": "2500"})
        monkeypatch.setattr(native_module.aiohttp, "ClientSession", _FakeSession)

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(
                client._openai_chat_http_fallback(self._request(), ChatOptions())
            )
        # The header hint rides the exception text in the shared protocol.
        assert "retry-after: 2.5s" in str(exc_info.value)
        assert parse_retry_after_seconds(exc_info.value) == 2.5

    def test_raise_carries_http_date_header(self, monkeypatch):
        import clearwing.llm.native as native_module

        client = _client()
        when = datetime.now(timezone.utc) + timedelta(seconds=600)
        _FakeSession.response = _FakeResponse(
            503, {"Retry-After": format_datetime(when, usegmt=True)}
        )
        monkeypatch.setattr(native_module.aiohttp, "ClientSession", _FakeSession)

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(
                client._openai_chat_http_fallback(self._request(), ChatOptions())
            )
        # Capped at 300s by the shared parser.
        assert parse_retry_after_seconds(exc_info.value) == 300.0

    def test_no_header_no_suffix(self, monkeypatch):
        import clearwing.llm.native as native_module

        client = _client()
        _FakeSession.response = _FakeResponse(500, {})
        monkeypatch.setattr(native_module.aiohttp, "ClientSession", _FakeSession)

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(
                client._openai_chat_http_fallback(self._request(), ChatOptions())
            )
        assert "retry-after" not in str(exc_info.value)


class TestRetryDelayIntegration:
    def test_retry_delay_uses_parsed_hint_over_backoff(self):
        client = _client()
        exc = RuntimeError("HTTP 503 (retry-after: 20)")
        # base = 20; jitter ≤ min(1, 20*0.2) = 1 → delay in [20, 21].
        delay = client._retry_delay_seconds(exc, attempt=0)
        assert 20.0 <= delay <= 21.0

    def test_retry_delay_uses_hint_from_nested_cause(self):
        client = _client()
        wrapped = RuntimeError("Web call failed")
        wrapped.__cause__ = RuntimeError("gateway says retry-after: 8")
        delay = client._retry_delay_seconds(wrapped, attempt=3)
        assert 8.0 <= delay <= 9.0

    def test_retry_delay_falls_back_to_exponential(self):
        client = _client(rate_limit_initial_backoff_seconds=1.0)
        delay = client._retry_delay_seconds(RuntimeError("no hint"), attempt=2)
        assert 4.0 <= delay <= 5.0  # 1 * 2^2 + jitter(≤0.8)
