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
- Codex PR-63 r1: the aiohttp raise sites ALSO stash the header hint on
  the exception's ``clearwing_retry_after`` attribute, which outranks
  every body-text token (a body ``retry-after: 1`` must never override a
  120 s header).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
from genai_pyo3 import ChatMessage, ChatOptions, ChatRequest

from clearwing.llm.native import (
    AsyncLLMClient,
    _attach_retry_after,
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

    def test_unitless_legacy_phrasings_do_not_match(self):
        # A number without an explicit ms/s unit is not a delay the
        # provider specified — "wait 5 minutes" must never read as 5s.
        assert parse_retry_after_seconds("please wait 5 minutes") is None
        assert parse_retry_after_seconds("try again in 5 minutes") is None
        assert parse_retry_after_seconds("wait 5") is None


class TestCleanNumberTail:
    """Codex PR-63 r3 (P2): a parsed number must END cleanly.

    The old seconds pattern matched any digit run after the header name:
    ``retry-after: 5 minutes`` half-matched as 5 s and ``retry-after: 1e3``
    as 1 s — premature retries that re-hit a rate-limited upstream. After
    an optional glued ms/s unit, no alphanumeric/dot may touch the number
    and none may follow intervening whitespace. The same guard now covers
    the ms header/suffix patterns, and the legacy unit can no longer eat
    the first letter of a longer word ("wait 5 seconds" is not "wait 5s").
    """

    def test_glued_unit_is_consumed(self):
        assert parse_retry_after_seconds("retry-after: 30s") == 30.0
        assert parse_retry_after_seconds("retry-after: 2.500s") == 2.5

    def test_fractional_seconds(self):
        assert parse_retry_after_seconds("retry-after: 0.5") == 0.5

    def test_unit_word_after_number_is_rejected(self):
        assert parse_retry_after_seconds("retry-after: 5 minutes") is None
        assert parse_retry_after_seconds("retry-after: 30 seconds") is None
        assert parse_retry_after_seconds("HTTP 429: retry-after: 10 minutes, cool down") is None

    def test_scientific_notation_is_rejected(self):
        assert parse_retry_after_seconds("retry-after: 1e3") is None
        assert parse_retry_after_seconds("retry-after: 2.5e2") is None

    def test_dotted_garbage_is_rejected(self):
        # A second dot cannot be part of the number — "1.5.2" is not 1.5 s.
        assert parse_retry_after_seconds("retry-after: 1.5.2") is None

    def test_trailing_punctuation_still_parses(self):
        # Punctuation is fine; only alnum-after-whitespace (a word) and
        # glued alnum/dot (scientific/dotted garbage) reject.
        assert parse_retry_after_seconds("retry-after: 30, server says slow") == 30.0
        assert parse_retry_after_seconds("HTTP 503 (retry-after: 20)") == 20.0

    def test_ms_header_gets_the_same_guard(self):
        assert parse_retry_after_seconds("retry-after-ms: 1e3") is None
        assert parse_retry_after_seconds("retry-after-ms: 500x") is None
        # The plain pass-through case is unchanged.
        assert parse_retry_after_seconds("retry-after-ms: 500") == 0.5

    def test_ms_unit_suffix_gets_the_same_guard(self):
        assert parse_retry_after_seconds("retry-after: 500 msx") is None
        # The plain pass-through case is unchanged.
        assert parse_retry_after_seconds("retry-after: 500 ms") == 0.5

    def test_legacy_unit_cannot_eat_a_longer_word(self):
        # "wait 5 seconds" used to match the bare "s" alternative and read
        # as 5 s by luck; with the guard it parses as nothing (a number
        # plus a word the patterns do not understand is not a delay).
        assert parse_retry_after_seconds("wait 5 seconds") is None
        assert parse_retry_after_seconds("try again in 10 moons") is None
        # The required-unit rejections stay pinned.
        assert parse_retry_after_seconds("try again in 5 minutes") is None
        assert parse_retry_after_seconds("please wait 5 minutes") is None


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
        # Codex PR-63 r1: the suffix takes the header-derived hint VALUE
        # (not the headers mapping) — the raise sites extract once and
        # feed both the text echo and the authoritative attribute.
        assert _retry_after_suffix(2.5) == " (retry-after: 2.500s)"
        assert _retry_after_suffix(30.0) == " (retry-after: 30.000s)"
        assert _retry_after_suffix(None) == ""
        assert _retry_after_suffix(0.0) == ""

    def test_suffix_fixed_point_round_trip_for_huge_hints(self):
        # ``:g`` formatting went scientific at ≥ 1e6 ("3e+06"), and the
        # downstream numeric regex truncated that to 3 s. The fixed-point
        # render (pre-capped at 300 since PR-63 r1) must round-trip:
        # embedded → parsed → capped at 300, never silently read as 3.
        suffix = _retry_after_suffix(3000000.0)
        assert suffix == " (retry-after: 300.000s)"
        exc = RuntimeError(f"HTTP 503 upstream exploded{suffix}")
        assert parse_retry_after_seconds(exc) == 300.0


class TestAttachedRetryAfterAttribute:
    """Codex PR-63 r1: the aiohttp raise sites stash the authoritative
    header-derived Retry-After on the exception's ``clearwing_retry_after``
    attribute; the parser prefers it (precedence 0) over every body-text
    token, so provider body chatter can never override the header."""

    def test_attribute_beats_body_seconds_token(self):
        exc = RuntimeError("HTTP 429 quota exceeded; body says retry-after: 1")
        exc.clearwing_retry_after = 120.0
        assert parse_retry_after_seconds(exc) == 120.0

    def test_attribute_beats_body_ms_token(self):
        exc = RuntimeError("rate limited (retry-after-ms: 500)")
        exc.clearwing_retry_after = 120.0
        assert parse_retry_after_seconds(exc) == 120.0

    def test_attribute_found_through_cause_chain(self):
        # The transport raise may be wrapped (retry plumbing, genai's
        # terse wrappers): the attribute walk covers the full chain.
        inner = RuntimeError("upstream exploded")
        inner.clearwing_retry_after = 90.0
        wrapped = RuntimeError("Web call failed for model m")
        wrapped.__cause__ = inner
        assert parse_retry_after_seconds(wrapped) == 90.0

    def test_attribute_still_capped_at_max_seconds(self):
        exc = RuntimeError("slow down")
        exc.clearwing_retry_after = 9999.0
        assert parse_retry_after_seconds(exc) == 300.0

    def test_bool_attribute_is_not_a_delay(self):
        # True is an int subclass but not a delay — the parser must skip
        # it and fall through to the text scan (None here).
        exc = RuntimeError("connection reset")
        exc.clearwing_retry_after = True
        assert parse_retry_after_seconds(exc) is None

    def test_attach_caps_hints_at_300(self):
        exc = _attach_retry_after(RuntimeError("HTTP 503"), 400.0)
        assert exc.clearwing_retry_after == 300.0

    def test_attach_sets_positive_hints_verbatim(self):
        exc = _attach_retry_after(RuntimeError("HTTP 429"), 120.0)
        assert exc.clearwing_retry_after == 120.0

    def test_attach_ignores_none_and_nonpositive_hints(self):
        for hint in (None, 0.0, -5.0):
            exc = _attach_retry_after(RuntimeError("HTTP 503"), hint)
            assert not hasattr(exc, "clearwing_retry_after")


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
        assert "retry-after: 2.500s" in str(exc_info.value)
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

    def test_header_attribute_beats_misleading_body_token(self, monkeypatch):
        """The exact Codex PR-63 r1 shape: the response HEADER says 120s
        while the error BODY text carries its own ``retry-after: 1``
        chatter. The text scan alone would parse 1 (the body token sits
        before the rendered suffix); the attached attribute must win."""
        import clearwing.llm.native as native_module

        client = _client()
        _FakeSession.response = _FakeResponse(
            503, {"Retry-After": "120"}, body="quota exceeded; retry-after: 1"
        )
        monkeypatch.setattr(native_module.aiohttp, "ClientSession", _FakeSession)

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(
                client._openai_chat_http_fallback(self._request(), ChatOptions())
            )
        # The authoritative copy rides the exception attribute, capped.
        assert exc_info.value.clearwing_retry_after == 120.0
        # And the shared parser prefers it over the body token.
        assert parse_retry_after_seconds(exc_info.value) == 120.0

    def test_header_attribute_beats_misleading_body_ms_token(self, monkeypatch):
        import clearwing.llm.native as native_module

        client = _client()
        _FakeSession.response = _FakeResponse(
            503, {"Retry-After": "120"}, body="slow down; retry-after-ms: 500"
        )
        monkeypatch.setattr(native_module.aiohttp, "ClientSession", _FakeSession)

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(
                client._openai_chat_http_fallback(self._request(), ChatOptions())
            )
        assert exc_info.value.clearwing_retry_after == 120.0
        assert parse_retry_after_seconds(exc_info.value) == 120.0


class TestRetryDelayIntegration:
    def test_retry_delay_uses_parsed_hint_over_backoff(self):
        client = _client()
        exc = RuntimeError("HTTP 503 (retry-after: 20)")
        # Server advice is honored verbatim — no jitter (the server named a
        # time; randomizing it can only land closer to the limit).
        assert client._retry_delay_seconds(exc, attempt=0) == 20.0

    def test_retry_delay_uses_hint_from_nested_cause(self):
        client = _client()
        wrapped = RuntimeError("Web call failed")
        wrapped.__cause__ = RuntimeError("gateway says retry-after: 8")
        assert client._retry_delay_seconds(wrapped, attempt=3) == 8.0

    def test_parsed_hint_bypasses_local_backoff_cap(self):
        # opencode retry.ts / codex parity: a server-suggested delay may
        # exceed the LOCAL rate_limit_max_backoff_seconds — only the
        # parser's 300s absolute cap applies.
        client = _client(rate_limit_max_backoff_seconds=60.0)
        exc = RuntimeError("HTTP 503 (retry-after: 200)")
        assert client._retry_delay_seconds(exc, attempt=0) == 200.0

    def test_no_hint_still_capped_at_local_backoff_max(self):
        client = _client(rate_limit_max_backoff_seconds=60.0)
        # attempt 10 → exponential 1*2^10 = 1024, capped at 60 (+ jitter).
        delay = client._retry_delay_seconds(RuntimeError("no hint"), attempt=10)
        assert 60.0 <= delay <= 61.0

    def test_retry_delay_falls_back_to_exponential(self):
        client = _client(rate_limit_initial_backoff_seconds=1.0)
        delay = client._retry_delay_seconds(RuntimeError("no hint"), attempt=2)
        assert 4.0 <= delay <= 5.0  # 1 * 2^2 + jitter(≤0.8)
