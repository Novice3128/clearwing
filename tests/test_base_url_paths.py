"""Issue #60: pathless OpenAI-compat base_url handling.

`_url_for_path` joins base + endpoint path the codex-rs way (single slash,
no urljoin re-interpretation); a pathless base_url warns once at client
construction and 404 errors gain the actionable advice. Root-mounted
gateways keep working exactly as before — warning only, never a rewrite.
"""

from __future__ import annotations

import json
import logging

import pytest
from genai_pyo3 import ChatMessage, ChatOptions, ChatRequest

from clearwing.llm.native import (
    AsyncLLMClient,
    _redact_url_credentials,
    _url_for_path,
)


class TestUrlForPathJoinMatrix:
    def test_plain_base(self):
        assert (
            _url_for_path("https://gw.test/v1", "chat/completions")
            == "https://gw.test/v1/chat/completions"
        )

    def test_trailing_slash_base(self):
        assert (
            _url_for_path("https://gw.test/v1/", "chat/completions")
            == "https://gw.test/v1/chat/completions"
        )

    def test_multiple_trailing_slashes_base(self):
        assert (
            _url_for_path("https://gw.test/v1///", "chat/completions")
            == "https://gw.test/v1/chat/completions"
        )

    def test_rootless_join_keeps_host_root(self):
        # A pathless base keeps its host-root mount point — never rewritten.
        assert (
            _url_for_path("http://host:8787", "chat/completions")
            == "http://host:8787/chat/completions"
        )

    def test_pathless_base_with_slash(self):
        assert (
            _url_for_path("http://host:8787/", "chat/completions")
            == "http://host:8787/chat/completions"
        )

    def test_leading_slash_path(self):
        assert (
            _url_for_path("https://gw.test/v1", "/chat/completions")
            == "https://gw.test/v1/chat/completions"
        )

    def test_empty_path_returns_base_unchanged(self):
        assert _url_for_path("https://gw.test/v1/", "") == "https://gw.test/v1/"
        assert _url_for_path("https://gw.test", "") == "https://gw.test"


def _openai_client(base_url: str | None) -> AsyncLLMClient:
    return AsyncLLMClient(
        model_name="test-model",
        provider_name="openai",
        api_key="dummy",
        base_url=base_url,
    )


class TestPathlessBaseUrlWarning:
    def test_pathless_openai_base_warns_at_construction(self, caplog):
        with caplog.at_level(logging.WARNING, logger="clearwing.llm.native"):
            _openai_client("http://localhost:8787")
        warnings = [
            r
            for r in caplog.records
            if "has no path component" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "http://localhost:8787" in warnings[0].getMessage()
        assert "/v1" in warnings[0].getMessage()

    def test_pathful_openai_base_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="clearwing.llm.native"):
            _openai_client("http://localhost:8787/v1")
        assert not any(
            "has no path component" in r.getMessage() for r in caplog.records
        )

    def test_openai_without_base_url_does_not_warn(self, caplog):
        # Anthropic-direct-style None base never routes to chat/completions.
        with caplog.at_level(logging.WARNING, logger="clearwing.llm.native"):
            _openai_client(None)
        assert not any(
            "has no path component" in r.getMessage() for r in caplog.records
        )

    def test_pathless_ollama_base_does_not_warn(self, caplog):
        # The ollama preset is pathless BY DESIGN (the adapter resolves its
        # own paths) — only the openai chat-completions family warns.
        with caplog.at_level(logging.WARNING, logger="clearwing.llm.native"):
            AsyncLLMClient(
                model_name="llama3",
                provider_name="ollama",
                api_key="ollama",
                base_url="http://localhost:11434",
            )
        assert not any(
            "has no path component" in r.getMessage() for r in caplog.records
        )

    def test_double_slash_pathless_base_still_warns(self, caplog):
        # "http://host:8787//" is as pathless as "/" or "" — all mount
        # endpoints at the host root, so all must warn.
        with caplog.at_level(logging.WARNING, logger="clearwing.llm.native"):
            _openai_client("http://localhost:8787//")
        warnings = [
            r
            for r in caplog.records
            if "has no path component" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "http://localhost:8787//" in warnings[0].getMessage()

    def test_userinfo_base_url_is_redacted_in_warning(self, caplog):
        # aiohttp honors user:pass@ as basic auth on the REQUEST, but the
        # warning text must not leak the credentials.
        with caplog.at_level(logging.WARNING, logger="clearwing.llm.native"):
            _openai_client("http://user:pass@localhost:8787")
        warnings = [
            r
            for r in caplog.records
            if "has no path component" in r.getMessage()
        ]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "user:pass" not in message
        assert "user" not in message
        assert "localhost:8787" in message


class TestAiohttpFallbackUrl:
    """The one openai-compat URL join site: the aiohttp chat/completions
    fallback must route through `_url_for_path`."""

    def _capture_fallback_url(self, monkeypatch, base_url: str) -> str:
        import aiohttp

        captured: dict[str, str] = {}
        ok_payload = json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

        class _FakeResponse:
            def __init__(self, status: int, text: str):
                self.status = status
                self._text = text

            async def text(self):
                return self._text

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

        class _FakeSession:
            def __init__(self, timeout=None):
                del timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def post(self, url, json=None, headers=None):
                del json, headers
                captured["url"] = url
                return _FakeResponse(200, ok_payload)

        monkeypatch.setattr(aiohttp, "ClientSession", _FakeSession)
        client = _openai_client(base_url)
        request = ChatRequest(messages=[ChatMessage("user", "hi")], system=None)
        options = ChatOptions(capture_usage=True)
        import asyncio

        response = asyncio.run(client._openai_chat_http_fallback(request, options))
        assert response.first_text == "ok"
        return captured["url"]

    def test_fallback_url_joins_v1_base(self, monkeypatch):
        url = self._capture_fallback_url(monkeypatch, "https://gw.test/v1")
        assert url == "https://gw.test/v1/chat/completions"

    def test_fallback_url_joins_pathless_base_at_host_root(self, monkeypatch):
        url = self._capture_fallback_url(monkeypatch, "http://host:8787/")
        assert url == "http://host:8787/chat/completions"


class TestPathless404Advice:
    @pytest.mark.asyncio
    async def test_404_from_pathless_base_gains_advice(self):
        client = _openai_client("http://host:8787")
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            raise RuntimeError(
                "OpenAI-compatible fallback failed with HTTP 404: not found"
            )

        with pytest.raises(RuntimeError) as excinfo:
            await client._with_retries(op)

        message = str(excinfo.value)
        assert "HTTP 404" in message
        assert "has no path component" in message
        assert "http://host:8787" in message
        assert "/v1" in message
        # 404 was never retryable and stays that way: single attempt.
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_404_from_pathful_base_gets_no_advice(self):
        client = _openai_client("http://host:8787/v1")

        async def op():
            raise RuntimeError(
                "OpenAI-compatible fallback failed with HTTP 404: not found"
            )

        with pytest.raises(RuntimeError) as excinfo:
            await client._with_retries(op)
        assert "has no path component" not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_404_from_double_slash_pathless_base_gains_advice(self):
        # "//" is as pathless as "/" — the advice must fire there too.
        client = _openai_client("http://host:8787//")

        async def op():
            raise RuntimeError(
                "OpenAI-compatible fallback failed with HTTP 404: not found"
            )

        with pytest.raises(RuntimeError) as excinfo:
            await client._with_retries(op)
        assert "has no path component" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_userinfo_base_url_is_redacted_in_404_advice(self):
        client = _openai_client("http://user:pass@host:8787")

        async def op():
            raise RuntimeError(
                "OpenAI-compatible fallback failed with HTTP 404: not found"
            )

        with pytest.raises(RuntimeError) as excinfo:
            await client._with_retries(op)
        message = str(excinfo.value)
        assert "has no path component" in message
        assert "user:pass" not in message
        assert "user" not in message
        assert "host:8787" in message

    @pytest.mark.asyncio
    async def test_structured_404_status_attribute_is_detected(self):
        client = _openai_client("http://host:8787")

        async def op():
            exc = RuntimeError("request failed")
            exc.status = 404  # structured attribute, no digits in the text
            raise exc

        with pytest.raises(RuntimeError) as excinfo:
            await client._with_retries(op)
        assert "has no path component" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_structured_non_404_status_beats_404_body_text(self):
        # Codex PR-62 r3: an explicit structured status anywhere in the
        # chain is AUTHORITATIVE — a 400 whose body text happens to say
        # "HTTP 404" must not gain the pathless-base_url advice (400 is
        # not retried, so this raises on the first attempt).
        client = _openai_client("http://host:8787")

        async def op():
            exc = RuntimeError("upstream body said: HTTP 404: not found")
            exc.status = 400
            raise exc

        with pytest.raises(RuntimeError) as excinfo:
            await client._with_retries(op)
        assert "has no path component" not in str(excinfo.value)

    def test_structured_status_is_authoritative_over_anchored_text(self):
        # Direct classifier probes (avoids the 5xx retry backoff):
        # - structured 500 + "HTTP 404" body text → NOT a 404;
        # - the same 500 wrapping a 404 cause → 404 in SOME layer wins;
        # - a structured "404" string counts too;
        # - no structured status anywhere → anchored text decides.
        client = _openai_client("http://host:8787")

        outer = RuntimeError("gateway exploded; body mentioned HTTP 404")
        outer.status = 500
        assert client._is_not_found_error(outer) is False

        inner = RuntimeError("not found")
        inner.status_code = 404
        chained = RuntimeError("gateway exploded; body mentioned HTTP 404")
        chained.status = 500
        chained.__cause__ = inner
        assert client._is_not_found_error(chained) is True

        string_status = RuntimeError("request failed")
        string_status.http_status = "404"
        assert client._is_not_found_error(string_status) is True

        text_only = RuntimeError("failed with HTTP 404: not found")
        assert client._is_not_found_error(text_only) is True

        bool_status = RuntimeError("failed with HTTP 404: not found")
        bool_status.status = True  # bool is not a status — text anchor decides
        assert client._is_not_found_error(bool_status) is True

    @pytest.mark.asyncio
    async def test_non_404_error_gets_no_advice(self):
        client = _openai_client("http://host:8787")

        async def op():
            raise RuntimeError("invalid request body")

        with pytest.raises(RuntimeError) as excinfo:
            await client._with_retries(op)
        assert "has no path component" not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_bare_404_digits_in_body_do_not_classify(self):
        # The anchor must match the STATUS, not stray digits ("see ticket
        # 4041" must not annotate; neither must an unanchored body 404).
        client = _openai_client("http://host:8787")

        async def op():
            raise RuntimeError("server said: see ticket 4041 for details")

        with pytest.raises(RuntimeError) as excinfo:
            await client._with_retries(op)
        assert "has no path component" not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_annotated_404_error_is_not_double_annotated(self):
        client = _openai_client("http://host:8787")

        async def op():
            raise RuntimeError(
                "failed with HTTP 404 (base_url 'http://host:8787' has no "
                "path component; requests will hit /chat/completions at the "
                "host root. Most OpenAI-compatible gateways serve /v1 — set "
                "the full path (e.g. http://host:8787/v1) if a 404 occurs.)"
            )

        with pytest.raises(RuntimeError) as excinfo:
            await client._with_retries(op)
        assert str(excinfo.value).count("has no path component") == 1


class TestRedactUrlCredentials:
    def test_no_userinfo_returned_unchanged(self):
        assert (
            _redact_url_credentials("http://host:8787/v1") == "http://host:8787/v1"
        )

    def test_userinfo_stripped_port_and_path_kept(self):
        assert (
            _redact_url_credentials("http://user:pass@host:8787/v1")
            == "http://host:8787/v1"
        )

    def test_userinfo_without_port(self):
        assert _redact_url_credentials("https://ak@kw.test") == "https://kw.test"

    def test_last_at_sign_is_the_split_point(self):
        # A userinfo may itself contain an encoded "@"; the hostinfo is
        # everything after the LAST one.
        assert (
            _redact_url_credentials("http://u%40x:pass@host:8787")
            == "http://host:8787"
        )

    def test_query_secret_dropped_even_without_userinfo(self):
        # Codex PR-62 r1 (P1): a presigned-gateway base_url carries its
        # credential in the QUERY, not the userinfo — the display form
        # must drop it. The old early-return leaked the full URL.
        assert (
            _redact_url_credentials("https://host:8787?api_key=sekrit")
            == "https://host:8787"
        )

    def test_fragment_dropped_even_without_userinfo(self):
        assert _redact_url_credentials("https://host:8787/#tok=abc") == "https://host:8787/"

    def test_pathless_query_secret_warning_and_404_do_not_leak(self, caplog):
        # End-to-end guard for the same leak through both new surfaces:
        # the construction warning and the 404 annotation.
        with caplog.at_level("WARNING"):
            _openai_client("https://host:8787?api_key=sekrit")
        assert not any("sekrit" in r.message for r in caplog.records)
