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

from clearwing.llm.native import AsyncLLMClient, _url_for_path


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
