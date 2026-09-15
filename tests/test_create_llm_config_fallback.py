"""_create_llm must consult env/config credentials even when a model name
rides in from the webui start frame (where it is always non-None).

Before the fix, the bare model name took resolve_llm_endpoint's
"CLI flags win" branch, config.yaml / env were never consulted, and every
webui chat start died with "no API key or base URL configured".
"""

from __future__ import annotations

import clearwing.agent.graph as graph_mod
from clearwing.providers.env import LLMEndpoint


def _patch(
    monkeypatch,
    *,
    source="config",
    model="glm-5.3",
    base_url="https://z.ai/api",
    api_key="sk-test",
):
    captured: dict = {}
    fake_ep = LLMEndpoint(
        provider="anthropic",
        model=model,
        base_url=base_url,
        api_key=api_key,
        source=source,
    )

    def fake_resolve(*args, **kwargs):
        captured["resolve_kwargs"] = kwargs
        return fake_ep

    def fake_for_endpoint(endpoint):
        captured["endpoint"] = endpoint

        class _PM:
            def get_native_client(self, task):
                return object()

        return _PM()

    monkeypatch.setattr(graph_mod, "resolve_llm_endpoint", fake_resolve)
    monkeypatch.setattr(
        graph_mod.ProviderManager, "for_endpoint", staticmethod(fake_for_endpoint)
    )
    return captured


class TestCreateLLMConfigFallback:
    def test_placeholder_model_keeps_configured_model(self, monkeypatch):
        captured = _patch(monkeypatch)
        graph_mod._create_llm("claude-sonnet-4-6")
        assert captured["endpoint"].model == "glm-5.3"
        assert captured["resolve_kwargs"].get("cli_model") is None

    def test_explicit_model_overrides_configured(self, monkeypatch):
        captured = _patch(monkeypatch)
        graph_mod._create_llm("kimi-k2")
        assert captured["endpoint"].model == "kimi-k2"

    def test_default_source_allows_model_override(self, monkeypatch):
        captured = _patch(
            monkeypatch, source="default", model="claude-sonnet-4-6",
            base_url=None, api_key=None,
        )
        graph_mod._create_llm("glm-5.3")
        assert captured["endpoint"].model == "glm-5.3"

    def test_explicit_credentials_skip_config_discovery(self, monkeypatch):
        captured = _patch(monkeypatch)
        graph_mod._create_llm("m", base_url="https://x", api_key="k")
        assert captured["resolve_kwargs"].get("config_provider") == {}
