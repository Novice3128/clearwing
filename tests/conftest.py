"""Repo-wide test fixtures.

Issue #72: an autouse tripwire makes "an offline test accidentally
resolves a REAL paid LLM endpoint from the host's ambient credentials"
impossible. ``tests/test_sourcehunt_runner.py::test_no_llm_at_all_runs_quick_path``
used to build a genuine Anthropic client whenever the developer shell
exported ``ANTHROPIC_API_KEY``, and the quick-depth ranker made paid
network calls on every test run.

Every test that is NOT marked ``@pytest.mark.requires_llm`` runs with:

1. The endpoint/credential env vars cleared — the CLEARWING_* env tier plus
   the ANTHROPIC_API_KEY default tier that ``resolve_llm_endpoint()``
   reads (``clearwing/providers/env.py``), and the preset/OAuth provider
   keys that ``ProviderManager.from_roles`` reads straight from the
   environment (``clearwing/providers/manager.py``). ``CLEARWING_BASE_URL``
   is included because a bare base_url alone can build a real
   openai-compat client (placeholder api_key).
2. ``CLEARWING_HOME`` redirected to an empty per-test directory, plus both
   ``Config`` path class attributes (``DEFAULT_CONFIG_PATH`` is frozen at
   import time, and ``_USER_CONFIG_PATH`` is loaded even under a
   ``CLEARWING_HOME`` override by design). Without this, the resolver's
   config tier auto-discovers the developer's real
   ``~/.clearwing/config.yaml`` — which on a configured host carries a
   complete paying endpoint — and clearing the env tier accomplishes
   nothing. The redirect also stops offline tests from writing into the
   real ``~/.clearwing/audit/`` tree. Tests that need a Clearwing home set
   their own ``CLEARWING_HOME`` (their monkeypatch runs after this fixture
   and wins); tests that need the REAL home opt in via the marker.

monkeypatch restores the original environment afterwards, so ambient
credentials survive for the developer's own commands.

Tests that genuinely need ambient real credentials must opt in:

    @pytest.mark.requires_llm
    def test_hits_a_real_endpoint(): ...
"""

from __future__ import annotations

import pytest

_LLM_ENDPOINT_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLEARWING_API_KEY",
    "CLEARWING_BASE_URL",
    "CLEARWING_MODEL",
    "DEEPSEEK_API_KEY",
    "FIREWORKS_API_KEY",
    "GROQ_API_KEY",
    "MINIMAX_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ORCAROUTER_API_KEY",
    "TOGETHER_API_KEY",
)


@pytest.fixture(autouse=True)
def _offline_llm_env(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Strip ambient LLM credentials (env tier AND config tier) unless the
    test opts in via ``requires_llm``."""
    if request.node.get_closest_marker("requires_llm") is not None:
        return
    for var in _LLM_ENDPOINT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Config tier, resolver path: point Clearwing's home at an empty dir so
    # _load_default_config_provider() finds no config.yaml instead of the
    # developer's real, fully-credentialed one.
    offline_home = tmp_path_factory.mktemp("cw-offline-home")
    monkeypatch.setenv("CLEARWING_HOME", str(offline_home))
    # Config tier, Config() path: Config deliberately loads the user's
    # personal ~/.clearwing/config.yaml even under a CLEARWING_HOME override
    # ("scan isolation shouldn't lock out LLM access" — core/config.py), and
    # DEFAULT_CONFIG_PATH is a class attribute frozen at import time, so the
    # env redirect alone cannot neutralize direct Config() construction.
    # Pin both to nonexistent files inside the offline home.
    from clearwing.core.config import Config

    monkeypatch.setattr(Config, "DEFAULT_CONFIG_PATH", offline_home / "config.yaml")
    monkeypatch.setattr(Config, "_USER_CONFIG_PATH", offline_home / "config.yaml")
