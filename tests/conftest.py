"""Repo-wide test fixtures.

Issue #72: an autouse tripwire makes "an offline test accidentally
resolves a REAL paid LLM endpoint from the host's ambient credentials"
impossible. ``tests/test_sourcehunt_runner.py::test_no_llm_at_all_runs_quick_path``
used to build a genuine Anthropic client whenever the developer shell
exported ``ANTHROPIC_API_KEY``, and the quick-depth ranker made paid
network calls on every test run.

Every test that is NOT marked ``@pytest.mark.requires_llm`` runs with the
endpoint/credential env vars cleared — the complete set the
``resolve_llm_endpoint()`` ladder reads (``clearwing/providers/env.py``):
the CLEARWING_* env tier plus the ANTHROPIC_API_KEY default tier.
``CLEARWING_BASE_URL`` is included because a bare base_url alone can
build a real openai-compat client (placeholder api_key). monkeypatch
restores the original environment afterwards, so ambient credentials
survive for the developer's own commands.

Tests that genuinely need ambient real credentials must opt in:

    @pytest.mark.requires_llm
    def test_hits_a_real_endpoint(): ...
"""

from __future__ import annotations

import pytest

_LLM_ENDPOINT_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "CLEARWING_API_KEY",
    "CLEARWING_BASE_URL",
    "CLEARWING_MODEL",
)


@pytest.fixture(autouse=True)
def _offline_llm_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Strip ambient LLM credentials unless the test opts in via
    ``requires_llm``."""
    if request.node.get_closest_marker("requires_llm") is not None:
        return
    for var in _LLM_ENDPOINT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
