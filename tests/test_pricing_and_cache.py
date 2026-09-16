"""Tests for the pricing table's prefix-aware resolution (issue #10) and the
per-step context-note assembly that keeps the request prefix cache-stable
(issue #36)."""

from __future__ import annotations

import logging

import pytest

from clearwing.llm.native import _mark_cache_prefix, _with_context_note
from clearwing.observability.telemetry import CostTracker


class TestPricingResolution:
    def setup_method(self):
        CostTracker._warned_pricing_models.clear()

    def teardown_method(self):
        CostTracker._warned_pricing_models.clear()

    def test_glm_53_entry_matches_confirmed_rates(self):
        # Rates cross-checked against actual v2-round billing (issue #10
        # live evidence: the Sonnet fallback overstated spend ~2.17x).
        pricing = CostTracker._resolve_pricing("glm-5.3")
        assert pricing == {"input": 1.40, "cached_input": 0.14, "output": 4.40}

    def test_mixed_case_keys_resolve(self):
        # Codex PR-39 P1: the table carries mixed-case gateway aliases;
        # lowercasing only the model name made "UnCut" miss its own entry
        # and bill at Sonnet rates.
        assert CostTracker._resolve_pricing("UnCut") == CostTracker.PRICING["UnCut"]
        assert CostTracker._resolve_pricing("uncut") == CostTracker.PRICING["UnCut"]
        assert CostTracker._resolve_pricing("UnCut-v2") == CostTracker.PRICING["UnCut"]
        assert CostTracker.estimate_cost(1_000_000, 0, "UnCut") == pytest.approx(1.40)

    def test_versioned_echo_resolves_by_prefix(self):
        assert CostTracker._resolve_pricing("claude-sonnet-4-6-20260901") == (
            CostTracker.PRICING["claude-sonnet-4-6"]
        )
        # Longest key wins: opus-4-7 must not be shadowed by a shorter key.
        assert CostTracker._resolve_pricing("claude-opus-4-7-20260901") == (
            CostTracker.PRICING["claude-opus-4-7"]
        )

    def test_endpoint_style_name_resolves_by_basename(self):
        assert CostTracker._resolve_pricing("org/glm-5.2") == (
            CostTracker.PRICING["glm-5.2"]
        )

    def test_unknown_model_returns_none_and_falls_back(self):
        assert CostTracker._resolve_pricing("totally-unknown-model") is None
        assert CostTracker.estimate_cost(1000, 1000, "totally-unknown-model") == (
            CostTracker.estimate_cost(1000, 1000, CostTracker._DEFAULT_MODEL)
        )

    def test_non_string_models_are_tolerated(self):
        # Hunter tests pass AsyncMock attributes as model names; the lookup
        # must not crash on them (the old dict.get silently fell back).
        assert CostTracker._resolve_pricing(None) is None  # type: ignore[arg-type]
        assert not CostTracker.has_pricing(object())

    def test_fallback_warns_once_per_model(self, caplog):
        with caplog.at_level(logging.WARNING):
            CostTracker.estimate_cost(10, 10, "mystery-model-x")
            CostTracker.estimate_cost(10, 10, "mystery-model-x")
        warnings = [r for r in caplog.records if "mystery-model-x" in r.message]
        assert len(warnings) == 1
        assert "reference" in warnings[0].message

    def test_cached_tokens_bill_at_cached_rate(self):
        cost = CostTracker.estimate_cost(1_000_000, 0, "glm-5.3", cached_tokens=500_000)
        # 500k uncached at 1.40 + 500k cached at 0.14 per 1M tokens.
        assert cost == pytest.approx(0.5 * 1.40 + 0.5 * 0.14)

    def test_anthropic_cache_reads_bill_at_ten_percent(self):
        # Codex PR-39 r2 P2: cache-capable rows must define cached_input —
        # Anthropic's official cache-read price is 10% of input.
        cost = CostTracker.estimate_cost(1_000_000, 0, "claude-sonnet-4-6", cached_tokens=1_000_000)
        assert cost == pytest.approx(0.30)
        opus = CostTracker.estimate_cost(1_000_000, 0, "claude-opus-4-7", cached_tokens=500_000)
        assert opus == pytest.approx(0.5 * 15.0 + 0.5 * 1.50)


class TestContextNoteAssembly:
    def test_note_rides_after_the_cache_breakpoint(self):
        from genai_pyo3 import ChatMessage

        history = [ChatMessage("user", "hello"), ChatMessage("assistant", "hi")]

        messages = _with_context_note(
            _mark_cache_prefix(list(history), True), "## Current Context\n..."
        )

        # The note is appended after the marked prefix: the breakpoint stays
        # on the last stable history message and the note itself is unmarked
        # (a changing tail must never enter the cached prefix).
        assert len(messages) == 3
        assert messages[2].role == "user"
        assert "Current Context" in messages[2].content
        assert messages[1].cache_control == "ephemeral"
        assert messages[2].cache_control is None
        # The caller's history list is not mutated with the note.
        assert len(history) == 2

    def test_empty_note_is_a_no_op(self):
        from genai_pyo3 import ChatMessage

        history = [ChatMessage("user", "hello")]
        assert _with_context_note(list(history), None) == history
        assert _with_context_note(list(history), "") == history
