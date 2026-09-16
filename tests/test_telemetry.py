"""Tests for the CostTracker telemetry module."""

import pytest

from clearwing.core.events import EventBus, EventType
from clearwing.observability.telemetry import CostSummary, CostTracker, ToolUsage


def _reset_tracker():
    """Reset the CostTracker singleton for test isolation."""
    CostTracker._instance = None


class TestToolUsage:
    def test_defaults(self):
        tu = ToolUsage(name="nmap")
        assert tu.name == "nmap"
        assert tu.calls == 0
        assert tu.total_duration_ms == 0


class TestCostSummary:
    def test_fields(self):
        cs = CostSummary(
            input_tokens=100,
            output_tokens=50,
            total_cost_usd=0.01,
            tool_calls=3,
            by_tool={},
        )
        assert cs.input_tokens == 100
        assert cs.total_cost_usd == 0.01


class TestCostTracker:
    def setup_method(self):
        _reset_tracker()

    def teardown_method(self):
        _reset_tracker()

    def test_singleton(self):
        t1 = CostTracker()
        t2 = CostTracker()
        assert t1 is t2

    def test_initial_state(self):
        t = CostTracker()
        assert t.input_tokens == 0
        assert t.output_tokens == 0
        assert t.total_cost_usd == 0.0
        assert t.tool_calls == 0
        assert t.by_tool == {}
        assert t.cost_limit is None

    def test_record_llm_call_sonnet(self):
        t = CostTracker()
        t.record_llm_call(1000, 500, "claude-sonnet-4-6")
        # Cost = (1000 * 3.0 + 500 * 15.0) / 1_000_000 = (3000 + 7500) / 1M = 0.0105
        assert t.input_tokens == 1000
        assert t.output_tokens == 500
        assert abs(t.total_cost_usd - 0.0105) < 1e-6

    def test_record_llm_call_opus(self):
        t = CostTracker()
        t.record_llm_call(1000, 500, "claude-opus-4-6")
        # Cost = (1000 * 15.0 + 500 * 75.0) / 1_000_000 = (15000 + 37500) / 1M = 0.0525
        assert abs(t.total_cost_usd - 0.0525) < 1e-6

    def test_record_llm_call_haiku(self):
        t = CostTracker()
        t.record_llm_call(1000, 500, "claude-haiku-4-5")
        # Cost = (1000 * 0.80 + 500 * 4.0) / 1_000_000 = (800 + 2000) / 1M = 0.0028
        assert abs(t.total_cost_usd - 0.0028) < 1e-6

    def test_unknown_model_uses_sonnet_pricing(self):
        t = CostTracker()
        t.record_llm_call(1000, 500, "unknown-model")
        assert abs(t.total_cost_usd - 0.0105) < 1e-6

    def test_cumulative_calls(self):
        t = CostTracker()
        t.record_llm_call(1000, 500, "claude-sonnet-4-6")
        t.record_llm_call(2000, 1000, "claude-sonnet-4-6")
        assert t.input_tokens == 3000
        assert t.output_tokens == 1500

    def test_record_tool_call(self):
        t = CostTracker()
        t.record_tool_call("scan_ports", 150)
        t.record_tool_call("scan_ports", 200)
        t.record_tool_call("detect_os", 50)
        assert t.tool_calls == 3
        assert t.by_tool["scan_ports"].calls == 2
        assert t.by_tool["scan_ports"].total_duration_ms == 350
        assert t.by_tool["detect_os"].calls == 1

    def test_get_summary(self):
        t = CostTracker()
        t.record_llm_call(100, 50, "claude-sonnet-4-6")
        t.record_tool_call("nmap", 100)
        summary = t.get_summary()
        assert isinstance(summary, CostSummary)
        assert summary.input_tokens == 100
        assert summary.output_tokens == 50
        assert summary.tool_calls == 1
        assert "nmap" in summary.by_tool

    def test_is_over_limit_no_limit(self):
        t = CostTracker()
        t.record_llm_call(1000000, 500000, "claude-opus-4-6")
        assert not t.is_over_limit()

    def test_is_over_limit_under(self):
        t = CostTracker()
        t.cost_limit = 1.0
        t.record_llm_call(1000, 500, "claude-sonnet-4-6")
        assert not t.is_over_limit()

    def test_is_over_limit_exceeded(self):
        t = CostTracker()
        t.cost_limit = 0.001
        t.record_llm_call(1000, 500, "claude-sonnet-4-6")
        assert t.is_over_limit()

    def test_reset(self):
        t = CostTracker()
        t.record_llm_call(1000, 500, "claude-sonnet-4-6")
        t.record_tool_call("nmap", 100)
        t.reset()
        assert t.input_tokens == 0
        assert t.output_tokens == 0
        assert t.total_cost_usd == 0.0
        assert t.tool_calls == 0
        assert t.by_tool == {}
        assert t.session_total("sess-a") == 0.0

    def test_session_totals_accumulate_per_session(self):
        """Operator cost-limit follow-up to #41: per-session accumulation.

        The global totals pool every session in the process; the per-session
        totals let an operator job's limit check see ITS spend (inner graph
        calls + attributed hunts) without cross-session pollution.
        """
        t = CostTracker()
        t.record_llm_call(1000, 500, "claude-sonnet-4-6", session_id="job-a")
        t.record_llm_call(1000, 500, "claude-sonnet-4-6", session_id="job-a")
        t.record_llm_call(1000, 0, "claude-sonnet-4-6", session_id="job-b")
        # Unattributed calls stay out of every session bucket...
        t.record_llm_call(1000, 0, "claude-sonnet-4-6")

        assert t.session_total("job-a") == pytest.approx(2 * 0.0105)
        assert t.session_total("job-b") == pytest.approx(0.003)
        assert t.session_total("job-c") == 0.0
        assert t.session_total(None) == 0.0
        assert t.session_total("") == 0.0
        # Token totals ride along in parallel: (input, output) per session.
        assert t.session_tokens("job-a") == (2000, 1000)
        assert t.session_tokens("job-b") == (1000, 0)
        assert t.session_tokens("job-c") == (0, 0)
        assert t.session_tokens(None) == (0, 0)
        assert t.session_tokens("") == (0, 0)
        # ...while the global total keeps pooling everything.
        assert t.total_cost_usd == pytest.approx(2 * 0.0105 + 0.003 + 0.003)

    def test_forget_session_zeroes_cost_and_tokens(self):
        """PR #44 review P2: owners must be able to retire a session entry.

        Long-lived processes (webui) had no completion path that cleared a
        session's totals; since webui/operator ids are 8-hex UUID prefixes,
        a colliding new session inherited the stale spend.
        """
        t = CostTracker()
        t.record_llm_call(1000, 500, "claude-sonnet-4-6", session_id="collide1")
        assert t.session_total("collide1") == pytest.approx(0.0105)
        assert t.session_tokens("collide1") == (1000, 500)

        t.forget_session("collide1")

        assert t.session_total("collide1") == 0.0
        assert t.session_tokens("collide1") == (0, 0)
        # Global counters are NOT rewound — the process-wide total keeps
        # every recorded call.
        assert t.total_cost_usd == pytest.approx(0.0105)
        assert t.input_tokens == 1000 and t.output_tokens == 500
        # Unknown / empty ids are no-ops.
        t.forget_session("never-recorded")
        t.forget_session(None)
        t.forget_session("")

    def test_forget_session_prevents_collision_reuse(self):
        """Simulated id collision: the second job must not eat the first
        job's spend once the first job's entry was forgotten."""
        t = CostTracker()
        t.record_llm_call(1_000_000, 0, "claude-sonnet-4-6", session_id="ab12cd34")
        t.forget_session("ab12cd34")  # first job ends

        t.record_llm_call(100, 20, "claude-sonnet-4-6", session_id="ab12cd34")

        assert t.session_total("ab12cd34") == pytest.approx((100 * 3 + 20 * 15) / 1_000_000)
        assert t.session_tokens("ab12cd34") == (100, 20)

    def test_reset_clears_session_token_entries(self):
        t = CostTracker()
        t.record_llm_call(10, 5, "claude-sonnet-4-6", session_id="s")
        t.reset()
        assert t.session_total("s") == 0.0
        assert t.session_tokens("s") == (0, 0)
        t.record_llm_call(10, 5, "claude-sonnet-4-6", session_id="s")
        assert t.session_tokens("s") == (10, 5)

    def test_record_llm_call_emits_cost_update_with_elapsed_and_provider(self):
        """New keyword args ride along in the COST_UPDATE payload."""
        received: list[dict] = []

        def handler(data):
            received.append(data)

        bus = EventBus()
        bus.subscribe(EventType.COST_UPDATE, handler)
        try:
            t = CostTracker()
            t.record_llm_call(
                1000,
                500,
                "claude-sonnet-4-6",
                cached_tokens=100,
                elapsed_ms=1234.5,
                provider="anthropic",
            )
        finally:
            bus.unsubscribe(EventType.COST_UPDATE, handler)

        assert len(received) == 1
        payload = received[0]
        assert payload["input_tokens"] == 1000
        assert payload["output_tokens"] == 500
        assert payload["cached_tokens"] == 100
        assert payload["model"] == "claude-sonnet-4-6"
        assert payload["provider"] == "anthropic"
        assert payload["elapsed_ms"] == 1234.5

    def test_record_llm_call_backward_compatible_without_new_kwargs(self):
        """Legacy callers without the new kwargs still emit a valid payload."""
        received: list[dict] = []

        def handler(data):
            received.append(data)

        bus = EventBus()
        bus.subscribe(EventType.COST_UPDATE, handler)
        try:
            t = CostTracker()
            t.record_llm_call(1000, 500, "claude-sonnet-4-6")
        finally:
            bus.unsubscribe(EventType.COST_UPDATE, handler)

        assert len(received) == 1
        payload = received[0]
        # Falls back to safe defaults.
        assert payload["provider"] == "unknown"
        assert payload["elapsed_ms"] == 0
        assert payload["cached_tokens"] == 0

    def test_pricing_table(self):
        assert "claude-sonnet-4-6" in CostTracker.PRICING
        assert "claude-opus-4-6" in CostTracker.PRICING
        assert "claude-haiku-4-5" in CostTracker.PRICING
        for model, prices in CostTracker.PRICING.items():
            assert "input" in prices
            assert "output" in prices
            assert prices["input"] > 0
            assert prices["output"] > 0
