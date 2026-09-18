"""Issue #57: FallbackChain member cooldown / dispatch stickiness.

A persistently-failing primary used to re-charge its FULL retry budget on
every call before the chain switched. Members that fail at chain level now
enter an exponentially growing cooldown (60s × 2^(n-1), max 600s — codex-rs
session failure flags + LiteLLM DEFAULT_COOLDOWN_TIME_SECONDS design), with
half-open recovery, Retry-After-aware windows, and an all-cooling fail-open.

Invariants that must NOT regress: per-member retry ownership, cancellation
never fails over (and never cools), spend-ledger collapse, the
served_by_primary flag, and the stream delta suppression policy.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import clearwing.llm.fallback as fallback_module
from clearwing.llm.fallback import FallbackChain


class _Member:
    """Duck-typed chain member with toggleable failure."""

    def __init__(self, model_name, provider_name="openai", response="ok"):
        self.model_name = model_name
        self.provider_name = provider_name
        self.spend_ledger = None
        self.response = response
        self.exc: Exception | None = None
        self.calls = 0

    async def achat(self, **kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.response

    async def achat_stream(self, **kwargs):
        return await self.achat(**kwargs)


class _Clock:
    """Deterministic time.monotonic stand-in."""

    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now


@pytest.fixture()
def clock(monkeypatch):
    clk = _Clock()
    monkeypatch.setattr(fallback_module.time, "monotonic", clk.monotonic)
    return clk


class TestCooldownSkip:
    def test_chain_level_failure_cools_primary_next_call_skips_it(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("status code 503")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        assert asyncio.run(chain.achat(messages=[])) == "saved"
        assert primary.calls == 1 and backup.calls == 1

        # The very next call must dispatch straight to the backup instead
        # of re-paying the primary's full retry budget.
        assert asyncio.run(chain.achat(messages=[])) == "saved"
        assert primary.calls == 1  # skipped, not re-charged
        assert backup.calls == 2

    def test_skip_notice_is_emitted(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("status code 503")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])
        notices: list[str] = []
        chain.set_retry_notice(notices.append)

        asyncio.run(chain.achat(messages=[]))
        asyncio.run(chain.achat(messages=[]))

        skips = [n for n in notices if "skipping" in n]
        assert skips, notices
        assert "cooldown" in skips[0]
        assert "primary-model" in skips[0]
        assert "backup-model" in skips[0]

    def test_achat_stream_skips_cooling_members_too(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("status code 503")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        asyncio.run(chain.achat_stream(messages=[]))
        asyncio.run(chain.achat_stream(messages=[]))
        assert primary.calls == 1
        assert backup.calls == 2


class TestCooldownExpiryAndReset:
    def test_expiry_restores_primary_half_open(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("status code 503")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        asyncio.run(chain.achat(messages=[]))
        clock.now += 61  # base window (60s) elapsed

        primary.exc = None  # recovered
        assert asyncio.run(chain.achat(messages=[])) == "ok"
        assert primary.calls == 2  # retried first again
        assert chain.served_by_primary is True

    def test_success_resets_the_streak(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("status code 503")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        asyncio.run(chain.achat(messages=[]))  # primary cools (streak 1)
        clock.now += 61
        primary.exc = None
        asyncio.run(chain.achat(messages=[]))  # success → streak reset

        primary.exc = RuntimeError("status code 503")
        asyncio.run(chain.achat(messages=[]))  # cools again (streak 1)

        remaining = chain._cooldown_remaining(primary)
        assert 59.0 <= remaining <= 60.0  # 60s window again, NOT 120s

    def test_consecutive_failures_double_the_window(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("status code 503")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        asyncio.run(chain.achat(messages=[]))  # streak 1 → 60s
        assert 59.0 <= chain._cooldown_remaining(primary) <= 60.0

        clock.now += 61  # expire; half-open retry
        asyncio.run(chain.achat(messages=[]))  # primary fails again → streak 2
        assert 119.0 <= chain._cooldown_remaining(primary) <= 120.0

        clock.now += 121  # expire; third failure → streak 3
        asyncio.run(chain.achat(messages=[]))
        assert 239.0 <= chain._cooldown_remaining(primary) <= 240.0

    def test_window_caps_at_600s(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("status code 503")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        for _ in range(12):  # streak would reach 4096s uncapped
            asyncio.run(chain.achat(messages=[]))
            clock.now += chain._cooldown_remaining(primary) + 1
        # Last failure: streak 12 → 60*2^11 = 122880 → capped at 600.
        assert chain._cooldown_remaining(primary) <= 600.0

    def test_huge_streak_never_overflows_the_exponent(self, clock):
        """Codex PR-63 r1: a provider down for weeks of capped half-open
        probes keeps growing its streak. 60.0 * 2**streak overflows the
        float BEFORE ``min()`` could apply the 600s cap — the cooldown
        itself must clamp the exponent first, so a provider failure can
        never surface as an OverflowError."""
        primary = _Member("primary-model")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        chain._cooldown_streak[id(primary)] = 5000
        chain._start_cooldown(primary, RuntimeError("status code 503"))
        # Window == the 600s cap exactly (60 * 2**10 already exceeds it).
        assert chain._cooldown_remaining(primary) == 600.0


class TestRetryAfterExtension:
    def test_retry_after_extends_the_window(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("HTTP 503 (retry-after: 300)")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        asyncio.run(chain.achat(messages=[]))

        remaining = chain._cooldown_remaining(primary)
        # max(exponential 60, Retry-After 300) = 300 (hint itself capped).
        assert 295.0 <= remaining <= 300.0

    def test_smaller_retry_after_does_not_shrink_the_window(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("retry-after: 5")
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        asyncio.run(chain.achat(messages=[]))
        assert 59.0 <= chain._cooldown_remaining(primary) <= 60.0

    def test_ms_unit_retry_after_is_respected(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("Retry-After-Ms: 90000")  # 90s
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        asyncio.run(chain.achat(messages=[]))
        assert 89.0 <= chain._cooldown_remaining(primary) <= 90.0


class TestFailOpen:
    def test_all_members_cooling_fails_open_to_normal_order(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("primary down")
        backup = _Member("backup-model")
        backup.exc = RuntimeError("backup down")
        chain = FallbackChain(primary, [backup])

        with pytest.raises(RuntimeError, match="backup down"):
            asyncio.run(chain.achat(messages=[]))
        assert primary.calls == 1 and backup.calls == 1
        # Both members cooled (the last member's chain-level failure cools
        # too) → the next call must fail OPEN to the normal order instead
        # of erroring with no candidates.
        with pytest.raises(RuntimeError, match="backup down"):
            asyncio.run(chain.achat(messages=[]))
        assert primary.calls == 2 and backup.calls == 2

        # Recovery via the fail-open tour: primary heals and is retried
        # FIRST (normal order), clearing its cooldown.
        primary.exc = None
        assert asyncio.run(chain.achat(messages=[])) == "ok"
        assert primary.calls == 3

    def test_fail_open_notice_emitted(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("primary down")
        backup = _Member("backup-model")
        backup.exc = RuntimeError("backup down")
        chain = FallbackChain(primary, [backup])
        notices: list[str] = []
        chain.set_retry_notice(notices.append)

        with pytest.raises(RuntimeError):
            asyncio.run(chain.achat(messages=[]))
        with pytest.raises(RuntimeError):
            asyncio.run(chain.achat(messages=[]))
        assert any("failing open" in n for n in notices)

    def test_fail_open_notice_fires_once_per_episode(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("primary down")
        backup = _Member("backup-model")
        backup.exc = RuntimeError("backup down")
        chain = FallbackChain(primary, [backup])
        notices: list[str] = []
        chain.set_retry_notice(notices.append)

        # Normal dispatch: both fail → both cool (streak 1, 60s windows).
        with pytest.raises(RuntimeError):
            asyncio.run(chain.achat(messages=[]))
        # Two consecutive all-cooling tours: the announcement fires ONCE.
        with pytest.raises(RuntimeError):
            asyncio.run(chain.achat(messages=[]))
        with pytest.raises(RuntimeError):
            asyncio.run(chain.achat(messages=[]))
        assert len([n for n in notices if "failing open" in n]) == 1

        # Cooldowns expire → a normal dispatch re-arms the announcement;
        # the members fail again (streak 2) and the NEXT all-cooling
        # dispatch announces itself again.
        clock.now += 61
        with pytest.raises(RuntimeError):
            asyncio.run(chain.achat(messages=[]))
        with pytest.raises(RuntimeError):
            asyncio.run(chain.achat(messages=[]))
        assert len([n for n in notices if "failing open" in n]) == 2

    def test_fail_open_tour_does_not_amplify_streaks(self, clock):
        primary = _Member("primary-model")
        primary.exc = RuntimeError("primary down")
        backup = _Member("backup-model")
        backup.exc = RuntimeError("backup down")
        chain = FallbackChain(primary, [backup])

        # Normal dispatch: streak 1, 60s window for each member.
        with pytest.raises(RuntimeError):
            asyncio.run(chain.achat(messages=[]))

        # Fail-open tours: each failure refreshes the window from the
        # EXISTING streak but must not increment it — a total outage would
        # otherwise grow every window per call toward the 600s cap and
        # mislead the logs about how long the provider has been down.
        for _ in range(5):
            with pytest.raises(RuntimeError):
                asyncio.run(chain.achat(messages=[]))

        assert chain._cooldown_streak[id(primary)] == 1
        assert chain._cooldown_streak[id(backup)] == 1
        # Window stays the 60s the existing streak sizes — not doubled,
        # not capped at 600s by call-volume amplification.
        assert 59.0 <= chain._cooldown_remaining(primary) <= 60.0
        assert 59.0 <= chain._cooldown_remaining(backup) <= 60.0


class TestNoCooldownPaths:
    def test_cancellation_never_cools(self, clock):
        primary = _Member("primary-model")
        primary.exc = asyncio.CancelledError()
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(chain.achat(messages=[]))
        assert backup.calls == 0  # never failed over
        assert chain._cooldown_remaining(primary) == 0.0  # and never cooled

        # The next call still dispatches to the primary first.
        primary.exc = None
        assert asyncio.run(chain.achat(messages=[])) == "ok"
        assert primary.calls == 2

    def test_cancellation_while_fallback_serves_keeps_cooldowns(self):
        """Primary already cooling, the backup is mid-dispatch when the
        task is cancelled: the primary's cooldown is untouched and the
        backup never enters one — cancellation is not provider failure
        (pinning the existing never-cool-on-cancel semantics).

        No ``clock`` fixture: it freezes ``time.monotonic`` globally,
        which also freezes the event loop's own clock and would hang
        ``asyncio.sleep`` forever. Real time keeps the 60s window at
        ~59-60s remaining for the whole (sub-second) scenario.
        """

        class _HangingBackup:
            model_name = "backup-model"
            provider_name = "openai"
            spend_ledger = None
            calls = 0

            async def achat(self, **kwargs):
                self.calls += 1
                await asyncio.sleep(3600)

        primary = _Member("primary-model")
        primary.exc = RuntimeError("status code 503")
        backup = _HangingBackup()
        chain = FallbackChain(primary, [backup])

        async def _cancelled_dispatch():
            task = asyncio.ensure_future(chain.achat(messages=[]))
            await asyncio.sleep(0.05)  # primary failed + cooled; backup serving
            task.cancel()
            await task

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(_cancelled_dispatch())

        assert backup.calls == 1  # the dispatch really reached the backup
        assert chain._cooldown_remaining(backup) == 0.0  # cancelled ≠ failed
        # Primary's own earlier cooldown is neither extended nor cleared.
        assert 59.0 <= chain._cooldown_remaining(primary) <= 60.0
        assert chain._cooldown_streak[id(primary)] == 1

    def test_spend_ledger_collapse_is_unaffected(self, clock, tmp_path):
        from clearwing.llm.budget import SpendLedger

        ledger = SpendLedger(
            limit_usd=0.0,
            session_id="cooldown-collapse",
            repo_url="/tmp/repo",
            output_dir=tmp_path,
            input_price_per_million=0.0,
            output_price_per_million=1.0,
        )

        class _RealishClient(_Member):
            def __init__(self):
                super().__init__("primary-model")
                self._ledger = None

            @property
            def spend_ledger(self):
                return self._ledger

            @spend_ledger.setter
            def spend_ledger(self, value):
                self._ledger = value

            def with_spend_ledger(self, ledger, *, stage):
                self.spend_ledger = ledger
                return self

        primary = _RealishClient()
        primary.exc = RuntimeError("status code 503")
        chain = FallbackChain(primary, [])
        collapsed = chain.with_spend_ledger(ledger, stage="test")
        assert collapsed.fallbacks == []

        notices: list[str] = []
        collapsed.set_retry_notice(notices.append)

        with pytest.raises(RuntimeError, match="503"):
            asyncio.run(collapsed.achat(messages=[]))
        with pytest.raises(RuntimeError, match="503"):
            asyncio.run(collapsed.achat(messages=[]))
        # The chain does not act on a collapsed (single-member) chain: no
        # cooldown bookkeeping, no skip/fail-open chatter.
        assert collapsed._cooldown_remaining(primary) == 0.0
        assert notices == []


class TestConsumerCallbackExceptions:
    """Codex PR-63 r1: a consumer-side exception — the CALLER's
    on_text_delta raising while the member streams — is not a provider
    failure. It must propagate to the caller verbatim, with no cooldown
    on the member that was serving and no failover (re-dispatching would
    double-serve a call whose partial output the consumer already saw)."""

    def _streaming_primary(self):
        class _StreamingPrimary:
            model_name = "primary-model"
            provider_name = "openai"
            spend_ledger = None
            calls = 0

            async def achat_stream(self, **kwargs):
                self.calls += 1
                callback = kwargs.get("on_text_delta")
                if callback is not None:
                    callback("live delta")
                return SimpleNamespace(first_text="answer", texts=["answer"])

        return _StreamingPrimary()

    def test_callback_exception_propagates_without_cooldown_or_failover(self, clock):
        primary = self._streaming_primary()
        backup = _Member("backup-model", response="saved")
        chain = FallbackChain(primary, [backup])

        def _boom(text: str) -> None:
            raise RuntimeError(f"consumer exploded on {text!r}")

        # (a) the caller's own exception surfaces unchanged.
        with pytest.raises(RuntimeError, match="consumer exploded on 'live delta'"):
            asyncio.run(chain.achat_stream(messages=[], on_text_delta=_boom))

        # (c) the chain never treated it as a provider failure, so no
        # failover: the fallback member was never dispatched.
        assert primary.calls == 1
        assert backup.calls == 0
        # (b) the primary entered no cooldown and keeps its place at the
        # head of the dispatch order.
        assert chain._cooldown_remaining(primary) == 0.0
        assert id(primary) not in chain._cooldown_streak

        # The next call still starts from the primary — healthy as far as
        # the chain knows.
        result = asyncio.run(chain.achat_stream(messages=[]))
        assert result.first_text == "answer"
        assert primary.calls == 2
        assert backup.calls == 0
        assert chain.served_by_primary is True

    def test_suppressed_reemission_exception_propagates_without_cooling_server(
        self, clock
    ):
        """The other consumer-side raise point: after a mid-stream failover
        (deltas already delivered), the serving member's full answer is
        re-emitted through the ORIGINAL callback — a raising consumer must
        not cool the member that served nor move the chain further."""
        emitted: list[str] = []

        class _FailMidStream:
            model_name = "primary-model"
            provider_name = "openai"
            spend_ledger = None

            async def achat_stream(self, **kwargs):
                callback = kwargs.get("on_text_delta")
                if callback is not None:
                    callback("partial")
                raise RuntimeError("stream died")

        class _ServingBackup:
            model_name = "backup-model"
            provider_name = "openai"
            spend_ledger = None
            calls = 0

            async def achat_stream(self, **kwargs):
                self.calls += 1
                assert kwargs.get("on_text_delta") is None  # suppressed
                return SimpleNamespace(first_text="full answer", texts=["full answer"])

        primary = _FailMidStream()
        backup = _ServingBackup()
        chain = FallbackChain(primary, [backup])

        def _accept_then_boom(text: str) -> None:
            emitted.append(text)
            if text == "full answer":  # the suppressed re-emission
                raise RuntimeError("consumer exploded on the full answer")

        with pytest.raises(RuntimeError, match="consumer exploded"):
            asyncio.run(chain.achat_stream(messages=[], on_text_delta=_accept_then_boom))

        # The primary's death was a genuine provider failure (it cools),
        # but the member that SERVED must not: no cooldown, no streak.
        assert backup.calls == 1
        assert chain._cooldown_remaining(backup) == 0.0
        assert id(backup) not in chain._cooldown_streak
        # The consumer saw the partial delta and the re-emission attempt.
        assert emitted == ["partial", "full answer"]


class TestMidDispatchCooldownRevival:
    """Codex PR-63 r2: a member SKIPPED at dispatch time whose cooldown
    EXPIRES while the dispatched members are still retrying must be
    re-dispatched at the tail instead of failing the call — the eligibility
    snapshot at call start is stale by the time the chain would give up.

    No ``clock`` fixture in this class: it monkeypatches the ``time``
    module's monotonic globally, which also freezes the event loop's clock
    (``asyncio.sleep`` would never advance) — and a frozen monotonic makes
    the injected cooldown NEVER expire. Real (sub-second) time only."""

    def _slow_failing_backup(self):
        class _SlowFailingBackup(_Member):
            # 0.35s in the member's own retry loop — past the primary's
            # 0.2s remaining cooldown at dispatch time.
            async def achat(self, **kwargs):
                self.calls += 1
                await asyncio.sleep(0.35)
                raise RuntimeError("backup still down")

        return _SlowFailingBackup("backup-model", response="unused")

    def test_achat_revives_member_cooled_during_dispatch(self):
        primary = _Member("primary-model", response="primary-answer")
        backup = self._slow_failing_backup()
        chain = FallbackChain(primary, [backup])
        # Primary cooling at dispatch time (0.2s left) → skipped.
        chain._cooldown_until[id(primary)] = time.monotonic() + 0.2

        started = time.monotonic()
        result = asyncio.run(chain.achat(messages=[]))

        # (a) The call SUCCEEDS via the revived primary — no raise, even
        # though the backup (the only dispatch-order member) failed.
        assert result == "primary-answer"
        # (b) The backup really was dispatched (and burned 0.35s).
        assert backup.calls == 1
        assert time.monotonic() - started >= 0.3
        assert primary.calls == 1
        assert chain.served_by_primary is True

    def test_achat_stream_revives_member_cooled_during_dispatch(self):
        primary = _Member("primary-model", response="primary-answer")
        backup = self._slow_failing_backup()
        chain = FallbackChain(primary, [backup])
        chain._cooldown_until[id(primary)] = time.monotonic() + 0.2

        emitted: list[str] = []
        result = asyncio.run(chain.achat_stream(messages=[], on_text_delta=emitted.append))

        assert result == "primary-answer"
        assert backup.calls == 1
        assert primary.calls == 1
        # No member streamed deltas here, so the revived primary (a later
        # index) was never wrongly suppressed.
        assert emitted == []
        assert chain.served_by_primary is True

    def test_unexpired_cooldown_is_not_awaited(self):
        # Reverse guard: a cooldown still well within its window must NOT
        # be slept on — the chain raises promptly instead of blocking (or
        # looping forever) waiting for the skipped member to recover.
        primary = _Member("primary-model", response="primary-answer")
        backup = _Member("backup-model", response="unused")
        backup.exc = RuntimeError("backup down")
        chain = FallbackChain(primary, [backup])
        chain._cooldown_until[id(primary)] = time.monotonic() + 60.0

        started = time.monotonic()
        with pytest.raises(RuntimeError, match="backup down"):
            asyncio.run(chain.achat(messages=[]))

        assert time.monotonic() - started < 5.0  # no cooldown waiting
        assert backup.calls == 1
        assert primary.calls == 0  # never dispatched while cooling


class TestStreamPolicyUnchanged:
    def test_suppression_survives_cooldown_reordering(self, clock):
        """Primary cooling → the backup streams first; when IT dies
        mid-stream the next member's deltas are suppressed and its full
        answer is emitted once (Codex PR-55 r4/r5 policy, unchanged)."""

        class _PartialThenFail:
            model_name = "backup-model"
            provider_name = "openai"
            spend_ledger = None

            async def achat_stream(self, **kwargs):
                callback = kwargs.get("on_text_delta")
                if callback is not None:
                    callback("partial backup text")
                raise RuntimeError("stream died")

        class _Third:
            model_name = "third-model"
            provider_name = "openai"
            spend_ledger = None

            async def achat_stream(self, **kwargs):
                callback = kwargs.get("on_text_delta")
                if callback is not None:
                    callback("third text")
                return SimpleNamespace(first_text="third answer", texts=["third answer"])

        primary = _Member("primary-model")
        primary.exc = RuntimeError("status code 503")
        chain = FallbackChain(primary, [_PartialThenFail(), _Third()])
        notices: list[str] = []
        chain.set_retry_notice(notices.append)
        emitted: list[str] = []

        # Call 1: primary fails (cools), backup dies mid-stream, third serves.
        result = asyncio.run(
            chain.achat_stream(messages=[], on_text_delta=emitted.append)
        )
        assert result.first_text == "third answer"
        assert emitted == ["partial backup text", "third answer"]
        assert any("abandoned" in n for n in notices)

        # Call 2: primary is skipped (cooldown), backup streams first and
        # succeeds — deltas flow live, exactly as a healthy first member.
        emitted.clear()

        class _FixedBackup(_PartialThenFail):
            async def achat_stream(self, **kwargs):
                callback = kwargs.get("on_text_delta")
                if callback is not None:
                    callback("whole answer")
                return SimpleNamespace(first_text="whole answer", texts=["whole answer"])

        chain2 = FallbackChain(primary, [_FixedBackup(), _Third()])
        result = asyncio.run(chain2.achat_stream(messages=[], on_text_delta=emitted.append))
        assert result.first_text == "whole answer"
        assert emitted == ["whole answer"]
        assert chain2.served_by_primary is False
        assert chain2.served_model_name == "backup-model"
