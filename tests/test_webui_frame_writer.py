"""Issue #45/#48: the webui's single outbound writer.

Every frame — bus events, terminal frames, heartbeats — leaves through one
FIFO queue and one writer coroutine:

- a frame enqueued before `complete` can never be delivered after it;
- an unserializable frame is dropped at ENQUEUE (logged), never queued as
  an immortal send that stalls every later frame;
- the #53 dedup gate still keys on DELIVERED agent-echo frames only;
- transcript recording is independent of the frame gate;
- out-of-turn cost frames carry session-scoped totals (#48-1), never the
  emitter's process-global ones;
- the session-cost accumulator is lock-protected against worker threads
  (#48-2);
- a late-booked call (arriving after the turn ended) re-bases the
  accumulator to the tracker's water level, so the next turn's in-turn
  frames never regress below it (Codex PR-62 r2);
- cost frames' accumulate→publish sequence is serialized under the
  session lock: the wire totals arrive in monotonically non-decreasing
  order (Codex PR-62 r2);
- totals are tracker-authoritative in EVERY turn window (Codex PR-62
  r3): the runtime books before emitting, so assigning from the session
  snapshot cannot double-count a handler that straddles a turn
  boundary;
- a terminal frame whose flush handshake was cancelled by `stop` is
  skipped by the writer — the client's first terminal frame is the
  authoritative `stopped`, never the cancelled turn's stale
  agent_message/complete (Codex PR-62 r3);
- a terminal frame gets exactly ONE send attempt (Codex PR-62 r4): a
  failed attempt discards the entry, so `_send_frame`'s False is a hard
  guarantee the frame never reaches the wire — a transport recovering
  after the failure cannot deliver a stale terminal frame;
- a failed send no longer sweeps pending flush handshakes (Codex PR-62
  r4): a waiter queued behind a stalled BUS frame stays pending until
  the transport recovers (the FIFO delivers everything and the turn ends
  with its `complete`) or no progress is possible at all (writer death,
  dead socket);
- an in-turn cost frame with NO session_id keeps the legacy
  accumulation on session-bound connections too (Codex PR-62 r4): its
  spend was never booked in the tracker, so the snapshot rewrite would
  silently drop it from the displayed totals — best-effort display, the
  tracker stays authoritative.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from clearwing.ui.web.app import create_app  # noqa: E402

API_KEY = "test-key-8f3a"
AUTH = {"X-API-Key": API_KEY}


def _receive_json_with_timeout(ws, *, timeout: float = 10.0) -> dict:
    """receive_json with a hard wall-clock cap (Codex PR-62 r1).

    A frame-count cap cannot advance past a blocking receive: when a
    regression means the expected frame never arrives at all, the first
    receive blocks forever and the suite hangs instead of failing. The
    receive runs on a worker thread. On timeout, fail IMMEDIATELY with a
    non-blocking pool shutdown: the worker only unblocks when the test's
    `websocket_connect` context exits (its ExitStack cancels the session
    task, which closes the receive stream) — ws.close() alone does NOT
    unblock it (starlette parks the session task after teardown), so
    waiting for the worker here would deadlock before the failure is
    even reported."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(ws.receive_json)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)
        pytest.fail(f"no frame arrived within {timeout}s")
    finally:
        pool.shutdown(wait=False)



@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("CLEARWING_WEB_API_KEY", API_KEY)
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    import clearwing.ui.web.session_report as session_report

    root = tmp_path / "results"
    monkeypatch.setattr(session_report, "default_results_dir", lambda sub: root / sub)
    return root


class _AI:
    def __init__(self, content):
        self.type = "ai"
        self.content = content


def _emit_message(content: str, msg_type: str = "agent") -> None:
    from clearwing.core.events import EventBus, EventType

    EventBus().emit(EventType.MESSAGE, {"content": content, "type": msg_type})


def _emit_tool_start(name: str, args) -> None:
    from clearwing.core.events import EventBus, EventType

    EventBus().emit(EventType.TOOL_START, {"tool": name, "args": args})


def _emit_cost(payload: dict) -> None:
    from clearwing.core.events import EventBus, EventType

    EventBus().emit(EventType.COST_UPDATE, payload)


class _EchoThenFinishGraph:
    """astream (running on the server loop) echoes the assistant text on
    the bus via call_soon_threadsafe, then yields the AI state event — the
    mirror-order shape where the enqueue races the turn's dedup gate."""

    def __init__(self, text, echo_type="agent"):
        self.text = text
        self.echo_type = echo_type

    def get_state(self, config):
        del config
        return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

    async def astream(self, input_msg, config, stream_mode="values"):
        del input_msg, config, stream_mode
        _emit_message(self.text[:200], self.echo_type)
        yield {"messages": [_AI(self.text)]}


class _LateThreadEmitGraph:
    """Cross-thread flavor of the #45 ordering race.

    The message turn's final stretch is: flush the writer, maybe send the
    inline agent_message, then — in a NO-AWAIT window — ask
    ``_graph_has_pending`` (``graph.get_state``) and enqueue the terminal
    ``complete``. This graph emits a bus event from a WORKER thread on the
    first get_state call after astream ran (i.e. from inside that window,
    by way of ``_turn_end_status``) and blocks the loop until the emit is
    scheduled: the call_soon_threadsafe enqueue sits in the ready queue,
    unrun, while the turn coroutine proceeds toward ``complete``. The
    single writer must still deliver that frame BEFORE ``complete`` —
    which is exactly what ``_send_frame``'s leading ``sleep(0)``
    guarantees."""

    def __init__(self):
        self._state = SimpleNamespace(values={"messages": []}, next=(), tasks=[])
        self._astream_done = False
        self._emitted = False

    def get_state(self, config):
        del config
        if self._astream_done and not self._emitted:
            self._emitted = True
            done = threading.Event()

            def _emit():
                _emit_tool_start("late_probe", {})
                done.set()

            threading.Thread(target=_emit, daemon=True).start()
            done.wait(timeout=5)
        return self._state

    async def astream(self, input_msg, config, stream_mode="values"):
        del input_msg, config, stream_mode
        yield {"messages": [_AI("done")]}
        self._astream_done = True


class TestFrameOrdering:
    def test_echo_precedes_complete_and_nothing_follows_it(self, client):
        """#45: the echo is enqueued (via call_soon_threadsafe) moments
        before the turn sends its terminal frames — it must be DELIVERED
        before complete, and no frame may arrive after complete."""
        text = "short answer"  # <= 200 chars: echo == full text
        with patch(
            "clearwing.ui.web.app.create_agent", lambda **kwargs: _EchoThenFinishGraph(text)
        ):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                assert ws.receive_json()["type"] == "started"
                ws.send_json({"type": "message", "content": "run"})

                frames = []
                while True:
                    frame = ws.receive_json()
                    frames.append(frame)
                    if frame["type"] == "complete":
                        break

                # Probe for late frames: the very next server frames must
                # be the stop handshake — a late echo would interleave here.
                ws.send_json({"type": "stop"})
                after1 = ws.receive_json()
                after2 = ws.receive_json()

        types = [f["type"] for f in frames]
        assert types[-1] == "complete"
        # The queued echo was delivered BEFORE complete (FIFO, single
        # writer), and the dedup gate (byte-equality on delivered agent
        # echoes) suppressed the redundant inline copy.
        assert types.count("agent_message") == 1
        assert frames[-2]["data"]["content"] == text
        assert after1["type"] == "stopped"
        assert after2["type"] == "complete"
        assert after2["data"]["status"] == "stopped"

    def test_worker_thread_emit_in_terminal_window_precedes_complete(self, client):
        """#45, cross-thread flavor: a bus event scheduled (but not yet
        run) while the turn coroutine is inside the no-await window before
        the `complete` enqueue must still be delivered BEFORE complete —
        _send_frame yields once so the scheduled enqueue lands in the FIFO
        first."""
        with patch(
            "clearwing.ui.web.app.create_agent",
            lambda **kwargs: _LateThreadEmitGraph(),
        ):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                assert ws.receive_json()["type"] == "started"
                ws.send_json({"type": "message", "content": "run"})

                frames = []
                while True:
                    frame = ws.receive_json()
                    frames.append(frame)
                    if frame["type"] == "complete":
                        break

        types = [f["type"] for f in frames]
        assert types[-1] == "complete"
        assert "tool_start" in types
        # The worker-thread emission raced the complete enqueue from
        # inside get_state — it must not land after the terminal frame.
        assert types.index("tool_start") < types.index("complete")


class TestUnserializableFrameDrop:
    def test_bad_frame_dropped_and_later_frames_still_flow(self, client, caplog):
        """An unserializable payload must be dropped at enqueue (logged),
        not queued as an immortal frame that stalls the queue."""

        class _BadThenGoodGraph:
            def get_state(self, config):
                del config
                return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

            async def astream(self, input_msg, config, stream_mode="values"):
                del input_msg, config, stream_mode
                from clearwing.core.events import EventBus, EventType

                # dict passes the shallow type check, but the nested set
                # only fails at json.dumps — the exact class #45 kills.
                EventBus().emit(
                    EventType.MESSAGE, {"content": {"bad": {1, 2}}, "type": "info"}
                )
                _emit_message("good frame", msg_type="info")
                yield {"messages": [_AI("done")]}

        with patch("clearwing.ui.web.app.create_agent", lambda **kwargs: _BadThenGoodGraph()):
            with caplog.at_level("ERROR", logger="clearwing.ui.web.app"):
                with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                    ws.send_json({"type": "start", "target": "10.0.0.9"})
                    assert ws.receive_json()["type"] == "started"
                    ws.send_json({"type": "message", "content": "run"})
                    frames = []
                    while True:
                        frame = ws.receive_json()
                        frames.append(frame)
                        if frame["type"] == "complete":
                            break

        # The poison frame was dropped with an error log …
        assert any(
            "Dropping unserializable" in r.getMessage() for r in caplog.records
        )
        contents = [
            f["data"].get("content")
            for f in frames
            if f["type"] == "agent_message"
        ]
        # … the later bus frame still went out (no stall), …
        assert "good frame" in contents
        # … and the poison content itself never reached the client.
        assert not any(
            isinstance(c, dict) and "bad" in c for c in contents
        )
        assert frames[-1]["type"] == "complete"

    def test_report_transcript_records_dropped_poison_and_good_tool(
        self, client, results_dir
    ):
        """Transcript recording is unconditional: the dropped
        (unserializable) tool STILL lands in the report in readable form
        (the report render str()s what json cannot encode), and the good
        one does too — in emission order."""

        class _ToolGraph:
            def get_state(self, config):
                del config
                return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

            async def astream(self, input_msg, config, stream_mode="values"):
                del input_msg, config, stream_mode
                _emit_tool_start("poison_tool", {"args": {1, 2}})  # set: dropped
                _emit_tool_start("good_tool", {"args": {"port": 22}})
                yield {"messages": [_AI("done")]}

        with patch("clearwing.ui.web.app.create_agent", lambda **kwargs: _ToolGraph()):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                assert ws.receive_json()["type"] == "started"
                ws.send_json({"type": "message", "content": "run"})
                while True:
                    frame = ws.receive_json()
                    if frame["type"] == "complete":
                        report_url = frame["data"]["report_url"]
                        break

        report = client.get(report_url, headers=AUTH).text
        assert "good_tool" in report
        # The poison frame never reached the socket, but the transcript
        # records it anyway — dropping it would also mis-stick the NEXT
        # tool_result's content_length onto the previous tool entry.
        assert "poison_tool" in report
        assert report.index("poison_tool") < report.index("good_tool")


class TestDedupGateWithSingleWriter:
    def test_agent_echo_suppresses_inline_copy(self, client):
        text = "即將執行需審批的掃描"  # <=200 chars: echo == full text
        with patch(
            "clearwing.ui.web.app.create_agent",
            lambda **kwargs: _EchoThenFinishGraph(text),
        ):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                frames = []
                while True:
                    frame = ws.receive_json()
                    frames.append(frame)
                    if frame["type"] == "complete":
                        break

        contents = [
            f["data"]["content"]
            for f in frames
            if f["type"] == "agent_message"
        ]
        assert contents.count(text) == 1  # delivered echo, no inline repeat

    def test_system_echo_is_never_deduped(self, client):
        """The gate keys on data.type == "agent" — a system/warning note
        with byte-identical content must NOT suppress the authoritative
        inline send."""
        text = "context note that matches the reply"
        with patch(
            "clearwing.ui.web.app.create_agent",
            lambda **kwargs: _EchoThenFinishGraph(text, echo_type="system"),
        ):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                frames = []
                while True:
                    frame = ws.receive_json()
                    frames.append(frame)
                    if frame["type"] == "complete":
                        break

        contents = [
            f["data"]["content"]
            for f in frames
            if f["type"] == "agent_message"
        ]
        assert contents.count(text) == 2  # echo AND authoritative inline

    def test_deduped_text_still_lands_in_report(self, client, results_dir):
        text = "dedup 但報告必須記錄"
        with patch(
            "clearwing.ui.web.app.create_agent",
            lambda **kwargs: _EchoThenFinishGraph(text),
        ):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                while True:
                    frame = ws.receive_json()
                    if frame["type"] == "complete":
                        report_url = frame["data"]["report_url"]
                        break

        report = client.get(report_url, headers=AUTH).text
        assert report.count(text) == 1


class _TrackerBookingGraph:
    """Books real tracker spend under the session id, then emits an
    in-turn cost frame (the runtime shape)."""

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")

    def get_state(self, config):
        del config
        return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

    async def astream(self, input_msg, config, stream_mode="values"):
        from clearwing.observability.telemetry import CostTracker

        del input_msg, config, stream_mode
        CostTracker().record_llm_call(1000, 100, "fake-model", session_id=self.session_id)
        _emit_cost(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cached_tokens": 0,
                "cost": 0.001,
                # The emitter's PROCESS-GLOBAL running total — the lie the
                # adapter must rewrite (#10/#48).
                "total_cost_usd": 999.0,
                "total_tokens": 999000,
                "model": "fake-model",
                "provider": "openai",
                "session_id": self.session_id,
                "elapsed_ms": 1,
            }
        )
        yield {"messages": [_AI("done")]}


class TestOutOfTurnCostScoping:
    """#48-1: out-of-turn matching frames are forwarded with THIS session's
    tracker totals; foreign frames are dropped."""

    def test_out_of_turn_frame_carries_session_totals_not_global(self, client):
        from clearwing.observability.telemetry import CostTracker

        with patch("clearwing.ui.web.app.create_agent", _TrackerBookingGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                sid = None
                while True:
                    frame = json.loads(ws.receive_text())
                    if frame["type"] == "complete":
                        sid = frame["data"]["session_id"]
                        break

                tracker = CostTracker()
                session_cost = tracker.session_total(sid)
                session_in, session_out = tracker.session_tokens(sid)
                assert session_cost > 0.0

                # Turn is over: emit from OUTSIDE any turn (this test
                # thread = the worker-thread shape) with the emitter's
                # process-global totals, attributed to this session.
                _emit_cost(
                    {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "cost": 0.002,
                        "total_cost_usd": 999.0,
                        "total_tokens": 999000,
                        "model": "fake-model",
                        "provider": "openai",
                        "session_id": sid,
                        "elapsed_ms": 0,
                    }
                )
                out_of_turn = None
                # Time- AND count-bounded receive: if the adapter ever
                # stops forwarding the frame (or nothing arrives at all),
                # fail fast instead of parking on a blocking receive.
                for _ in range(20):
                    frame = _receive_json_with_timeout(ws, timeout=10.0)
                    if frame["type"] == "cost_update":
                        out_of_turn = frame
                        break
                if out_of_turn is None:
                    pytest.fail("no out-of-turn cost_update within 20 frames")

                # Forwarded, but with session-scoped totals — never the
                # emitter's 999.0 process-global numbers.
                assert out_of_turn["data"]["total_cost_usd"] == pytest.approx(
                    session_cost
                )
                assert out_of_turn["data"]["total_tokens"] == session_in + session_out
                assert out_of_turn["data"]["total_cost_usd"] != pytest.approx(999.0)

                # A foreign out-of-turn frame never reaches this socket:
                # the next frames are the stop handshake, nothing else.
                _emit_cost(
                    {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cost": 0.5,
                        "session_id": "someone-else",
                    }
                )
                ws.send_json({"type": "stop"})
                assert ws.receive_json()["type"] == "stopped"
                assert ws.receive_json()["type"] == "complete"


class TestSessionCostThreadSafety:
    """#48-2 + Codex PR-62 r3: cost frames are written from worker
    threads; the lock serializes snapshot-read and publish, and the
    tracker-authoritative snapshot makes the delivered totals exact under
    concurrency — the final frame equals the booked total, no lost or
    double-counted call."""

    def test_concurrent_worker_emissions_accumulate_exactly(self, client):
        from clearwing.observability.telemetry import CostTracker

        threads_n, per_thread = 8, 50
        per_call = CostTracker.estimate_cost(1, 0, "fake-model")

        class _HammerGraph:
            def __init__(self, **kwargs):
                self.session_id = kwargs.get("session_id")

            def get_state(self, config):
                del config
                return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

            async def astream(self, input_msg, config, stream_mode="values"):
                from clearwing.observability.telemetry import CostTracker

                del input_msg, config, stream_mode

                # The runtime shape (Codex PR-62 r3): each worker books a
                # real tracker call under the session id — record_llm_call
                # books FIRST and then emits the frame itself.
                def worker():
                    for _ in range(per_thread):
                        CostTracker().record_llm_call(
                            1, 0, "fake-model", session_id=self.session_id
                        )

                workers = [
                    threading.Thread(target=worker, daemon=True)
                    for _ in range(threads_n)
                ]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join()
                yield {"messages": [_AI("done")]}

        with patch("clearwing.ui.web.app.create_agent", _HammerGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                cost_frames = []
                while True:
                    frame = json.loads(ws.receive_text())
                    if frame["type"] == "cost_update":
                        cost_frames.append(frame)
                    if frame["type"] == "complete":
                        break

        total_emissions = threads_n * per_thread
        # Every emission's frame was delivered BEFORE complete (FIFO).
        assert len(cost_frames) == total_emissions
        last = cost_frames[-1]["data"]
        assert last["total_cost_usd"] == pytest.approx(total_emissions * per_call)
        assert last["total_tokens"] == total_emissions


class _BookingGraph:
    """Books ONE real tracker call per astream under this session's id —
    ``record_llm_call`` also emits the COST_UPDATE frame carrying the
    exact booked cost, which is what makes in-turn accumulation and
    tracker totals line up exactly."""

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")

    def get_state(self, config):
        del config
        return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

    async def astream(self, input_msg, config, stream_mode="values"):
        from clearwing.observability.telemetry import CostTracker

        del input_msg, config, stream_mode
        CostTracker().record_llm_call(
            1000, 100, "fake-model", session_id=self.session_id
        )
        yield {"messages": [_AI("done")]}


class TestLateBookingRebase:
    """Codex PR-62 r2/r3: an LLM call attributed to this session that
    completes AFTER the turn ended must still be captured — every frame's
    totals come from the tracker-authoritative session snapshot, so the
    next turn's in-turn frames sit on the late-booked water level instead
    of a stale base, and the wire totals never regress below spend that
    already happened."""

    def test_next_turn_accumulates_from_late_booked_water_level(self, client):
        from clearwing.observability.telemetry import CostTracker

        call_cost = CostTracker.estimate_cost(1000, 100, "fake-model")
        late_cost = CostTracker.estimate_cost(200, 50, "fake-model")

        with patch("clearwing.ui.web.app.create_agent", _BookingGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                assert ws.receive_json()["type"] == "started"

                # Turn 1: one booked call, one in-turn cost frame.
                ws.send_json({"type": "message", "content": "run"})
                turn1 = None
                sid = None
                while True:
                    frame = json.loads(ws.receive_text())
                    if frame["type"] == "cost_update":
                        turn1 = frame["data"]
                    if frame["type"] == "complete":
                        sid = frame["data"]["session_id"]
                        break
                assert turn1 is not None
                assert turn1["total_cost_usd"] == pytest.approx(call_cost)
                assert turn1["total_tokens"] == 1100

                # Late booking: the call completes on a worker thread
                # AFTER the turn ended — the tracker books it and the
                # frame arrives out-of-turn (this test thread is the
                # worker-thread shape).
                CostTracker().record_llm_call(
                    200, 50, "fake-model", session_id=sid
                )
                late = None
                for _ in range(20):
                    frame = _receive_json_with_timeout(ws, timeout=10.0)
                    if frame["type"] == "cost_update":
                        late = frame["data"]
                        break
                if late is None:
                    pytest.fail("no late-booked cost_update within 20 frames")
                assert late["total_cost_usd"] == pytest.approx(
                    call_cost + late_cost
                )
                assert late["total_tokens"] == 1350

                # Turn 2: its in-turn frame must build on the LATE-BOOKED
                # water level — pre-fix the accumulator still held turn
                # 1's total and the wire regressed below the late frame.
                ws.send_json({"type": "message", "content": "again"})
                turn2 = None
                while True:
                    frame = json.loads(ws.receive_text())
                    if frame["type"] == "cost_update":
                        turn2 = frame["data"]
                    if frame["type"] == "complete":
                        break
                assert turn2 is not None
                assert turn2["total_cost_usd"] >= late["total_cost_usd"]
                expected = 2 * call_cost + late_cost
                assert turn2["total_cost_usd"] == pytest.approx(expected)
                assert turn2["total_tokens"] == 2450
                # The final water level equals the tracker's authoritative
                # session total: two turns' calls plus the late booking.
                assert turn2["total_cost_usd"] == pytest.approx(
                    CostTracker().session_total(sid)
                )

                # Clean stop handshake — also proves no stray cost frame
                # was still in flight after turn 2.
                ws.send_json({"type": "stop"})
                assert ws.receive_json()["type"] == "stopped"
                assert ws.receive_json()["type"] == "complete"


class TestTrackerAuthoritativeTotals:
    """Codex PR-62 r3: the tracker's session snapshot is authoritative in
    EVERY turn window. An in-turn frame's own per-call cost/token fields
    are informational and are never ADDED into the running totals — a
    handler delayed across a turn boundary could otherwise double-count a
    call an out-of-turn frame had already captured into the snapshot."""

    def test_in_turn_frame_totals_come_from_tracker_not_frame_fields(self, client):
        from clearwing.observability.telemetry import CostTracker

        real_call = CostTracker.estimate_cost(500, 50, "fake-model")

        class _LyingFrameGraph:
            def __init__(self, **kwargs):
                self.session_id = kwargs.get("session_id")

            def get_state(self, config):
                del config
                return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

            async def astream(self, input_msg, config, stream_mode="values"):
                from clearwing.observability.telemetry import CostTracker

                del input_msg, config, stream_mode
                # One real, booked call (the runtime shape: book first,
                # the frame is emitted by record_llm_call itself)…
                CostTracker().record_llm_call(
                    500, 50, "fake-model", session_id=self.session_id
                )
                # …then an emission whose per-call fields LIE (never booked
                # in the tracker): pre-fix the in-turn branch +=-ed them
                # into the running totals; the snapshot must ignore them.
                _emit_cost(
                    {
                        "input_tokens": 1_000_000,
                        "output_tokens": 1_000_000,
                        "cost": 999.0,
                        "total_cost_usd": 999.0,
                        "total_tokens": 999,
                        "model": "fake-model",
                        "provider": "openai",
                        "session_id": self.session_id,
                        "elapsed_ms": 0,
                    }
                )
                yield {"messages": [_AI("done")]}

        with patch("clearwing.ui.web.app.create_agent", _LyingFrameGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                frames = []
                while True:
                    frame = json.loads(ws.receive_text())
                    if frame["type"] == "cost_update":
                        frames.append(frame)
                    if frame["type"] == "complete":
                        break

        assert len(frames) == 2
        booked, lying = (f["data"] for f in frames)
        assert booked["total_cost_usd"] == pytest.approx(real_call)
        assert booked["total_tokens"] == 550
        # The lying frame's totals are rewritten to the SAME tracker water
        # level — its own cost/token fields never enter the totals.
        assert lying["total_cost_usd"] == pytest.approx(real_call)
        assert lying["total_tokens"] == 550
        assert lying["total_cost_usd"] != pytest.approx(real_call + 999.0)
        # Per-call fields still ride along untouched.
        assert lying["cost"] == 999.0


class TestCostFramePublishOrdering:
    """Codex PR-62 r2 (Finding 2): snapshot-read and publish are ONE
    serialized sequence under the session lock — the enqueue (wire) order
    of concurrent cost frames matches their snapshot order, so the
    delivered totals sequence never regresses."""

    def test_concurrent_cost_frames_wire_totals_monotonic(self, client):
        import sys

        from clearwing.observability.telemetry import CostTracker

        threads_n, per_thread = 8, 50

        class _ConcurrentCostGraph:
            def __init__(self, **kwargs):
                self.session_id = kwargs.get("session_id")

            def get_state(self, config):
                del config
                return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

            async def astream(self, input_msg, config, stream_mode="values"):
                from clearwing.observability.telemetry import CostTracker

                del input_msg, config, stream_mode

                # The runtime shape (Codex PR-62 r3): each worker books a
                # real tracker call (book first, emit after). Distinct
                # per-thread token counts widen any out-of-order dip the
                # test could observe.
                def worker(tokens):
                    for _ in range(per_thread):
                        CostTracker().record_llm_call(
                            tokens, 0, "fake-model", session_id=self.session_id
                        )

                workers = [
                    threading.Thread(target=worker, args=(t + 1,), daemon=True)
                    for t in range(threads_n)
                ]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join()
                yield {"messages": [_AI("done")]}

        with patch("clearwing.ui.web.app.create_agent", _ConcurrentCostGraph):
            # Shrink the GIL switch interval: the pre-fix race window
            # (lock released after the snapshot read, enqueue scheduled only
            # after dumps) is a handful of bytecodes wide and almost never
            # hit at the default 5ms interval — at ~1µs a preempted holder
            # is the common case, so the monotonicity assertion actually
            # exercises the serialization it guards.
            old_interval = sys.getswitchinterval()
            sys.setswitchinterval(1e-6)
            try:
                with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                    ws.send_json({"type": "start", "target": "10.0.0.9"})
                    ws.send_json({"type": "message", "content": "run"})
                    cost_frames = []
                    while True:
                        frame = json.loads(ws.receive_text())
                        if frame["type"] == "cost_update":
                            cost_frames.append(frame)
                        if frame["type"] == "complete":
                            break
            finally:
                sys.setswitchinterval(old_interval)

        total_emissions = threads_n * per_thread
        assert len(cost_frames) == total_emissions
        costs = [f["data"]["total_cost_usd"] for f in cost_frames]
        tokens = [f["data"]["total_tokens"] for f in cost_frames]
        # Monotonic non-decreasing: the single writer delivers in enqueue
        # order, and (post-fix) enqueue order == snapshot-read order — a
        # smaller total never follows a larger one on the wire.
        assert all(b >= a for a, b in zip(costs, costs[1:]))
        assert all(b >= a for a, b in zip(tokens, tokens[1:]))
        # Final consistency: the last delivered frame carries the exact
        # tracker-authoritative totals for every booking.
        expected_cost = sum(
            CostTracker.estimate_cost(t + 1, 0, "fake-model") * per_thread
            for t in range(threads_n)
        )
        expected_tokens = sum((t + 1) * per_thread for t in range(threads_n))
        assert costs[-1] == pytest.approx(expected_cost)
        assert tokens[-1] == expected_tokens


class TestWriterFailureTeardown:
    """A terminal frame whose send keeps failing must drive the handler
    into its teardown path (connection closed) — never a hang."""

    def test_terminal_frame_send_failure_tears_down_connection(
        self, client, monkeypatch
    ):
        import threading

        from fastapi import WebSocket as FastAPIWebSocket

        from clearwing.core.events import EventBus

        monkeypatch.setattr("clearwing.ui.web.app._PUMP_SEND_RETRY_SECONDS", 0.02)

        # Since #45 every frame leaves via the single writer's send_text
        # (pre-serialized JSON), so the dead transport is injected there:
        # healthy until the flag flips, then every send raises — the
        # "client vanished mid-session" shape.
        real_send_text = FastAPIWebSocket.send_text
        transport_dead = {"now": False}

        async def dying_send_text(self_ws, data):
            if transport_dead["now"]:
                raise RuntimeError("client vanished mid-send")
            return await real_send_text(self_ws, data)

        monkeypatch.setattr(FastAPIWebSocket, "send_text", dying_send_text)

        # Teardown runs bus.unsubscribe for every handler this connection
        # registered. starlette's TestClient synthesizes no close frame
        # when the handler merely returns, so the client socket cannot
        # observe the teardown — but the handler firing unsubscribe while
        # the client is still connected and idle proves it exited its
        # loop on the failed terminal send instead of parking forever.
        torn_down = threading.Event()
        real_unsubscribe = EventBus.unsubscribe

        def recording_unsubscribe(bus_self, event_type, handler):
            real_unsubscribe(bus_self, event_type, handler)
            torn_down.set()

        monkeypatch.setattr(EventBus, "unsubscribe", recording_unsubscribe)

        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "10.0.0.9"})
            assert ws.receive_json()["type"] == "started"

            # Kill the transport, then ask for a stop: its `stopped`
            # terminal frame can never be delivered, and the flush
            # handshake must resolve False (the pre-#45 single-strike
            # signal) so the handler breaks out and tears down.
            transport_dead["now"] = True
            ws.send_json({"type": "stop"})

            assert torn_down.wait(timeout=10), (
                "handler must tear down when a terminal frame's send keeps "
                "failing — not hang on a dead socket"
            )


class TestWriterDeathWithoutWaiters:
    """Codex PR-62 r1: the writer dying while NO flush future is pending
    must not strand the NEXT waiter — a later terminal frame registers a
    fresh future nobody will ever resolve. _send_frame's writer_task.done()
    guard has to fail it immediately instead."""

    def test_command_after_writer_death_fails_fast_not_forever(
        self, client, monkeypatch
    ):
        import time

        from fastapi import WebSocket as FastAPIWebSocket

        from clearwing.core.events import EventBus

        class _WriterKilled(BaseException):
            # _safe_send_text swallows `Exception` only — a BaseException
            # escapes it and kills the writer coroutine for real.
            pass

        monkeypatch.setattr("clearwing.ui.web.app._PUMP_SEND_RETRY_SECONDS", 0.02)

        real_send_text = FastAPIWebSocket.send_text
        kill = {"now": False}

        async def killed_send_text(self_ws, data):
            if kill["now"]:
                raise _WriterKilled("writer coroutine died")
            return await real_send_text(self_ws, data)

        monkeypatch.setattr(FastAPIWebSocket, "send_text", killed_send_text)

        torn_down = threading.Event()
        real_unsubscribe = EventBus.unsubscribe

        def recording_unsubscribe(bus_self, event_type, handler):
            real_unsubscribe(bus_self, event_type, handler)
            torn_down.set()

        monkeypatch.setattr(EventBus, "unsubscribe", recording_unsubscribe)

        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "10.0.0.9"})
            assert ws.receive_json()["type"] == "started"

            # Kill the writer while NOBODY is awaiting a flush: emit a bus
            # event whose delivery kills the writer coroutine (no terminal
            # frame in flight → no pending future at death time).
            kill["now"] = True
            _emit_tool_start("after_death_probe", {})
            time.sleep(0.5)  # let the loop process the poisoned send

            # A later command's terminal frame must fail FAST (the done()
            # guard) — without it the handler parks on a future that died
            # with the writer and teardown never runs.
            ws.send_json({"type": "stop"})
            assert torn_down.wait(timeout=10), (
                "a command arriving after the writer died must tear the "
                "connection down via the writer_task.done() guard — not "
                "wait forever on a future nothing will resolve"
            )


class TestCancelledTerminalFrameResidue:
    """Codex PR-62 r3 (Finding 1), r4 shape: a terminal entry whose flush
    handshake was cancelled by `stop` must never reach the wire. The turn
    passes its flush (transport healthy); then a bus echo lands in the
    FIFO BETWEEN the flush and the turn's inline agent_message, the
    transport congests, and the writer stalls on that bus frame's retry
    backoff while the turn parks on the inline handshake. Stop cancels the
    turn task — the parked handshake is cancelled, its entry stranded in
    the queue — and only then does the transport recover: the stalled echo
    goes out, the cancelled entry is SKIPPED, and the client's first
    terminal frame is the authoritative `stopped` +
    `complete(status=stopped)` handshake."""

    def test_stale_terminal_frame_skipped_after_stop_cancels_flush(
        self, client, monkeypatch
    ):
        import threading

        from fastapi import WebSocket as FastAPIWebSocket

        from clearwing.ui.web import app as app_module

        # A wide retry interval: the writer's next bus-frame retry attempt
        # must stay a safe distance past the stop's round trip, so the
        # writer is still stalled (and the turn still parked on its
        # handshake) when the stop cancels the turn task.
        monkeypatch.setattr("clearwing.ui.web.app._PUMP_SEND_RETRY_SECONDS", 1.0)

        # Turn-side anchor with a deterministic hand-off: add_agent fires
        # right after the flush resolved and right before the dedup gate
        # and the inline agent_message enqueue. While the turn is parked
        # HERE (blocking the loop thread), the test congests the transport
        # and emits a second bus echo — its call_soon_threadsafe enqueue is
        # scheduled on the loop BEFORE the turn's post-sleep(0) resume, so
        # the echo provably enters the FIFO ahead of the inline frame.
        flushed = threading.Event()
        echo_scheduled = threading.Event()
        real_add_agent = app_module.SessionTranscript.add_agent

        def handoff_add_agent(transcript_self, content):
            real_add_agent(transcript_self, content)
            flushed.set()
            # Block the loop thread until the test thread has congested
            # the transport and scheduled the second echo's enqueue.
            assert echo_scheduled.wait(timeout=10), (
                "test thread never scheduled the second echo"
            )

        monkeypatch.setattr(
            app_module.SessionTranscript, "add_agent", handoff_add_agent
        )

        # Stop-side anchor: the stop handler records the operator stop in
        # the transcript AFTER the cancelled turn task was joined and
        # BEFORE enqueueing `stopped`.
        stopped_recorded = threading.Event()
        real_add_error = app_module.SessionTranscript.add_error

        def signaling_add_error(transcript_self, message):
            real_add_error(transcript_self, message)
            if "stopped by operator" in str(message):
                stopped_recorded.set()

        monkeypatch.setattr(
            app_module.SessionTranscript, "add_error", signaling_add_error
        )

        # Congestion, then recovery: every send fails while the flag is
        # down (the writer sits in its retry backoff on the second echo)
        # and succeeds again once it flips back up.
        real_send_text = FastAPIWebSocket.send_text
        congested = {"now": False}

        async def congested_send_text(self_ws, data):
            if congested["now"]:
                raise RuntimeError("simulated send congestion")
            return await real_send_text(self_ws, data)

        monkeypatch.setattr(FastAPIWebSocket, "send_text", congested_send_text)

        # > 200 chars: the pre-flush echo is a 200-char preview that never
        # equals the full text, so the dedup gate cannot suppress the
        # inline send the turn parks on.
        text = "stale-terminal-frame-probe " * 40
        with patch(
            "clearwing.ui.web.app.create_agent",
            lambda **kwargs: _EchoThenFinishGraph(text),
        ):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                assert ws.receive_json()["type"] == "started"

                ws.send_json({"type": "message", "content": "run"})
                assert flushed.wait(timeout=10), (
                    "turn never passed its flush handshake"
                )
                # Congest the transport, then schedule a bus echo that
                # will stall the writer AHEAD of the turn's inline frame.
                congested["now"] = True
                _emit_message("second-echo-before-inline", "agent")
                echo_scheduled.set()

                # Stop inside the backoff window: the turn is parked on
                # the inline handshake, the writer on the stalled echo.
                ws.send_json({"type": "stop"})
                # Once the stop is recorded the turn task has been
                # cancelled and joined: its handshake future is cancelled
                # and the entry is stranded in the queue. Only now may
                # the transport recover.
                assert stopped_recorded.wait(timeout=10), (
                    "stop handler never cancelled+joined the turn"
                )
                congested["now"] = False

                frames = []
                while True:
                    frame = _receive_json_with_timeout(ws)
                    frames.append(frame)
                    if frame["type"] == "complete":
                        break

        stopped_frame = next(f for f in frames if f["type"] == "stopped")
        # The turn really was cancelled mid-park (not ended early by a
        # failed-send sweep): the residue scenario was exercised.
        assert stopped_frame["data"]["cancelled_turn"] is True
        types = [f["type"] for f in frames]
        stopped_at = types.index("stopped")
        # The first terminal frame is the authoritative stopped — no
        # complete (and no stale agent_message) may precede it.
        assert "complete" not in types[:stopped_at]
        contents = [
            f["data"]["content"] for f in frames if f["type"] == "agent_message"
        ]
        # The pre-flush echo was already delivered (healthy transport) and
        # the stalled second echo goes out after recovery; the cancelled
        # inline agent_message is skipped — the full text never lands.
        assert text not in contents
        assert text[:200] in contents
        assert "second-echo-before-inline" in contents
        # The handshake completes as stopped, and no complete frame ever
        # claims status ok.
        assert frames[-1]["data"]["status"] == "stopped"
        assert all(
            f["data"].get("status") != "ok" for f in frames if f["type"] == "complete"
        )


class TestTerminalFrameSingleAttempt:
    """Codex PR-62 r4 (P1): a terminal frame gets exactly ONE send
    attempt. When that attempt fails the writer DISCARDS the entry and
    resolves the handshake False — so a transport that recovers afterwards
    can never deliver the frame the caller already gave up on. Pre-fix the
    writer kept retrying the failed terminal frame in the background and
    delivered it once the transport recovered, out of order with the stop
    handshake that had already closed the turn."""

    def test_failed_terminal_frame_never_sent_after_recovery(
        self, client, monkeypatch
    ):
        import threading

        from fastapi import WebSocket as FastAPIWebSocket

        monkeypatch.setattr("clearwing.ui.web.app._PUMP_SEND_RETRY_SECONDS", 0.05)

        # Anchor on the transport itself: the patched send_text records
        # the first RAISED attempt, so the test knows the terminal frame's
        # one attempt already failed before it flips the transport back on.
        real_send_text = FastAPIWebSocket.send_text
        congested = {"now": False}
        failed_attempt = threading.Event()

        async def congested_send_text(self_ws, data):
            if congested["now"]:
                failed_attempt.set()
                raise RuntimeError("simulated send congestion")
            return await real_send_text(self_ws, data)

        monkeypatch.setattr(FastAPIWebSocket, "send_text", congested_send_text)

        # No bus echo: the queue is empty at flush time, so the turn sails
        # through `_flush_writer` (the sentinel resolves without a send)
        # and parks on the inline agent_message's own handshake — THAT
        # frame is the one the writer attempts (and fails) while
        # congested. The heartbeat's first frame is +10s away, so nothing
        # else can be the failed send.
        class _SilentGraph:
            def get_state(self, config):
                del config
                return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

            async def astream(self, input_msg, config, stream_mode="values"):
                del input_msg, config, stream_mode
                yield {"messages": [_AI("silent turn output")]}

        with patch(
            "clearwing.ui.web.app.create_agent", lambda **kwargs: _SilentGraph()
        ):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                assert ws.receive_json()["type"] == "started"

                congested["now"] = True
                ws.send_json({"type": "message", "content": "run"})
                # The writer attempted (and failed) the turn's terminal
                # agent_message exactly once; the turn saw the hard False
                # and returned without ever enqueueing a complete.
                assert failed_attempt.wait(timeout=10)

                # The stop arrives AFTER the failed attempt, and only now
                # does the transport recover — the discarded frame must
                # still never be sent.
                congested["now"] = False
                ws.send_json({"type": "stop"})

                frames = []
                while True:
                    frame = _receive_json_with_timeout(ws)
                    frames.append(frame)
                    if frame["type"] == "complete":
                        break

        types = [f["type"] for f in frames]
        # Nothing but the stop handshake: the discarded agent_message is
        # never delivered — pre-fix it was retried in the background and
        # arrived as a stale terminal frame after (or before) the
        # authoritative stopped/complete.
        assert types[0] == "stopped"
        assert "agent_message" not in types
        assert frames[-1]["type"] == "complete"
        assert frames[-1]["data"]["status"] == "stopped"


class TestStalledBusFrameKeepsLaterWaitersPending:
    """Codex PR-62 r4 (P2): a waiter queued behind a BUS frame stalled in
    its retry backoff must STAY pending. The old failed-send sweep
    resolved it False, the turn returned without sending `complete`, and
    the frames still went out once the transport recovered — the client
    sat waiting for a complete that never came. On recovery the FIFO
    delivers everything and the turn ends normally with complete(ok)."""

    def test_recovery_delivers_fifo_and_turn_completes(self, client, monkeypatch):
        import threading

        from fastapi import WebSocket as FastAPIWebSocket

        monkeypatch.setattr("clearwing.ui.web.app._PUMP_SEND_RETRY_SECONDS", 0.05)

        # Congestion lifts only after the stalled echo has failed a SECOND
        # attempt: with a single failure the pre-fix sweep of the flush
        # sentinel might not have reached the inline handshake yet, and
        # the recovered retry would deliver everything before the turn
        # gave up — a false pass on the old code. Two failed attempts
        # guarantee the old sweep already failed BOTH the sentinel and the
        # inline handshake parked behind it.
        real_send_text = FastAPIWebSocket.send_text
        congested = {"now": False}
        failures = {"n": 0}
        second_failure = threading.Event()

        async def congested_send_text(self_ws, data):
            if congested["now"]:
                failures["n"] += 1
                if failures["n"] >= 2:
                    second_failure.set()
                raise RuntimeError("simulated send congestion")
            return await real_send_text(self_ws, data)

        monkeypatch.setattr(FastAPIWebSocket, "send_text", congested_send_text)

        text = "fifo-recovery-probe " * 30  # > 200 chars: echo is a preview
        with patch(
            "clearwing.ui.web.app.create_agent",
            lambda **kwargs: _EchoThenFinishGraph(text),
        ):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                assert ws.receive_json()["type"] == "started"

                congested["now"] = True
                ws.send_json({"type": "message", "content": "run"})
                # The echo (bus frame, no handshake) is stalled in the
                # writer's retry backoff; the turn's flush sentinel and
                # then its inline handshake queue up behind it and stay
                # PENDING (no sweep).
                assert second_failure.wait(timeout=10)
                congested["now"] = False

                # Recovery: the FIFO delivers the echo, the flush resolves,
                # the inline full text goes out, and the turn ends with
                # complete(ok) — pre-fix the client hung here with no
                # complete ever arriving.
                frames = []
                while True:
                    frame = _receive_json_with_timeout(ws)
                    frames.append(frame)
                    if frame["type"] == "complete":
                        break

                # The connection is fully healthy afterwards — a clean
                # stop handshake round-trips on the same socket.
                ws.send_json({"type": "stop"})
                assert ws.receive_json()["type"] == "stopped"
                assert ws.receive_json()["type"] == "complete"

        contents = [
            f["data"]["content"] for f in frames if f["type"] == "agent_message"
        ]
        assert text[:200] in contents  # the stalled echo went out
        assert text in contents  # the full inline text followed in FIFO order
        assert frames[-1]["data"]["status"] == "ok"


class TestLegacyUnmarkedCostFrame:
    """Codex PR-62 r4 (P2): an in-turn cost frame with NO session_id (a
    legacy emitter) carries spend the tracker never booked — no bucket
    holds it. On a session-bound connection the adapter must still count
    it into the DISPLAYED totals (the legacy in-turn accumulation) instead
    of rewriting it to the tracker water level that lacks it. The
    accumulation is best-effort display only: the next scoped frame's
    tracker snapshot stays authoritative and does not include the unbooked
    legacy spend."""

    def test_in_turn_unmarked_frame_counted_then_scoped_snapshot(self, client):
        from clearwing.observability.telemetry import CostTracker

        class _LegacyThenScopedGraph:
            def __init__(self, **kwargs):
                self.session_id = kwargs.get("session_id")

            def get_state(self, config):
                del config
                return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

            async def astream(self, input_msg, config, stream_mode="values"):
                del input_msg, config, stream_mode
                # Legacy shape: no session_id, nothing booked in the
                # tracker — the spend exists only in the frame's fields.
                _emit_cost(
                    {
                        "input_tokens": 11,
                        "output_tokens": 4,
                        "cost": 0.003,
                        "total_cost_usd": 777.0,  # the emitter's process-global lie
                        "total_tokens": 777000,
                        "model": "fake-model",
                        "provider": "openai",
                        "elapsed_ms": 0,
                    }
                )
                # Scoped shape: a real booking under this session's id
                # (record_llm_call emits its frame itself).
                CostTracker().record_llm_call(
                    500, 50, "fake-model", session_id=self.session_id
                )
                yield {"messages": [_AI("done")]}

        with patch(
            "clearwing.ui.web.app.create_agent", _LegacyThenScopedGraph
        ):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                assert ws.receive_json()["type"] == "started"
                ws.send_json({"type": "message", "content": "run"})
                cost_frames = []
                while True:
                    frame = json.loads(ws.receive_text())
                    if frame["type"] == "cost_update":
                        cost_frames.append(frame)
                    if frame["type"] == "complete":
                        break

        assert len(cost_frames) == 2
        legacy, scoped = (f["data"] for f in cost_frames)
        # The unmarked in-turn frame's spend IS counted into the display —
        # pre-fix the snapshot branch rewrote it to the (empty) tracker
        # bucket and the spend vanished from the wire.
        assert legacy["total_cost_usd"] == pytest.approx(0.003)
        assert legacy["total_tokens"] == 15
        # The next scoped frame is tracker-authoritative: the booked call
        # only — the legacy spend is not in any bucket and does not ride
        # along (best-effort display; the tracker stays authoritative).
        booked_cost = CostTracker.estimate_cost(500, 50, "fake-model")
        assert scoped["total_cost_usd"] == pytest.approx(booked_cost)
        assert scoped["total_tokens"] == 550


class TestSessionSnapshotAtomicity:
    """Codex PR-62 r1: out-of-turn frames read cost and tokens through ONE
    tracker snapshot — two separately locked reads could straddle a
    concurrent booking and report internally inconsistent totals."""

    def test_snapshot_returns_cost_and_tokens_together(self):
        from clearwing.observability.telemetry import CostTracker

        tracker = CostTracker()
        sid = "snap-atomic-1"
        tracker.record_llm_call(11, 7, "claude-sonnet-4-6", session_id=sid)
        cost, in_tok, out_tok = tracker.session_snapshot(sid)
        assert in_tok == 11 and out_tok == 7
        assert cost == tracker.session_total(sid)
        assert (in_tok, out_tok) == tracker.session_tokens(sid)
        tracker.forget_session(sid)

    def test_snapshot_unknown_and_none_ids(self):
        from clearwing.observability.telemetry import CostTracker

        tracker = CostTracker()
        assert tracker.session_snapshot("never-recorded") == (0.0, 0, 0)
        assert tracker.session_snapshot(None) == (0.0, 0, 0)
