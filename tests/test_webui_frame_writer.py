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
  (#48-2).
"""

from __future__ import annotations

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

    def test_report_transcript_records_good_tool_and_skips_dropped_one(
        self, client, results_dir
    ):
        """Transcript recording follows successful enqueue: the dropped
        (unserializable) tool never lands in the report; the good one
        does."""

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
        assert "poison_tool" not in report


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
                while out_of_turn is None:
                    frame = json.loads(ws.receive_text())
                    if frame["type"] == "cost_update":
                        out_of_turn = frame

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
    """#48-2: the session accumulator is written from worker threads; the
    lock makes the accumulated totals exact under concurrency."""

    def test_concurrent_worker_emissions_accumulate_exactly(self, client):
        threads_n, per_thread = 8, 50

        class _HammerGraph:
            def __init__(self, **kwargs):
                self.session_id = kwargs.get("session_id")

            def get_state(self, config):
                del config
                return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

            async def astream(self, input_msg, config, stream_mode="values"):
                del input_msg, config, stream_mode

                def worker():
                    for _ in range(per_thread):
                        _emit_cost(
                            {
                                "input_tokens": 1,
                                "output_tokens": 0,
                                "cost": 0.01,
                                "total_cost_usd": 0.0,
                                "total_tokens": 0,
                                "model": "fake-model",
                                "provider": "openai",
                                "session_id": self.session_id,
                                "elapsed_ms": 0,
                            }
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
        assert last["total_cost_usd"] == pytest.approx(total_emissions * 0.01)
        assert last["total_tokens"] == total_emissions
