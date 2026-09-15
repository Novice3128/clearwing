"""Tests for the web UI module."""

from unittest.mock import MagicMock, patch

import pytest

# Import guard - tests skip if fastapi not installed
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from clearwing.ui.web.app import _cors_origins, create_app

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
def unauth_app(monkeypatch):
    monkeypatch.delenv("CLEARWING_WEB_API_KEY", raising=False)
    return create_app()


@pytest.fixture
def unauth_client(unauth_app):
    return TestClient(unauth_app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "clearwing"


class TestSessionEndpoints:
    def test_list_sessions_empty(self, client):
        with patch("clearwing.data.memory.SessionStore") as mock_store:
            mock_store.return_value.list_sessions.return_value = []
            resp = client.get("/api/sessions")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_list_sessions_with_data(self, client):
        mock_session = MagicMock()
        mock_session.session_id = "abc123"
        mock_session.target = "10.0.0.1"
        mock_session.model = "claude-sonnet-4-6"
        mock_session.status = "completed"
        mock_session.start_time = "2024-01-01T00:00:00"
        mock_session.cost_usd = 0.05
        mock_session.token_count = 1000

        with patch("clearwing.data.memory.SessionStore") as mock_store:
            mock_store.return_value.list_sessions.return_value = [mock_session]
            resp = client.get("/api/sessions")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["session_id"] == "abc123"
            assert data[0]["target"] == "10.0.0.1"

    def test_get_session(self, client):
        mock_session = MagicMock()
        mock_session.session_id = "abc123"
        mock_session.target = "10.0.0.1"
        mock_session.model = "claude-sonnet-4-6"
        mock_session.status = "completed"
        mock_session.start_time = "2024-01-01T00:00:00"
        mock_session.cost_usd = 0.05
        mock_session.token_count = 1000
        mock_session.open_ports = [{"port": 22}]
        mock_session.services = []
        mock_session.vulnerabilities = []
        mock_session.exploit_results = []
        mock_session.flags_found = []

        with patch("clearwing.data.memory.SessionStore") as mock_store:
            mock_store.return_value.load.return_value = mock_session
            resp = client.get("/api/sessions/abc123")
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == "abc123"
            assert data["open_ports"] == [{"port": 22}]

    def test_get_session_not_found(self, client):
        with patch("clearwing.data.memory.SessionStore") as mock_store:
            mock_store.return_value.load.return_value = None
            resp = client.get("/api/sessions/nonexistent")
            assert resp.status_code == 404


class TestMetricsEndpoints:
    def test_get_metrics(self, client):
        with patch("clearwing.observability.telemetry.CostTracker") as mock_tracker:
            mock_summary = MagicMock()
            mock_summary.input_tokens = 1000
            mock_summary.output_tokens = 500
            mock_summary.total_cost_usd = 0.05
            mock_summary.tool_calls = 10
            mock_tracker.return_value.get_summary.return_value = mock_summary
            resp = client.get("/api/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert data["input_tokens"] == 1000

    def test_prometheus_metrics(self, client):
        resp = client.get("/api/metrics/prometheus")
        assert resp.status_code == 200


class TestOperateEndpoints:
    def test_start_operator_missing_fields(self, client):
        resp = client.post("/api/operate", json={}, headers=AUTH)
        assert resp.status_code == 400

    def test_start_operator(self, client):
        resp = client.post(
            "/api/operate",
            json={
                "target": "10.0.0.1",
                "goals": ["Scan ports"],
            },
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["status"] == "running"

    def test_get_operator_status_not_found(self, client):
        resp = client.get("/api/operate/nonexistent", headers=AUTH)
        assert resp.status_code == 404

    def test_get_operator_status(self, client):
        # Start a session first
        resp = client.post(
            "/api/operate",
            json={
                "target": "10.0.0.1",
                "goals": ["Scan"],
            },
            headers=AUTH,
        )
        sid = resp.json()["session_id"]

        resp = client.get(f"/api/operate/{sid}", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["target"] == "10.0.0.1"


class TestWebSocketEndpoint:
    def test_websocket_connect(self, client):
        with client.websocket_connect("/ws/agent", headers=AUTH):
            # Just connect and disconnect
            pass

    def test_websocket_invalid_json(self, client):
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_text("not json")
            # Should disconnect gracefully


class TestCreateApp:
    def test_app_has_routes(self, app):
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/api/health" in routes
        assert "/api/sessions" in routes
        assert "/api/metrics" in routes
        assert "/ws/agent" in routes

    def test_cors_enabled(self, app):
        [type(m) for m in app.user_middleware]
        # Just verify the app was created without error
        assert app is not None


class TestOperateAuth:
    """c16: /api/operate* and /ws/agent require X-API-Key."""

    def test_operate_requires_key(self, client):
        resp = client.post("/api/operate", json={"target": "x", "goals": ["g"]})
        assert resp.status_code == 401

    def test_operate_rejects_wrong_key(self, client):
        resp = client.post(
            "/api/operate",
            json={"target": "x", "goals": ["g"]},
            headers={"X-API-Key": "wrong"},
        )
        assert resp.status_code == 401

    def test_operate_status_requires_key(self, client):
        resp = client.get("/api/operate/abc")
        assert resp.status_code == 401

    def test_websocket_rejects_without_key(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/agent"):
                pass

    def test_websocket_rejects_wrong_key(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/agent", headers={"X-API-Key": "nope"}):
                pass

    def test_auto_approve_exploits_ignored(self, client):
        with patch("clearwing.ui.web.app.OperatorConfig") as mock_cfg:
            client.post(
                "/api/operate",
                json={
                    "target": "10.0.0.1",
                    "goals": ["g"],
                    "auto_approve_exploits": True,
                },
                headers=AUTH,
            )
            # Background task may or may not have run yet under TestClient;
            # if it did, auto_approve_exploits must be False regardless of body.
            for call in mock_cfg.call_args_list:
                assert call.kwargs.get("auto_approve_exploits") is False


class TestOperateDisabledWithoutKey:
    """c16: routes refuse to mount when CLEARWING_WEB_API_KEY is unset."""

    def test_operate_503_without_env(self, unauth_client):
        resp = unauth_client.post("/api/operate", json={"target": "x", "goals": ["g"]})
        assert resp.status_code == 503
        assert "CLEARWING_WEB_API_KEY" in resp.json()["detail"]

    def test_operate_status_503_without_env(self, unauth_client):
        resp = unauth_client.get("/api/operate/abc")
        assert resp.status_code == 503

    def test_health_still_works_without_env(self, unauth_client):
        assert unauth_client.get("/api/health").status_code == 200

    def test_websocket_closed_without_env(self, unauth_client):
        with pytest.raises(Exception):
            with unauth_client.websocket_connect("/ws/agent"):
                pass

    def test_log_emitted(self, monkeypatch, caplog):
        monkeypatch.delenv("CLEARWING_WEB_API_KEY", raising=False)
        with caplog.at_level("ERROR", logger="clearwing.ui.web.app"):
            create_app()
        assert any("CLEARWING_WEB_API_KEY" in r.message for r in caplog.records)


class TestCorsOrigins:
    def test_default_empty(self, monkeypatch):
        monkeypatch.delenv("CLEARWING_WEB_CORS_ORIGINS", raising=False)
        assert _cors_origins() == []

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "CLEARWING_WEB_CORS_ORIGINS", "https://a.example, https://b.example "
        )
        assert _cors_origins() == ["https://a.example", "https://b.example"]


class TestQueryKeyAuth:
    """Browsers cannot set headers; every key-gated route must accept ?api_key=."""

    def test_operate_accepts_query_key(self, client):
        resp = client.post(
            f"/api/operate?api_key={API_KEY}",
            json={"target": "10.0.0.1", "goals": ["g"]},
        )
        assert resp.status_code == 200

    def test_operate_rejects_wrong_query_key(self, client):
        resp = client.post(
            "/api/operate?api_key=wrong",
            json={"target": "10.0.0.1", "goals": ["g"]},
        )
        assert resp.status_code == 401

    def test_reports_requires_key(self, client):
        assert client.get("/api/reports/whatever").status_code == 401

    def test_non_ascii_key_is_401_not_500(self, client):
        resp = client.get("/api/reports/abc?api_key=%C3%A9llo")
        assert resp.status_code == 401


class _FakeAI:
    def __init__(self, content="Task done: report attached"):
        self.type = "ai"
        self.content = content


class _FakeGraph:
    def __init__(self, events=None, resume_messages=None):
        self.events = events or []
        self.resume_messages = resume_messages or []

    async def astream(self, input_msg, config, stream_mode="values"):
        for ev in self.events:
            yield ev

    async def ainvoke(self, input_data, config):
        from clearwing.agent.runtime import GraphStateSnapshot

        return GraphStateSnapshot(values={"messages": list(self.resume_messages)})


class _SlowFakeGraph(_FakeGraph):
    """Graph whose first turn sleeps far beyond a test's patience."""

    def __init__(self):
        super().__init__()
        self.turn_started = False
        self.turn_cancelled = False

    async def astream(self, input_msg, config, stream_mode="values"):
        import asyncio

        self.turn_started = True
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.turn_cancelled = True
            raise
        yield {"messages": [_FakeAI()]}


class _PendingFakeGraph(_FakeGraph):
    """Graph suspended awaiting an approve frame (stop must discard it)."""

    def __init__(self):
        super().__init__()
        self.discard_calls = 0

    def discard_interrupt(self, config):
        self.discard_calls += 1
        # Real semantics: True only while something is actually pending.
        return self.discard_calls == 1


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    import clearwing.ui.web.session_report as session_report

    root = tmp_path / "results"
    monkeypatch.setattr(session_report, "default_results_dir", lambda sub: root / sub)
    return root


class TestAgentSessionFlow:
    """config → prompt → (tools) → complete → deterministic report artifact."""

    def test_message_turn_completes_with_report(self, client, monkeypatch, results_dir):
        fake = _FakeGraph(events=[{"messages": [_FakeAI()]}])
        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: fake)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "10.0.0.1", "model": "m"})
            started = ws.receive_json()
            assert started["type"] == "started"

            ws.send_json({"type": "message", "content": "scan it and report"})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "complete":
                    break
            types = [f["type"] for f in frames]
            assert "agent_message" in types
            complete = frames[-1]
            assert complete["data"]["report_url"].startswith("/api/reports/")

        sid = complete["data"]["session_id"]
        report = results_dir / "sessions" / sid / "report.md"
        assert report.is_file()
        content = report.read_text(encoding="utf-8")
        assert "scan it and report" in content
        assert "Task done" in content

    def test_start_failure_sends_error_frame(self, client, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("no llm configured")

        monkeypatch.setattr("clearwing.ui.web.app.create_agent", boom)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "x", "model": "m"})
            frame = ws.receive_json()
            assert frame["type"] == "error"
            assert "Failed to start agent" in frame["data"]["message"]

    def test_report_written_on_disconnect_after_start(self, client, monkeypatch, results_dir):
        import time

        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: _FakeGraph())
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": "m"})
            started = ws.receive_json()
            sid = started["session_id"]
        report = results_dir / "sessions" / sid / "report.md"
        for _ in range(100):
            if report.is_file():
                break
            time.sleep(0.02)
        assert report.is_file(), "session report must be written even without any turn"

    def test_message_before_start_gets_error_reply(self, client):
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "message", "content": "hello?"})
            frame = ws.receive_json()
            assert frame["type"] == "error"
            assert "start" in frame["data"]["message"]

    def test_approve_records_decision_and_response(self, client, monkeypatch, results_dir):
        fake = _FakeGraph(resume_messages=[_FakeAI("post approval answer")])
        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: fake)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": "m"})
            started = ws.receive_json()
            sid = started["session_id"]

            ws.send_json({"type": "approve", "approved": True})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "complete":
                    break
            types = [f["type"] for f in frames]
            assert "agent_message" in types

        report = results_dir / "sessions" / sid / "report.md"
        content = report.read_text(encoding="utf-8")
        assert "[approval approved by operator]" in content
        assert "post approval answer" in content


class TestStopFrame:
    """#6: an operator must be able to cancel a running/pending agent turn."""

    def test_stop_cancels_running_turn(self, client, monkeypatch, results_dir):
        import time

        fake = _SlowFakeGraph()
        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: fake)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": "m"})
            assert ws.receive_json()["type"] == "started"

            t0 = time.monotonic()
            ws.send_json({"type": "message", "content": "long task"})
            ws.send_json({"type": "stop"})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "complete":
                    break

        assert time.monotonic() - t0 < 10, "stop must cancel the 30s turn promptly"
        stopped = next(f for f in frames if f["type"] == "stopped")
        assert stopped["data"]["cancelled_turn"] is True
        assert fake.turn_started
        assert fake.turn_cancelled

    def test_message_during_running_turn_is_rejected(self, client, monkeypatch):
        fake = _SlowFakeGraph()
        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: fake)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": "m"})
            assert ws.receive_json()["type"] == "started"

            ws.send_json({"type": "message", "content": "first"})
            ws.send_json({"type": "message", "content": "second"})
            busy = ws.receive_json()
            assert busy["type"] == "error"
            assert "already running" in busy["data"]["message"]

            ws.send_json({"type": "stop"})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "complete":
                    break
            assert any(f["type"] == "stopped" for f in frames)

    def test_stop_discards_pending_approval(self, client, monkeypatch, results_dir):
        fake = _PendingFakeGraph()
        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: fake)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": "m"})
            assert ws.receive_json()["type"] == "started"

            ws.send_json({"type": "stop"})
            stopped = ws.receive_json()
            assert stopped["type"] == "stopped"
            assert stopped["data"]["cancelled_turn"] is False
            assert stopped["data"]["discarded_approval"] is True
            assert ws.receive_json()["type"] == "complete"

        # Once for the stop frame; the handler's disconnect cleanup may
        # call it again (idempotent no-op on a real graph).
        assert fake.discard_calls >= 1

    def test_stop_before_start_is_harmless(self, client):
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "stop"})
            stopped = ws.receive_json()
            assert stopped["type"] == "stopped"
            assert stopped["data"] == {
                "cancelled_turn": False,
                "discarded_approval": False,
            }
            complete = ws.receive_json()
            assert complete["type"] == "complete"
            assert complete["data"] == {}


class TestSessionReportHardening:
    def test_report_redacts_key_shapes_and_is_owner_only(self, monkeypatch, tmp_path):
        import stat as stat_module

        import clearwing.ui.web.session_report as session_report

        monkeypatch.setattr(
            session_report, "default_results_dir", lambda sub: tmp_path / sub
        )
        transcript = session_report.SessionTranscript("sec00001", model="m")
        transcript.add_user("my key is sk-abcdefghijklmnopqrst")
        transcript.add_user("token: supersecret123")
        transcript.add_user("audit hash aabbccdd00112233aabbccdd00112233")
        path = transcript.write()

        assert stat_module.S_IMODE(path.stat().st_mode) == 0o600
        content = path.read_text(encoding="utf-8")
        assert "sk-abcdefghijklmnopqrst" not in content
        assert "supersecret123" not in content
        assert "aabbccdd00112233aabbccdd00112233" not in content

    def test_non_string_content_renders(self, monkeypatch, tmp_path):
        """#25: content null/list/object must not crash report rendering."""
        import clearwing.ui.web.session_report as session_report

        monkeypatch.setattr(
            session_report, "default_results_dir", lambda sub: tmp_path / sub
        )
        transcript = session_report.SessionTranscript("sec00002", model="m")
        transcript.add_user(None)
        transcript.add_user([{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}])
        transcript.add_user({"tool_use": {"name": "x"}})
        transcript.add_agent(None)
        transcript.add_error({"message": "structured boom"})
        path = transcript.write()

        content = path.read_text(encoding="utf-8")
        assert "part one\npart two" in content
        assert '"tool_use"' in content
        assert "structured boom" in content
        assert "(empty)" in content  # null content keeps its placeholder

    def test_write_falls_back_when_render_is_poisoned(self, monkeypatch, tmp_path):
        """#25: even an unrenderable transcript must still produce the artifact."""
        import clearwing.ui.web.session_report as session_report

        monkeypatch.setattr(
            session_report, "default_results_dir", lambda sub: tmp_path / sub
        )
        transcript = session_report.SessionTranscript("sec00003", model="m")
        transcript.add_user("hello")
        # Bypass add_user() with a legacy non-coerced value (int) — render()
        # cannot process it and raises.
        transcript.user_messages.append(12345)
        path = transcript.write()  # must not raise

        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        assert "render failed" in content
        assert "sec00003" in content

    def test_ws_message_with_null_content_still_writes_report(
        self, client, monkeypatch, results_dir
    ):
        """#25 end-to-end: `content: null` must not permanently break the report."""
        fake = _FakeGraph(events=[{"messages": [_FakeAI()]}])
        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: fake)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": "m"})
            assert ws.receive_json()["type"] == "started"

            ws.send_json({"type": "message", "content": None})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "complete":
                    break
            assert any(f["type"] == "agent_message" for f in frames)

        sid = frames[-1]["data"]["session_id"]
        report = results_dir / "sessions" / sid / "report.md"
        assert report.is_file()


class TestReportDownloadEndpoint:
    def test_serves_report_with_query_key(self, client, monkeypatch, results_dir):
        import clearwing.ui.web.session_report as session_report

        transcript = session_report.SessionTranscript("abc12345", target="t", model="m")
        transcript.add_user("hello")
        path = transcript.write()
        assert path.is_file()

        resp = client.get(f"/api/reports/abc12345?api_key={API_KEY}")
        assert resp.status_code == 200
        assert b"hello" in resp.content

    def test_404_for_unknown_session(self, client):
        resp = client.get(f"/api/reports/zzzzzzzz?api_key={API_KEY}")
        assert resp.status_code == 404

    def test_400_for_malformed_session_id(self, client):
        resp = client.get("/api/reports/..%2Fetc%2Fpasswd?api_key=" + API_KEY)
        assert resp.status_code in (400, 404)
