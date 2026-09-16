"""Tests for the web UI module."""

from types import SimpleNamespace
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

    def get_state(self, config):
        from clearwing.agent.runtime import GraphStateSnapshot

        return GraphStateSnapshot(values={"messages": []}, next=("tools",), tasks=[])


class _FailingResumeGraph(_FakeGraph):
    """Graph whose approve-resume always raises."""

    async def ainvoke(self, input_data, config):
        raise RuntimeError("resume exploded")


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

    def test_message_turn_carries_target_into_graph_state(
        self, client, monkeypatch, results_dir
    ):
        """Issue #19: the start-frame target must ride along with every
        message so episodes are attributable (recall used to find nothing
        because every row landed with target='unknown')."""
        captured: dict = {}

        class _CaptureGraph(_FakeGraph):
            async def astream(self, input_msg, config, stream_mode="values"):
                captured.update(input_msg)
                for ev in self.events:
                    yield ev

        fake = _CaptureGraph(events=[{"messages": [_FakeAI()]}])
        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: fake)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "10.9.8.7", "model": "m"})
            assert ws.receive_json()["type"] == "started"

            ws.send_json({"type": "message", "content": "scan it"})
            while True:
                if ws.receive_json()["type"] == "complete":
                    break

        assert captured["target"] == "10.9.8.7"
        assert captured["messages"][0]["content"] == "scan it"

    def test_start_frame_reports_resolved_model(self, client, monkeypatch):
        """Issue #21/#22 surface: `started` echoes the model that actually
        got configured, and an empty model field defers to config."""

        class _LabeledGraph(_FakeGraph):
            llm = SimpleNamespace(model_name="glm-5.3")

        fake = _LabeledGraph()
        captured: dict = {}

        def make_agent(**kwargs):
            captured.update(kwargs)
            return fake

        monkeypatch.setattr("clearwing.ui.web.app.create_agent", make_agent)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": ""})
            started = ws.receive_json()
            assert started["type"] == "started"
            assert started["model"] == "glm-5.3"
            assert captured["model_name"] is None
            assert captured["model_explicit"] is False

    def test_start_frame_with_non_string_model_is_treated_as_unset(
        self, client, monkeypatch
    ):
        """Codex P2: a truthy non-string model used to crash `.strip()`
        before the error-frame path, tearing down the socket silently."""
        captured: dict = {}

        def make_agent(**kwargs):
            captured.update(kwargs)
            return _FakeGraph()

        monkeypatch.setattr("clearwing.ui.web.app.create_agent", make_agent)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": 12345})
            started = ws.receive_json()
            assert started["type"] == "started"
            assert captured["model_name"] is None
            assert captured["model_explicit"] is False


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

    def test_approve_failure_still_sends_complete(self, client, monkeypatch, results_dir):
        """Review P1: a failed resume used to die on an unbound `snapshot`,
        leaving the client waiting for a complete frame that never came."""
        fake = _FailingResumeGraph()
        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: fake)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": "m"})
            assert ws.receive_json()["type"] == "started"

            ws.send_json({"type": "approve", "approved": True})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "complete":
                    break

        types = [f["type"] for f in frames]
        assert "error" in types
        assert types[-1] == "complete"
        assert "resume exploded" in next(f for f in frames if f["type"] == "error")["data"][
            "message"
        ]

    def test_message_while_approval_pending_is_rejected(self, client, monkeypatch):
        """A user message on a suspended batch would orphan its tool_use."""
        fake = _PendingFakeGraph()
        monkeypatch.setattr("clearwing.ui.web.app.create_agent", lambda **kwargs: fake)
        with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
            ws.send_json({"type": "start", "target": "t", "model": "m"})
            assert ws.receive_json()["type"] == "started"

            ws.send_json({"type": "message", "content": "hello?"})
            frame = ws.receive_json()
            assert frame["type"] == "error"
            assert "approval is still pending" in frame["data"]["message"]


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

    def test_json_shaped_content_is_redacted(self, monkeypatch, tmp_path):
        """coerce_text JSON-fallen dict content must still hit the redaction
        patterns (quote between key and colon used to bypass them)."""
        import clearwing.ui.web.session_report as session_report

        monkeypatch.setattr(
            session_report, "default_results_dir", lambda sub: tmp_path / sub
        )
        transcript = session_report.SessionTranscript("sec00004", model="m")
        transcript.add_user({"token": "abcdef1234567890", "secret": "xyzvalue9876"})
        path = transcript.write()

        content = path.read_text(encoding="utf-8")
        assert "abcdef1234567890" not in content
        assert "xyzvalue9876" not in content

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


class _FakeCostGraph:
    """Stands in for create_agent(): its astream fires COST_UPDATE events on
    the process-wide bus (the way the real runtime's tracker does) and yields
    one assistant state event. Emits: own-session call, foreign-session call,
    own-session call."""

    session_id = None

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")

    async def astream(self, input_data, config, stream_mode="values"):
        from clearwing.core.events import EventBus, EventType

        del input_data, config, stream_mode
        bus = EventBus()
        emissions = [
            (0.01, 1000, 100, self.session_id),
            (0.02, 2000, 200, "someone-elses-session"),
            (0.03, 3000, 300, self.session_id),
        ]
        for cost, in_, out, origin in emissions:
            bus.emit(
                EventType.COST_UPDATE,
                {
                    "input_tokens": in_,
                    "output_tokens": out,
                    "cached_tokens": 0,
                    "cost": cost,
                    # Process-global running total (polluted by other
                    # sessions in a long-lived webui) — the adapter must
                    # rewrite this per session (issue #10).
                    "total_cost_usd": 99.0 + cost,
                    "model": "glm-5.3",
                    "provider": "openai",
                    "session_id": origin,
                    "elapsed_ms": 10,
                },
            )
        yield {
            "messages": [SimpleNamespace(type="ai", content="done", text="done")]
        }


class TestCostSessionScoping:
    def test_cost_update_frames_are_session_scoped(self, client):
        import json

        with patch("clearwing.ui.web.app.create_agent", _FakeCostGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                frames = []
                while True:
                    msg = json.loads(ws.receive_text())
                    frames.append(msg)
                    if msg["type"] == "complete":
                        break

        cost_frames = [f for f in frames if f["type"] == "cost_update"]
        # The foreign session's frame never reaches this socket at all —
        # only this session's two own frames do.
        assert len(cost_frames) == 2
        # Session-scoped totals replace the tracker's process-global ones.
        assert cost_frames[0]["data"]["total_cost_usd"] == pytest.approx(0.01)
        assert cost_frames[1]["data"]["total_cost_usd"] == pytest.approx(0.04)
        assert cost_frames[1]["data"]["total_tokens"] == 4400
        # Per-call fields and attribution ride along untouched.
        assert cost_frames[1]["data"]["cost"] == 0.03
        assert cost_frames[1]["data"]["model"] == "glm-5.3"

    def test_session_report_carries_scoped_totals(self, client):
        import json

        with patch("clearwing.ui.web.app.create_agent", _FakeCostGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                complete = None
                while True:
                    msg = json.loads(ws.receive_text())
                    if msg["type"] == "complete":
                        complete = msg
                        break

        report_url = complete["data"]["report_url"]
        report = client.get(report_url, headers=AUTH).text
        # Session-scoped cost (0.01 + 0.03) and session-scoped tokens
        # (4400) — not the last call's per-call token count.
        assert "| Cost | $0.0400 |" in report
        assert "| Tokens | 4400 |" in report

    def test_second_start_rearms_session_totals(self, client):
        import json

        with patch("clearwing.ui.web.app.create_agent", _FakeCostGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                first_costs = []
                for _round in range(2):
                    ws.send_json({"type": "start", "target": "10.0.0.9"})
                    ws.send_json({"type": "message", "content": "run"})
                    while True:
                        msg = json.loads(ws.receive_text())
                        if msg["type"] == "cost_update":
                            first_costs.append(msg)
                        if msg["type"] == "complete":
                            break
                # Both rounds start from a fresh budget: each session's
                # first frame is 0.01/1100, not the previous session's tail.
                # (2 frames per round — the foreign frame is dropped.)
                assert first_costs[0]["data"]["total_cost_usd"] == pytest.approx(0.01)
                assert first_costs[2]["data"]["total_cost_usd"] == pytest.approx(0.01)
                assert first_costs[2]["data"]["total_tokens"] == 1100




class _SlowGraph:
    """astream takes noticeably longer than the (shortened) heartbeat cadence."""

    def __init__(self, **kwargs):
        del kwargs

    async def astream(self, input_data, config, stream_mode="values"):
        import asyncio

        del input_data, config, stream_mode
        await asyncio.sleep(0.2)
        yield {"messages": [SimpleNamespace(type="ai", content="done", text="done")]}


class _FailingGraph:
    """astream raises an exception carrying the retry-count attribute."""

    def __init__(self, **kwargs):
        del kwargs

    async def astream(self, input_data, config, stream_mode="values"):
        del input_data, config, stream_mode
        exc = RuntimeError("connection timed out")
        exc._clearwing_attempts = 3
        raise exc
        yield  # pragma: no cover — keeps this an async generator


class TestLlmProgressHeartbeat:
    def test_heartbeat_frames_during_slow_turn(self, client, monkeypatch):
        import json

        monkeypatch.setattr("clearwing.ui.web.app._LLM_PROGRESS_INTERVAL_SECONDS", 0.05)
        with patch("clearwing.ui.web.app.create_agent", _SlowGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "slow"})
                types = []
                while True:
                    msg = json.loads(ws.receive_text())
                    types.append(msg["type"])
                    if msg["type"] == "complete":
                        break
        assert "llm_progress" in types
        # The heartbeat is cancelled before the terminal frames, so it can
        # never land after complete.
        assert types[-1] == "complete"

    def test_error_frame_carries_retry_count(self, client):
        import json

        with patch("clearwing.ui.web.app.create_agent", _FailingGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "boom"})
                error = None
                while True:
                    msg = json.loads(ws.receive_text())
                    if msg["type"] == "error":
                        error = msg
                    if msg["type"] == "complete":
                        break
        assert error is not None
        assert error["data"]["retries"] == 3
        assert "gave up after 3 retries" in error["data"]["message"]


class _SlowApproveGraph:
    """get_state for the no-op-resume probe; ainvoke takes longer than the
    (shortened) heartbeat cadence."""

    def __init__(self, **kwargs):
        del kwargs

    def get_state(self, config):
        del config
        return SimpleNamespace(values={"messages": []}, next=(), tasks=[])

    async def ainvoke(self, command, config):
        import asyncio

        del command, config
        await asyncio.sleep(0.2)
        return SimpleNamespace(values={"messages": [SimpleNamespace(type="ai", content="ok")]})


class _InstantGraph:
    def __init__(self, **kwargs):
        del kwargs

    async def astream(self, input_data, config, stream_mode="values"):
        del input_data, config, stream_mode
        yield {"messages": [SimpleNamespace(type="ai", content="done", text="done")]}


class _AmbientSessionProbeGraph:
    """Stands in for create_agent(): records the ambient session id its
    astream runs under (the context the agent's tool calls inherit)."""

    seen = None

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")

    async def astream(self, input_data, config, stream_mode="values"):
        from clearwing.agent.tooling import current_session_id

        del input_data, config, stream_mode
        _AmbientSessionProbeGraph.seen = current_session_id()
        yield {
            "messages": [SimpleNamespace(type="ai", content="done", text="done")]
        }


class TestAmbientSessionAttribution:
    def test_turn_binds_session_id_for_spawned_tools(self, client):
        """Issue #41: the turn context must carry the session id so hunts
        launched by the agent's tools attribute their spend to this session
        (and land in this session's scoped footer), not to whatever other
        session happens to be running."""
        import json

        with patch("clearwing.ui.web.app.create_agent", _AmbientSessionProbeGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                session_id = None
                while True:
                    msg = json.loads(ws.receive_text())
                    if msg["type"] == "complete":
                        session_id = msg["data"].get("session_id")
                        break

        assert session_id
        assert _AmbientSessionProbeGraph.seen == session_id


class TestLlmProgressEdgeCases:
    def test_approve_turn_also_heartbeats(self, client, monkeypatch):
        import json

        monkeypatch.setattr("clearwing.ui.web.app._LLM_PROGRESS_INTERVAL_SECONDS", 0.05)
        with patch("clearwing.ui.web.app.create_agent", _SlowApproveGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "approve", "approved": True})
                types = []
                while True:
                    msg = json.loads(ws.receive_text())
                    types.append(msg["type"])
                    if msg["type"] == "complete":
                        break
        assert "llm_progress" in types
        assert types[-1] == "complete"

    def test_fast_turn_emits_no_heartbeat(self, client, monkeypatch):
        # The t=0 rule: the busy-rejection frame must stay the first frame
        # a racing second message sees, so the heartbeat never fires for a
        # turn that finishes within one interval.
        import json

        monkeypatch.setattr("clearwing.ui.web.app._LLM_PROGRESS_INTERVAL_SECONDS", 0.5)
        with patch("clearwing.ui.web.app.create_agent", _InstantGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "quick"})
                types = []
                while True:
                    msg = json.loads(ws.receive_text())
                    types.append(msg["type"])
                    if msg["type"] == "complete":
                        break
        assert "llm_progress" not in types


class TestSessionCostTeardown:
    """PR #44 review P2: the WS teardown forgets the session's tracker entry.

    Webui session ids are 8-hex UUID prefixes; in a long-lived process a
    colliding new session would inherit the stale spend (bogus scoped cost
    frames / limits). The handler's teardown finally calls
    ``CostTracker().forget_session(session_id)``.
    """

    def test_disconnect_forgets_session_cost_entry(self, client):
        import json

        from clearwing.observability.telemetry import CostTracker

        class _RecordingCostGraph:
            """Fake graph that books real tracker spend under its session
            id — the way hunts spawned inside the session's turn do."""

            session_id = None

            def __init__(self, **kwargs):
                self.session_id = kwargs.get("session_id")

            async def astream(self, input_data, config, stream_mode="values"):
                from clearwing.core.events import EventBus, EventType

                del input_data, config, stream_mode
                CostTracker().record_llm_call(
                    1000, 100, "glm-5.3", session_id=self.session_id
                )
                EventBus().emit(
                    EventType.COST_UPDATE,
                    {
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "cached_tokens": 0,
                        "cost": 0.001,
                        "total_cost_usd": 0.001,
                        "model": "glm-5.3",
                        "provider": "openai",
                        "session_id": self.session_id,
                        "elapsed_ms": 1,
                    },
                )
                yield {"messages": [SimpleNamespace(type="ai", content="done", text="done")]}

        with patch("clearwing.ui.web.app.create_agent", _RecordingCostGraph):
            with client.websocket_connect("/ws/agent", headers=AUTH) as ws:
                ws.send_json({"type": "start", "target": "10.0.0.9"})
                ws.send_json({"type": "message", "content": "run"})
                started = None
                while True:
                    msg = json.loads(ws.receive_text())
                    if msg["type"] == "started":
                        started = msg["session_id"]
                    if msg["type"] == "complete":
                        break
                # While connected, the session's entry exists.
                assert CostTracker().session_total(started) > 0.0

        # After the connection tears down, the entry is gone.
        assert CostTracker().session_total(started) == 0.0
        assert CostTracker().session_tokens(started) == (0, 0)
