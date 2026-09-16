"""FastAPI web UI backend for Clearwing."""

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import clearwing.data.memory as memory_data
import clearwing.observability.telemetry as telemetry
from clearwing.agent.graph import create_agent
from clearwing.agent.operator import OperatorAgent, OperatorConfig
from clearwing.agent.runtime import Command
from clearwing.core.events import EventBus, EventType
from clearwing.observability import MetricsCollector
from clearwing.observability.integration import ObservabilityIntegration
from clearwing.ui.web.session_report import (
    SessionTranscript,
    coerce_text,
    session_report_path,
)

logger = logging.getLogger(__name__)

# Cadence of the llm_progress heartbeat frames during a running turn
# (issue #4). Module-level so tests can shorten it.
_LLM_PROGRESS_INTERVAL_SECONDS = 10

# Backoff before the event pump retries a failed frame send (issue #11): a
# transient send error must not kill the pump for the rest of the session.
# Module-level so tests can shorten it.
_PUMP_SEND_RETRY_SECONDS = 0.5


def _last_ai_content(values: dict[str, Any] | None) -> str:
    """Last non-empty AI message text from a graph state snapshot."""
    messages = (values or {}).get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", "") != "ai":
            continue
        content = message.content
        if isinstance(content, list):
            content = "\n".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if content:
            return content
    return ""


def _state_fingerprint(graph_ref: Any, config_ref: dict) -> str | None:
    """Last-AI-content fingerprint of the graph state (issue #33).

    Used to tell a turn that surfaced new assistant text from one that
    left the previous turn's last AI message as-is (stale resume). None
    when the state cannot be read (e.g. graphs without ``get_state``) —
    callers treat that as "no before-sample available".
    """
    try:
        values = graph_ref.get_state(config_ref).values
    except Exception:
        return None
    return _last_ai_content(values) or None


def _make_session_store():
    return memory_data.SessionStore()


def _make_cost_tracker():
    return telemetry.CostTracker()


def _cors_origins() -> list[str]:
    raw = os.environ.get("CLEARWING_WEB_CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Clearwing API",
        description="REST and WebSocket API for the Clearwing penetration testing agent",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auto-wire standard OTLP tracing when configured and flush the SDK batch
    # processor on graceful shutdown so nothing gets dropped.
    _obs = ObservabilityIntegration.bootstrap_from_env()
    if _obs is not None:
        @app.on_event("shutdown")
        def _flush_observability() -> None:
            _obs.disconnect()

    # ---------------------------------------------------------------
    # Authentication for privileged endpoints
    # ---------------------------------------------------------------

    _api_key = os.environ.get("CLEARWING_WEB_API_KEY")

    def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        api_key: str | None = Query(default=None),
    ) -> None:
        # Browsers cannot set headers on WebSocket/REST calls, so the query
        # param is accepted everywhere the header is (same as _ws_authorized).
        provided = x_api_key or api_key
        if not _api_key:
            # Should be unreachable because routes are not mounted without a key,
            # but keep a defensive fallback.
            raise HTTPException(status_code=503, detail="API key not configured")
        if not _key_matches(provided, _api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def _key_matches(provided: str | None, expected: str) -> bool:
        if provided is None:
            return False
        try:
            return hmac.compare_digest(provided.encode(), expected.encode())
        except (TypeError, AttributeError, UnicodeEncodeError):
            # Non-ASCII junk must be a 401, not a 500.
            return False

    def _ws_authorized(websocket: WebSocket) -> bool:
        provided = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
        return bool(_api_key and _key_matches(provided, _api_key))

    # Serve the single-page frontend
    _static_dir = Path(__file__).parent / "static"

    @app.get("/")
    async def index():
        return FileResponse(_static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    # In-memory session registry
    _sessions: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------------
    # REST endpoints
    # ---------------------------------------------------------------

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "clearwing"}

    @app.get("/api/sessions")
    async def list_sessions():
        """List all known sessions."""
        store = _make_session_store()
        sessions = store.list_sessions()
        return [
            {
                "session_id": s.session_id,
                "target": s.target,
                "model": s.model,
                "status": s.status,
                "start_time": str(s.start_time),
                "cost_usd": s.cost_usd,
                "token_count": s.token_count,
            }
            for s in sessions
        ]

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get details for a specific session."""
        store = _make_session_store()
        try:
            session = store.load(session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.session_id,
            "target": session.target,
            "model": session.model,
            "status": session.status,
            "start_time": str(session.start_time),
            "cost_usd": session.cost_usd,
            "token_count": session.token_count,
            "open_ports": session.open_ports,
            "services": session.services,
            "vulnerabilities": session.vulnerabilities,
            "exploit_results": session.exploit_results,
            "flags_found": session.flags_found,
        }

    @app.get("/api/metrics")
    async def get_metrics():
        """Get current metrics in JSON format."""
        tracker = _make_cost_tracker()
        summary = tracker.get_summary()
        return {
            "input_tokens": summary.input_tokens,
            "output_tokens": summary.output_tokens,
            "total_cost_usd": summary.total_cost_usd,
            "tool_calls": summary.tool_calls,
        }

    @app.get("/api/metrics/prometheus")
    async def get_prometheus_metrics():
        """Get metrics in Prometheus exposition format."""
        collector = MetricsCollector()
        return JSONResponse(
            content=collector.format_prometheus(),
            media_type="text/plain",
        )

    if not _api_key:
        logger.error(
            "CLEARWING_WEB_API_KEY is not set; refusing to mount /api/operate* "
            "and /ws/agent. Set the environment variable and restart to enable "
            "the operator endpoints."
        )

        @app.post("/api/operate")
        async def _operate_disabled_post():
            raise HTTPException(
                status_code=503,
                detail="Operator API disabled: CLEARWING_WEB_API_KEY is not set",
            )

        @app.get("/api/operate/{session_id}")
        async def _operate_disabled_get(session_id: str):
            raise HTTPException(
                status_code=503,
                detail="Operator API disabled: CLEARWING_WEB_API_KEY is not set",
            )

        @app.websocket("/ws/agent")
        async def _ws_disabled(websocket: WebSocket):
            await websocket.close(code=1008)

        return app

    @app.post("/api/operate", dependencies=[Depends(require_api_key)])
    async def start_operator(request_body: dict):
        """Start an autonomous Operator agent session.

        Requires the ``X-API-Key`` header to match ``CLEARWING_WEB_API_KEY``.

        Request body:
        {
            "target": "10.0.0.1",
            "goals": ["Scan ports", "Find vulnerabilities"],
            "model": "claude-sonnet-4-6",
            "max_turns": 50,
            "timeout_minutes": 30
        }

        ``auto_approve_exploits`` is ignored if supplied; exploit approval is
        always required.
        """
        target = request_body.get("target")
        goals = request_body.get("goals", [])
        if not target or not goals:
            raise HTTPException(status_code=400, detail="target and goals are required")

        session_id = uuid.uuid4().hex[:8]
        _sessions[session_id] = {
            "status": "running",
            "target": target,
            "goals": goals,
            "result": None,
        }

        # Run operator in background
        async def run_operator():
            try:
                config = OperatorConfig(
                    goals=goals,
                    target=target,
                    model=request_body.get("model", "claude-sonnet-4-6"),
                    base_url=request_body.get("base_url"),
                    api_key=request_body.get("api_key"),
                    max_turns=request_body.get("max_turns", 50),
                    timeout_minutes=request_body.get("timeout_minutes", 30),
                    # auto_approve_exploits is intentionally not accepted from
                    # the request body; a remote caller must never be able to
                    # bypass the exploit approval gate.
                    auto_approve_exploits=False,
                    lhost=request_body.get("lhost", "host.docker.internal"),
                    lport=request_body.get("lport", 9999),
                )
                operator = OperatorAgent(config)
                result = await operator.arun()
                _sessions[session_id]["status"] = result.status
                _sessions[session_id]["result"] = {
                    "status": result.status,
                    "turns": result.turns,
                    "findings": result.findings,
                    "flags_found": result.flags_found,
                    "cost_usd": result.cost_usd,
                    "tokens_used": result.tokens_used,
                    "duration_seconds": result.duration_seconds,
                    "escalation_question": result.escalation_question,
                    "error": result.error,
                }
            except Exception as e:
                _sessions[session_id]["status"] = "error"
                _sessions[session_id]["result"] = {"error": str(e)}

        asyncio.create_task(run_operator())

        return {"session_id": session_id, "status": "running"}

    # ---------------------------------------------------------------
    # Disclosure REST endpoints
    # ---------------------------------------------------------------

    @app.get("/api/disclosure/queue")
    async def disclosure_queue(state: str | None = None, repo: str | None = None):
        from clearwing.sourcehunt.disclosure_db import DisclosureDB
        db = DisclosureDB()
        try:
            return db.get_queue(state=state, repo_url=repo)
        finally:
            db.close()

    @app.post("/api/disclosure/{finding_id}/validate")
    async def disclosure_validate(finding_id: str, body: dict):
        from clearwing.sourcehunt.disclosure_db import DisclosureDB
        from clearwing.sourcehunt.disclosure_workflow import DisclosureWorkflow
        db = DisclosureDB()
        try:
            wf = DisclosureWorkflow(db)
            reviewer = body.get("reviewer", "web")
            notes = body.get("notes", "")
            wf.validate(finding_id, reviewer, notes)
            return {"status": "validated", "finding_id": finding_id}
        finally:
            db.close()

    @app.post("/api/disclosure/{finding_id}/reject")
    async def disclosure_reject(finding_id: str, body: dict):
        from clearwing.sourcehunt.disclosure_db import DisclosureDB
        from clearwing.sourcehunt.disclosure_workflow import DisclosureWorkflow
        db = DisclosureDB()
        try:
            wf = DisclosureWorkflow(db)
            reviewer = body.get("reviewer", "web")
            reason = body.get("reason", "")
            wf.reject(finding_id, reviewer, reason)
            return {"status": "rejected", "finding_id": finding_id}
        finally:
            db.close()

    @app.post("/api/disclosure/{finding_id}/send")
    async def disclosure_send(finding_id: str, body: dict):
        from clearwing.sourcehunt.disclosure_db import DisclosureDB
        from clearwing.sourcehunt.disclosure_workflow import DisclosureWorkflow
        db = DisclosureDB()
        try:
            wf = DisclosureWorkflow(db)
            templates = wf.send_disclosure(
                finding_id,
                reviewer=body.get("reviewer", "web"),
                reporter_name=body.get("reporter_name", "(your name)"),
                reporter_affiliation=body.get("reporter_affiliation", "(your affiliation)"),
                reporter_email=body.get("reporter_email", "(your email)"),
            )
            return {"status": "sent", "finding_id": finding_id, "templates": list(templates.keys())}
        finally:
            db.close()

    @app.get("/api/disclosure/status")
    async def disclosure_status():
        from clearwing.sourcehunt.disclosure_db import DisclosureDB
        from clearwing.sourcehunt.disclosure_workflow import DisclosureWorkflow
        db = DisclosureDB()
        try:
            wf = DisclosureWorkflow(db)
            return wf.get_dashboard()
        finally:
            db.close()

    @app.get("/api/operate/{session_id}", dependencies=[Depends(require_api_key)])
    async def get_operator_status(session_id: str):
        """Get the status of an operator session."""
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return _sessions[session_id]

    @app.get("/api/reports/{session_id}", dependencies=[Depends(require_api_key)])
    async def get_session_report(session_id: str):
        """Download the deterministic markdown report for an agent session."""
        import re

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise HTTPException(status_code=400, detail="Invalid session id")
        path = session_report_path(session_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Report not found")
        return FileResponse(path, media_type="text/markdown", filename=path.name)

    # ---------------------------------------------------------------
    # WebSocket endpoint for real-time streaming
    # ---------------------------------------------------------------

    @app.websocket("/ws/agent")
    async def agent_websocket(websocket: WebSocket):
        """WebSocket endpoint for interactive agent sessions.

        Protocol:
        - Client sends: {"type": "start", "target": "10.0.0.1", "model": "..."}
        - Client sends: {"type": "message", "content": "scan ports"}
        - Client sends: {"type": "approve", "approved": true}
        - Client sends: {"type": "stop"} to cancel the running turn
        - Server sends: {"type": "agent_message", "content": "..."}
        - Server sends: {"type": "tool_start", "tool": "scan_ports", "args": {...}}
        - Server sends: {"type": "tool_result", "tool": "scan_ports", "content": "..."}
        - Server sends: {"type": "flag_found", "flag": "...", "context": "..."}
        - Server sends: {"type": "cost_update", "data": {"input_tokens": int,
                       "output_tokens": int, "cached_tokens": int, "cost": float,
                       "total_cost_usd": float, "total_tokens": int,
                       "model": str, "provider": str, "elapsed_ms": int}} —
                       cost/tokens are per call; total_* are session-scoped
                       running totals for this connection while its turn is
                       active (frames attributed to other sessions are not
                       forwarded here; out-of-turn unscoped frames pass
                       through raw)
        - Server sends: {"type": "approval_needed", "prompt": "..."}
        - Server sends: {"type": "llm_progress", "data": {"elapsed_seconds": int}}
          every ~10s while a turn runs (first at +10s), so slow LLM calls
          are a visible wait instead of a silent stall
        - Server sends: {"type": "error", "data": {"message": "...",
                       "retries": int}} — retries is the transport retry
                       count consumed before giving up (0/absent on
                       fast-fail and handler-rejection frames)
        - Server sends: {"type": "stopped", "data": {"cancelled_turn": bool,
                       "discarded_approval": bool}} after a stop frame
        - Server sends: {"type": "complete", "data": {"status": "ok" |
                       "awaiting_approval" | "stopped" | "error",
                       "produced_new": bool, "session_id": "...",
                       "report_path": "...", "report_url": "/api/reports/<id>"}}
          (emitted after each message/approve/stop turn; report_path/report_url
          appear once the agent session has been started; status "ok" is
          assumed by clients when the field is absent — see docs/web-api.md)

        Turns run as background tasks so the receive loop stays responsive;
        a `message`/`approve` sent while a turn is still running is rejected
        with an error frame. On stop or client disconnect the running turn is
        cancelled and any pending approval is discarded.
        """
        if not _ws_authorized(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()

        # Sync tools emit bus events from worker threads (asyncio.to_thread);
        # asyncio.Queue is not thread-safe, so frames must hop onto this
        # loop via call_soon_threadsafe instead of a raw cross-thread
        # put_nowait (issue #11: such frames could be stranded until the
        # next loop wakeup, or lost to the queue's internals).
        ws_loop = asyncio.get_running_loop()
        message_queue: asyncio.Queue = asyncio.Queue()
        transcript: SessionTranscript | None = None
        session_id: str | None = None
        # EventBus is a process-wide singleton whose payloads carry no session
        # id, so bus events can only be attributed while THIS session's turn
        # is running; record nothing outside the window.
        turn_state = {"active": False}

        # Session-scoped cost totals (issue #10): CostTracker is a
        # process-wide singleton whose running totals accumulate across
        # webui sessions. While THIS session's turn is running, accumulate
        # the per-call cost/tokens and rewrite a shallow copy for the wire
        # and the transcript; the shared bus payload itself stays untouched
        # (metrics gauges keep reading the process-global totals).
        session_cost = {"cost_usd": 0.0, "tokens": 0}

        def _session_scope_cost_update(data: dict) -> dict | None:
            """Returns the frame to enqueue, or None to drop it.

            Outside this session's turn the bus is unattributable
            (process-wide singleton): unscoped frames pass through raw.
            Frames carrying ANOTHER session's id never belong on this
            socket — dropping them keeps a concurrent session's spend from
            flashing into this session's footer. Emissions without a
            session id (hunter, older callers) still accumulate while this
            turn runs.
            """
            if not turn_state["active"]:
                origin = data.get("session_id")
                if origin is not None and origin != session_id:
                    return None
                return data
            origin = data.get("session_id")
            if origin is not None and origin != session_id:
                return None
            call_cost = data.get("cost")
            if isinstance(call_cost, (int, float)):
                session_cost["cost_usd"] += float(call_cost)
            per_call_tokens = (data.get("input_tokens") or 0) + (
                data.get("output_tokens") or 0
            )
            if isinstance(per_call_tokens, int) and per_call_tokens > 0:
                session_cost["tokens"] += per_call_tokens
            scoped = dict(data)
            scoped["total_cost_usd"] = round(session_cost["cost_usd"], 10)
            scoped["total_tokens"] = session_cost["tokens"]
            return scoped

        def _record_in_transcript(event_name: str, data: Any) -> None:
            """Mirror bus events into the session report transcript."""
            if transcript is None or not turn_state["active"] or not isinstance(data, dict):
                return
            if event_name == "tool_start":
                transcript.add_tool(
                    data.get("tool") or data.get("name") or "tool", args=data.get("args")
                )
            elif event_name == "tool_result":
                if transcript.tool_calls and transcript.tool_calls[-1].get("content_length") is None:
                    transcript.tool_calls[-1]["content_length"] = data.get("content_length")
            elif event_name == "cost_update":
                # A concurrent session's frame passing through raw must not
                # land in this session's report (last-write-wins would
                # otherwise let a foreign total be the recorded one).
                origin = data.get("session_id")
                if origin is not None and origin != session_id:
                    return
                # The session-scoped copy carries running totals; fall back
                # to the legacy flat keys, then per-call sums, for raw
                # frames that skipped the adapter.
                tokens = data.get("total_tokens")
                if not isinstance(tokens, int):
                    tokens = data.get("tokens")
                if not isinstance(tokens, int):
                    tokens = (data.get("input_tokens") or 0) + (data.get("output_tokens") or 0)
                transcript.set_cost(
                    data.get("total_cost_usd", data.get("cost_usd")), tokens
                )
            elif event_name == "error":
                transcript.add_error(data.get("message") or str(data))

        # Subscribe to EventBus and forward events to the WebSocket
        try:
            bus = EventBus()

            def on_event(event_type_name: str):
                def handler(data):
                    try:
                        if isinstance(data, dict | list | str | int | float | bool | type(None)):
                            serializable = data
                        elif hasattr(data, "__dataclass_fields__"):
                            from dataclasses import asdict
                            serializable = asdict(data)
                        else:
                            serializable = str(data)
                        if (
                            event_type_name == "cost_update"
                            and isinstance(serializable, dict)
                        ):
                            serializable = _session_scope_cost_update(serializable)
                            if serializable is None:
                                # A concurrent session's frame — never
                                # reaches this socket or its transcript.
                                return
                        item = {"type": event_type_name, "data": serializable}
                        try:
                            ws_loop.call_soon_threadsafe(
                                message_queue.put_nowait, item
                            )
                        except RuntimeError:
                            # The loop is already closed — the socket is
                            # tearing down; there is nothing left to
                            # deliver to.
                            return
                        _record_in_transcript(event_type_name, serializable)
                    except Exception:
                        logger.debug("Failed to enqueue event", exc_info=True)

                return handler

            handlers = {}
            event_map = {
                EventType.MESSAGE: "agent_message",
                EventType.TOOL_START: "tool_start",
                EventType.TOOL_RESULT: "tool_result",
                EventType.FLAG_FOUND: "flag_found",
                EventType.COST_UPDATE: "cost_update",
                EventType.ERROR: "error",
                EventType.APPROVAL_NEEDED: "approval_needed",
                EventType.CAMPAIGN_PROGRESS: "campaign_progress",
                EventType.SOURCEHUNT_STAGE: "sourcehunt_stage",
                EventType.HUNT_PROGRESS: "hunt_progress",
                EventType.VALIDATION_RESULT: "validation_result",
                EventType.DISCLOSURE_UPDATE: "disclosure_update",
                EventType.BENCHMARK_PROGRESS: "benchmark_progress",
                EventType.EVAL_PROGRESS: "eval_progress",
            }
            for et, name in event_map.items():
                h = on_event(name)
                handlers[et] = h
                bus.subscribe(et, h)

        except ImportError:
            bus = None
            handlers = {}

        graph = None
        config = None
        turn_task: asyncio.Task | None = None
        # Target from the start frame; rides along with every message so the
        # graph state (episodic memory, system prompt recall) can attribute
        # turns to the target the operator declared.
        handler_target = ""

        def _turn_active() -> bool:
            return turn_task is not None and not turn_task.done()

        def _graph_has_pending(graph_ref: Any, config_ref: dict) -> bool:
            # Fake graphs in tests may lack get_state; treat as no pending.
            try:
                return bool(getattr(graph_ref.get_state(config_ref), "next", ()))
            except Exception:
                return False

        async def _reject_busy_frame() -> bool:
            """True when the handler may keep serving; False when the client is gone."""
            return await _safe_send(
                {
                    "type": "error",
                    "data": {
                        "message": (
                            "A turn is already running — "
                            'send {"type": "stop"} to cancel it first'
                        )
                    },
                }
            )

        async def _reject_pending_approval_frame() -> bool:
            return await _safe_send(
                {
                    "type": "error",
                    "data": {
                        "message": (
                            "An approval is still pending — answer it with "
                            '{"type": "approve", "approved": true|false} or '
                            'discard it with {"type": "stop"} first'
                        )
                    },
                }
            )

        async def _safe_send(payload: dict) -> bool:
            try:
                await websocket.send_json(payload)
                return True
            except Exception:
                # Client went away mid-send; the receive loop will observe the
                # disconnect. Never let a send failure kill the handler.
                return False

        async def _pump_events() -> None:
            # Forward queued bus events while the agent loop is running —
            # the receive loop below cannot drain the queue mid-turn, so
            # without this pump, tool progress and approvals arrive late.
            # A failed send must not kill the pump for the rest of the
            # session (issue #11): retry the same frame after a short
            # backoff. A genuinely dead connection is reaped by the
            # receive loop's disconnect path, which cancels this task.
            while True:
                msg = await message_queue.get()
                # Retry the SAME frame until it goes out (or the task is
                # cancelled by the disconnect path).
                while not await _safe_send(msg):
                    await asyncio.sleep(_PUMP_SEND_RETRY_SECONDS)

        async def _drain_events() -> None:
            while True:
                try:
                    msg = message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not await _safe_send(msg):
                    return

        async def _llm_progress_heartbeat() -> None:
            # Issue #4: a turn against a slow or flaky endpoint used to be a
            # multi-minute silent stall. First frame at +interval (never at
            # t=0 — the busy-rejection frame must stay the first frame a
            # racing second message sees), then on the same cadence; exit
            # silently once the socket is gone.
            start = time.monotonic()
            while True:
                await asyncio.sleep(_LLM_PROGRESS_INTERVAL_SECONDS)
                elapsed = int(time.monotonic() - start)
                if not await _safe_send(
                    {"type": "llm_progress", "data": {"elapsed_seconds": elapsed}}
                ):
                    return

        def _cancel_heartbeat(heartbeat: asyncio.Task) -> None:
            # Fire-and-forget: awaiting the cancel inside the turn's cleanup
            # would let a second cancellation (stop/disconnect) interrupt it.
            heartbeat.cancel()

            def _swallow(task: asyncio.Task) -> None:
                if not task.cancelled() and task.exception() is not None:
                    logger.debug("llm_progress heartbeat ended: %s", task.exception())

            heartbeat.add_done_callback(_swallow)

        def _error_payload(e: Exception) -> dict:
            # _with_retries attaches the retry count to the final exception
            # so the operator can tell a fast-fail from an exhausted
            # backoff (issue #4).
            attempts = getattr(e, "_clearwing_attempts", 0) or 0
            message = str(e)
            if attempts:
                message += f" (gave up after {attempts} retr{'y' if attempts == 1 else 'ies'})"
            return {"type": "error", "data": {"message": message, "retries": attempts}}

        def _complete_payload(
            status: str = "ok", produced_new: bool | None = None
        ) -> dict:
            # `status` tells the client how the turn actually ended (issue
            # #33): "ok" (turn finished), "awaiting_approval" (graph is
            # suspended at an approval gate — the session is NOT finished),
            # "stopped" (operator stop), "error" (an error frame was already
            # sent this turn). Clients written before the field existed
            # treat its absence as "ok".
            data: dict[str, Any] = {"status": status}
            if produced_new is not None:
                data["produced_new"] = produced_new
            if transcript is not None:
                try:
                    report_path = transcript.write()
                except Exception:
                    logger.debug("Failed to write session report", exc_info=True)
                else:
                    data.update(
                        {
                            "session_id": transcript.session_id,
                            "report_path": str(report_path),
                            "report_url": f"/api/reports/{transcript.session_id}",
                        }
                    )
            return {"type": "complete", "data": data}

        def _turn_end_status(
            graph_ref: Any, config_ref: dict, sent_error: bool
        ) -> str:
            if _graph_has_pending(graph_ref, config_ref):
                return "awaiting_approval"
            if sent_error:
                return "error"
            return "ok"

        async def _run_message_turn(
            graph_ref: Any,
            config_ref: dict,
            input_msg: dict[str, Any],
            transcript_ref: SessionTranscript | None,
        ) -> None:
            # Parameters bind the session objects at task creation: a `start`
            # frame arriving mid-turn rebinds the handler variables but must
            # not swap the graph under a running turn.
            sent_error = False
            produced_new = False
            try:
                turn_state["active"] = True
                last_content = ""
                heartbeat = asyncio.create_task(_llm_progress_heartbeat())
                try:
                    async for event in graph_ref.astream(
                        input_msg, config_ref, stream_mode="values"
                    ):
                        msgs = event.get("messages", [])
                        if msgs:
                            last = msgs[-1]
                            if hasattr(last, "content") and last.type == "ai":
                                c = last.content
                                if isinstance(c, list):
                                    c = "\n".join(
                                        p["text"]
                                        for p in c
                                        if isinstance(p, dict) and p.get("type") == "text"
                                    )
                                if c:
                                    last_content = c
                finally:
                    _cancel_heartbeat(heartbeat)
                # Stream events are produced by this turn, so non-empty
                # last_content is by construction fresh (issue #33:
                # produced_new reports it instead of a message count).
                produced_new = bool(last_content)
                if produced_new:
                    if transcript_ref:
                        transcript_ref.add_agent(last_content)
                    if not await _safe_send(
                        {
                            "type": "agent_message",
                            "data": {"content": last_content},
                        }
                    ):
                        return
            except Exception as e:
                sent_error = True
                logger.exception("Agent turn failed")
                if transcript_ref:
                    transcript_ref.add_error(str(e))
                if not await _safe_send(_error_payload(e)):
                    return
            finally:
                turn_state["active"] = False
            # Complete follows both success and failure, so the client never
            # waits on a turn that died mid-stream. A turn parked at an
            # approval gate completes with status "awaiting_approval", not
            # "ok" (issue #33).
            status = _turn_end_status(graph_ref, config_ref, sent_error)
            await _drain_events()
            await _safe_send(
                _complete_payload(status=status, produced_new=produced_new)
            )

        async def _run_approve_turn(
            graph_ref: Any,
            config_ref: dict,
            approved: bool,
            transcript_ref: SessionTranscript | None,
        ) -> None:
            sent_error = False
            produced_new = False
            try:
                # Fingerprint the last AI text before the resume (issue #33):
                # a resume that only appends tool output — or a stale approve
                # with nothing pending — leaves it unchanged, and the
                # previous turn's text must not be replayed as a fresh
                # agent_message. The old message-count probe was permanently
                # True whenever get_state failed and silently replayed.
                before = _state_fingerprint(graph_ref, config_ref)
                turn_state["active"] = True
                heartbeat = asyncio.create_task(_llm_progress_heartbeat())
                try:
                    snapshot = await graph_ref.ainvoke(Command(resume=approved), config_ref)
                finally:
                    _cancel_heartbeat(heartbeat)
            except Exception as e:
                sent_error = True
                logger.exception("Agent resume failed")
                if transcript_ref:
                    transcript_ref.add_error(str(e))
                if not await _safe_send(_error_payload(e)):
                    return
            else:
                values = getattr(snapshot, "values", None)
                after = _last_ai_content(values) or None
                produced_new = after is not None and after != before
                if produced_new:
                    content = after
                    if transcript_ref:
                        transcript_ref.add_agent(content)
                    if not await _safe_send(
                        {
                            "type": "agent_message",
                            "data": {"content": content},
                        }
                    ):
                        return
            finally:
                turn_state["active"] = False
            # Complete follows both success and failure, so the client never
            # waits on a resume that died mid-stream. A resume that parks the
            # graph at the next approval gate completes with
            # "awaiting_approval", not "ok" (issue #33).
            status = _turn_end_status(graph_ref, config_ref, sent_error)
            await _drain_events()
            await _safe_send(
                _complete_payload(status=status, produced_new=produced_new)
            )

        pump = asyncio.create_task(_pump_events())

        try:
            while True:
                # Receive client message with timeout
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    data = json.loads(raw)
                except asyncio.TimeoutError:
                    continue
                except (WebSocketDisconnect, json.JSONDecodeError):
                    break

                msg_type = data.get("type")

                if msg_type == "start":
                    if _turn_active():
                        if not await _reject_busy_frame():
                            break
                        continue
                    # Initialize agent. An empty model field means "defer to
                    # config.yaml/env"; a non-empty one is an explicit choice
                    # that must win over the configured model (issue #22).
                    # Non-string junk counts as "not provided" rather than
                    # crashing the handler before the error frame can be sent.
                    raw_model = data.get("model")
                    model = (
                        raw_model.strip() or None
                        if isinstance(raw_model, str)
                        else None
                    )
                    target = data.get("target", "")
                    handler_target = target
                    session_id = uuid.uuid4().hex[:8]
                    # A start frame begins a new session on this connection:
                    # re-arm the session-scoped cost totals (issue #10).
                    session_cost.update(cost_usd=0.0, tokens=0)

                    try:
                        graph = create_agent(
                            model_name=model,
                            session_id=session_id,
                            base_url=data.get("base_url"),
                            api_key=data.get("api_key"),
                            model_explicit=model is not None,
                        )
                    except Exception as e:
                        logger.exception("Failed to create agent for ws session")
                        graph = None
                        config = None
                        if not await _safe_send(
                            {
                                "type": "error",
                                "data": {"message": f"Failed to start agent: {e}"},
                            }
                        ):
                            break
                        continue
                    config = {"configurable": {"thread_id": f"ws-{session_id}"}}
                    # Report the model that actually got configured (config
                    # fallback may resolve a different one than the frame).
                    resolved_model = (
                        getattr(getattr(graph, "llm", None), "model_name", None)
                        or model
                        or ""
                    )
                    transcript = SessionTranscript(
                        session_id, target=target, model=resolved_model
                    )

                    if not await _safe_send(
                        {
                            "type": "started",
                            "session_id": session_id,
                            "target": target,
                            "model": resolved_model,
                        }
                    ):
                        break

                elif msg_type == "message" and graph and config:
                    if _turn_active():
                        if not await _reject_busy_frame():
                            break
                        continue
                    if _graph_has_pending(graph, config):
                        # A user message on top of a suspended tool batch
                        # would orphan its tool_use (provider 400 next turn).
                        if not await _reject_pending_approval_frame():
                            break
                        continue

                    # WS clients can send content: null / list / object; the
                    # transcript and the LLM both need a plain string.
                    content = coerce_text(data.get("content", ""))
                    if transcript:
                        transcript.add_user(content)
                    input_msg: dict[str, Any] = {
                        "messages": [{"role": "user", "content": content}]
                    }
                    if handler_target:
                        input_msg["target"] = handler_target

                    turn_task = asyncio.create_task(
                        _run_message_turn(graph, config, input_msg, transcript)
                    )

                elif msg_type == "approve" and graph and config:
                    if _turn_active():
                        if not await _reject_busy_frame():
                            break
                        continue

                    approved = data.get("approved", False)
                    if transcript:
                        transcript.add_user(
                            f"[approval {'approved' if approved else 'denied'} by operator]"
                        )

                    turn_task = asyncio.create_task(
                        _run_approve_turn(graph, config, approved, transcript)
                    )

                elif msg_type == "stop":
                    cancelled_turn = _turn_active()
                    if cancelled_turn:
                        turn_task.cancel()
                        try:
                            await turn_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            logger.debug("Cancelled turn task raised", exc_info=True)
                    turn_task = None
                    discarded = False
                    if graph is not None and config is not None:
                        # Covers a turn paused awaiting approve (no running
                        # task to cancel) and any dangling tool batch the
                        # cancellation path could not reach.
                        try:
                            discarded = bool(graph.discard_interrupt(config))
                        except Exception:
                            logger.debug("Failed to discard pending interrupt", exc_info=True)
                    if transcript and (cancelled_turn or discarded):
                        transcript.add_error("[session stopped by operator]")
                    await _drain_events()
                    if not await _safe_send(
                        {
                            "type": "stopped",
                            "data": {
                                "cancelled_turn": cancelled_turn,
                                "discarded_approval": discarded,
                            },
                        }
                    ):
                        break
                    if not await _safe_send(_complete_payload(status="stopped")):
                        break

                elif msg_type in ("message", "approve"):
                    # The frame needs an agent, but no start succeeded yet —
                    # never leave the client waiting in silence.
                    if not await _safe_send(
                        {
                            "type": "error",
                            "data": {
                                "message": "No active agent — send a start frame first"
                            },
                        }
                    ):
                        break

        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Agent websocket handler crashed")
        finally:
            # A disconnect must not leave the agent burning tokens in the
            # background: cancel the running turn and answer any pending
            # approval/tool batch before tearing the session down.
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
                try:
                    await turn_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug("Turn task cancellation error", exc_info=True)
            if graph is not None and config is not None:
                try:
                    graph.discard_interrupt(config)
                except Exception:
                    logger.debug("Failed to discard pending interrupt", exc_info=True)
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Event pump shutdown error", exc_info=True)
            if transcript is not None:
                try:
                    transcript.write()
                except Exception:
                    logger.debug("Failed to write final session report", exc_info=True)
            # Cleanup subscriptions
            if bus and handlers:
                for et, h in handlers.items():
                    bus.unsubscribe(et, h)

    return app
