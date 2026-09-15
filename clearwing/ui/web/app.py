"""FastAPI web UI backend for Clearwing."""

import asyncio
import hmac
import json
import logging
import os
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
from clearwing.ui.web.session_report import SessionTranscript, session_report_path

logger = logging.getLogger(__name__)


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
        - Server sends: {"type": "agent_message", "content": "..."}
        - Server sends: {"type": "tool_start", "tool": "scan_ports", "args": {...}}
        - Server sends: {"type": "tool_result", "tool": "scan_ports", "content": "..."}
        - Server sends: {"type": "flag_found", "flag": "...", "context": "..."}
        - Server sends: {"type": "cost_update", "cost_usd": 0.05, "tokens": 1000}
        - Server sends: {"type": "approval_needed", "prompt": "..."}
        - Server sends: {"type": "error", "message": "..."}
        - Server sends: {"type": "complete", "data": {"session_id": "...",
                       "report_path": "...", "report_url": "/api/reports/<id>"}}
          (emitted after each message/approve turn; report_path/report_url
          appear once the agent session has been started)
        """
        if not _ws_authorized(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()

        message_queue: asyncio.Queue = asyncio.Queue()
        transcript: SessionTranscript | None = None
        # EventBus is a process-wide singleton whose payloads carry no session
        # id, so bus events can only be attributed while THIS session's turn
        # is running; record nothing outside the window.
        turn_state = {"active": False}

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
                # CostTracker emits total_cost_usd / input_tokens / output_tokens.
                tokens = data.get("tokens")
                if tokens is None:
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
                        message_queue.put_nowait(
                            {"type": event_type_name, "data": serializable}
                        )
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
            while True:
                msg = await message_queue.get()
                if not await _safe_send(msg):
                    return

        async def _drain_events() -> None:
            while True:
                try:
                    msg = message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not await _safe_send(msg):
                    return

        def _complete_payload() -> dict:
            data: dict[str, Any] = {}
            if transcript is not None:
                try:
                    report_path = transcript.write()
                except Exception:
                    logger.debug("Failed to write session report", exc_info=True)
                else:
                    data = {
                        "session_id": transcript.session_id,
                        "report_path": str(report_path),
                        "report_url": f"/api/reports/{transcript.session_id}",
                    }
            return {"type": "complete", "data": data}

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
                    # Initialize agent
                    model = data.get("model", "claude-sonnet-4-6")
                    target = data.get("target", "")
                    session_id = uuid.uuid4().hex[:8]

                    try:
                        graph = create_agent(
                            model_name=model,
                            session_id=session_id,
                            base_url=data.get("base_url"),
                            api_key=data.get("api_key"),
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
                    transcript = SessionTranscript(session_id, target=target, model=model)

                    if not await _safe_send(
                        {
                            "type": "started",
                            "session_id": session_id,
                            "target": target,
                            "model": model,
                        }
                    ):
                        break

                elif msg_type == "message" and graph and config:
                    content = data.get("content", "")
                    if transcript:
                        transcript.add_user(content)
                    input_msg = {"messages": [{"role": "user", "content": content}]}

                    try:
                        turn_state["active"] = True
                        last_content = ""
                        async for event in graph.astream(input_msg, config, stream_mode="values"):
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
                        if last_content:
                            if transcript:
                                transcript.add_agent(last_content)
                            if not await _safe_send(
                                {
                                    "type": "agent_message",
                                    "data": {"content": last_content},
                                }
                            ):
                                break
                    except Exception as e:
                        logger.exception("Agent turn failed")
                        if transcript:
                            transcript.add_error(str(e))
                        if not await _safe_send(
                            {
                                "type": "error",
                                "data": {"message": str(e)},
                            }
                        ):
                            break
                    finally:
                        turn_state["active"] = False
                    # Complete follows both success and failure, so the
                    # client never waits on a turn that died mid-stream.
                    await _drain_events()
                    if not await _safe_send(_complete_payload()):
                        break

                elif msg_type == "approve" and graph and config:
                    approved = data.get("approved", False)
                    if transcript:
                        transcript.add_user(
                            f"[approval {'approved' if approved else 'denied'} by operator]"
                        )
                    resume_ok = True
                    try:
                        # Detect no-op resumes (stale approve with nothing
                        # pending): the graph reports the same message count.
                        try:
                            before = len(
                                (graph.get_state(config).values or {}).get("messages", [])
                            )
                        except Exception:
                            before = None
                        turn_state["active"] = True
                        snapshot = await graph.ainvoke(Command(resume=approved), config)
                    except Exception as e:
                        resume_ok = False
                        logger.exception("Agent resume failed")
                        if transcript:
                            transcript.add_error(str(e))
                        if not await _safe_send(
                            {
                                "type": "error",
                                "data": {"message": str(e)},
                            }
                        ):
                            break
                    finally:
                        turn_state["active"] = False
                    if resume_ok:
                        values = getattr(snapshot, "values", None)
                        messages_after = (values or {}).get("messages", [])
                        produced_new = before is None or len(messages_after) > before
                        if produced_new:
                            content = _last_ai_content(values)
                            if content:
                                if transcript:
                                    transcript.add_agent(content)
                                if not await _safe_send(
                                    {
                                        "type": "agent_message",
                                        "data": {"content": content},
                                    }
                                ):
                                    break
                    # Complete follows both success and failure, so the
                    # client never waits on a resume that died mid-stream.
                    await _drain_events()
                    if not await _safe_send(_complete_payload()):
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
