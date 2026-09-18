"""FastAPI web UI backend for Clearwing."""

import asyncio
import hmac
import json
import logging
import os
import threading
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
from clearwing.agent.tooling import session_scope
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

# Backoff before the outbound writer retries a failed frame send (issue #11):
# a transient send error must not kill the writer for the rest of the session.
# Module-level so tests can shorten it.
_PUMP_SEND_RETRY_SECONDS = 0.5


def _ai_text_stats(values: dict[str, Any] | None) -> tuple[int, str]:
    """(count of AI messages carrying text, last such text) for a snapshot.

    The count matters as much as the text: a resume whose new reply is
    byte-identical to the previous turn's last AI text (a terse "Done.")
    still increments it, so a (count, last_text) fingerprint sees the new
    message where a text-only fingerprint would not.
    """
    messages = (values or {}).get("messages", [])
    count = 0
    last_text = ""
    for message in messages:
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
            count += 1
            last_text = content
    return count, last_text


def _last_ai_content(values: dict[str, Any] | None) -> str:
    """Last non-empty AI message text from a graph state snapshot."""
    return _ai_text_stats(values)[1]


def _is_foreign_session_message(payload: dict, current_session_id: str | None) -> bool:
    """True when a bus MESSAGE payload is stamped for ANOTHER session.

    LLM retry/fallback notices (issue #17) carry the emitting session's id
    on the payload; without this check every connected socket would see
    every session's retry chatter on the process-wide bus. Unstamped
    payloads (legacy emitters, session-less subsystems) stay broadcast —
    same rule the session-scoped cost frames follow.
    """
    origin = payload.get("session_id")
    return origin is not None and origin != current_session_id


def _state_fingerprint(graph_ref: Any, config_ref: dict) -> tuple[int, str] | None:
    """(AI-text-count, last-AI-text) fingerprint of the graph state (#33).

    Used to tell a turn that surfaced new assistant text from one that
    left the previous turn's last AI message as-is (stale resume). None
    only when the state cannot be read (e.g. graphs without ``get_state``)
    — callers treat that as unreadable and do not replay anything.
    """
    try:
        values = graph_ref.get_state(config_ref).values
    except Exception:
        return None
    return _ai_text_stats(values)


def _make_session_store():
    return memory_data.SessionStore()


def _make_cost_tracker():
    return telemetry.CostTracker()


def _cors_origins() -> list[str]:
    raw = os.environ.get("CLEARWING_WEB_CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _state_dir_status() -> tuple[bool, str]:
    """Probe the clearwing state directory for writability (issue #7).

    An unwritable CLEARWING_HOME (a container whose HOME is /nonexistent,
    a read-only volume) breaks SessionStore and every state-writing
    endpoint; /api/health must surface that instead of reporting "ok"
    while /api/sessions 500s on every call.

    The probe file name is unique per call: concurrent health polls used
    to race on one fixed `.health_probe` path (one caller's unlink hit
    another's write → FileNotFoundError → spurious 503s).
    """
    from clearwing.core.config import clearwing_home

    home = clearwing_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / f".health_probe.{uuid.uuid4().hex}"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return False, f"state dir {home} is not writable: {exc}"
    return True, ""


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
        # Issue #7: the state directory's writability is part of health —
        # degraded answers 503 so orchestration stops trusting a container
        # whose /api/sessions would 500 on every call. The detail stays
        # generic on the wire (this endpoint is unauthenticated — no
        # internal paths/errno); the full reason goes to the server log.
        ok, reason = _state_dir_status()
        if not ok:
            logger.warning("Health degraded: %s", reason)
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "service": "clearwing",
                    "detail": "state directory unavailable",
                },
            )
        return {"status": "ok", "service": "clearwing"}

    @app.get("/api/sessions")
    async def list_sessions():
        """List all known sessions."""
        store = _make_session_store()
        if not store.available:
            # Unauthenticated endpoint: generic detail on the wire, full
            # reason (paths, errno) in the server log only.
            logger.warning("Session store unavailable: %s", store.unavailable_reason)
            raise HTTPException(status_code=503, detail="session store unavailable")
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
        if not store.available:
            logger.warning("Session store unavailable: %s", store.unavailable_reason)
            raise HTTPException(status_code=503, detail="session store unavailable")
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
                       running totals (in-turn frames accumulate per call;
                       out-of-turn frames carry the tracker's totals for this
                       session — never the process-global ones). Frames
                       attributed to other sessions are not forwarded here.
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

        Every outbound frame — bus events, llm_progress heartbeats, and
        terminal frames alike — leaves through ONE writer coroutine consuming
        a single FIFO queue (issue #45), so no frame can reach the client
        after a terminal frame that was enqueued before it.
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
        # Single outbound queue (issue #45): every frame this connection
        # sends — bus events, llm_progress heartbeats, and terminal frames
        # (started/error/agent_message/complete/stopped, busy-rejections)
        # alike — is pre-serialized to wire text at enqueue time and
        # delivered by exactly one writer coroutine (`_writer_loop`) for the
        # whole connection lifetime. Entries are `(text, payload, fut)`:
        # - `text` is the exact JSON wire form; pre-serialization means a
        #   send can now only fail for transport reasons — the
        #   deterministic TypeError class (an unserializable payload
        #   poisoning send_json on every retry) is dropped at enqueue
        #   instead of stalling the queue forever;
        # - `payload` is the original frame dict (None for flush-only
        #   sentinels), kept so the #53 dedup gate can observe frames that
        #   were actually DELIVERED, never merely enqueued;
        # - `fut` (optional) is the enqueue-side flush handshake: it
        #   resolves True once the writer has sent the entry (or, for a
        #   sentinel, everything before it), and False as soon as a send
        #   attempt fails — the same single-strike signal the old inline
        #   `_safe_send` callers used to detect a gone client.
        outbound_queue: asyncio.Queue = asyncio.Queue()
        # The single writer task (created after the handler's helpers are
        # defined, before the receive loop starts). _send_frame and
        # _flush_writer consult it: once the writer is done, no queued
        # frame or sentinel will ever be processed, so new flush
        # handshakes must fail fast instead of parking forever (the
        # done-callback only drains futures that were pending at death).
        writer_task: asyncio.Task | None = None
        # Flush futures currently pending in the queue. The writer resolves
        # them all False the moment a send attempt fails: they provably
        # cannot be delivered until the stuck frame goes out, and without
        # this an enqueue-side `await fut` could hang forever on a dead
        # socket (the receive loop would never return to receive_text to
        # notice the disconnect).
        flush_futures: set[asyncio.Future] = set()
        transcript: SessionTranscript | None = None
        session_id: str | None = None
        # Every session id this connection has minted, plus the spend each
        # retired id had at retire time. A re-`start` retires previous ids
        # immediately; teardown reclaims a retired id ONLY when its tracker
        # entry still matches the recorded spend — an id re-minted by
        # another connection (8-hex collision) is that owner's entry now,
        # and unconditionally popping it would reset their accounting
        # (Codex PR-56 r2).
        connection_session_ids: list[str] = []
        retired_session_totals: dict[str, float] = {}
        # EventBus is a process-wide singleton whose payloads carry no session
        # id, so bus events can only be attributed while THIS session's turn
        # is running; record nothing outside the window.
        turn_state = {"active": False}

        # Session-scoped cost totals (issue #10): CostTracker is a
        # process-wide singleton whose running totals accumulate across
        # webui sessions. The per-session snapshot from the tracker is
        # authoritative: every cost_update frame this connection
        # forwards carries it (rewritten into a shallow copy for the
        # wire and the transcript), while the shared bus payload stays
        # untouched (metrics gauges keep reading the process-global
        # totals). The accumulator is written from worker threads
        # (on_event handlers run in the emitting thread) and reset from
        # the event loop (start frames), so every access takes the lock
        # (issue #48: the unlocked read-modify-write could drop
        # concurrent bookings).
        session_cost = {"cost_usd": 0.0, "tokens": 0}
        session_cost_lock = threading.Lock()

        # Codex PR-62 r2 (Finding 2): for cost_update frames the ENTIRE
        # sequence — scope-rewrite, assemble, dumps, schedule the enqueue,
        # record the transcript — runs under this lock (see the
        # cost_update branch in `on_event`). The enqueue order IS the wire
        # order (single writer, FIFO), so the snapshot read and publish
        # must be one serialized step: with the lock around the arithmetic
        # only, a worker holding a SMALLER total could schedule its
        # enqueue after a worker holding a larger one, and the wire totals
        # would regress.
        def _session_scope_cost_update_locked(data: dict) -> dict | None:
            """Returns the frame to enqueue, or None to drop it.

            Caller MUST hold ``session_cost_lock`` — this function mutates
            the accumulator without re-acquiring it (not an RLock; the
            publish path that calls it owns the acquisition).

            Frames now carry an attribution id (the runtime since PR #39,
            hunts since issue #41): only frames whose id matches THIS
            session are scoped here — a mismatching id (another webui
            session, a standalone hunt's sh-* id) is dropped so concurrent
            sessions never swallow each other's spend. Emissions without a
            session id (older callers) keep the transitional behaviour and
            still accumulate while this turn runs.

            Totals are TRACKER-AUTHORITATIVE (Codex PR-62 r3): the runtime
            books every call under the tracker's lock BEFORE emitting its
            frame (``CostTracker.record_llm_call``), so whichever turn
            window the handler runs in, the session snapshot already
            includes the call. Both branches therefore ASSIGN from one
            locked snapshot instead of the in-turn branch ``+=``-ing the
            frame's own per-call fields: a handler delayed across a turn
            boundary could not double-count a call an out-of-turn frame
            already captured (the r2 rebase race), and whatever order the
            handlers run in, the wire totals are rewritten to the same
            authoritative water level. A sessionless (None) id keeps the
            legacy in-turn ``+=`` transitional semantics — the tracker has
            no bucket to read there (out-of-turn it snapshots to zero,
            the accepted r1 behaviour).
            """
            origin = data.get("session_id")
            if origin is not None and origin != session_id:
                return None
            scoped = dict(data)
            if session_id is None and turn_state["active"]:
                # Legacy no-id transition: accumulate the frame's own
                # per-call fields while this turn runs (unchanged).
                call_cost = data.get("cost")
                per_call_tokens = (data.get("input_tokens") or 0) + (
                    data.get("output_tokens") or 0
                )
                if isinstance(call_cost, (int, float)):
                    session_cost["cost_usd"] += float(call_cost)
                if isinstance(per_call_tokens, int) and per_call_tokens > 0:
                    session_cost["tokens"] += per_call_tokens
                scoped["total_cost_usd"] = round(session_cost["cost_usd"], 10)
                scoped["total_tokens"] = session_cost["tokens"]
                return scoped
            # One locked snapshot, not two independent reads: a booking
            # landing between separate session_total and session_tokens
            # calls would yield a frame whose tokens include a call its
            # cost does not (Codex PR-62 r1). For a None id out of turn
            # this is (0, 0, 0) — the accepted r1 behaviour.
            scoped_cost, in_tokens, out_tokens = telemetry.CostTracker().session_snapshot(
                session_id
            )
            session_cost["cost_usd"] = scoped_cost
            session_cost["tokens"] = in_tokens + out_tokens
            scoped["total_cost_usd"] = round(scoped_cost, 10)
            scoped["total_tokens"] = in_tokens + out_tokens
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

        def _publish_bus_event(event_name: str, serializable: Any) -> None:
            """Assemble, pre-serialize, schedule, and record one bus event.

            Extracted from ``on_event`` (Codex PR-62 r2, Finding 2) so the
            cost_update path can run this whole publish sequence UNDER
            ``session_cost_lock``: the enqueue order IS the wire order
            (single writer, FIFO), so accumulate and publish must be one
            serialized step or the wire totals can regress. Every other
            event type publishes unlocked, exactly as before.
            """
            item = {"type": event_name, "data": serializable}
            # Pre-serialize BEFORE enqueue (issue #45): an
            # unserializable payload used to reach the writer and fail
            # send_json with the same TypeError on every retry — an
            # immortal frame blocking every later one. Dropping it here
            # keeps the queue moving; the shallow type check in
            # `on_event` cannot catch nested non-JSON values (e.g. a set
            # inside a dict), which is exactly what this dumps call
            # proves out.
            try:
                text = json.dumps(item)
            except (TypeError, ValueError):
                logger.error(
                    "Dropping unserializable %s frame — it would "
                    "stall the outbound writer; later frames still go out",
                    event_name,
                    exc_info=True,
                )
                # Dropped from the SOCKET, not from history: transcript
                # recording is unconditional and independent of the frame
                # gate (#53), and the report render falls back to
                # str()/default=str for values json cannot encode, so the
                # event stays legible there. Recording the dropped
                # tool_start also keeps its own tool_result's
                # content_length from mis-sticking onto the previous
                # tool entry.
                _record_in_transcript(event_name, serializable)
                return
            try:
                ws_loop.call_soon_threadsafe(
                    outbound_queue.put_nowait, (text, item, None)
                )
            except RuntimeError:
                # The loop is already closed — the socket is tearing
                # down; there is nothing left to deliver to.
                return
            _record_in_transcript(event_name, serializable)

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
                            # Codex PR-62 r2 (Finding 2): hold the session
                            # lock across the scope-rewrite AND the whole
                            # publish sequence (assembly, dumps, enqueue
                            # scheduling, transcript) so the wire order of
                            # cost frames always matches the snapshot-read
                            # order — totals never regress. Safe to call
                            # under the lock: call_soon_threadsafe is
                            # non-blocking, and the transcript's
                            # cost_update branch is dict writes. No RLock —
                            # the locked helpers simply never re-acquire.
                            with session_cost_lock:
                                scoped = _session_scope_cost_update_locked(
                                    serializable
                                )
                                if scoped is None:
                                    # A concurrent session's frame — never
                                    # reaches this socket or its transcript.
                                    return
                                _publish_bus_event(event_type_name, scoped)
                            return
                        if event_type_name == "agent_message" and isinstance(
                            serializable, dict
                        ):
                            # Retry/fallback notices (issue #17) carry the
                            # emitting session's id; every other socket
                            # must not receive this session's LLM chatter.
                            if _is_foreign_session_message(serializable, session_id):
                                return
                        _publish_bus_event(event_type_name, serializable)
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
            return await _send_frame(
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
            return await _send_frame(
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

        def _enqueue_frame(
            payload: dict, fut: asyncio.Future | None = None
        ) -> bool:
            # Pre-serialize at enqueue (issue #45): the writer only ever
            # sends text that already proved serializable, so a send
            # failure can only be a transport failure. A terminal frame
            # that fails here reports the same False an inline send_json
            # TypeError used to (the handler tears down instead of
            # queueing an immortal frame).
            try:
                text = json.dumps(payload)
            except (TypeError, ValueError):
                logger.error(
                    "Dropping unserializable outbound frame %r — it would "
                    "stall the outbound writer",
                    payload.get("type"),
                    exc_info=True,
                )
                if fut is not None and not fut.done():
                    fut.set_result(False)
                return False
            outbound_queue.put_nowait((text, payload, fut))
            return True

        async def _send_frame(payload: dict) -> bool:
            """Enqueue a terminal frame and wait for the writer to deliver it.

            Returns True when the writer sent it (every frame enqueued
            earlier went out first — FIFO, single writer, so a `complete`
            can never overtake queued bus frames again, issue #45). Returns
            False on the frame's first failed send attempt — the same
            single-strike "client is gone" signal the old inline
            `_safe_send` gave its callers. The writer keeps retrying the
            frame in the background, but a False here means teardown is
            imminent.
            """
            # Yield once before entering the FIFO: a bus event emitted just
            # before the turn ended (a late worker-thread booking) reaches
            # the queue via call_soon_threadsafe, and without this yield
            # the code between the last flush and this enqueue contains no
            # await — the terminal frame would enter the queue ahead of the
            # already-scheduled echo and deliver after `complete`
            # (issue #45's symptom, cross-thread flavor).
            await asyncio.sleep(0)
            if writer_task is None or writer_task.done():
                # No consumer will ever pick this frame up (the writer died
                # or was torn down): report the same single-strike False a
                # failed send gives, instead of registering a future that
                # nothing will resolve.
                return False
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            flush_futures.add(fut)
            try:
                if not _enqueue_frame(payload, fut):
                    return False
                return await fut
            finally:
                flush_futures.discard(fut)

        async def _flush_writer() -> None:
            """Wait until the writer has delivered every frame enqueued so far.

            Replaces the old `_drain_events` tail flushes (#53): a sentinel
            sits in the same FIFO queue, so the writer only reaches it after
            every earlier frame was actually sent (and noted for the dedup
            gate). If a send is failing, the writer resolves the sentinel
            False on its next failed attempt rather than parking here
            forever.
            """
            if writer_task is None or writer_task.done():
                # Same fast-fail as _send_frame: a dead writer never
                # reaches the sentinel, and awaiting it would park here.
                return
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            flush_futures.add(fut)
            try:
                # Flush-only sentinel: empty text, no payload.
                outbound_queue.put_nowait(("", None, fut))
                await fut
            finally:
                flush_futures.discard(fut)

        def _note_delivered_frame(msg: dict) -> None:
            # Issue #53: remember the last bus-delivered assistant echo so
            # a turn's closing inline agent_message can skip re-sending
            # text the client already received verbatim (approval-pause
            # turns let the writer win the race against the inline send).
            # Only frames actually DELIVERED count (issue #45: the writer
            # notes after its send succeeds, never at enqueue) — a
            # queued-but-lost echo must not suppress the authoritative
            # full-text send.
            # Assistant echoes carry type:"agent" (runtime emit_message);
            # warning/system notes map to the same frame type but are not
            # assistant text and must not participate in the dedup. The
            # runtime also echoes EMPTY assistant steps; an empty echo must
            # not overwrite the last real one (fail-open, review P3).
            if msg.get("type") != "agent_message":
                return
            data = msg.get("data")
            if not isinstance(data, dict):
                return
            content = data.get("content")
            if data.get("type") == "agent" and content:
                turn_state["last_bus_agent_text"] = content

        async def _safe_send_text(text: str) -> bool:
            try:
                await websocket.send_text(text)
            except Exception:
                # Client went away mid-send; the receive loop will observe the
                # disconnect. Never let a send failure kill the writer.
                return False
            return True

        async def _writer_loop() -> None:
            # The single outbound writer (issue #45): one coroutine owns
            # websocket.send for the whole connection, so frames leave in
            # FIFO order — a terminal frame enqueued last can never be
            # overtaken by an older queued frame the way the old inline
            # one-shot sends could deliver after `complete`. A failed send
            # is retried after a short backoff (issue #11): pre-serialization
            # at enqueue removed the deterministic failure class, so only
            # transient/disconnect failures remain, and a genuinely dead
            # connection is reaped by the receive loop's disconnect path,
            # which cancels this task.
            while True:
                text, payload, fut = await outbound_queue.get()
                if fut is not None and fut.cancelled():
                    # Codex PR-62 r3 (Finding 1): this entry's flush
                    # handshake was CANCELLED — the coroutine that
                    # enqueued it (a turn task killed by `stop`) is gone
                    # and no longer claims the frame. `_send_frame`'s
                    # finally already removed the future from
                    # flush_futures, but the entry kept the object, so
                    # cancelled() is still observable here. Sending it
                    # would deliver a STALE terminal frame — the
                    # cancelled turn's agent_message/complete — after
                    # the receive loop moved on, ahead of the stop
                    # handshake's authoritative `stopped` +
                    # `complete(status=stopped)`: a client treating the
                    # first complete as authoritative would read a
                    # wrong turn state. Skip it; a cancelled flush
                    # sentinel is equally dead (its waiter is gone) and
                    # skipping it changes nothing on the wire.
                    continue
                if payload is None:
                    # Flush sentinel: everything enqueued before it has been
                    # delivered (the writer is strictly sequential — a
                    # still-retrying frame would be holding us here).
                    if not fut.done():
                        fut.set_result(True)
                    continue
                while not await _safe_send_text(text):
                    # A send failure — this frame's own flush handshake (if
                    # any) and every handshake queued behind it cannot be
                    # delivered until this frame goes out, so report the
                    # single-strike False the old inline sends gave their
                    # callers instead of parking the enqueue side forever
                    # on a dead socket.
                    for pending in flush_futures:
                        if not pending.done():
                            pending.set_result(False)
                    await asyncio.sleep(_PUMP_SEND_RETRY_SECONDS)
                _note_delivered_frame(payload)
                if fut is not None and not fut.done():
                    fut.set_result(True)

        async def _llm_progress_heartbeat() -> None:
            # Issue #4: a turn against a slow or flaky endpoint used to be a
            # multi-minute silent stall. First frame at +interval (never at
            # t=0 — the busy-rejection frame must stay the first frame a
            # racing second message sees), then on the same cadence. Frames
            # ride the same outbound queue as everything else (issue #45);
            # the task itself is cancelled at turn end / teardown.
            start = time.monotonic()
            while True:
                await asyncio.sleep(_LLM_PROGRESS_INTERVAL_SECONDS)
                elapsed = int(time.monotonic() - start)
                _enqueue_frame(
                    {"type": "llm_progress", "data": {"elapsed_seconds": elapsed}}
                )

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
                # Issue #53: dedup only against echoes delivered within
                # THIS turn — a previous turn's echo must never suppress
                # this turn's inline send.
                turn_state.pop("last_bus_agent_text", None)
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
                # Issue #53, mirror order: on a normally-ending turn the
                # final step's bus echo is emitted with NO await boundary
                # before this point, so the dedup gate cannot have seen it
                # (the inline frame would go out first and the echo would
                # follow via the queue — a verbatim duplicate for texts
                # ≤200 chars). Yield one loop turn so the scheduled
                # enqueue lands, flush the writer (delivery, not just
                # dequeue), THEN decide. On pause turns the writer has
                # already delivered the echo and this is a no-op.
                await asyncio.sleep(0)
                await _flush_writer()
                # Stream events are produced by this turn, so non-empty
                # last_content is by construction fresh (issue #33:
                # produced_new reports it instead of a message count).
                produced_new = bool(last_content)
                if produced_new:
                    if transcript_ref:
                        transcript_ref.add_agent(last_content)
                    # Issue #53: approval-pause turns let the writer deliver
                    # the bus echo of this very text before the turn ends;
                    # re-sending it inline duplicated short messages
                    # verbatim. Skip only the redundant FRAME — the
                    # transcript above still records the text, and a long
                    # text (echo is a 200-char preview, never the full
                    # text) still gets its authoritative full send.
                    if turn_state.get("last_bus_agent_text") != last_content:
                        if not await _send_frame(
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
                if not await _send_frame(_error_payload(e)):
                    return
            finally:
                turn_state["active"] = False
            # Complete follows both success and failure, so the client never
            # waits on a turn that died mid-stream. A turn parked at an
            # approval gate completes with status "awaiting_approval", not
            # "ok" (issue #33). Enqueueing it behind everything already
            # queued (issue #45) means no earlier frame can follow it out.
            status = _turn_end_status(graph_ref, config_ref, sent_error)
            await _send_frame(
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
                # Fingerprint the AI-text stats before the resume (#33): a
                # resume that only appends tool output — or a stale approve
                # with nothing pending — leaves it unchanged, and the
                # previous turn's text must not be replayed as a fresh
                # agent_message. The (count, last_text) pair still catches a
                # new reply whose text is byte-identical to the last one.
                # An unreadable before-state is conservative: nothing is
                # replayed (the turn's own error path already covers real
                # failures).
                before = _state_fingerprint(graph_ref, config_ref)
                turn_state["active"] = True
                # Issue #53: dedup only against echoes delivered within
                # THIS turn (see _run_message_turn).
                turn_state.pop("last_bus_agent_text", None)
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
                if not await _send_frame(_error_payload(e)):
                    return
            else:
                # Issue #53, mirror order (see _run_message_turn): the
                # resumed step's echo may still be sitting in the
                # call_soon_threadsafe queue with no await boundary since
                # the emit — flush the writer before the dedup gate
                # decides.
                await asyncio.sleep(0)
                await _flush_writer()
                after = _ai_text_stats(getattr(snapshot, "values", None))
                produced_new = before is not None and after != before
                if produced_new:
                    content = after[1]
                    if transcript_ref:
                        transcript_ref.add_agent(content)
                    # Issue #53: skip the redundant inline frame when the
                    # bus echo of this text was already delivered verbatim
                    # during this resume (same gate as message turns).
                    if turn_state.get("last_bus_agent_text") != content:
                        if not await _send_frame(
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
            # "awaiting_approval", not "ok" (issue #33). Enqueueing it behind
            # everything already queued (issue #45) means no earlier frame
            # can follow it out.
            status = _turn_end_status(graph_ref, config_ref, sent_error)
            await _send_frame(
                _complete_payload(status=status, produced_new=produced_new)
            )

        async def _run_turn_scoped(
            bound_session_id: str | None, turn_fn, *turn_args
        ) -> None:
            # Ambient session attribution (issue #41): bind this session's id
            # around the turn so LLM spend from anything the turn spawns —
            # including sourcehunt hunts via the agent's tool calls — is
            # attributed to this session instead of leaking onto the bus
            # unscoped (where it would land in whichever other session has
            # an active turn).
            with session_scope(bound_session_id):
                await turn_fn(*turn_args)

        def _writer_died(task: asyncio.Task) -> None:
            # Zombie-connection guard: the writer is only ever cancelled in
            # teardown, so a death outside cancellation means it crashed.
            # Every pending flush handshake would then wait forever — and
            # because the handler parks inside `await fut`, it would never
            # return to receive_text to notice the client is gone, leaving
            # a connection that neither sends nor reaps. Resolving all
            # pending futures False routes each waiter onto the same
            # teardown path a failed send takes. add_done_callback runs on
            # the event loop like every other flush_futures touch, and the
            # normal cancel path finds the set already drained (callers
            # discard their future in `finally`), so this stays a no-op
            # there.
            if task.cancelled():
                return
            exc = task.exception()
            if exc is None:
                return
            logger.warning(
                "Outbound writer died unexpectedly; failing pending frame "
                "flushes to trigger connection teardown",
                exc_info=exc,
            )
            for pending in flush_futures:
                if not pending.done():
                    pending.set_result(False)

        writer_task = asyncio.create_task(_writer_loop())
        writer_task.add_done_callback(_writer_died)

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
                    # Issue #51: a start frame begins a NEW session on this
                    # connection — retire the previous session's tracker
                    # entry immediately instead of waiting for socket
                    # teardown (which only forgets the final id). Without
                    # this, every re-start left the earlier entry in the
                    # process-global tracker forever, and an 8-hex id
                    # collision later in the process lifetime would inherit
                    # the stale spend. The busy-frame guard above ensures no
                    # turn of the old session is still running.
                    tracker = telemetry.CostTracker()
                    for prior_id in connection_session_ids:
                        try:
                            # Record the spend at retire time so teardown can
                            # re-forget ONLY entries that are still ours (an
                            # id re-minted by another connection after a
                            # collision is that owner's entry now — Codex
                            # PR-56 r2).
                            retired_session_totals[prior_id] = (
                                tracker.session_total(prior_id)
                            )
                            tracker.forget_session(prior_id)
                        except Exception:
                            logger.debug(
                                "Failed to retire prior session cost entry",
                                exc_info=True,
                            )
                    session_id = uuid.uuid4().hex[:8]
                    connection_session_ids.append(session_id)
                    # A start frame begins a new session on this connection:
                    # re-arm the session-scoped cost totals (issue #10).
                    with session_cost_lock:
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
                        if not await _send_frame(
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

                    if not await _send_frame(
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
                        _run_turn_scoped(
                            session_id, _run_message_turn, graph, config, input_msg, transcript
                        )
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
                        _run_turn_scoped(
                            session_id, _run_approve_turn, graph, config, approved, transcript
                        )
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
                    # `stopped`/`complete` ride the same FIFO queue as the
                    # cancelled turn's pending bus frames (issue #45): they
                    # go out only after every earlier frame, without the old
                    # explicit tail drain.
                    if not await _send_frame(
                        {
                            "type": "stopped",
                            "data": {
                                "cancelled_turn": cancelled_turn,
                                "discarded_approval": discarded,
                            },
                        }
                    ):
                        break
                    if not await _send_frame(_complete_payload(status="stopped")):
                        break

                elif msg_type in ("message", "approve"):
                    # The frame needs an agent, but no start succeeded yet —
                    # never leave the client waiting in silence.
                    if not await _send_frame(
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
            writer_task.cancel()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass
            except BaseException:
                # The writer died on its own (e.g. a poisoned transport
                # raising a BaseException that _safe_send_text's Exception
                # guard cannot swallow) — its death was already logged by
                # the done-callback; joining must not abort the teardown
                # still ahead (transcript write, bus unsubscribe).
                logger.debug("Outbound writer shutdown error", exc_info=True)
            if transcript is not None:
                try:
                    transcript.write()
                except Exception:
                    logger.debug("Failed to write final session report", exc_info=True)
            # Cleanup subscriptions
            if bus and handlers:
                for et, h in handlers.items():
                    bus.unsubscribe(et, h)
            # PR #44 review P2: retire this session's cost/token entry —
            # 8-hex ids collide in a long-lived webui, and a stale entry
            # would hand the colliding session this session's spend. Retired
            # ids are reclaimed only when their entry still matches the
            # spend recorded at retire time: an entry that grew (a late
            # worker-thread booking) is skipped as residue, and an id since
            # re-minted by ANOTHER connection is that owner's entry now —
            # unconditionally popping it would reset their accounting.
            tracker = telemetry.CostTracker()
            for used_id in connection_session_ids:
                if used_id == session_id:
                    continue  # handled below
                recorded = retired_session_totals.get(used_id)
                if recorded is not None and tracker.session_total(used_id) == recorded:
                    tracker.forget_session(used_id)
            if session_id is not None:
                tracker.forget_session(session_id)

    return app
