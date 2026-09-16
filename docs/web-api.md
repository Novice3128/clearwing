# Web API — `/ws/agent` event schema

The Clearwing web UI backend (`clearwing.ui.web.app`, started by
`clearwing serve`) exposes a FastAPI server with a single real-time
WebSocket endpoint. Commit `dd5f093` wired the in-process
[`EventBus`](architecture.md) into that WebSocket so external
consumers (dashboards, CI tailers, custom TUIs) can follow campaign /
sourcehunt / validator / disclosure / benchmark / eval progress
without polling the REST API.

This page documents every message that crosses the wire. Field types
are drawn from the dataclasses in `clearwing/core/event_payloads.py`
and the `emit_*` call sites — no "optional" marker appears here
unless the underlying payload type is genuinely `T | None`.

## Connection

| | |
|---|---|
| URL | `ws://<host>:<port>/ws/agent` (default host/port: whatever `clearwing serve` binds to) |
| Subprotocol | none — plain JSON-text frames |
| Auth | none when `CLEARWING_WEB_API_KEY` is unset. When set, the socket must present the key either as an `X-API-Key` header or as an `?api_key=` query parameter (browsers cannot set WebSocket headers, so the served frontend uses the query parameter, forwarding it from the page URL). Unauthorized sockets are closed with code `1008` before being accepted. |
| CORS | `allow_origins=["*"]` — the frontend is served from the same FastAPI app |

### Lifecycle

1. **Accept.** Client opens the WebSocket. The server calls
   `websocket.accept()` and subscribes handlers to the 14 `EventBus`
   event types listed in [Server → client events](#server--client-events).
2. **Start.** Client sends a `start` frame to create an agent and a
   session id. Server replies with a `started` frame.
3. **Stream.** Client sends `message` and `approve` frames; server
   streams `agent_message` frames from the LangGraph `astream` loop
   **plus** any bus-emitted events produced by the agent's tools,
   the sourcehunt pipeline, the validator, etc. Events are drained
   from an internal `asyncio.Queue` every ~100 ms between client
   receives.
4. **Disconnect.** Either side closes. On disconnect the server
   unsubscribes every bus handler it registered for this connection.

## Client → server messages

Every client frame is a JSON object with a `type` discriminator.

### `start`

Initializes an agent session. Must be sent before `message` or
`approve` (both require a live graph/config).

```json
{
  "type": "start",
  "target": "10.0.0.1",
  "model": "claude-sonnet-4-6",
  "base_url": "https://api.anthropic.com",
  "api_key": "sk-ant-..."
}
```

`target`, `base_url`, and `api_key` are forwarded to
`create_agent(...)`; they default to empty / `None` if omitted.
`model` empty or omitted means "defer to config.yaml / env"; a
non-empty value is an explicit choice that wins over the configured
model even when it equals the placeholder. The `started` frame echoes
the model that was actually configured (which may differ from the
frame's value).

### `message`

Sends a user turn into the ReAct graph.

```json
{"type": "message", "content": "scan ports on the target"}
```

`content` must be a string (or omitted). `null` / list / object values
are coerced to text rather than rejected. Only one turn may run at a
time: a `message` (or `approve`) sent while a turn is still running is
rejected with an inline `error` frame.

### `approve`

Resumes a graph that paused at an approval interrupt.

```json
{"type": "approve", "approved": true}
```

### `stop`

Cancels the running turn (if any) and discards any approval still
pending. The server answers the unanswered tool calls of the abandoned
batch so the session stays usable — the next `message` starts a fresh
turn instead of failing on an orphaned `tool_use`.

```json
{"type": "stop"}
```

Stopping is idempotent: a `stop` with nothing running still gets a
`stopped` frame with both flags false.

Limitation: `stop` cancels the agent loop promptly, but a synchronous
tool already in flight (most scanning tools run in a worker thread)
runs to completion on the server; its result is recorded as skipped.
LLM calls stop immediately.

## Server → client envelope

All bus-forwarded events share one envelope:

```ts
interface BusEnvelope<T> {
  type: string;   // one of the event names listed below
  data: T;        // payload — a plain JSON object (dataclasses are `asdict`-ed)
}
```

Four server frames do **not** use this envelope because they are
emitted inline by the WebSocket handler itself rather than forwarded
from the bus — `started`, the streaming `agent_message` reply, the
inline `error` frame, and `llm_progress`. Their shapes are documented
under [Inline server frames](#inline-server-frames).

Clients MUST ignore frames with an unrecognized `type` (and unknown
fields within known frames) — the server may add frames like
`llm_progress` at any time.

Payload serialization rules (`clearwing/ui/web/app.py` lines
303–317):

- `dict`, `list`, `str`, `int`, `float`, `bool`, `None` → passed
  through unchanged.
- dataclass (`__dataclass_fields__`) → `dataclasses.asdict(...)`.
- Anything else → `str(data)`.

## Server → client events

The bus supports 17 `EventType` values; the WebSocket handler
forwards 14 of them. Three are deliberately **not** forwarded:
`STATE_CHANGED`, `USER_INPUT`, `USER_COMMAND` (they describe
client-driven transitions the client already knows about).

### `agent_message`

Fires whenever the agent (or any subsystem) calls
`EventBus().emit_message(...)`.

Payload:

| Field | Type | Notes |
|---|---|---|
| `content` | `string` | Human-readable message text. |
| `type` | `string` | Message category — `"info"`, `"warn"`, `"error"`, or other caller-defined tag. Defaults to `"info"`. |

```json
{
  "type": "agent_message",
  "data": {"content": "Enumerating services on 10.0.0.1", "type": "info"}
}
```

Note: the inline non-bus `agent_message` frame produced by the
LangGraph streaming loop has a different payload — see
[Inline server frames](#inline-server-frames).

### `tool_start`

Fires when a tool invocation begins
(`emit_tool(name, "start", data)`).

| Field | Type | Notes |
|---|---|---|
| `tool` | `string` | Tool name (e.g. `"scan_ports"`). |
| `phase` | `string` | Always `"start"` for this event. |
| `data` | `any` | Caller-supplied context — usually the tool's input args. |

```json
{
  "type": "tool_start",
  "data": {
    "tool": "scan_ports",
    "phase": "start",
    "data": {"target": "10.0.0.1", "ports": "1-1024"}
  }
}
```

### `tool_result`

Fires when a tool invocation finishes
(`emit_tool(name, "end", data)`).

| Field | Type | Notes |
|---|---|---|
| `tool` | `string` | Tool name. |
| `phase` | `string` | Typically `"end"` / `"result"`. |
| `data` | `any` | Tool output, truncated for transport where applicable. |

```json
{
  "type": "tool_result",
  "data": {
    "tool": "scan_ports",
    "phase": "end",
    "data": {"open": [22, 80, 443], "duration_s": 4.2}
  }
}
```

### `flag_found`

Fires whenever the agent detects a flag-like token
(`emit_flag(flag, context)`).

| Field | Type | Notes |
|---|---|---|
| `flag` | `string` | The captured flag literal. |
| `context` | `any` | Surrounding evidence — usually the file path or tool output line. |

```json
{
  "type": "flag_found",
  "data": {
    "flag": "flag{sql_injection_on_login}",
    "context": "HTTP response body at /login?id=1'"
  }
}
```

### `cost_update`

Emitted by the cost tracker after every LLM call (`CostTracker.record_llm_call`).
Per-call counts plus session-scoped running totals for this connection
(the process-wide tracker's cross-session totals are deliberately NOT what
the wire reports — issue #10). Frames attributed to a concurrent session
are not forwarded to this socket at all; unscoped frames emitted while
this session has NO running turn (e.g. an `/api/operate` job in the same
process) pass through raw with the emitter's process-global totals, and
`elapsed_ms` is 0 unless the caller supplied a latency.

| Field | Type | Notes |
|---|---|---|
| `input_tokens` | `integer` | Prompt tokens for this call. |
| `output_tokens` | `integer` | Completion tokens for this call. |
| `cached_tokens` | `integer` | Subset of input served from the prompt cache. |
| `cost` | `number` | USD cost of this call (reference pricing). |
| `total_cost_usd` | `number` | Session-scoped running cost. |
| `total_tokens` | `number` | Session-scoped running token count. |
| `model` | `string` | Model that served the call. |
| `provider` | `string` | Provider name. |
| `elapsed_ms` | `integer` | Wall-clock latency. |

```json
{"type": "cost_update", "data": {"input_tokens": 23085, "output_tokens": 625,
 "cached_tokens": 0, "cost": 0.0351, "total_cost_usd": 0.0702,
 "total_tokens": 47420, "model": "glm-5.3", "provider": "openai",
 "elapsed_ms": 3410}}
```

### `approval_needed`

Fires when the guardrail / approval layer pauses on a destructive
operation and waits for a human `approve` frame.

| Field | Type | Notes |
|---|---|---|
| `prompt` | `string` | Human-readable description of the pending action. |

Other implementation-specific fields (tool name, args) may accompany
`prompt`; consumers should preserve unknown fields.

```json
{
  "type": "approval_needed",
  "data": {"prompt": "Run `msfconsole exploit/multi/http/struts2_namespace_ognl`?"}
}
```

### `error`

Fires on `EventType.ERROR`. Unlike the inline error frame, bus-routed
errors go through the standard `{type, data}` envelope.

| Field | Type | Notes |
|---|---|---|
| `data` | `any` | Whatever the emitter passed — typically a string message or a `{"message": ..., "where": ...}` dict. |

```json
{"type": "error", "data": "provider timeout after 30s"}
```

### `campaign_progress`

Fires from `clearwing/sourcehunt/campaign.py` whenever a project in a
multi-project campaign transitions (start, finish, error). Payload
is `CampaignProgressPayload`.

| Field | Type | Notes |
|---|---|---|
| `campaign_name` | `string` | |
| `projects_completed` | `integer` | Count of targets whose state is `"completed"`. |
| `projects_total` | `integer` | Total targets declared in the campaign config. |
| `current_project` | `string` | Repo currently being processed — `""` between projects. |
| `status` | `string` | One of `"running"`, `"completed"`, `"error"`. |
| `cost_usd` | `number` | Budget spent so far across the campaign. |
| `findings_total` | `integer` | Sum of findings across all projects. |
| `verified_total` | `integer` | Subset of `findings_total` that passed validator gating. |

```json
{
  "type": "campaign_progress",
  "data": {
    "campaign_name": "q2-oss-audit",
    "projects_completed": 3,
    "projects_total": 12,
    "current_project": "https://github.com/FFmpeg/FFmpeg",
    "status": "running",
    "cost_usd": 14.82,
    "findings_total": 27,
    "verified_total": 9
  }
}
```

### `sourcehunt_stage`

Fires at every sourcehunt pipeline transition in
`clearwing/sourcehunt/runner.py`: `preprocess`, `rank`, `hunt`,
`exploit`, `report`. Payload is `SourcehuntStagePayload`.

| Field | Type | Notes |
|---|---|---|
| `session_id` | `string` | Per-run id (hex). |
| `repo` | `string` | Target repo URL. |
| `stage` | `string` | `"preprocess"` \| `"rank"` \| `"hunt"` \| `"exploit"` \| `"report"`. |
| `status` | `string` | `"started"` \| `"completed"` \| `"degraded"` \| `"error"`. |
| `findings_so_far` | `integer` | Running count at the moment the stage emitted. |
| `cost_usd` | `number` | Running per-session LLM cost. |
| `detail` | `string` | Free-form stage context, e.g. `"Enumerated 1842 files"`. |

```json
{
  "type": "sourcehunt_stage",
  "data": {
    "session_id": "7f3a2c91",
    "repo": "https://github.com/FFmpeg/FFmpeg",
    "stage": "hunt",
    "status": "started",
    "findings_so_far": 0,
    "cost_usd": 0.0,
    "detail": "1842 files"
  }
}
```

### `hunt_progress`

Fires from the hunter pool (`clearwing/sourcehunt/pool.py`) after
every per-file worker returns. Payload is `HuntProgressPayload`.

| Field | Type | Notes |
|---|---|---|
| `session_id` | `string` | Matches the parent `sourcehunt_stage` session. |
| `tier` | `string` | Pool tier name (e.g. `"high"`, `"medium"`, `"deep"`). |
| `band` | `string` | File priority band within the tier. |
| `files_completed` | `integer` | Workers in `completed`/`error`/`timeout` state. |
| `files_total` | `integer` | Total files in the hunt config. |
| `findings_this_tier` | `integer` | Findings from `completed` workers in this pool. |
| `cost_usd` | `number` | Cumulative pool spend. |
| `budget_remaining` | `number` | Remaining budget — never negative (`max(0, budget − spent)`). |

```json
{
  "type": "hunt_progress",
  "data": {
    "session_id": "7f3a2c91",
    "tier": "high",
    "band": "A",
    "files_completed": 42,
    "files_total": 180,
    "findings_this_tier": 3,
    "cost_usd": 1.87,
    "budget_remaining": 8.13
  }
}
```

### `validation_result`

Fires from `clearwing/sourcehunt/validator.py` after the adversarial
second-pass validator issues a verdict. Payload is
`ValidationResultPayload`.

| Field | Type | Notes |
|---|---|---|
| `finding_id` | `string` | Stable id of the finding being validated. |
| `axes` | `object<string, boolean>` | Axis-name → pass/fail map (`reachability`, `exploitability`, `impact`, etc.). |
| `advance` | `boolean` | Whether the verdict promotes the finding past the validator gate. |
| `severity` | `string \| null` | Validator-assigned severity; `null` if the verdict refused to rate. |
| `evidence_level` | `string` | One of the evidence-ladder rungs — `"suspicion"`, `"static_corroboration"`, `"crash_reproduced"`, `"root_cause_explained"`, `"exploit_demonstrated"`, `"patch_validated"`. |

```json
{
  "type": "validation_result",
  "data": {
    "finding_id": "sh-7f3a2c91-0004",
    "axes": {"reachability": true, "exploitability": true, "impact": false},
    "advance": false,
    "severity": "medium",
    "evidence_level": "static_corroboration"
  }
}
```

### `disclosure_update`

Fires from `clearwing/ui/commands/disclose.py` on every disclosure
workflow transition. Payload is `DisclosureUpdatePayload`.

| Field | Type | Notes |
|---|---|---|
| `finding_id` | `string` | |
| `action` | `string` | `"validated"` \| `"rejected"` \| `"sent"` (extensible — other workflow transitions may be added). |
| `reviewer` | `string \| null` | Operator handle that triggered the action; `null` for automated transitions. |
| `days_remaining` | `integer \| null` | CVD-clock days left; `90` on `sent`, `null` until the clock starts. |
| `detail` | `string` | Free-form — notes, rejection reason, or status text such as `"CVD 90-day timeline started"`. |

```json
{
  "type": "disclosure_update",
  "data": {
    "finding_id": "sh-7f3a2c91-0004",
    "action": "sent",
    "reviewer": "rob",
    "days_remaining": 90,
    "detail": "CVD 90-day timeline started"
  }
}
```

### `benchmark_progress`

Fires from `clearwing/bench/ossfuzz.py` after every benchmark target
finishes. Payload is `BenchmarkProgressPayload`.

| Field | Type | Notes |
|---|---|---|
| `mode` | `string` | Benchmark mode name (e.g. `"ossfuzz"`). |
| `targets_completed` | `integer` | Targets processed so far. |
| `targets_total` | `integer` | Total targets in this benchmark run. |
| `current_project` | `string` | Project of the target that just finished. |
| `tier_distribution` | `object<string, integer>` | Tier-name → count of findings landing in that tier. |
| `cost_usd` | `number` | Cumulative benchmark cost. |

```json
{
  "type": "benchmark_progress",
  "data": {
    "mode": "ossfuzz",
    "targets_completed": 5,
    "targets_total": 20,
    "current_project": "libxml2",
    "tier_distribution": {"high": 1, "medium": 2, "low": 3},
    "cost_usd": 2.14
  }
}
```

### `eval_progress`

Fires from `clearwing/eval/preprocessing.py` on every eval run —
once per run with `status="running"` or `"cached"`, and once again
with `"completed"` or `"error"` when the run settles. Payload is
`EvalProgressPayload`.

| Field | Type | Notes |
|---|---|---|
| `project` | `string` | Project being evaluated. |
| `config_name` | `string` | Named config under test. |
| `run_index` | `integer` | 0-based run ordinal within this config. |
| `runs_total` | `integer` | Configured runs-per-config. |
| `configs_completed` | `integer` | Configs finished before the current one. |
| `configs_total` | `integer` | Total configs in the eval. |
| `status` | `string` | `"running"` \| `"cached"` \| `"completed"` \| `"error"`. |
| `cost_usd` | `number` | USD cost for this run — `0.0` for `"running"`/`"cached"`/`"error"`. |

```json
{
  "type": "eval_progress",
  "data": {
    "project": "FFmpeg",
    "config_name": "deep-reasoning-v2",
    "run_index": 2,
    "runs_total": 5,
    "configs_completed": 1,
    "configs_total": 4,
    "status": "completed",
    "cost_usd": 0.84
  }
}
```

## Inline server frames

These frames originate in the WebSocket handler itself rather than
the `EventBus`. They share the top-level `type` discriminator but
their fields sit alongside `type`, not nested under `data` (with three
exceptions — the inline streaming `agent_message`, inline `error`, and
`llm_progress` frames **do** nest under `data`).

### `llm_progress`

Emitted every 10 seconds while a message/approve turn is still running
(first frame at +10s) — this covers tool executions between LLM rounds
too — so a slow or flaky endpoint is a visible wait instead of a
multi-minute silent stall. The turn's terminal `agent_message` /
`error` / `complete` frame follows it.

```json
{"type": "llm_progress", "data": {"elapsed_seconds": 30}}
```

### `started`

Sent exactly once in response to a client `start` frame. `model` is
the model that was actually configured (resolved from config.yaml /
env when the start frame's model field was empty), not necessarily the
frame's value.

```json
{
  "type": "started",
  "session_id": "a1b2c3d4",
  "target": "10.0.0.1",
  "model": "glm-5.3"
}
```

### `agent_message` (inline)

Produced by the LangGraph streaming loop after each `message` turn.
Carries only the final assistant text; intermediate tool turns reach
the client as `tool_start` / `tool_result` bus frames.

```json
{
  "type": "agent_message",
  "data": {"content": "I scanned ports 1-1024; 22, 80, and 443 are open."}
}
```

### `error` (inline)

Produced when the `message` or `approve` handler catches an
exception while driving the graph. `retries` is the number of
transport retries the LLM layer consumed before giving up (0 for
fast-fail errors; absent on handler-rejection frames like busy or
missing-agent); the message carries a human-readable suffix when
retries happened.

```json
{
  "type": "error",
  "data": {"message": "ProviderTimeout: no response in 30s (gave up after 2 retries)", "retries": 2}
}
```

### `stopped`

Sent in response to a client `stop` frame (see [`stop`](#stop)).

```json
{
  "type": "stopped",
  "data": {
    "cancelled_turn": true,
    "discarded_approval": false
  }
}
```

A `complete` frame follows every `stopped` frame, same as after
`message` / `approve` turns.

### `complete`

Emitted after every `message` / `approve` / `stop` turn, regardless of
how the turn ended — the client must never be left waiting on a turn
that died mid-stream. The `data.status` field says how it ended:

| `status`              | Meaning                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| `"ok"`                | The turn ran to completion; nothing is pending.                         |
| `"awaiting_approval"` | The graph is suspended at an approval gate. The session is **not** finished — answer the pending `approval_needed` frame with `approve`, or discard it with `stop`. |
| `"stopped"`           | The turn ended because the operator sent `stop`.                        |
| `"error"`             | The turn already emitted an `error` frame this turn.                    |

`status` is backward compatible: clients written before the field
existed treat its absence as `"ok"`.

`produced_new` (boolean, present on `message` / `approve` turns) reports
whether the turn surfaced new assistant text (a fresh `agent_message`
frame was sent for it). `false` means no new assistant content arrived —
for example a stale `approve` with nothing pending, or a resume that
only appended tool output — and no previous turn's text was replayed.

`session_id` / `report_path` / `report_url` appear once the agent
session has been started (they ride along on every subsequent
`complete`).

```json
{
  "type": "complete",
  "data": {
    "status": "awaiting_approval",
    "produced_new": true,
    "session_id": "a1b2c3d4",
    "report_path": "results/sessions/a1b2c3d4/report.md",
    "report_url": "/api/reports/a1b2c3d4"
  }
}
```

## Events that are emitted but not forwarded

For completeness — these `EventType` values exist on the bus but the
`/ws/agent` handler does **not** subscribe to them, so they never
reach WebSocket clients:

- `state_changed`
- `user_input`
- `user_command`

If you need them, subscribe directly in-process via
`EventBus().subscribe(...)`.
