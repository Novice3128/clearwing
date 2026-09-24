# Subsystems map (fork-local)

> Fork-local architecture map. The upstream `docs/architecture.md` stays
> untouched to avoid widening the merge-conflict surface; this file is the
> fork's own living snapshot. Any PR that changes a subsystem boundary, an
> API surface, or the deployment topology should update this file in the
> same PR.
>
> Snapshot verified against code at commit `fa04772` (tool count measured
> via `len(get_all_tools())`).

## Entry chain

```
clearwing (console script, pyproject.toml [project.scripts])
  └─ clearwing/__init__.py::main()          # bootstraps OTLP observability first
      └─ clearwing/ui/cli.py::CLI           # argparse dispatcher
          └─ clearwing/ui/commands/__init__.py::ALL_COMMANDS
              # 23 command modules, imported and registered by module name
```

A legacy `clearwing.py` shim still exists at the repo root (same `CLI()`
entry); new code should use the `clearwing` console script or
`python -m clearwing`.

## Subsystem table

21 packages plus one capability module under `clearwing/`:

| Path | Responsibility | Entry point | Key modules |
|---|---|---|---|
| `agent/` | ReAct agent runtime, tool registry, specialists | `graph.py::create_agent`, `runtime.py::NativeAgentGraph` | `runtime.py` (astream loop, parallel tool batches, approval interrupts, budget guards), `graph.py`, `prompts.py`, `tooling.py`, `tools/` (scan, exploit, recon, ops, data, meta, hunt), `specialists/` (recon/planner/exploit/blue/reporter agents), `operator.py` (autonomous OperatorAgent) |
| `analysis/` | Static source analysis helpers | `source_analyzer.py` | `source_analyzer.py`, `taint_tracker.py` |
| `bench/` | OSS-Fuzz crash benchmarking | `ossfuzz.py` | `ossfuzz.py`, `crash_classifier.py`, `results.py` |
| `capabilities.py` (module) | Runtime capability detection gating optional features | `capabilities.py` | detects memory summarization, event bus, telemetry, audit, knowledge-graph availability |
| `core/` | Engine, config, event bus, shared models | `engine.py::CoreEngine`, `events.py::EventBus` | `engine.py`, `config.py`, `events.py` + `event_payloads.py`, `logger.py`, `models.py`, `module_loader.py`, `skills/` |
| `crypto/` | Cryptographic attack/math helpers | `srp.py` | `srp.py`, `stats.py` |
| `data/` | Persistence: DB, knowledge graph, memory | `memory/session_store.py` | `database/models.py`, `knowledge/graph.py`, `memory/` (`episodic_memory.py`, `semantic_memory.py`, `session_store.py`, `summarizer.py`) |
| `eval/` | Evaluation harnesses and gates | `metrics.py` | `metrics.py`, `preprocessing.py`, `proof_flow.py`, `sourcehunt.py` |
| `exploitation/` | Exploit integrations and payload building | `exploiters/` | `exploiters/` (`metasploit_bridge.py`, `password_crackers.py`, `privilege_escalation.py`, `rce_exploits.py`), `payloads/` (`encoder.py`, `obfuscator.py`, `beacon.py`, `watermark.py`, `corpus.py`, `authorization.py`) |
| `findings/` | The `Finding` dataclass shared by every subsystem | `types.py` | `types.py` |
| `llm/` | LLM client, spend ledger, fallback chain | `native.py::AsyncLLMClient` | `native.py` (genai-pyo3 bridge, cache-breakpoint context notes), `budget.py` (SpendLedger), `fallback.py::FallbackChain`, `chat.py`, `json_extract.py`, `messages.py` |
| `mcp/` | Model Context Protocol server + client | `server.py::MCPServer` | `server.py` (stdio JSON-RPC server), `client.py::MCPClient` (external MCP servers) |
| `observability/` | Cost tracking, audit bookkeeping, telemetry | `bookkeeping.py::book_llm_call` | `bookkeeping.py` (every metered LLM call: CostTracker row + audit row move together), `telemetry.py::CostTracker`, `metrics.py` (Prometheus), `otel.py`, `tracer.py`, `integration.py` |
| `providers/` | LLM endpoint resolution and role binding | `env.py::resolve_llm_endpoint` | `env.py`, `manager.py::ProviderManager`, `catalog.py`, `roles.py` (model_roles), `binding.py`, `runtime.py`, `local_llm.py`, `openai_oauth.py` |
| `remediation/` | Remediation lifecycle: proposals, review, apply | `workflow.py` | `models.py`, `policy.py`, `store.py`, `transaction.py`, `workflow.py`, `panel.py`, `dynamic.py` |
| `reporting/` | Report generation (md/html/json, SARIF upstream) | `report_generator.py` | `report_generator.py`, `safety.py` (report redaction), `remediation/generator.py`, `templates/` |
| `runners/` | Batch orchestration for scans and hunts | `parallel/executor.py` | `parallel/executor.py` (`ParallelScanConfig`, `TierBudget` 70/25/5 split), `cicd/runner.py::CICDRunner` + `sarif.py`, `workflow/engine.py` |
| `safety/` | Guardrails, audit logging, scoring | `guardrails/` | `guardrails/` (input/output), `audit/logger.py`, `auth/config.py`, `scoring/` (`cvss.py`, `dedup.py`) |
| `sandbox/` | Sandboxed execution backends | `backend.py` | `backend.py`, `rpc_backend.py` (JSON-RPC over Unix socket), `dind.py`, `container.py`, `hunter_sandbox.py::HunterSandbox`, `builders.py`, `registry.py`, `seccomp_profiles.py` |
| `scanning/` | Network scanners | `vulnerability_scanner.py` | `port_scanner.py`, `service_scanner.py`, `vulnerability_scanner.py`, `os_scanner.py`, `ot_scanner.py` |
| `sourcehunt/` | Source-code hunting pipeline (largest subsystem) | `runner.py::SourceHuntRunner` | see below |
| `ui/` | CLI, TUI, WebUI | `ui/cli.py::CLI` | `cli.py`, `commands/` (23 modules), `tui/` (Textual app), `web/app.py` (FastAPI + `/ws/agent`), `web/session_report.py`, `machine.py`, `llm_activity.py` |

### sourcehunt pipeline

`SourceHuntRunner` (`clearwing/sourcehunt/runner.py`) orchestrates:

```
preprocess → sandbox build → rank → tiered hunt → verify → exploit → proof
```

- **rank** — `ranker.py` scores files by attack surface (surface/influence/reach weights).
- **hunt** — `pool.py::HunterPool`: files are tiered A/B/C by `assign_tier()`
  and executed with the shared `runners/parallel/executor.py::TierBudget`
  split **A 70% / B 25% / C 5%** of the total USD budget (rollover between
  phases, per-tier per-file cost caps).
- **verify** — `verifier.py`: independent-context adversarial verifier (never
  sees hunter reasoning; steel-mans both sides; optional patch-oracle truth
  test — fix, recompile, re-run the PoC).
- **exploit / proof** — `exploiter.py` exploit generation, then
  `proof/engine.py::ProofFlowRunner` evidence upgrading.
- **Specialists** invoked by the runner (each a separately metered LLM role):
  `ranker`, `hunter`, `verifier`, `exploiter` (`exploit`), patcher
  (`patcher.py::AutoPatcher`), `harness_generator.py::HarnessGenerator`,
  `variant_loop.py`, `stability.py::StabilityVerifier`,
  `mechanism_memory.py`, `elaboration.py`, `proof/*`.
- Supporting modules: `checkpoints.py` (resume), `audit.py`
  (`SecurityAuditLog`, append-only sensitive-op log — currently standalone,
  not wired into the runner), `disclosure*.py` (CVE disclosure workflow),
  `campaign.py`, `calibration.py`, `retro_hunt.py`, `nday_*.py`,
  `reveng*.py`, `semgrep_sidecar.py`, `webhook_server.py`.

## Data flow

### WebUI chat session

```
client ── WS /ws/agent ──────────────────────────────────────────────────┐
  start frame ──► create_agent() (agent/graph.py)                        │
  message frame ─► NativeAgentGraph.astream (agent/runtime.py)           │
                    ├─ assistant step → LLM (llm/native.py)              │
                    ├─ tool batch: parallel dispatch + approval          │
                    │  interrupts (guarded_tools_node)                   │
                    └─ every step echoed on EventBus (core/events.py)    │
                               │                                         │
  EventBus ──► per-socket handlers ──► single FIFO writer coroutine      │
               (pre-serialized frames, one _writer_loop per socket)      │
                    ├─► SessionTranscript (ui/web/session_report.py)     │
                    │     └─ results/sessions/<id>/report.md             │
                    │        └─ GET /api/reports/{id} serves it          │
                    └─► cost path: every metered LLM call →              │
                          book_llm_call (observability/bookkeeping.py)   │
                          ├─ CostTracker (process-global totals)         │
                          ├─ audit row ~/.clearwing/audit/<sid>/audit.jsonl
                          └─ cost_update frame (session-scoped totals)   │
```

### sourcehunt run

```
clearwing sourcehunt (ui/commands/sourcehunt.py)
  └─ SourceHuntRunner.run (sourcehunt/runner.py)
       ├─ stage events: _emit_stage → EventBus.SOURCEHUNT_STAGE
       │    (preprocess / rank / hunt / verify / exploit / proof)
       ├─ rank ──► ranker.py ──► FileTarget list (tier A/B/C)
       ├─ hunt ──► HunterPool (pool.py, tiered 70/25/5 budget)
       │            └─ hunter.py per-file agents
       │                 └─ HUNT_PROGRESS events (pool.py)
       ├─ verify ─► verifier.py (fresh independent-context agent
       │            per finding — adversarial second pass)
       ├─ exploit ► exploiter.py / patcher.py / variant_loop.py /
       │            harness_generator.py / stability.py / proof/
       │            (each books through book_llm_call with role tags)
       ├─ builds/compiles/re-runs PoCs inside HunterSandbox
       │            (sandbox/hunter_sandbox.py → backend.py /
       │            rpc_backend.py JSON-RPC / dind.py)
       └─ session audit: init_session_audit_logger →
            ~/.clearwing/audit/<session_id>/audit.jsonl

WebUI forwarding: SOURCEHUNT_STAGE → "sourcehunt_stage" frame,
HUNT_PROGRESS → "hunt_progress" frame (ws subscription map in
ui/web/app.py). Reports land under results/sourcehunt/<sh-id>/.
```

## API surfaces

### 1. WebUI REST + WebSocket (`clearwing/ui/web/app.py`)

REST routes (mounted by `create_app`; the `/api/operate*`, `/api/reports`
routes are only mounted when `CLEARWING_WEB_API_KEY` is set, otherwise a
503 stub answers):

| Method | Path | Auth | Contract |
|---|---|---|---|
| GET | `/` | none | Serves the single-page frontend (`static/index.html`) |
| GET | `/static/*` (mount) | none | Static assets |
| GET | `/api/health` | none | `200 {"status":"ok"}`; `503 degraded` when state dir or session store unwritable (write-probed) |
| GET | `/api/sessions` | none | Session summaries; `503` when session store unavailable |
| GET | `/api/sessions/{id}` | none | Session detail; `404` unknown id, `503` store unavailable |
| GET | `/api/metrics` | none | Process-global CostTracker summary JSON |
| GET | `/api/metrics/prometheus` | none | Prometheus exposition format |
| POST | `/api/operate` | **key** | Start autonomous OperatorAgent; `400` missing target/goals; returns `{session_id, status:"running"}`; 503 stub when key unset |
| GET | `/api/operate/{id}` | **key** | Operator session status; `404` unknown; 503 stub when key unset |
| GET | `/api/disclosure/queue` | none | Disclosure queue (filterable by `state`, `repo`) |
| POST | `/api/disclosure/{id}/validate` | none | Mark disclosure validated |
| POST | `/api/disclosure/{id}/reject` | none | Mark disclosure rejected |
| POST | `/api/disclosure/{id}/send` | none | Render+send disclosure templates |
| GET | `/api/disclosure/status` | none | Disclosure workflow dashboard |
| GET | `/api/reports/{id}` | **key** | Deterministic session markdown report; `400` invalid id charset, `404` not found |

**Auth face**: `CLEARWING_WEB_API_KEY` gates `/api/operate*`, `/api/reports/{id}`,
and `/ws/agent`. The key is accepted as an `X-API-Key` header or an
`?api_key=` query parameter (browsers cannot set WS headers); mismatches
answer `401`, and an unauthorized WebSocket is closed with code `1008`
before accept. uvicorn access logs redact `api_key=` values.

**WebSocket `/ws/agent`**: client sends 4 frame types (`start`, `message`,
`approve`, `stop`); the server forwards 14 EventBus event types as envelope
frames (`agent_message`, `tool_start`, `tool_result`, `flag_found`,
`cost_update`, `error`, `approval_needed`, `campaign_progress`,
`sourcehunt_stage`, `hunt_progress`, `validation_result`, `disclosure_update`,
`benchmark_progress`, `eval_progress`) plus inline frames it originates
itself (`started`, turn-end `agent_message`, `llm_progress` heartbeat,
`error`, `stopped`, `complete`). Full per-field contract:
[web-api.md](web-api.md).

### 2. CLI surface (`clearwing/ui/commands/`)

23 command modules registered in `ALL_COMMANDS` (order as registered):

| Command | Purpose |
|---|---|
| `setup` | First-run configuration |
| `doctor` | Environment + dependency diagnostics |
| `scan` | One-shot network scan |
| `report` | Generate reports from stored results |
| `history` | Browse past runs |
| `config` | Inspect/edit config.yaml |
| `interactive` | Interactive agent (Textual TUI via `ui/tui`, `--no-tui` for legacy loop) |
| `graph` | Knowledge-graph inspection |
| `models` | List/resolve configured LLM models |
| `sessions` | List recorded sessions |
| `ci` | CI/CD scan mode (SARIF output) |
| `parallel` | Multi-target parallel scans (`runners/parallel`) |
| `remediate` | Remediation lifecycle UI |
| `mcp` | Run the MCP stdio server |
| `operate` | Headless OperatorAgent run |
| `webui` | Start the FastAPI WebUI (default port 8899) |
| `sourcehunt` | Source-code hunting pipeline |
| `tool` | Tool registry inspection (list/describe) |
| `disclose` | Disclosure workflow from CLI |
| `campaign` | Multi-repo sourcehunt campaigns |
| `bench` | OSS-Fuzz benchmarking |
| `eval` | Evaluation harness |
| `visualize` | Result visualization |

### 3. MCP surface (`clearwing/mcp/`) — previously undocumented

- **`server.py::MCPServer`** — a stdio JSON-RPC MCP server (protocol
  version `2024-11-05`, server name `clearwing`). Launched via
  `clearwing mcp`. It converts the agent tool registry
  (`agent/tools/get_all_tools`) into MCP tool definitions (`initialize`,
  `tools/list`, `tools/call`; tool results are JSON-serialized into a text
  content block). **Denylist**: tool names listed in the
  `CLEARWING_MCP_DENIED_TOOLS` env var (comma-separated) are never
  registered; the default is
  `create_custom_tool,connect_mcp_server,kali_cleanup`, so the MCP surface
  cannot spawn processes, register runtime code, or tear down the isolation
  container.
- **`client.py::MCPClient`** — the inverse direction: spawns an external
  MCP server over stdio and exposes its tools to the agent (backs the
  `connect_mcp_server` agent tool in `agent/tools/ops`).

### 4. Sandbox JSON-RPC surface

Sandbox backends speak a JSON-RPC protocol over a Unix socket or inherited
fd (`sandbox/rpc_backend.py`), implemented by local, dind, and container
backends (`sandbox/backend.py`, `dind.py`, `container.py`; seccomp profiles
in `seccomp_profiles.py`). Contract details:
[sandbox-backends.md](sandbox-backends.md).

## Deployment topology (operator host)

Three webui instances may run simultaneously on this host:

| Instance | Port | Origin |
|---|---|---|
| venv checkout instance | 8899 | `.venv/bin/clearwing webui --host 0.0.0.0 --port 8899` (log `~/clearwing-webui-8899.log`, key `~/.clearwing/webui-8899.key`) |
| compose stack | 8443 | `docker compose` caddy reverse proxy (self-signed TLS), backing + edge networks |
| system-installed `clearwing` | 8080 | system package — leave alone |

Compose services (owner-maintained `docker-compose.yml`, not committed
upstream):

| Service | Image | Notes |
|---|---|---|
| `dind` | `docker:24-dind` | privileged DinD sidecar, host network, Docker over Unix socket (no TLS) |
| `clearwing-cli` | `clearwing:local` | backing network; `mem_limit: 8g`; depends on healthy `dind` |
| `clearwing-webui` | `clearwing:local` | `webui --host 0.0.0.0 --port 8080`; attached to `backing` + `edge` |
| `caddy` | `caddy:2` | reverse proxy, self-signed TLS on `127.0.0.1:8443`, edge network |
| `otel-collector` | `otel/opentelemetry-collector-contrib:0.96.0` | backing network |

Networks: `backing` (internal services) and `edge` (caddy → webui only).
Containers keep running pre-merge code until restarted after merges.

## Test and verification assets

Two distinct asset families:

- **`tests/`** — the development gate. Run
  `.venv/bin/python -m pytest -q --strict-markers --strict-config -m "not integration"`.
  Sandbox-integration files need Docker and are excluded locally when the
  daemon is broken.
- **`e2e/`** — the *verifier's* assets (kept separate from the product):
  `test_suite.py` (60 offline tests over protocol/frames/artifacts),
  `suite.yaml` (declarative tiers `quick` / `full` / `deep-cold`, hard
  thresholds — duplicate pairs, late frames, cache-prefix medians — and the
  `hot_paths` drift contract), plus `runner.py`/`analyze.py`/`chaos.py`
  tooling. Release bar: **Full tier passing twice consecutively + at least
  one Deep-cold sample** (`e2e/SPEC.md`), with an R3 manual spot-check
  round.
