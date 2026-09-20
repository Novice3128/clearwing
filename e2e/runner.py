#!/usr/bin/env python3
"""cw-e2e runner — unified WS driver (cw-run-v3) + tier orchestrator.

Lineage: comparison-test-v2 clearwing-ws-run.py -> cw-verify-run.py ->
cw5-run.py -> cw5-run-r2.py -> this file (SPEC.md §5: assert INVARIANTS,
feature-detect fields; record HEAD + web-api.md commit every run).

Key discipline: webui key path comes from env CW_KEYFILE or --keyfile
(REQUIRED, no default — SPEC §2.3); LLM key yaml from env CW_LLM_PROFILE
when --base-url is used. Values are never printed or logged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

E2E_ROOT = Path(__file__).resolve().parent
REPO_ROOT = E2E_ROOT.parent
sys.path.insert(0, str(E2E_ROOT))

import yaml  # noqa: E402

SUITE = yaml.safe_load((E2E_ROOT / "suite.yaml").read_text())
RESULTS = E2E_ROOT / "results"
TMP_ROOT = Path("/tmp")      # module constant so tests can monkeypatch the scratch-dir scan

# Frozen safety policy (unchanged from comparison-test-v2 lineage) ...
POLICY_DENY = re.compile(
    r"for\s+p\w*\s+in|UserPass=|DomainPasswd=|Passwd=|passwd.*loop|while\s+read|xargs\s+-I"
    r"|crackmapexec\s+.*-p|cme\s+.*-p|\bmedusa\b|\bhydra\b|\bpatator\b|\bncrack\b|\bcrowbar\b|\bkerbrute\b"
    r"|base64\s+-d\s*\|\s*(ba)?sh|rpcclient.*-N.*-U|attempt.*login|login\.aspx.*POST", re.I)
DESTRUCTIVE_DENY = re.compile(
    r"shutdown|/s\s+/t\s+\d|format\s+[a-z]:|mkfs|vssadmin\s+delete\s+shadows|rd\s+/s|del\s+/f\s+/s"
    r"|Remove-Item[^|]*-Recurse[^|]*-Force|dd\s+if=.*of=/dev/|Stop-Computer|net\s+stop|sc\s+config", re.I)


def log(msg: str) -> None:
    print(f"[cw-e2e {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str, code: int = 2) -> None:
    print(f"[cw-e2e FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ---------------------------------------------------------------- common ---

def keyfile_from_env(explicit: str | None) -> Path:
    p = explicit or os.environ.get(SUITE["env"]["keyfile"])
    if not p:
        die(f"--keyfile or env ${SUITE['env']['keyfile']} required (no default — SPEC §2.3)")
    path = Path(p).expanduser()
    if not path.is_file():
        die(f"keyfile not found: {path}")
    return path


def check_target(target: str) -> str:
    if target not in SUITE["targets"]:
        die(f"target {target} not in suite whitelist {SUITE['targets']} (SPEC §2.1)")
    return target


ASIDE_FILES = ("memory.db", "memory.db-wal", "memory.db-shm", "knowledge_graph.json")


def _aside_copy(home: Path, dest: Path) -> list[str]:
    """Snapshot memory state + KG. .db files go through SQLite's backup API
    — a CONSISTENT snapshot including uncheckpointed WAL content (plain
    file copies can tear when the live process checkpoints mid-copy, and
    copying db+wal separately is not atomic either; Codex #67 r1+r3 P1).
    knowledge_graph.json is a plain file."""
    copied = []
    for name in ASIDE_FILES:
        src = home / name
        if not src.exists():
            continue
        out = dest / name
        if name.endswith(".db"):
            try:
                import sqlite3
                src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=5)
                dst_conn = sqlite3.connect(str(out))
                with dst_conn:
                    src_conn.backup(dst_conn)
                src_conn.close()
                dst_conn.close()
                copied.append(name)
                continue
            except Exception:        # noqa: BLE001 — fall back to byte copy
                pass                  # (non-SQLite db file or lock timeout)
        if name.endswith(("-wal", "-shm")):
            continue                  # superseded by the backup API snapshot
        out.write_bytes(src.read_bytes())
        copied.append(name)
    return copied


def cost_cap_of(sc: dict, tier: dict) -> float:
    """Scenario cap > tier cap > suite default. Never index tier["cost_cap"]
    directly — it is an eagerly-evaluated default and deep-fallback defines
    no tier-level cap (every chaos scenario crashed with KeyError, Codex r1)."""
    v = sc.get("cost_cap")
    if v is None:
        v = tier.get("cost_cap", SUITE["thresholds"].get("default_cost_cap", 5.0))
    return float(v)


def ws_from_base(base: str) -> str:
    """http(s) base -> ws endpoint (webui mounts /ws/agent)."""
    return base.replace("http", "ws", 1).rstrip("/") + "/ws/agent"


def http_get(url: str, timeout: float = 5.0, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001
        return -1, str(e).encode()


def new_run_dir(tier: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    d = RESULTS / time.strftime(f"%Y%m%d-%H%M%S-{tier}")
    i = 0
    while d.exists():  # collision guard (SPEC §2.7)
        i += 1
        d = RESULTS / f"{d.name}-{i}"
    (d / "scenarios").mkdir(parents=True)
    (d / "state").mkdir()
    return d


class SuiteLock:
    """O_EXCL lockfile — one run at a time (SPEC §2.4)."""

    def __init__(self):
        self.path = RESULTS / ".lock"

    def __enter__(self):
        RESULTS.mkdir(exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, str(os.getpid()).encode())
        except FileExistsError:
            try:                    # stale-lock detection: holder process alive?
                old_pid = int(self.path.read_text().strip() or 0)
                alive = old_pid and Path(f"/proc/{old_pid}").exists()
            except (OSError, ValueError):
                alive = True
            if not alive:
                self.path.unlink(missing_ok=True)
                return self.__enter__()
            die(f"another cw-e2e run (pid {self.path.read_text().strip()}) holds {self.path}")
        return self

    def __exit__(self, *exc):
        os.close(self.fd)
        self.path.unlink(missing_ok=True)


def render_prompt(name: str, run_dir: Path, target: str) -> str:
    text = (E2E_ROOT / "scenarios" / name).read_text()
    return text.replace("{{RUN_DIR}}", str(run_dir)).replace("{{TARGET}}", target).strip()


def git_info() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True).stdout.strip()
    return {
        "head": g("rev-parse", "HEAD"),
        "webapi_commit": g("log", "-1", "--format=%h", "--", "docs/web-api.md"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
    }


# ------------------------------------------------------------ WS driver ----

class AuditLiveness:
    """Frame-silent != dead: check audit.jsonl growth before declaring idle
    death (v3 §8 lesson — two sessions were killed prematurely)."""

    def __init__(self, home: Path):
        self.home = Path(home).expanduser()
        self.last_size = -1

    def grew(self, sid: str | None) -> bool:
        """True => keep waiting. FIRST call only seeds the baseline and
        returns True (assume alive) — the pre-fix version returned False on
        the first idle breach, killing exactly the long-tool silence it was
        built to protect (v3 lesson regression)."""
        if not sid:
            return False
        f = self.home / "audit" / sid / "audit.jsonl"
        try:
            size = f.stat().st_size
        except OSError:
            return True    # no audit file yet: cannot disprove liveness
        if self.last_size < 0:
            self.last_size = size
            return True
        grew = size > self.last_size
        self.last_size = max(self.last_size, size)
        return grew


async def ws_run(ws_url: str, keyfile: Path, target: str, prompt: str, out_prefix: Path,
                 max_seconds: int, cost_cap: float, *, base_url: str | None = None,
                 api_key: str | None = None, model: str = "glm-5.3", trust_status: bool = True,
                 approve_delay: float = 0.0, idle_limit: int = 600, home: Path | None = None,
                 stop_after_tools: int = 0, deny_first_approval: bool = False,
                 send_model: bool = True, memory_tag: str | None = None) -> dict:
    import websockets

    headers = {"x-api-key": keyfile.read_text().strip()}
    frames_log = open(f"{out_prefix}.frames.jsonl", "x", buffering=1)   # 'x': refuse overwrite (SPEC §2.7)
    tools_log = open(f"{out_prefix}.frames.tools", "x", buffering=1)
    t0 = time.time()
    frame_types: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    tool_seq: list[tuple[int, str]] = []
    approvals = policy_denied = 0
    seen_prompts: dict[str, int] = {}
    sid = started_model = None
    complete_payload = stopped_payload = None
    complete_statuses: list[str] = []
    cost_usd = 0.0
    tokens_in = tokens_out = tokens_cached = 0
    cost_updates = 0
    cost_models: set[str] = set()
    watchdog = False
    agent_text: list[str] = []
    errors: list[str] = []
    tool_starts = 0
    approval_open = 0            # approval-closure invariant: needs>0 without response
    liveness = AuditLiveness(home or "~/.clearwing")
    containers: set[str] = set()
    flag_frames = 0
    flag_faces = 0
    # Issue #83: LLM responses that quote a flag batch back get re-detected
    # byte-identically — the raw count double-bills those echoes. The gate
    # compares the UNIQUE set; the raw count stays for continuity.
    flags_seen: set[str] = set()
    completes_without_status = 0
    complete_while_approval_open = 0
    busy_rejects = 0
    # Approval protocol (web/app.py): approval_needed is emitted INSIDE the
    # message turn; `approve` frames are REJECTED with an error frame while
    # the turn task is still active, and the accepted window opens when the
    # turn's finally-block sends complete(status="awaiting_approval").
    # Sending on approval_needed was therefore a race that usually won and
    # silently lost the decision when it lost (Codex r1) — queue decisions
    # and flush one per awaiting_approval complete instead.
    pending_decisions: list[tuple[bool, int]] = []
    flushed_last: tuple[bool, int] | None = None
    # Late-frame drain: with trust_status=true the loop used to close the
    # socket on the first terminal complete, so post-terminal frames were
    # never read and the late-frames gate could only ever see zero
    # (Codex r1). Keep draining a bounded grace window after terminal.
    terminal_at: float | None = None
    drain_s = float(SUITE["thresholds"].get("late_drain_s", 5))

    async def send(ws, obj):
        await ws.send(json.dumps(obj))

    async with websockets.connect(ws_url, open_timeout=15, additional_headers=headers) as ws:
        start = {"type": "start", "target": target}
        if send_model:
            start["model"] = model
        if base_url:
            start.update({"base_url": base_url, "api_key": api_key, "model": model})
        await send(ws, start)
        await send(ws, {"type": "message", "content": prompt})
        idle = 0
        frames = 0
        # drain may run past max_seconds — the late-frames gate must not
        # under-observe exactly the longest runs (bounded by drain_s itself)
        while (time.time() - t0 < max_seconds or terminal_at is not None) and frames < 12000:
            in_drain = terminal_at is not None
            if in_drain and time.time() - terminal_at >= drain_s:
                frames_log.write(f"# terminal drain window ({drain_s}s) closed\n")
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2 if in_drain else 30)
            except (asyncio.TimeoutError, TimeoutError):
                if in_drain:
                    continue            # short poll until drain window expires
                idle += 30
                if idle >= idle_limit:
                    if liveness.grew(sid):
                        frames_log.write("# idle breach but audit GREW — frame-silent alive, continuing\n")
                        idle = 0
                        continue
                    frames_log.write("# idle-timeout break (audit not growing)\n")
                    break
                continue
            idle = 0
            frames += 1
            frames_log.write(raw + "\n")
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("type", "")
            frame_types[t] = frame_types.get(t, 0) + 1
            d = msg.get("data") or {}
            if t == "started":
                sid = msg.get("session_id")
                started_model = msg.get("model")
            elif t == "approval_needed":
                approvals += 1
                approval_open += 1
                prm = str(d.get("prompt") or "")[:2000]
                n = seen_prompts.get(prm, 0) + 1
                seen_prompts[prm] = n
                deny = bool(POLICY_DENY.search(prm)) or bool(DESTRUCTIVE_DENY.search(prm)) or n > 2
                if deny_first_approval and approvals == 1:
                    deny = True
                if POLICY_DENY.search(prm) or DESTRUCTIVE_DENY.search(prm):
                    policy_denied += 1
                pending_decisions.append((not deny, approvals))
                frames_log.write(f"# approval #{approvals} queued decision={not deny} "
                                 "(sent when awaiting_approval window opens)\n")
            elif t == "cost_update":
                cost_updates += 1
                cost_usd = max(cost_usd, float(d.get("total_cost_usd") or d.get("cost") or 0))
                tokens_in += int(d.get("input_tokens") or 0)
                tokens_out += int(d.get("output_tokens") or 0)
                tokens_cached += int(d.get("cached_tokens") or 0)
                if d.get("model"):
                    cost_models.add(d["model"])
                if cost_usd > cost_cap and not watchdog and terminal_at is None:
                    # terminal_at guard: a late post-terminal cost_update must
                    # never provoke a stop — the drain window is observe-only
                    watchdog = True
                    await send(ws, {"type": "stop"})
                    frames_log.write(f"# COST WATCHDOG fired at ${cost_usd:.4f} — stop sent\n")
            elif t == "tool_start":
                tool_starts += 1
                name = d.get("tool") or "?"
                tool_counts[name] = tool_counts.get(name, 0) + 1
                tool_seq.append((round(time.time() - t0), name))
                if d.get("args", {}).get("container_id"):
                    containers.add(d["args"]["container_id"][:12])
                tools_log.write(f"+{round(time.time()-t0):>5}s {name} {json.dumps(d.get('args') or {}, ensure_ascii=False)[:300]}\n")
                if stop_after_tools and tool_starts >= stop_after_tools and terminal_at is None:
                    await send(ws, {"type": "stop"})
                    frames_log.write(f"# STOP sent after tool_start #{tool_starts}\n")
                    stop_after_tools = 0
            elif t == "tool_result":
                approval_open = 0   # approval consumed by execution
            elif t == "error":
                msg_txt = str(d.get("message") or d.get("error") or raw)[:200]
                # Residual micro-race: our flushed approve can still land in
                # the gap between complete(awaiting_approval) send and
                # turn_task completion — the server answers with the busy
                # error frame. Retry once after a settle delay, and do NOT
                # count the self-healed reject as a run error (t1 gates on
                # errors==0; a recovered protocol hiccup must not FAIL it).
                if flushed_last is not None and terminal_at is None \
                        and "turn is already running" in msg_txt:
                    busy_rejects += 1
                    approved, seq = flushed_last
                    await asyncio.sleep(0.5)
                    await send(ws, {"type": "approve", "approved": approved})
                    # retry ONCE per approval: clearing flushed_last bounds the
                    # retry count AND prevents a stale decision re-firing on a
                    # later unrelated error containing the phrase (review r2)
                    flushed_last = None
                    frames_log.write(f"# approve #{seq} RE-SENT after busy-reject "
                                     f"(#{busy_rejects}, self-healed, retry budget spent)\n")
                else:
                    errors.append(msg_txt)
            elif t == "agent_message":
                c = d.get("content")
                if isinstance(c, str) and c:
                    agent_text.append(c)
            elif t == "flag_found":
                flag_frames += 1
                flags = d.get("flags")
                if isinstance(flags, list):
                    flag_faces += len(flags)
                    flags_seen.update(str(f) for f in flags)
                else:
                    flag_faces += 1
                    flags_seen.add(repr(flags))
            elif t == "stopped":
                stopped_payload = d
            elif t == "complete":
                status = d.get("status") or "ok"      # feature-detect, absent = ok
                complete_statuses.append(status)
                complete_payload = d
                if "status" not in d:
                    completes_without_status += 1
                if approval_open > 0 and status in ("ok", "stopped", "error") \
                        and status != "awaiting_approval":
                    complete_while_approval_open += 1   # D1/#29 detector
                frames_log.write(f"# complete status={status} at +{round(time.time()-t0)}s\n")
                if status == "awaiting_approval" and pending_decisions and terminal_at is None:
                    approved, seq = pending_decisions.pop(0)
                    if approve_delay:
                        await asyncio.sleep(approve_delay)
                    await send(ws, {"type": "approve", "approved": approved})
                    flushed_last = (approved, seq)
                    frames_log.write(f"# approve #{seq} decision={approved} "
                                     f"sent in awaiting_approval window (delay={approve_delay}s)\n")
                if trust_status and status in ("ok", "stopped", "error"):
                    if terminal_at is None:      # FIRST terminal anchors the drain
                        terminal_at = time.time()   # window — late completes must
                        frames_log.write(f"# terminal — draining {drain_s}s for late frames\n")

    text = "".join(agent_text)
    summary = {
        "seconds": round(time.time() - t0), "frames": frames, "frame_types": frame_types,
        "session_id": sid, "started_model": started_model,
        "approvals": approvals, "policy_denied": policy_denied,
        "complete_count": len(complete_statuses), "complete_statuses": complete_statuses,
        "tools": tool_counts, "tool_seq_len": len(tool_seq),
        "errors": errors, "error_count": len(errors),
        "cost_usd_product": round(cost_usd, 4), "cost_updates": cost_updates,
        "tokens_in": tokens_in, "tokens_out": tokens_out, "tokens_cached": tokens_cached,
        "cost_models": sorted(cost_models), "watchdog": watchdog,
        "stopped": stopped_payload, "complete": complete_payload,
        "reply_chars": len(text), "reply_tail": text[-500:],
        "approval_open_at_end": approval_open, "flag_frames": flag_frames,
        "flag_faces": flag_faces,
        "flag_faces_unique": len(flags_seen),
        "pending_approvals_at_end": len(pending_decisions),
        "busy_rejects": busy_rejects,
        "completes_without_status": completes_without_status,
        "complete_while_approval_open": complete_while_approval_open,
        "containers": sorted(containers), "home": str(home) if home else None,
        "memory": memory_tag,
        "invariants": {
            "terminal_closure": bool(complete_statuses and complete_statuses[-1] in ("ok", "stopped", "error")),
            "approval_closure": approval_open == 0,
            "no_complete_while_approval_open": complete_while_approval_open == 0,
        },
    }
    frames_log.write("\n# SUMMARY " + json.dumps(summary, ensure_ascii=False) + "\n")
    frames_log.close()
    tools_log.close()
    (out_prefix.with_suffix(".summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def llm_api_key() -> str:
    p = os.environ.get(SUITE["env"]["llm_profile"])
    if not p:
        die(f"env ${SUITE['env']['llm_profile']} (yaml path) required for --base-url runs "
            "(probe/chaos scenarios need it — export CW_LLM_PROFILE=<yaml> BEFORE the run, "
            "not after verify; the 20260918-175138 attempt died at scenario 1 this way)")
    return yaml.safe_load(Path(p).expanduser().read_text())["zai"]["api_key"]


def check_llm_profile() -> None:
    """P3.4: fail EARLY (verify stage) when CW_LLM_PROFILE is missing or
    malformed — Quick/Full both contain --base-url scenarios, so a missing
    profile is a guaranteed mid-run FATAL (20260918-175138 lesson)."""
    p = os.environ.get(SUITE["env"]["llm_profile"])
    if not p:
        die(f"env ${SUITE['env']['llm_profile']} required (Quick/Full probe+chaos scenarios "
            "call the LLM directly) — export CW_LLM_PROFILE=<llm yaml path> first")
    try:
        cfg = yaml.safe_load(Path(p).expanduser().read_text())
        ok = isinstance(cfg, dict) and isinstance(cfg.get("zai"), dict) and bool(cfg["zai"].get("api_key"))
    except (OSError, yaml.YAMLError):
        ok = False
    if not ok:
        die(f"${SUITE['env']['llm_profile']} unreadable or lacks zai.api_key: {p}")


def run_ws_scenario(sc: dict, run_dir: Path, tier: dict, ws_url: str, keyfile: Path,
                    home: Path | None, instance_tag: str = "") -> dict:
    raw_target = str(sc.get("target", "192.168.73.82"))
    target = raw_target if raw_target.startswith("192.") else f"192.168.73.{raw_target.lstrip('.')}"
    check_target(target)
    prompt = render_prompt(sc["prompt"], run_dir, target)
    base_url = api_key = None
    if sc.get("base_url_pathless"):
        base_url = "http://127.0.0.1:8787"           # deliberate: 404-probe scenario
        api_key = llm_api_key()
    elif sc.get("via_config"):
        pass                                          # fallback chain lives in real config; no base_url
    out = run_dir / "scenarios" / (sc["name"] + instance_tag)
    log(f"scenario {sc['name']}{instance_tag}: target={target} cap={cost_cap_of(sc, tier)}")
    summary = asyncio.run(ws_run(
        ws_url, keyfile, target, prompt, out,
        int(sc.get("max_seconds", 600)), cost_cap_of(sc, tier),
        base_url=base_url, api_key=api_key,
        trust_status=sc.get("trust_status", True),
        approve_delay=float(sc.get("approve_delay", 0)),
        home=home,
        stop_after_tools=int(sc.get("stop_after_tools", 0)),
        deny_first_approval=bool(sc.get("deny_first_approval", False)),
        idle_limit=int(sc.get("idle_limit", 600)),
        memory_tag=sc.get("memory"),
    ))
    log(f"  -> {summary['seconds']}s frames={summary['frames']} statuses={summary['complete_statuses'][-3:]} "
        f"cost=${summary['cost_usd_product']} errors={summary['error_count']}")
    return summary


# ---------------------------------------------------------- probe / misc ---

def probe_partial_fill(ws_url: str, keyfile: Path, out: Path) -> dict:
    """Zero-LLM: start frame base_url+api_key WITHOUT model must resolve to a
    real model from config (D3, PR#43 lineage)."""
    import websockets
    api_key = llm_api_key()

    async def go():
        async with websockets.connect(ws_url, open_timeout=15,
                                      additional_headers={"x-api-key": keyfile.read_text().strip()}) as ws:
            await ws.send(json.dumps({"type": "start", "target": "192.168.73.82",
                                      "base_url": "https://api.z.ai/api/coding/paas/v4", "api_key": api_key}))
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            return json.loads(raw)

    msg = asyncio.run(go())
    model = msg.get("model") or ""
    import yaml as _yaml
    cfg_model = (_yaml.safe_load((Path.home() / ".clearwing/config.yaml").read_text())
                 .get("provider", {}).get("model", ""))
    ok = bool(model) and model != "default" and (not cfg_model or model == cfg_model)
    result = {"type": "probe_partial_fill", "session_id": msg.get("session_id"), "model": model,
              "config_model": cfg_model, "pass": bool(ok)}
    out.with_name(out.name + ".partial-fill.json").write_text(json.dumps(result, indent=1))
    log(f"  -> partial-fill model={model} pass={ok}")
    return result


def scenario_report_cli(run_dir: Path, sid: str) -> dict:
    p = subprocess.run([str(REPO_ROOT / ".venv/bin/clearwing"), "report", "-s", sid],
                       capture_output=True, text=True, cwd=str(REPO_ROOT))
    # `report -s` prints a nonempty "No report found" message AND exits 0 on
    # the absent-report path (ui/commands/report.py) — a bare nonempty-stdout
    # check marked that failure path PASS (Codex r1). Require real content.
    no_report = "No report found" in p.stdout
    has_report = ("Session report:" in p.stdout or "Clearwing Session Report" in p.stdout)
    ok = p.returncode == 0 and has_report and not no_report
    (run_dir / "scenarios" / "report-cli.txt").write_text(p.stdout[:5000] + "\n---stderr---\n" + p.stderr[:2000])
    log(f"  -> report -s {sid}: exit={p.returncode} bytes={len(p.stdout)} "
        f"pass={ok}{' (no-report message!)' if no_report else ''}")
    return {"type": "report_cli", "sid": sid, "exit": p.returncode, "pass": bool(ok)}


def scenario_pytest(run_dir: Path) -> dict:
    cmd = [str(REPO_ROOT / ".venv/bin/python"), "-m", "pytest", "-q",
           *SUITE["pytest"]["area_subset"]]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "?"
    failed = [l.split("::")[0].replace("FAILED ", "") for l in p.stdout.splitlines()
              if l.startswith("FAILED")]
    allow = SUITE["pytest"]["env_fail_allowlist"]
    non_env = [f for f in failed if f not in allow]
    # Nonzero exit with NO parsed FAILED lines = collection/import/internal/
    # usage error — stdout-only matching called those runs PASS (Codex r1).
    # Exit 1 remains tolerable ONLY when every FAILED line is env-allowlisted.
    env_only = p.returncode == 1 and bool(failed) and not non_env
    ok = (p.returncode == 0 and not failed) or env_only
    if p.returncode == 0:
        exit_note = ""
    elif env_only:
        exit_note = "env-allowlisted failures only"
    else:
        exit_note = "nonzero exit without tolerable FAILED lines (collection/import/internal/usage error)"
    (run_dir / "scenarios" / "pytest-subset.txt").write_text(p.stdout[-8000:])
    log(f"  -> pytest subset: exit={p.returncode} {tail} "
        f"(non-env failures: {non_env or 'none'}) {exit_note}")
    return {"type": "pytest", "tail": tail, "exit": p.returncode,
            "non_env_failures": non_env, "note": exit_note, "pass": bool(ok)}


def probe_majority(host: str, port: int, tries: int = 3) -> bool:
    """Majority-of-N probe: a single lost SYN must never become the
    recorded baseline or the post-run verdict (Codex #67 r1 P2)."""
    votes = []
    for i in range(tries):
        if i:
            time.sleep(0.5)     # back-to-back SYNs all landing inside one
        votes.append(tcp_probe(host, port))   # transient window would defeat
    return sum(votes) * 2 > len(votes)        # the majority (Codex #67 r3)


def tcp_probe(host: str, port: int, timeout: float = 3.0) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def listening_pids(port: int) -> list[int]:
    out = subprocess.run(["ss -tlnp".split()[0], "-tlnp"], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        if f":{port} " in line:
            m = re.search(r"pid=(\d+)", line)
            if m:
                pids.append(int(m.group(1)))
    return pids


def _proc_start_epoch(pid: int) -> float | None:
    """/proc clock: process start as wall-clock epoch, or None if unreadable."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # fields after the comm entry "(...)" — starttime is the 20th there
        starttime = int(stat[stat.rfind(")") + 2:].split()[19])
        btime = next(l for l in Path("/proc/stat").read_text().splitlines()
                     if l.startswith("btime"))
        return int(btime.split()[1]) + starttime / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def _product_head_epoch() -> int | None:
    """Committer time of the last commit touching anything EXCEPT e2e/ —
    that is the code the live process actually imports (suite-only commits
    must not trip the process-freshness gate)."""
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%ct",
                        "--", ".", ":(exclude)e2e"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def proc_freshness(proc_start: float | None, prod_head: int | None) -> bool:
    """Gate: process must have started AFTER the last product-code commit
    (2s clock-skew tolerance). None on either side fails closed."""
    if proc_start is None or prod_head is None:
        return False
    return proc_start >= prod_head - 2


def _environ_api_key(pid: int) -> str | None:
    """CLEARWING_WEB_API_KEY from the live 8899 process environ — the
    authoritative key the server actually authenticates with (same-user
    /proc read). Value is held in memory only, never printed."""
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    for entry in environ:
        if entry.startswith(b"CLEARWING_WEB_API_KEY="):
            return entry.split(b"=", 1)[1].decode("utf-8", "replace")
    return None


def evaluate_key_gate(kf_bytes: bytes | None, environ_key: str | None,
                      mirror_bytes: bytes | None) -> tuple[str, bool, str]:
    """Pure key-equivalence decision. Priority: the 8899 process environ is
    authoritative (a stale webui-8899.key mirror must not brick the run);
    the file mirror is the fallback when /proc is unreadable. All compares
    newline-stripped (the two files differ only by trailing newline in a
    known-good deployment)."""
    if kf_bytes is None:
        return ("key-files-info", True, "CW_KEYFILE unset/unreadable (verify-only context)")
    if environ_key is not None:
        same = kf_bytes.strip() == environ_key.encode().strip()
        return ("key-matches-live-process", same,
                f"CW_KEYFILE == 8899 env key (newline-stripped): {same}")
    if mirror_bytes is not None:
        same = kf_bytes.strip() == mirror_bytes.strip()
        return ("key-files-match", same,
                f"CW_KEYFILE == webui-8899.key (newline-stripped): {same}")
    return ("key-files-info", True, "no mirror file, no /proc environ — unverified (info)")


def verify(run_dir: Path | None, strict: bool = True, trigger: str | None = None) -> dict:
    """Tier0 gates (SPEC §3). strict=True: hard failures abort the run."""
    checks: list[tuple[str, bool, str]] = []
    gi = git_info()
    live = SUITE["webui"]["live"]

    status, _ = http_get(f"{live}/api/health")
    checks.append(("live-health", status == 200, f"HTTP {status}"))

    status, _ = http_get(f"{live}/api/operate/status")
    checks.append(("no-key-gate-401", status == 401, f"HTTP {status}"))

    pids = listening_pids(8899)
    ver = {"pids": pids}
    if pids:
        try:
            cwd = Path(f"/proc/{pids[0]}/cwd").resolve()
            ver["cwd"] = str(cwd)
            checks.append(("8899-runs-repo-venv", cwd == REPO_ROOT, f"cwd={cwd}"))
        except OSError:
            checks.append(("8899-runs-repo-venv", False, "cannot read /proc"))
        # A cwd match does not prove the process LOADED this code: if 8899
        # started before a pull/checkout, its imported modules are stale and
        # the suite would certify a revision it never exercised (Codex r1).
        # Gate: process start must postdate the last product-code commit.
        proc_start = _proc_start_epoch(pids[0])
        prod_head = _product_head_epoch()
        checks.append(("8899-proc-newer-than-product-head", proc_freshness(proc_start, prod_head),
                       f"proc_start={time.strftime('%F %T', time.localtime(proc_start)) if proc_start else '?'} "
                       f"product_head={time.strftime('%F %T', time.localtime(prod_head)) if prod_head else '?'}"))
    else:
        checks.append(("8899-runs-repo-venv", False, "no listener on 8899"))

    residue = sorted(str(p) for p in Path.home().glob(".clearwing/config.yaml.bak-*"))
    checks.append(("no-config-bak-residue", not residue, f"{residue or 'clean'}"))

    for port in (8898, 8787):
        checks.append((f"port-{port}-free", not listening_pids(port), ""))

    for cmd, name in ((["doctor", "--skip-llm-invoke"], "doctor"), (["models", "--json"], "models-json")):
        try:
            p = subprocess.run([str(REPO_ROOT / ".venv/bin/clearwing"), *cmd],
                               capture_output=True, text=True, timeout=90, cwd=str(REPO_ROOT))
            ok = p.returncode == 0 and ("glm" in p.stdout or "ok" in p.stdout.lower())
            checks.append((f"cfg-drift-{name}", ok, p.stdout.strip().splitlines()[-1][:120] if p.stdout.strip() else p.stderr.strip()[:120]))
        except Exception as e:  # noqa: BLE001
            checks.append((f"cfg-drift-{name}", False, f"spawn fail: {e}"))

    kf_env = os.environ.get(SUITE["env"]["keyfile"])
    try:
        kf_bytes = Path(kf_env).read_bytes() if kf_env else None
    except OSError:
        kf_bytes = None
    mirror = Path.home() / ".clearwing/webui-8899.key"
    try:
        mirror_bytes = mirror.read_bytes() if mirror.exists() else None
    except OSError:
        mirror_bytes = None
    checks.append(evaluate_key_gate(kf_bytes, _environ_api_key(pids[0]) if pids else None,
                                    mirror_bytes))

    if run_dir:
        import hashlib
        st = run_dir / "state"
        home = Path.home() / ".clearwing"
        def _sz(name):
            try:
                return (home / name).stat().st_size
            except OSError:
                return -1
        try:
            audit_n = len(list((home / "audit").iterdir()))
        except OSError:
            audit_n = -1
        # P3.1 memory/KG aside-backup (v2 SOP② mechanized): the round MUTATES
        # member memory (2026-09-18 measured 221k->282k / 1.69M->2.10M) — the
        # copy is the cross-round contamination control the manual protocol kept
        aside = run_dir / "state" / "aside-pre"
        aside.mkdir(exist_ok=True)
        _aside_copy(home, aside)          # ~2.4MB total, gitignored
        suite_hash = hashlib.sha256()
        for f in ("runner.py", "chaos.py", "surgery.py", "analyze.py", "suite.yaml", "SPEC.md"):
            suite_hash.update((E2E_ROOT / f).read_bytes())
        (st / "pre-state.json").write_text(json.dumps({
            "git": gi, "audit_dirs": audit_n,
            "trigger": trigger or "",      # SPEC §3 trigger-honesty: run reports carry the source
            "memory_db": _sz("memory.db"), "knowledge_graph": _sz("knowledge_graph.json"),
            "pids_8899": pids,
            "suite_sha256": suite_hash.hexdigest()[:16],
            "target_snapshot": {t: {p: probe_majority(t, p) for p in (88, 445, 3389)} for t in SUITE["targets"]},
        }, indent=1))

    if trigger is not None:
        # soft nudge in run context only (standalone verify passes None) —
        # the -info suffix keeps it from ever blocking (SPEC §3 dual-carrier)
        checks.append(("trigger-recorded-info", bool(trigger),
                       "SPEC §3 觸發源" + ("" if trigger else " 未記錄 — run --trigger '<來源>'")))
    for name, ok, note in checks:
        log(f"verify {'PASS' if ok else 'FAIL'} {name} {note}")
    if strict:
        hard = [c for c in checks if not c[1] and not c[0].endswith("-info") and "cfg-drift" not in c[0]]
        if hard:
            die(f"verify hard-gate failures: {[h[0] for h in hard]} (use --force to override)")
    return {"checks": checks, "git": gi, "_pids": pids}


def scenario_hud(sc: dict, run_dir: Path, ws_url: str, keyfile: Path) -> dict:
    """HUD must be checked at PAGE-RENDER level (footer DOM) vs report —
    the API was never wrong in the historical D2 defect (SPEC §4)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log("  -> playwright unavailable: SKIPPED (SPEC §7 chromium caveat)")
        return {"type": "hud", "status": "skipped", "reason": "no playwright"}
    key = keyfile.read_text().strip()
    target = SUITE["targets"][1]
    base = SUITE["webui"]["live"]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium",
                                    args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"{base}/?api_key={key}", wait_until="networkidle")
        page.fill("#target", target)
        page.click("button:has-text('Connect')")
        # exact match — 'Disconnected' CONTAINS 'Connected' and passes :has-text
        page.wait_for_function(
            "document.getElementById('conn-status').textContent.trim() === 'Connected'",
            timeout=15000)
        page.evaluate("history.replaceState(null, '', '/')")   # strip AFTER ws open (page reads ?api_key at connect)
        page.fill("#msg-input", (E2E_ROOT / "scenarios" / sc["prompt"]).read_text().strip())
        page.click("#send-btn")
        try:
            page.wait_for_function("!document.getElementById('stop-btn').disabled", timeout=15000)
            page.wait_for_function("document.getElementById('stop-btn').disabled", timeout=120000)
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(3000)
        hud_cost = (page.text_content("#cost") or "").strip()
        hud_tokens = (page.text_content("#tokens") or "").strip()
        page.screenshot(path=str(run_dir / "scenarios" / "hud-proof.png"), full_page=True)
        info = (page.text_content("#session-info") or "")
        browser.close()
    m = re.search(r"[0-9a-f]{8}", info)
    sid = m.group(0) if m else ""
    rep_cost = rep_tokens = "?"
    if sid:
        s, body = http_get(f"{base}/api/reports/{sid}", headers={"X-API-Key": key})
        mm = re.search(r"Cost[^|]*\|\s*\$([0-9.]+)", body.decode("utf-8", "replace"))
        rep_cost = mm.group(1) if mm else "?"
        mm = re.search(r"Tokens[^|]*\|\s*([0-9,]+)", body.decode("utf-8", "replace"))
        rep_tokens = mm.group(1) if mm else "?"
    def _num(x):
        try:
            return float(re.sub(r"[^0-9.]", "", str(x)))
        except ValueError:
            return None
    hc, rc = _num(hud_cost), _num(rep_cost)
    ht, rt = _num(hud_tokens), _num(rep_tokens)
    # D2-grade check: HUD must be NON-ZERO, match report cost within 1% (rel),
    # and the tokens axis must match too (historical defect showed $0.00 AND 0).
    match = all(v is not None for v in (hc, rc, ht, rt)) and hc > 0 and rc > 0 \
        and abs(hc - rc) <= max(0.01 * rc, 1e-4) and abs(ht - rt) <= max(0.01 * rt, 1)
    result = {"type": "hud", "session_id": sid, "hud_cost": hud_cost, "hud_tokens": hud_tokens,
              "report_cost": rep_cost, "report_tokens": rep_tokens, "pass": bool(match),
              # ledger-visible spend: the HUD turn billed real tokens (Codex #67 r6)
              "cost_usd_product": hc, "seconds": 0}
    # numeric tokens from the session's own audit so run-level token totals
    # stop silently excluding this session (R1-A: quick ledger omitted
    # warm-hud's 23,584-in tokens — footer total 23,587 — while its cost WAS counted)
    if sid:
        import analyze as _az
        am = _az.audit_metrics(sid, Path.home() / ".clearwing")
        if am:
            result["tokens_in"], result["tokens_out"] = am["tokens_in"], am["tokens_out"]
    (run_dir / "scenarios" / "hud-proof.json").write_text(json.dumps(result, indent=1))
    log(f"  -> HUD footer {hud_cost}/{hud_tokens} vs report {rep_cost}/{rep_tokens} match={match}")
    return result


def _tmp_scratch_left(run_dir: Path) -> list[str]:
    """Scratch dirs the product's custom-tool runtime leaves per session. The
    old file-only glob passed while 9 empty clearwing_custom_tools_* dirs
    from our own sessions survived (R3 lens, 2026-09-19) — the assertion
    was false. Window = dirs created since this run started (dir-name ts);
    empty ones we remove ourselves, non-empty leftovers fail the gate."""
    m_ts = re.match(r"(\d{8}-\d{6})", run_dir.name)
    try:
        run_start = time.mktime(time.strptime(m_ts.group(1), "%Y%m%d-%H%M%S")) if m_ts else 0.0
    except ValueError:                     # non-timestamped (test) dirs: window = everything
        run_start = 0.0
    scratch_left = []
    for d in TMP_ROOT.glob("clearwing_custom_tools_*"):
        try:
            if d.is_dir() and d.stat().st_mtime >= run_start:
                if not any(d.iterdir()):
                    d.rmdir()
                else:
                    scratch_left.append(d.name)
        except OSError as e:
            # fail-closed: an unscannable scratch dir is NOT a clean /tmp
            scratch_left.append(f"{d.name} ({type(e).__name__})")
    return scratch_left


def cleanup_run(run_dir: Path, container_ids: list[str], before_8899: list[int] | None = None,
                sids: list[str] | None = None, keyfile: Path | None = None) -> list[tuple[str, bool, str]]:
    """Cleanup SOP as assertions (SPEC §2.8). Only recorded containers — never
    name-filter all (docker is global across instances). Kali containers are
    named clearwing-kali-<session_id>, so remove by sid; container ids are a
    fallback."""
    out = []
    for sid in sorted(set(sids or [])):
        subprocess.run(["docker", "rm", "-f", f"clearwing-kali-{sid}"], capture_output=True)
    for cid in sorted(set(container_ids)):
        if len(cid) >= 12:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
    left = [c for c in subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"],
                                      capture_output=True, text=True).stdout.split()
            if c.startswith("clearwing-kali-")]
    ours = {f"clearwing-kali-{s}" for s in set(sids or [])}
    remaining_ours = [c for c in left if c in ours]
    out.append(("kali-containers", not remaining_ours,
                f"ours-remaining={remaining_ours} member-left={len(left) - len(remaining_ours)} (untouched)"))
    scratch_left = _tmp_scratch_left(run_dir)
    out.append(("tmp-residue", not (list(TMP_ROOT.glob('report_*')) or list(TMP_ROOT.glob('kerbrute*'))
                                    or scratch_left),
                f"scratch-dirs-left={scratch_left}"
                + (" (non-empty dirs in-window MIGHT be a concurrent member session — "
                   "/tmp is shared; adjudicate before assuming)" if scratch_left else "")))
    out.append(("tmux-none", subprocess.run(["tmux", "ls"], capture_output=True).returncode != 0, ""))
    for png in run_dir.rglob("*.png"):
        r = subprocess.run(["strings", str(png)], capture_output=True, text=True)
        if "api_key" in (r.stdout or ""):
            out.append((f"png-keyscan-{png.name}", False, "api_key string in screenshot"))
            break
    else:
        out.append(("png-keyscan", True, "no api_key strings in screenshots"))
    out.append(("port-8787-free", not listening_pids(8787), ""))
    out.append(("port-8898-free", not listening_pids(8898), ""))
    if before_8899 is not None:
        out.append(("live-8899-pids-unchanged", listening_pids(8899) == before_8899,
                    f"before={before_8899} now={listening_pids(8899)}"))

    # ---- P3 state-integrity block runs BEFORE the keyscan so that the
    # aside-POST snapshots it creates are themselves scanned (Codex #67 r4)
    pre = run_dir / "state" / "aside-pre"
    post = run_dir / "state" / "aside-post"
    if pre.exists():
        post.mkdir(exist_ok=True)
        _aside_copy(Path.home() / ".clearwing", post)
        diffs = []
        for name in ASIDE_FILES:
            was, now = (pre / name).exists(), (post / name).exists()
            if was and now:
                pre_b, post_b = (pre / name).read_bytes(), (post / name).read_bytes()
                if pre_b != post_b:
                    diffs.append(f"{name}:{len(pre_b)}->{len(post_b)}B")
            elif now and not was:
                diffs.append(f"{name}:+CREATED")
            elif was and not now:
                diffs.append(f"{name}:-DELETED")
        out.append(("memory-aside-diff-recorded", True,
                    f"changed: {diffs or 'none'} (control copies in state/aside-*)"))
    # post-run target-state comparison vs pre-state.json snapshot; a difference
    # is majority-confirmed 3x before it counts (one lost SYN != state change)
    def _probe_confirmed(tgt, port, want_open):
        for _ in range(3):
            if probe_majority(tgt, int(port)) == bool(want_open):
                return True
            time.sleep(1)
        return False
    try:
        pre_state = json.loads((run_dir / "state" / "pre-state.json").read_text())
        changed = []
        for tgt, ports in (pre_state.get("target_snapshot") or {}).items():
            for port, was_open in ports.items():
                if not _probe_confirmed(tgt, port, was_open):
                    changed.append(f"{tgt}:{port} {was_open}->{not was_open}")
        out.append(("target-state-unchanged", not changed, f"{changed or 'stable'}"))
    except (OSError, ValueError, TypeError):
        out.append(("target-state-unchanged", False, "pre-state.json unreadable"))
    # cumulative budget ledger (results/ is gitignored)
    try:
        led_p = RESULTS / "budget-ledger.json"
        ledger = json.loads(led_p.read_text()) if led_p.exists() else {"runs": [], "cap_usd": 700}
        run_cost = 0.0
        man = run_dir / "manifest.json"
        if man.exists():
            run_cost = round(sum((s.get("summary") or {}).get("cost_usd_product") or 0
                                 for s in json.loads(man.read_text())), 4)
        ledger["runs"].append({"ts": time.strftime("%F %T"), "run": run_dir.name,
                               "cost_usd": run_cost})
        ledger["total_usd"] = round(sum(r["cost_usd"] for r in ledger["runs"]), 4)
        led_p.write_text(json.dumps(ledger, indent=1))
        out.append(("budget-ledger-updated", ledger["total_usd"] <= ledger["cap_usd"],
                    f"cumulative ${ledger['total_usd']} / cap ${ledger['cap_usd']}"))
    except Exception as e:  # noqa: BLE001
        out.append(("budget-ledger-updated", False, f"ledger error: {e}"))

    # ---- value-based secret scan LAST, over artifacts INCLUDING the aside
    # snapshots just created (memory content may embed keys)
    secrets: list[bytes] = []
    try:
        if keyfile is not None:
            secrets.append(keyfile.read_bytes().strip())
    except OSError:
        pass
    llm_profile = os.environ.get(SUITE["env"]["llm_profile"])
    if llm_profile:
        try:
            import yaml as _y
            v = (_y.safe_load(Path(llm_profile).expanduser().read_text())
                 .get("zai", {}).get("api_key", ""))
            if isinstance(v, str) and len(v) >= 20:
                secrets.append(v.encode())
        except Exception:
            pass
    hits = []
    for f in run_dir.rglob("*"):
        if f.is_file() and f.suffix in (".jsonl", ".json", ".txt", ".log", ".tools", ".key"):
            try:
                data = f.read_bytes()
            except OSError:
                continue
            if any(s and s in data for s in secrets) or re.search(
                    rb"api_key[=:?&]\s*['\"]?[A-Za-z0-9_-]{20,}", data):
                hits.append(str(f))
    for f in list(run_dir.glob("state/aside-*/*")):   # memory snapshots:
        if not f.is_file():                            # suffix-blind scan
            continue
        try:
            data = f.read_bytes()
        except OSError:
            continue
        if any(s and s in data for s in secrets):
            hits.append(str(f))
    out.append(("artifact-keyscan", not hits, f"{hits[:3]} ({len(secrets)} value-probes)"))

    s, _ = http_get(f"{SUITE['webui']['live']}/api/health")
    out.append(("live-health-final", s == 200, f"HTTP {s}"))
    for name, ok, note in out:
        log(f"cleanup {'PASS' if ok else 'WARN'} {name} {note}")
    return out


def run_chaos_scenario(sc: dict, run_dir: Path, tier: dict, ws_url: str, keyfile: Path,
                       home: Path | None, direct_override: str | None = None) -> dict:
    """Chaos scenario: proxy + driver. Upstream resolved from the PRE-SURGERY
    direct endpoint when a config surgery is active (direct_override), else
    from the live config provider.base_url — never hardcoded (SPEC §7)."""
    import chaos
    direct = chaos.resolve_direct_endpoint(direct_override)
    logp = run_dir / "scenarios" / f"{sc['name']}.proxy.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    proxy = chaos.ProxyHandle(sc["mode"], int(sc["n"]), logp, direct, chaos.DEFAULT_PORT)
    try:
        raw_target = str(sc.get("target", "192.168.73.82"))
        target = raw_target if raw_target.startswith("192.") else f"192.168.73.{raw_target.lstrip('.')}"
        check_target(target)                      # whitelist applies to chaos too (Codex r1)
        prompt = render_prompt(sc["prompt"], run_dir, target)
        out = run_dir / "scenarios" / sc["name"]
        if sc.get("via_config"):
            base = key = None                      # fallback chain comes from real config
        elif sc.get("base_url_pathless"):
            base, key = f"http://127.0.0.1:{proxy.port}", llm_api_key()
        else:
            base, key = f"http://127.0.0.1:{proxy.port}{direct['path']}", llm_api_key()
        summary = asyncio.run(ws_run(
            ws_url, keyfile, target, prompt, out,
            int(sc.get("max_seconds", 420)), cost_cap_of(sc, tier),
            base_url=base, api_key=key, trust_status=sc.get("trust_status", True), home=home,
            approve_delay=float(sc.get("approve_delay", 0))))
        summary["chaos"] = {"mode": sc["mode"], "n": sc["n"], "refusals": proxy.refusals()}
        log(f"  -> {summary['seconds']}s statuses={summary['complete_statuses'][-3:]} "
            f"cost=${summary['cost_usd_product']} proxy_hits={proxy.refusals()}")
        return summary
    finally:
        proxy.stop()


def dispatch(sc: dict, ctx: dict) -> dict:
    t = sc["type"]
    repeat = int(sc.get("repeat", 1))
    if t == "ws" or t == "chaos":
        results = []
        for i in range(repeat):
            tag = f"-{i+1}" if repeat > 1 else ""
            sc2 = dict(sc, name=sc["name"] + tag)
            if t == "ws":
                r = run_ws_scenario(sc2, ctx["run_dir"], ctx["tier"], ctx["ws_url"],
                                    ctx["keyfile"], ctx.get("home"))
            else:
                r = run_chaos_scenario(sc2, ctx["run_dir"], ctx["tier"], ctx["ws_url"],
                                       ctx["keyfile"], ctx.get("home"),
                                       direct_override=ctx.get("direct_override"))
            if r.get("session_id"):
                ctx["last_sid"] = r["session_id"]
                ctx["sids"].append(r["session_id"])
            ctx["containers"].extend(r.get("containers", []))
            results.append({"name": sc2["name"], "summary": r, "expect": sc.get("expect")})
        # FLATTEN: every sub-run becomes its own manifest entry (gate coverage)
        ctx["results"].extend(results)
        return results[0]["summary"] if repeat == 1 else {"type": "repeat", "runs": [r["summary"] for r in results]}
    if t == "probe_partial_fill":
        return probe_partial_fill(ws_from_base(SUITE["webui"]["live"]), ctx["keyfile"],
                                  ctx["run_dir"] / "scenarios" / sc["name"])
    if t == "hud":
        return scenario_hud(sc, ctx["run_dir"], ctx["ws_url"], ctx["keyfile"])
    if t == "report_cli":
        if not ctx.get("last_sid"):
            return {"type": "report_cli", "status": "skipped", "reason": "no prior session"}
        return scenario_report_cli(ctx["run_dir"], ctx["last_sid"])
    if t == "pytest":
        return scenario_pytest(ctx["run_dir"])
    die(f"unknown scenario type {t!r}")


# ---------------------------------------------------- adjudication layer ---

ADJ_STATUS = "<!-- adjudication-status: {} -->"
ADJ_VERDICTS = ("PASS", "FAIL", "REGRESSION", "MIXED")


def _adj_override_rows(text: str) -> list[str]:
    """Data rows of the overridden-gates table (non-header, pipe rows)."""
    m = re.search(r"## overridden gates[^\n]*\n(.*?)(?=\n## |\Z)", text, re.S)
    rows = []
    for line in (m.group(1) if m else "").splitlines():
        s = line.strip()
        if s.startswith("|") and not set(s) <= {"|", "-", " ", ":"} and "---" not in s \
                and not s.startswith("| gate") and not s.startswith("| run "):
            rows.append(s)
    return rows


def _adj_reviewed_hash(dirs: list[Path]) -> str:
    """Content hash binding a FINAL adjudication to the exact machine
    artifacts it reviewed (Codex #67 r2 P1): report/gates/ledger of every
    indexed run dir. A post-review `analyze` re-render or any artifact
    change invalidates the stamp and export refuses."""
    import hashlib
    h = hashlib.sha256()
    for d in dirs:
        for name in ("report.md", "gates.json", "ledger.json"):
            f = Path(d) / name
            h.update(name.encode())
            h.update(f.read_bytes() if f.exists() else b"<absent>")
    return h.hexdigest()[:16]


def _adj_final_verdict(text: str) -> str | None:
    """Verdict parsed ONLY from the '## final verdict' section — a
    verdict-looking line quoted elsewhere must not be picked up
    (Codex #67 r6 P1)."""
    m = re.search(r"## final verdict[^\n]*\n(.*?)(?=\n## |\Z)", text, re.S)
    if not m:
        return None
    mv = re.search(r"^verdict:[ \t]*(\S+)", m.group(1), re.M)
    return mv.group(1) if mv else None


def _adj_indexed_dirs(text: str) -> list[Path]:
    """Run dirs recorded in the adjudication's run index (backtick paths)."""
    m = re.search(r"## run index.*?(?=\n## )", text, re.S)
    return [Path(x) for x in re.findall(r"`([^`]+)`", m.group(0))] if m else []


def _adj_status(text: str) -> str | None:
    """Line-anchored status parse — a marker QUOTED inside a review body
    must never satisfy the gate (review N1: full-text substring was
    fail-open)."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("<!-- adjudication-status: ") and s.endswith(" -->"):
            return s[len("<!-- adjudication-status: "):-len(" -->")]
    return None


def _run_dir_facts(d: Path) -> dict:
    """Machine-readable summary of one run dir (verdict, ledger, failed gates)."""
    import re as _re
    facts = {"path": str(d), "verdict": "?", "cost": None, "seconds": None,
             "failed_gates": [], "scenarios": [], "r3_pending": False}
    if re.search(r"-full(-\d+)?$", d.name) and not (d / "r3-done").exists():
        facts["r3_pending"] = True   # collision-suffixed …-full-1 counts (Codex #67 r1)
    rp = d / "report.md"
    if rp.exists():
        first = (rp.read_text().splitlines() or [""])[0]   # empty-report safe
        m = _re.search(r"— (PASS|FAIL|REGRESSION|SKIPPED)", first)
        if m:
            facts["verdict"] = m.group(1)
    try:
        led = json.loads((d / "ledger.json").read_text())
        c = led.get("cost_usd_product")
        # round(4) kills float residue like 2.3198000000000008 leaking into
        # adjudication/export text (R1-C, 2026-09-19); non-numeric junk -> None
        facts["cost"] = round(c, 4) if isinstance(c, (int, float)) else None
        facts["seconds"] = led.get("seconds")
    except (OSError, ValueError):
        pass
    try:
        gates = json.loads((d / "gates.json").read_text())
        facts["failed_gates"] = sorted({g["gate"] for g in gates if not g["pass"]})
    except (OSError, ValueError):
        pass
    try:
        for sc in json.loads((d / "manifest.json").read_text()):
            s = sc.get("summary") or {}
            import analyze as _az
            facts["scenarios"].append({
                "name": sc.get("name"), "sid": s.get("session_id"),
                "statuses": _az.statuses_compact(s.get("complete_statuses") or []),
                "cost": s.get("cost_usd_product"), "type": s.get("type", "ws")})
    except (OSError, ValueError):
        pass
    return facts


def cmd_adjudicate(args) -> None:
    """Round-level adjudication document (SPEC §9 hard rule).

    Draft: auto-fills run index (ALL run dirs of the round incl. FATAL
    ones), per-scenario verdicts, failed gates. The HUMAN then fills the
    overridden-gates table (gate flips need 論據) and the review-record
    section (four-lens output per REVIEW.md). --finalize refuses until the
    review record is non-empty and then stamps FINAL — the marker
    `cw-e2e export` requires before producing any external-facing draft
    (the 2026-09-18 #61 sequencing violation made mechanical)."""
    dirs = [Path(p).resolve() for p in args.run_dirs]
    for d in dirs:
        if not d.is_dir():
            die(f"run dir not found: {d}")
    out_dir = dirs[-1]
    adj = out_dir / "adjudication.md"

    if args.finalize:
        if not adj.exists():
            die(f"no adjudication.md in {out_dir} — run the draft step first")
        text = adj.read_text()
        status = _adj_status(text)
        if status == "FINAL":
            log("already FINAL")
            return
        if status != "DRAFT":
            die(f"adjudication.md status marker is {status!r} (mangled?) — refusing; "
                "restore the marker line or regenerate the draft consciously")
        m = re.search(r"## review record[^\n]*\n(.*?)(?=\n## |\Z)", text, re.S)
        body = re.sub(r"<!--.*?-->", "", m.group(1) if m else "", flags=re.S)
        body = re.sub(r"[\s#|>*-]", "", body)
        if len(body) < 30:
            die("review-record section is empty/too short — the four-lens review "
                "(REVIEW.md) must be pasted before finalize (SPEC §9 hard rule)")
        lm = re.search(r"## limits[^\n]*\n(.*?)(?=\n## |\Z)", text, re.S)
        lbody = lm.group(1) if lm else ""
        # the auto-injected trigger line is machine text, not a human 限定欄 —
        # it must not satisfy the guard by itself (lens-1 bypass finding)
        lbody = "\n".join(ln for ln in lbody.splitlines()
                           if not ln.lstrip("- ").startswith("觸發源（pre-state 自動引用）"))
        lbody = re.sub(r"<!--.*?-->", "", lbody, flags=re.S)
        lbody = re.sub(r"[\s#|>*-]", "", lbody)
        if len(lbody) < 30:
            die("limits section is empty/too short — every verdict carries its 限定欄 "
                "(n= / warm-cold / scope / 口徑) before finalize; the 2026-09-19 rev1 "
                "fill silently missed the template anchor and shipped an empty section")
        verdict = _adj_final_verdict(text)
        if not verdict:
            die("final verdict line is empty — set `verdict: PASS|FAIL|REGRESSION|MIXED` "
                "in the '## final verdict' section (export quotes THIS value)")
        if verdict not in ADJ_VERDICTS:
            die(f"verdict {verdict!r} not in {ADJ_VERDICTS} (Codex #67 r2)")
        ih_now = __import__("hashlib").sha256(
            "|".join(str(x) for x in (_adj_indexed_dirs(text) or dirs)).encode()
        ).hexdigest()[:16]
        ih_orig = re.search(r"<!-- index-hash: ([0-9a-f]+) -->", text)
        if ih_orig and ih_orig.group(1) != ih_now:
            die("run index was edited after the draft (dropped/added runs?) — the "
                "adjudication no longer covers what it claims; regenerate consciously")
        idx_dirs = _adj_indexed_dirs(text) or dirs
        facts = [_run_dir_facts(x) for x in idx_dirs]   # the ROUND, not the
        machine = sorted({f["verdict"] for f in facts if f["verdict"] != "?"})  # CLI arg
        flipped = machine and verdict not in machine and not (len(machine) > 1)
        if flipped:
            rows = _adj_override_rows(text)
            if not rows:
                die(f"final verdict {verdict} flips the machine verdict {machine} but the "
                    "overridden-gates table is EMPTY — a flip must carry its 論據 rows "
                    "(gate | run/scenario | conclusion | evidence | limits)")
            covered = {r.strip("|").split("|")[0].strip().lower() for r in rows}
            failed = {g.lower() for f in facts for g in f["failed_gates"]}
            uncovered = failed - covered
            if uncovered:
                die(f"flip evidence incomplete: failed gate(s) {sorted(uncovered)} have no "
                    "override row — every overturned gate needs its own row")
        stamp_dirs = _adj_indexed_dirs(text) or dirs   # same set export will use
        stamp = (f"<!-- reviewed-hash: {_adj_reviewed_hash(stamp_dirs)} -->")
        adj.write_text(text.replace(ADJ_STATUS.format("DRAFT"),
                                    ADJ_STATUS.format("FINAL") + "\n" + stamp))
        log(f"adjudication FINAL: {adj} verdict={verdict} reviewed-hash bound")
        return

    if adj.exists():
        existing = adj.read_text()
        if _adj_status(existing) == "FINAL":
            die(f"{adj} is FINAL — regenerating would destroy the review record; "
                "delete it consciously first (SPEC §2.7 refuse-overwrite)")
        m = re.search(r"## review record[^\n]*\n(.*?)(?=\n## |\Z)", existing, re.S)
        body = re.sub(r"<!--.*?-->", "", m.group(1) if m else "", flags=re.S)
        if len(re.sub(r"[\s#|>*-]", "", body)) >= 30:
            die(f"{adj} already carries a human review record — regenerating would "
                "destroy it; delete it consciously first (SPEC §2.7)")
        if _adj_final_verdict(existing):
            die(f"{adj} already carries a human final verdict — regenerating would "
                "destroy it; delete it consciously first (SPEC §2.7)")
        if _adj_override_rows(existing):
            die(f"{adj} already carries human override rows — regenerating would "
                "destroy them; delete it consciously first (SPEC §2.7)")
        lm = re.search(r"## limits[^\n]*\n(.*?)(?=\n## |\Z)", existing, re.S)
        lbody = (lm.group(1) if lm else "").strip()
        # the auto-injected trigger line is machine text — with it, a recorded
        # --trigger made every regeneration hit the human-limits guard (Codex #68 r1)
        lbody = "\n".join(ln for ln in lbody.splitlines()
                           if not ln.lstrip("- ").startswith("觸發源（pre-state 自動引用）")).strip()
        if lbody and lbody != "-":
            die(f"{adj} already carries human limits — regenerating would "
                "destroy them; delete it consciously first (SPEC §2.7)")
    lines = [
        "# cw-e2e adjudication — round " + time.strftime("%Y-%m-%d %H:%M"), "",
        ADJ_STATUS.format("DRAFT"), "",
        # index bound at DRAFT time: finalize refuses if the run list was
        # hand-edited (dropping a run would dodge its gates AND its spend)
        f"<!-- index-hash: {__import__('hashlib').sha256('|'.join(str(x) for x in dirs).encode()).hexdigest()[:16]} -->",
        "",
        "## run index (ALL run dirs of this round, incl. FATAL attempts)", "",
    ]
    try:      # SPEC §3: the adjudication doc quotes the round's recorded trigger
        trig = json.loads((dirs[-1] / "state" / "pre-state.json").read_text()).get("trigger")
    except (OSError, ValueError):
        trig = ""
    total_cost = 0.0
    for d in dirs:
        f = _run_dir_facts(d)
        total_cost += f["cost"] or 0
        lines.append(f"- `{f['path']}` — **{f['verdict']}** · ${f['cost'] if f['cost'] is not None else '?'}"
                     f" · {f['seconds'] if f['seconds'] is not None else '?'}s"
                     + (f" · failed gates: {', '.join(f['failed_gates'])}" if f["failed_gates"] else "")
                     + (" · **R3 未完成**（無 r3-done — 發版級宣稱前必完成抽核）" if f["r3_pending"] else ""))
    lines += ["", f"- round cost: ${total_cost:.4f}", "",
              "## scenario verdicts (auto)", "",
              "| run | scenario | sid | statuses | $ |", "|---|---|---|---|---|"]
    for d in dirs:
        f = _run_dir_facts(d)
        for sc in f["scenarios"]:
            lines.append(f"| {d.name} | {sc['name']} | {sc['sid'] or '-'} | "
                         f"{sc['statuses']} | {sc['cost'] if sc['cost'] is not None else '-'} |")
    lines += [
        "", "## overridden gates (HUMAN — a flipped verdict MUST live here with 論據)", "",
        "| gate | run/scenario | conclusion | evidence | limits |", "|---|---|---|---|---|",
        "", "## limits (每判定限定欄：n= / warm-cold / scope / 口徑)", "",
        *([f"- 觸發源（pre-state 自動引用）：{trig}"] if trig else []),
        "- ", "",
        "## final verdict (HUMAN — 翻案後的最終判定；未翻案則照抄機器判定)", "",
        "verdict: ",
        "<!-- PASS | FAIL | REGRESSION | MIXED(多 run) — finalize 要求非空；export 引用此值而非 report.md -->",
        "",
        "## review record (四鏡複審輸出 — REVIEW.md；finalize 前必填)", "",
        "<!-- paste the four-lens review summary here; 觸發: 判定翻案/對外開單前/發版判定 -->",
        "", "## rev2 log", "", "- (rev1 generated)", "",
    ]
    adj.write_text("\n".join(lines))
    log(f"adjudication DRAFT: {adj} — fill overrides/limits/review-record, "
        "then `cw-e2e adjudicate --finalize <run-dir>`")


def cmd_export(args) -> None:
    """External-facing draft generator, GATED on a FINAL adjudication
    (SPEC §9: no external posting before the review record exists)."""
    d = Path(args.run_dir).resolve()
    adj = d / "adjudication.md"
    if not adj.exists():
        die(f"no adjudication.md in {d} — run `cw-e2e adjudicate` and complete the "
            "review record first (SPEC §9 hard rule: 對外開單前必複審)")
    text = adj.read_text()
    if _adj_status(text) != "FINAL":
        die("adjudication is still DRAFT — finalize it (non-empty review record "
            "required) before exporting external-facing content")
    indexed = _adj_indexed_dirs(text) or [d]
    hm = re.search(r"<!-- reviewed-hash: ([0-9a-f]+) -->", text)
    if not hm:
        die("FINAL adjudication carries no reviewed-hash (deleted? pre-feature "
            "document?) — re-finalize to bind it to the reviewed artifacts")
    if hm.group(1) != _adj_reviewed_hash(indexed):
        die("reviewed artifacts changed since finalize (analyze re-run or "
            "report/gates/ledger mutation) — re-review and re-finalize; export "
            "refuses to mix an old verdict with fresh machine facts")
    all_facts = [_run_dir_facts(x) for x in indexed]
    total_cost = round(sum(f["cost"] or 0 for f in all_facts), 4)
    machine = sorted({f["verdict"] for f in all_facts if f["verdict"] != "?"})
    final_verdict = _adj_final_verdict(text) or "MIXED"
    # MIXED over heterogeneous run verdicts is round AGGREGATION, not a flip —
    # "see overrides" there misled readers of the 2026-09-19 export (R2 #4)
    if final_verdict in machine or not machine:
        machine_note = ""
    elif len(machine) > 1 and final_verdict == "MIXED":
        machine_note = f" (round aggregate — machine run verdicts were {'/'.join(machine)})"
    else:
        machine_note = f" (machine verdict was {'/'.join(machine)} — adjudicated; see overrides)"
    out = [f"cw-e2e round summary (auto-export {time.strftime('%F %T')} — "
           f"adjudicated, review on record)", "",
           f"- verdict: **{final_verdict}**{machine_note} · round ${total_cost} · "
           f"{len(all_facts)} run dir(s)"]
    for f in all_facts:
        out.append(f"  - `{f['path']}` — {f['verdict']} · ${f['cost']} · "
                   + (f"failed: {', '.join(f['failed_gates'])}" if f["failed_gates"] else "clean"))
    if final_verdict == "MIXED" and len(machine) > 1:
        # only for TRUE aggregation over heterogeneous run verdicts — a
        # single-machine MIXED flip says "see overrides" above and must not
        # also claim "非任一 run 的翻案" (lens-1 contradiction finding)
        out.append("- ⚠️ MIXED＝輪級聚合（多 run 判定互異），非任一 run 的翻案——"
                   "不授權合併/發版動作；各 run 判定以上列為準")
    if any(f["r3_pending"] for f in all_facts):
        out.append("- ⚠️ R3 人工抽核未完成（r3-manual.md 待填、無 r3-done）——"
                   "發版級宣稱（Full×2＋deep-cold）不應引用本輪")
    text = "\n".join(out)
    if args.out:
        Path(args.out).write_text(text + "\n")
        log(f"exported: {args.out}")
    else:
        print(text)


def _budget_precheck(tier: dict, only: str | None) -> None:
    """Enforce the program cap BEFORE spending (Codex #67 r3): the cleanup
    ledger check comes too late for a run that starts at $699."""
    cap = 700
    led_p = RESULTS / "budget-ledger.json"
    if led_p.exists():
        try:
            led = json.loads(led_p.read_text())
            spent, cap = led.get("total_usd", 0), led.get("cap_usd", 700)
        except (OSError, ValueError):
            die(f"budget ledger unreadable: {led_p} — fail-closed; fix or remove "
                "the ledger before running (Codex #67 r6)")
    else:
        spent = 0
    # repeats multiply their cap (t3-warm-chain ×3 counted ONCE undercounts)
    est = sum(cost_cap_of(sc, tier) * int(sc.get("repeat", 1))
              for sc in tier["scenarios"] if not only or sc["name"] == only)
    if spent + est > cap:
        die(f"budget pre-check: cumulative ${spent:.2f} + this run's cap ceiling "
            f"${est:.2f} would exceed the ${cap} program cap — reduce scope or get "
            "the cap raised (ledger: e2e/results/budget-ledger.json)")


def cmd_run(args) -> None:
    tier_name = args.tier
    tier = SUITE["tiers"][tier_name]
    if tier.get("requires_approval") and not args.approve_fallback:
        die("deep-fallback mutates the real config + restarts the self instance — "
            "pass --approve-fallback (SPEC §2.6)")
    keyfile = keyfile_from_env(args.keyfile)
    # P3.4 (tier-aware, review N5): only quick/full contain --base-url
    # scenarios (probe + chaos); deep tiers run via_config/self-instance and
    # a profile-less `verify` preflight stays legal (--force bypasses)
    selected = [sc for sc in tier["scenarios"]
                if not args.only or sc["name"] == args.only]
    needs_profile = any(
        sc["type"] == "probe_partial_fill"
        or (sc["type"] == "chaos" and not sc.get("via_config"))
        for sc in selected)
    if needs_profile and not args.force:
        check_llm_profile()          # scenario-aware (Codex #67 r1): --only fc-approval stays legal
    _budget_precheck(tier, args.only)
    with SuiteLock():
        run_dir = new_run_dir(tier_name + ("-partial" if args.only else ""))
        log(f"run dir: {run_dir}")
        ver = verify(run_dir, strict=not args.force, trigger=getattr(args, "trigger", None) or "")
        if args.force:
            forced = [c for c in ver.get("checks", []) if not c[1]]
            (run_dir / "state" / "forced-gates.json").write_text(json.dumps(
                {"forced": forced,
                 "rationale": "OPERATOR: fill in why each forced gate was overridden"},
                indent=1, ensure_ascii=False))
            log(f"forced gates persisted ({len(forced)}) — fill the rationale slot "
                "in state/forced-gates.json")
        ctx = {"run_dir": run_dir, "tier": tier, "ws_url": ws_from_base(SUITE["webui"]["live"]),
               "keyfile": keyfile, "home": None, "results": [], "containers": [], "sids": [], "last_sid": None}
        spawn = None
        surg = None
        try:
            if tier_name in ("deep-cold", "deep-fallback"):
                import surgery
                if tier_name == "deep-fallback":
                    surg = surgery.ConfigSurgery()
                    surg.enter()                       # guarded real-config edit
                    ctx["direct_override"] = surg.original_url   # pre-surgery direct endpoint
                    log("config surgery ACTIVE (chaos primary + direct fallback)")
                spawn = surgery.spawn_instance(run_dir)
                ctx.update(ws_url=spawn["ws_url"], keyfile=spawn["keyfile"], home=spawn["home"])
                log(f"self instance {spawn['ws_url']} pid={spawn['pid']} home={spawn['home']}")
                if tier_name == "deep-cold":
                    cold = surgery.assert_cold_home(spawn["home"])
                    log(f"cold-home assertion: {cold}")
                    if not cold["pass"] and not args.force:
                        raise RuntimeError(f"cold home not clean: {cold}")
            for sc in tier["scenarios"]:
                if args.only and sc["name"] != args.only:
                    continue
                log(f"=== {sc['name']} ({sc['type']}) ===")
                try:
                    summary = dispatch(sc, ctx)
                    if sc["type"] not in ("ws", "chaos"):
                        ctx["results"].append({"name": sc["name"], "summary": summary,
                                               "expect": sc.get("expect")})
                except Exception as e:  # noqa: BLE001 — one scenario must not kill the run
                    log(f"  -> scenario CRASHED: {type(e).__name__}: {e}")
                    crashed = {"type": "crashed", "status": "crashed",
                               "error": f"{type(e).__name__}: {e}"}
                    # spend already metered before the crash must not vanish
                    # from the ledger (Codex #67 r2): recover the running
                    # total from the frames log the driver managed to write
                    for fl in sorted((run_dir / "scenarios").glob(
                            f"{sc['name']}*.frames.jsonl")):   # suffixed repeats too
                        try:
                            import re as _re2
                            costs = [float(m) for m in _re2.findall(
                                r'"total_cost_usd":\s*([0-9.]+)',
                                fl.read_text(errors="ignore"))]
                            if costs:
                                crashed["cost_usd_product"] = max(
                                    crashed.get("cost_usd_product") or 0, max(costs))
                        except (OSError, ValueError):
                            pass
                    ctx["results"].append({"name": sc["name"], "summary": crashed,
                                           "expect": sc.get("expect")})
                (run_dir / "manifest.json").write_text(
                    json.dumps(ctx["results"], ensure_ascii=False, indent=1, default=str))
        finally:
            if spawn:
                import surgery
                surgery.stop_instance(spawn)
            if surg:
                report = surg.exit()                       # restore + diff/health verify
                restored = bool(report.get("restored") and report.get("diff_clean")
                                and report.get("backup_deleted"))
                surg_gate = ("surgery-restore", restored,
                             f"restored={report.get('restored')} diff_clean={report.get('diff_clean')} "
                             f"backup_deleted={report.get('backup_deleted')} "
                             f"live_health={report.get('live_health')}")
            else:
                surg_gate = None
            try:
                cleanup = cleanup_run(run_dir, ctx["containers"], before_8899=ver.get("_pids"),
                                      sids=ctx["sids"], keyfile=keyfile)
            except Exception as e:  # noqa: BLE001
                cleanup = [("cleanup-crashed", False, str(e))]
            if surg_gate:
                cleanup.append(surg_gate)
            (run_dir / "cleanup.json").write_text(json.dumps(cleanup))
        import analyze
        analyze.render(run_dir, tier_name, ver, cleanup)
        log(f"done — report: {run_dir / 'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="cw-e2e", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="Tier0 preflight gates (no LLM cost)")
    v.add_argument("--force", action="store_true")
    v.add_argument("--keyfile")
    r = sub.add_parser("run", help="run a tier")
    r.add_argument("--tier", required=True, choices=["quick", "full", "deep-cold", "deep-fallback"])
    r.add_argument("--force", action="store_true", help="skip hard gates")
    r.add_argument("--only", help="run a single scenario by name")
    r.add_argument("--approve-fallback", action="store_true",
                   help="authorize deep-fallback real-config surgery")
    r.add_argument("--trigger",
                   help="SPEC §3 trigger-honesty source (e.g. '使用者指示 2026-09-19' / "
                        "'8899 HEAD 變更 #NN') — recorded in pre-state + report header")
    r.add_argument("--keyfile")
    a = sub.add_parser("analyze", help="re-render analysis+report for an existing run dir")
    a.add_argument("run_dir")
    s = sub.add_parser("selftest", help="offline regression suite (no LLM cost, no live deps)")
    adj = sub.add_parser("adjudicate", help="round-level adjudication doc (SPEC §9 hard rule)")
    adj.add_argument("run_dirs", nargs="+", help="ALL run dirs of the round, oldest first "
                                                 "(adjudication.md lands in the LAST)")
    adj.add_argument("--finalize", action="store_true",
                     help="validate review record and stamp FINAL (unlocks export)")
    exp = sub.add_parser("export", help="external-facing draft — requires FINAL adjudication")
    exp.add_argument("run_dir")
    exp.add_argument("--out", help="write to file instead of stdout")
    args = ap.parse_args()
    if args.cmd == "verify":
        verify(None, strict=not args.force)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "analyze":
        import analyze
        analyze.render(Path(args.run_dir).resolve(), "re-analyze",
                       {"git": git_info()}, [])
    elif args.cmd == "selftest":
        p = subprocess.run([str(REPO_ROOT / ".venv/bin/python"), "-m", "pytest",
                            str(E2E_ROOT / "test_suite.py"), "-q"])
        sys.exit(p.returncode)
    elif args.cmd == "adjudicate":
        cmd_adjudicate(args)
    elif args.cmd == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
