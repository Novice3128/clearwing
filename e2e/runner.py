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
    completes_without_status = 0
    complete_while_approval_open = 0

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
        while time.time() - t0 < max_seconds and frames < 12000:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except (asyncio.TimeoutError, TimeoutError):
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
                if approve_delay:
                    await asyncio.sleep(approve_delay)
                await send(ws, {"type": "approve", "approved": not deny})
                frames_log.write(f"# approve #{approvals} decision={not deny}\n")
            elif t == "cost_update":
                cost_updates += 1
                cost_usd = max(cost_usd, float(d.get("total_cost_usd") or d.get("cost") or 0))
                tokens_in += int(d.get("input_tokens") or 0)
                tokens_out += int(d.get("output_tokens") or 0)
                tokens_cached += int(d.get("cached_tokens") or 0)
                if d.get("model"):
                    cost_models.add(d["model"])
                if cost_usd > cost_cap and not watchdog:
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
                if stop_after_tools and tool_starts >= stop_after_tools:
                    await send(ws, {"type": "stop"})
                    frames_log.write(f"# STOP sent after tool_start #{tool_starts}\n")
                    stop_after_tools = 0
            elif t == "tool_result":
                approval_open = 0   # approval consumed by execution
            elif t == "error":
                errors.append(str(d.get("message") or d.get("error") or raw)[:200])
            elif t == "agent_message":
                c = d.get("content")
                if isinstance(c, str) and c:
                    agent_text.append(c)
            elif t == "flag_found":
                flag_frames += 1
                flags = d.get("flags")
                flag_faces += len(flags) if isinstance(flags, list) else 1
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
                if trust_status and status in ("ok", "stopped", "error"):
                    break

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
        die(f"env ${SUITE['env']['llm_profile']} (yaml path) required for --base-url runs")
    return yaml.safe_load(Path(p).expanduser().read_text())["zai"]["api_key"]


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
    log(f"scenario {sc['name']}{instance_tag}: target={target} cap={sc.get('cost_cap', tier['cost_cap'])}")
    summary = asyncio.run(ws_run(
        ws_url, keyfile, target, prompt, out,
        int(sc.get("max_seconds", 600)), float(sc.get("cost_cap", tier["cost_cap"])),
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
    ok = p.returncode == 0 and len(p.stdout.strip()) > 0
    (run_dir / "scenarios" / "report-cli.txt").write_text(p.stdout[:5000] + "\n---stderr---\n" + p.stderr[:2000])
    log(f"  -> report -s {sid}: exit={p.returncode} bytes={len(p.stdout)} pass={ok}")
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
    (run_dir / "scenarios" / "pytest-subset.txt").write_text(p.stdout[-8000:])
    log(f"  -> pytest subset: {tail} (non-env failures: {non_env or 'none'})")
    return {"type": "pytest", "tail": tail, "non_env_failures": non_env, "pass": not non_env}


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


def verify(run_dir: Path | None, strict: bool = True) -> dict:
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

    try:
        kf_env = os.environ.get(SUITE["env"]["keyfile"])
        k2 = Path.home() / ".clearwing/webui-8899.key"
        same = (kf_env and Path(kf_env).read_bytes().strip() == k2.read_bytes().strip())
        checks.append(("key-files-info", True, f"env-key == webui-8899.key (newline-stripped): {same}"))
    except OSError:
        checks.append(("key-files-info", True, "webui-8899.key absent (env-key deployment)"))

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
        suite_hash = hashlib.sha256()
        for f in ("runner.py", "chaos.py", "surgery.py", "analyze.py", "suite.yaml", "SPEC.md"):
            suite_hash.update((E2E_ROOT / f).read_bytes())
        (st / "pre-state.json").write_text(json.dumps({
            "git": gi, "audit_dirs": audit_n,
            "memory_db": _sz("memory.db"), "knowledge_graph": _sz("knowledge_graph.json"),
            "pids_8899": pids,
            "suite_sha256": suite_hash.hexdigest()[:16],
            "target_snapshot": {t: {p: tcp_probe(t, p) for p in (88, 445, 3389)} for t in SUITE["targets"]},
        }, indent=1))

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
              "report_cost": rep_cost, "report_tokens": rep_tokens, "pass": bool(match)}
    (run_dir / "scenarios" / "hud-proof.json").write_text(json.dumps(result, indent=1))
    log(f"  -> HUD footer {hud_cost}/{hud_tokens} vs report {rep_cost}/{rep_tokens} match={match}")
    return result


def cleanup_run(run_dir: Path, container_ids: list[str], before_8899: list[int] | None = None,
                sids: list[str] | None = None) -> list[tuple[str, bool, str]]:
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
    out.append(("tmp-residue", not (list(Path('/tmp').glob('report_*')) or list(Path('/tmp').glob('kerbrute*'))), ""))
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
    hits = []
    for f in run_dir.rglob("*"):
        if f.is_file() and f.suffix in (".jsonl", ".json", ".txt", ".log", ".tools"):
            try:
                if re.search(r"api_key[=:?&]\s*['\"]?[A-Za-z0-9_-]{20,}", f.read_text(errors="ignore")):
                    hits.append(str(f))
            except OSError:
                pass
    out.append(("artifact-keyscan", not hits, f"{hits[:3]}"))
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
            int(sc.get("max_seconds", 420)), float(sc.get("cost_cap", tier["cost_cap"])),
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


def cmd_run(args) -> None:
    tier_name = args.tier
    tier = SUITE["tiers"][tier_name]
    if tier.get("requires_approval") and not args.approve_fallback:
        die("deep-fallback mutates the real config + restarts the self instance — "
            "pass --approve-fallback (SPEC §2.6)")
    keyfile = keyfile_from_env(args.keyfile)
    with SuiteLock():
        run_dir = new_run_dir(tier_name + ("-partial" if args.only else ""))
        log(f"run dir: {run_dir}")
        ver = verify(run_dir, strict=not args.force)
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
                    ctx["results"].append({"name": sc["name"],
                                           "summary": {"type": "crashed", "status": "crashed",
                                                       "error": f"{type(e).__name__}: {e}"},
                                           "expect": sc.get("expect")})
                (run_dir / "manifest.json").write_text(
                    json.dumps(ctx["results"], ensure_ascii=False, indent=1, default=str))
        finally:
            if spawn:
                import surgery
                surgery.stop_instance(spawn)
            if surg:
                surg.exit()                            # restore + diff/health verify
            try:
                cleanup = cleanup_run(run_dir, ctx["containers"], before_8899=ver.get("_pids"),
                                      sids=ctx["sids"])
            except Exception as e:  # noqa: BLE001
                cleanup = [("cleanup-crashed", False, str(e))]
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
    r.add_argument("--keyfile")
    a = sub.add_parser("analyze", help="re-render analysis+report for an existing run dir")
    a.add_argument("run_dir")
    args = ap.parse_args()
    if args.cmd == "verify":
        verify(None, strict=not args.force)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "analyze":
        import analyze
        analyze.render(Path(args.run_dir).resolve(), "re-analyze",
                       {"git": git_info()}, [])


if __name__ == "__main__":
    main()
