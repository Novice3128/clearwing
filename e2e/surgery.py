#!/usr/bin/env python3
"""cw-e2e surgery — self-instance lifecycle (8898) + guarded real-config
surgery for the fallback tier.

Design constraints baked from the 2026-09-17/18 reviews (SPEC §2.5–2.6):
- CLEARWING_HOME does NOT isolate provider config (the real ~/.clearwing/config.yaml
  always overlays home config, config.py:146-161) — fallback topology therefore
  REQUIRES a real-config edit; the self instance merely re-reads it on ITS start,
  so the member's 8899 process is never restarted by us.
- The self instance MUST be launched via the `clearwing webui` CLI entry point
  (installs the api_key= log filter), bind 127.0.0.1 (unauthenticated
  /api/sessions* + /api/metrics), cwd = run dir (keeps ./results inside the run),
  CLEARWING_MCP_SERVERS_DIR pointed at an empty dir, and be killed ONLY by its
  pidfile pid — never `pkill -f clearwing` (would kill 8899 and the 8080 system
  instance too).
- Config surgery: timestamped 0600 backup, atomic write + fsync, signal/exit
  traps that restore, byte-identical diff verify on exit, backup deleted after.
"""
from __future__ import annotations

import atexit
import json
import os
import secrets
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
E2E_ROOT = Path(__file__).resolve().parent
LIVE_CFG = Path.home() / ".clearwing" / "config.yaml"
import yaml as _yaml
_SELF_URL = _yaml.safe_load((E2E_ROOT / "suite.yaml").read_text())["webui"]["self"]
SELF_PORT = int(_SELF_URL.rsplit(":", 1)[-1])


def _log(msg: str) -> None:
    print(f"[cw-e2e-surgery {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_status(url: str, timeout: float = 3.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return -1


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass                            # best-effort: not all filesystems allow it


# ------------------------------------------------------- self instance ----

def spawn_instance(run_dir: Path, port: int = SELF_PORT) -> dict:
    key = secrets.token_urlsafe(36)
    keyfile = run_dir / f"webui-{port}.key"
    kfd = os.open(keyfile, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(kfd, "w") as fh:
        fh.write(key)
    home = run_dir / "home"
    (home / "audit").mkdir(parents=True, exist_ok=True)
    mcp = run_dir / "empty-mcp"
    mcp.mkdir(exist_ok=True)
    logf = open(run_dir / f"webui-{port}.log", "ab")

    env = dict(os.environ)
    env.update({
        "CLEARWING_WEB_API_KEY": key,
        "CLEARWING_HOME": str(home),
        "CLEARWING_MCP_SERVERS_DIR": str(mcp),
    })
    proc = subprocess.Popen(
        [str(REPO_ROOT / ".venv/bin/clearwing"), "webui", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(run_dir), env=env, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True, stdin=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(40):
        if http_status(f"{base}/api/health") == 200:
            break
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError(f"self instance died at startup — see {run_dir / f'webui-{port}.log'}")
        time.sleep(1)
    else:
        proc.terminate()
        proc.wait(timeout=10)
        logf.close()
        raise RuntimeError("self instance health timeout")
    _log(f"self instance up: {base} pid={proc.pid} home={home}")
    return {"ws_url": f"ws://127.0.0.1:{port}/ws/agent", "http": base, "keyfile": keyfile,
            "home": home, "pid": proc.pid, "_proc": proc, "_logf": logf}


def stop_instance(meta: dict) -> None:
    proc = meta.get("_proc")
    if not proc:
        return
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:
            pass
    deadline = time.time() + 10
    while time.time() < deadline and http_status(f"{meta['http']}/api/health") != -1:
        time.sleep(0.5)
    _log(f"self instance stopped (pid={meta['pid']}, port free={http_status(meta['http'] + '/api/health') == -1})")
    try:
        meta["_logf"].close()
    except Exception:
        pass
    try:
        # ephemeral instance key must not survive the run as a key-bearing
        # artifact (review r2)
        Path(meta["keyfile"]).unlink(missing_ok=True)
    except (OSError, KeyError):
        pass


def assert_cold_home(home: Path) -> dict:
    """Deep-cold precondition: memory must actually be empty (SPEC §3)."""
    checks = {
        "memory_db_absent": not (home / "memory.db").exists(),
        "knowledge_graph_absent": not (home / "knowledge_graph.json").exists(),
        "audit_empty": not any((home / "audit").iterdir()),
        "results_empty": not any((home.parent / "results").glob("**/*")) if (home.parent / "results").exists() else True,
    }
    return {"pass": all(checks.values()), "checks": checks}


# ------------------------------------------------------ config surgery ----

def build_surgical_config(original: str, proxy_base: str) -> tuple[str, str]:
    """Pure config rewrite: returns (new_yaml_text, chaos_primary_url).

    Structural injection — mutate the parsed provider mapping and dump, so
    `fallbacks` lands UNDER provider no matter what keys follow it. The old
    text-append glued indented `fallbacks:` lines onto whatever top-level
    block was last; with any mapping after `provider` the fallback became
    its child (or invalid YAML) and was never read by resolve_fallback_
    endpoints (Codex r1). Comments are dropped in the surgical state —
    acceptable: exit() restores the original bytes.
    """
    cfg = _yaml.safe_load(original)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("provider"), dict):
        raise RuntimeError("live config has no provider mapping — refusing surgery")
    prov = cfg["provider"]
    direct, api_key, model = prov.get("base_url"), prov.get("api_key"), prov.get("model")
    if not (isinstance(direct, str) and direct and api_key and model):
        raise RuntimeError("provider mapping lacks base_url/api_key/model — refusing surgery")
    from urllib.parse import urlsplit
    _u = urlsplit(direct)
    if _u.scheme == "http" and _u.hostname in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("live config already points at a local proxy — refusing")
    chaos_primary = f"{proxy_base}{urlsplit(direct).path.rstrip('/')}"
    prov["base_url"] = chaos_primary
    prov["fallbacks"] = [{
        "adapter": "openai",
        "base_url": direct,
        "model": model,
        "api_key": api_key,
    }]
    return _yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=4096), chaos_primary


def _sessions_running_8899() -> bool:
    """True when any 8899 session is running/paused; False when the list is
    verifiably clean. /api/sessions returns objects with status strings —
    there is no boolean `true` to string-match (Codex r1). Any non-200
    (incl. 503 store-unavailable) REFUSES: right before editing the real
    config, "cannot verify" must not degrade into "proceed"."""
    code = http_status("http://127.0.0.1:8899/api/sessions")
    if code != 200:
        raise RuntimeError(f"/api/sessions answered HTTP {code} — surgery refused "
                           "(cannot verify no running sessions)")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8899/api/sessions", timeout=5) as r:
            body = r.read()
        sessions = json.loads(body)
    except json.JSONDecodeError:
        raise RuntimeError("/api/sessions returned non-JSON — surgery refused "
                           "(cannot verify no running sessions)")
    except OSError as e:
        raise RuntimeError(f"/api/sessions unreadable ({type(e).__name__}) — surgery refused")
    if not isinstance(sessions, list):
        raise RuntimeError("/api/sessions returned non-list — surgery refused")
    # `paused` = parked awaiting an approval decision — still an active
    # mid-turn session that surgery would strand
    return any(isinstance(s, dict) and s.get("status") in ("running", "paused")
               for s in sessions)


class ConfigSurgery:
    """Guarded edit of the REAL ~/.clearwing/config.yaml (fallback tier).

    enter(): backup + rewrite provider.base_url to the chaos proxy (with full
             path) and append ONE fallback member = the original direct
             endpoint (api key copied in memory, never printed).
    exit():  byte-identical restore + diff assert + backup deletion.
    """

    def __init__(self, proxy_base: str = f"http://127.0.0.1:8787"):
        self.proxy_base = proxy_base
        self.backup: Path | None = None
        self._original: str | None = None
        self.original_url: str | None = None
        self._armed = False
        self._trapped: list = []

    def _write_atomic(self, text: str) -> None:
        tmp = LIVE_CFG.with_name("config.yaml.cwe2e-tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, LIVE_CFG)

    def _restore_now(self) -> None:
        if not self._armed or self._original is None:
            return
        self._write_atomic(self._original)
        LIVE_CFG.with_name("config.yaml.cwe2e-tmp").unlink(missing_ok=True)
        self._armed = False
        if self.backup is not None:
            self.backup.unlink(missing_ok=True)
            self.backup = None
        for sig in self._trapped:
            try:
                signal.signal(sig, signal.SIG_DFL)
            except (ValueError, OSError):
                pass
        self._trapped = []
        _log("config RESTORED (atomic write, backup deleted, traps disarmed)")

    def enter(self) -> "ConfigSurgery":
        residue = sorted(str(p) for p in Path.home().glob(".clearwing/config.yaml.bak-*")) + \
            ([str(LIVE_CFG.with_name("config.yaml.cwe2e-tmp"))]
             if LIVE_CFG.with_name("config.yaml.cwe2e-tmp").exists() else [])
        if residue:
            raise RuntimeError(f"stale config surgery residue present (contains secrets): "
                               f"{residue} — verify live config integrity, then delete before surgery")
        if LIVE_CFG.is_symlink():
            raise RuntimeError("live config is a symlink — atomic replace would destroy it "
                               "(dotfile-managed deployment); surgery refused")
        original = LIVE_CFG.read_text()
        # pre-flight: member 8899 must have no running sessions (we never
        # restart it, but the risk window disclosure demands a clean field)
        running = _sessions_running_8899()
        if running:
            raise RuntimeError("member 8899 has running sessions — surgery refused")
        new_text, chaos_primary = build_surgical_config(original, self.proxy_base)
        # direct endpoint snapshot for the chaos proxy, taken pre-surgery
        self.original_url = _yaml.safe_load(original)["provider"]["base_url"]
        self._original = original
        self.backup = LIVE_CFG.with_name(
            f"config.yaml.bak-cwe2e-{time.strftime('%Y%m%d-%H%M%S')}")
        bfd = os.open(self.backup, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(bfd, "w") as fh:
            fh.write(original)
            fh.flush()
            os.fsync(fh.fileno())          # the backup is the only recovery
                                           # asset after SIGKILL — it must be
                                           # durable BEFORE the swap happens
        _fsync_dir(LIVE_CFG.parent)
        self._armed = True                     # armed BEFORE the swap: any crash from here restores
        atexit.register(self._restore_now)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, lambda *_: (self._restore_now(), sys_exit(130)))
                self._trapped.append(sig)
            except (ValueError, OSError):
                pass
        self._write_atomic(new_text)
        # post-write structural self-verify before declaring surgery active
        reread = _yaml.safe_load(LIVE_CFG.read_text())
        reread_prov = (reread or {}).get("provider") or {}
        ok_shape = (reread_prov.get("base_url") == chaos_primary
                    and isinstance(reread_prov.get("fallbacks"), list)
                    and reread_prov["fallbacks"]
                    and reread_prov["fallbacks"][0].get("base_url") == self.original_url)
        if not ok_shape:
            self._restore_now()
            fb = reread_prov.get("fallbacks")
            fb_shape = (f"list[{len(fb)}] first-keys="
                        f"{sorted(fb[0].keys()) if isinstance(fb[0], dict) else type(fb[0]).__name__}"
                        if isinstance(fb, list) and fb else type(fb).__name__)
            # NEVER include fallback values here — they contain the live
            # api_key and this message lands in crash output (review r2 P1)
            raise RuntimeError("post-write self-verify failed: fallbacks not under provider — "
                               f"restored original (shape={fb_shape}, "
                               f"provider_keys={sorted(reread_prov.keys())})")
        _log(f"config surgery ACTIVE: primary->{chaos_primary}, fallback->direct, "
             f"backup={self.backup.name} — config is FROZEN for the duration; "
             f"member edits inside the window are reverted on restore")
        return self

    def exit(self) -> dict:
        self._restore_now()
        report = {"restored": self._original is not None and not self._armed and self.backup is None}
        if report["restored"]:
            reread = LIVE_CFG.read_text()
            report["diff_clean"] = reread == (self._original or "")
            report["live_health"] = http_status("http://127.0.0.1:8899/api/health") == 200
            report["backup_deleted"] = not self.backup
            _log(f"surgery exit verify: diff_clean={report['diff_clean']} "
                 f"live_health={report['live_health']} backup_deleted={report['backup_deleted']}")
        return report


def sys_exit(code: int) -> None:
    raise SystemExit(code)
