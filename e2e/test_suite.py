"""cw-e2e offline regression tests — one test per fixed defect.

Every test here maps to a finding (Codex PR-65 r1 or suite self-review) and
runs hermetically: no LLM spend, no live 8899 dependency, no real config
writes. Run explicitly (repo pytest testpaths only covers tests/):

    .venv/bin/python -m pytest e2e/test_suite.py -q
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

E2E_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(E2E_ROOT))

import analyze  # noqa: E402
import runner  # noqa: E402
import surgery  # noqa: E402


# ------------------------------------------------- suite.yaml invariants ---

def test_cost_cap_lookup_never_keyerrors():
    """Codex r1 P1: tier['cost_cap'] was an eagerly-evaluated default and
    deep-fallback defines no tier cap — every chaos scenario crashed."""
    for tier in runner.SUITE["tiers"].values():
        for sc in tier["scenarios"]:
            assert runner.cost_cap_of(sc, tier) > 0
    # the yaml-level mitigation is pinned too — a tier without a cap must
    # not come back via suite.yaml edits
    assert "cost_cap" in runner.SUITE["tiers"]["deep-fallback"]


def test_chaos_scenarios_declare_recovery_expectations():
    """Codex r1 P1: a provider failing immediately (or bypassing the proxy)
    could still PASS the chaos scenarios."""
    chaos = [(sc, tier) for tier in runner.SUITE["tiers"].values()
             for sc in tier["scenarios"] if sc["type"] == "chaos"]
    assert chaos, "chaos matrix disappeared from suite.yaml"
    for sc, _ in chaos:
        exp = sc.get("expect") or {}
        if sc.get("base_url_pathless"):
            continue                            # deliberate 404 probe — no recovery
        assert exp.get("complete_status") == "ok", f"{sc['name']}: recovery not required"
        assert "chaos_hits" in exp, f"{sc['name']}: injected-fault count not gated"


def test_deep_cold_measures_before_warming():
    """Codex r1 P1: smoke-first warmed the isolated home, so the 'cold'
    coefficient was measured on a warm environment."""
    scenarios = runner.SUITE["tiers"]["deep-cold"]["scenarios"]
    assert scenarios[0]["name"] == "cold-fulldepth"


def test_chaos_target_whitelist_enforced(monkeypatch, tmp_path):
    """Codex r1 P1: the chaos path derived a target without check_target —
    an out-of-whitelist target could reach ws_run."""
    import chaos
    monkeypatch.setattr(chaos, "resolve_direct_endpoint",
                        lambda u=None: {"host": "h", "port": 1, "path": "", "scheme": "https"})
    monkeypatch.setattr(chaos, "ProxyHandle", lambda *a: types.SimpleNamespace(
        refusals=lambda: 0, stop=lambda: None))
    sc = {"name": "evil", "mode": "refuse", "n": 1, "prompt": "quick-scan.txt",
          "target": "8.8.8.8"}
    keyfile = tmp_path / "k"
    keyfile.write_text("x")
    with pytest.raises(SystemExit):
        runner.run_chaos_scenario(sc, tmp_path, {}, "ws://127.0.0.1:1", keyfile, None)


# ----------------------------------------------------- subprocess gates ----

def _fake_run(monkeypatch, *, returncode, stdout):
    def fake(cmd, **kwargs):
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(runner.subprocess, "run", fake)


def _scenario_dir(tmp_path):
    d = tmp_path / "scenarios"
    d.mkdir(exist_ok=True)
    return tmp_path


def test_pytest_gate_fails_on_nonzero_exit_without_failed_lines(monkeypatch, tmp_path):
    """Codex r1 P1: collection/internal errors exit nonzero with no FAILED
    lines — stdout-only matching marked them PASS."""
    _fake_run(monkeypatch, returncode=2,
              stdout="ImportError while importing test module\nERROR tests/x.py")
    assert runner.scenario_pytest(_scenario_dir(tmp_path))["pass"] is False


def test_pytest_gate_env_allowlist_still_passes(monkeypatch, tmp_path):
    allow = runner.SUITE["pytest"]["env_fail_allowlist"][0]
    _fake_run(monkeypatch, returncode=1, stdout=f"FAILED {allow}::test_x - OSError\n1 failed")
    assert runner.scenario_pytest(_scenario_dir(tmp_path))["pass"] is True


def test_report_cli_rejects_no_report_message(monkeypatch, tmp_path):
    """Codex r1 P1: `report -s` prints a nonempty no-report message and
    exits 0 — the old check called that PASS."""
    _fake_run(monkeypatch, returncode=0,
              stdout="No report found for session deadbeef\nLooked under ~/.clearwing/results\n")
    assert runner.scenario_report_cli(_scenario_dir(tmp_path), "deadbeef")["pass"] is False


def test_report_cli_accepts_real_report(monkeypatch, tmp_path):
    _fake_run(monkeypatch, returncode=0,
              stdout="Session report: /x/report.md\n\n# Clearwing Session Report\n## 一\n")
    assert runner.scenario_report_cli(_scenario_dir(tmp_path), "cafe0001")["pass"] is True


# --------------------------------------------------------- WS protocol ----

class _FakeWS:
    """Inter-frame gap 0.2s — large enough that an immediate-send revert
    (approve on approval_needed instead of the window) is provably EARLIER
    than the awaiting_approval delivery and the zero-slack assertion
    catches it (review r2: 0.01s gap + 0.05s slack was vacuous)."""

    GAP = 0.2

    def __init__(self, script):
        self.script = list(script)
        self.sent: list[tuple[float, dict]] = []
        self.delivered: dict[str, float] = {}

    async def recv(self):
        if self.script:
            item = self.script.pop(0)
            m = json.loads(item)
            if m.get("type") == "approval_needed":
                self.delivered["approval_needed"] = time.time()
            if m.get("type") == "complete" and (m.get("data") or {}).get("status") == "awaiting_approval":
                self.delivered["awaiting_approval"] = time.time()
            await asyncio.sleep(self.GAP)
            return item
        await asyncio.sleep(60)             # wait_for turns this into a timeout
        raise AssertionError("unreachable")

    async def send(self, s):
        self.sent.append((time.time(), json.loads(s)))

    async def close(self):
        pass


class _FakeConnect:
    def __init__(self, ws):
        self.ws = ws

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *exc):
        return False


def _drive(monkeypatch, tmp_path, script, *, cap=1.0, prompt="hi"):
    ws = _FakeWS(script)
    fake_mod = types.SimpleNamespace(connect=lambda *a, **k: _FakeConnect(ws))
    monkeypatch.setitem(sys.modules, "websockets", fake_mod)
    monkeypatch.setitem(runner.SUITE["thresholds"], "late_drain_s", 1)
    keyfile = tmp_path / f"k{time.time_ns()}"
    keyfile.write_text("x")
    out = tmp_path / f"scen{time.time_ns()}"
    summary = asyncio.run(runner.ws_run(
        "ws://x", keyfile, "192.168.73.82", prompt, out, 60, cap, home=tmp_path))
    return ws, summary, out


def test_approval_sent_only_after_awaiting_window(monkeypatch, tmp_path):
    """Codex r1 P1: approve-on-approval_needed raced the server's turn
    teardown (busy-reject loses the decision). The driver must queue and
    flush on complete(awaiting_approval) — and keep draining after the
    terminal frame so late frames become visible."""
    ws, summary, out = _drive(monkeypatch, tmp_path, [
        json.dumps({"type": "started", "session_id": "testsid0001", "model": "glm-5.3"}),
        json.dumps({"type": "approval_needed", "data": {"prompt": "run nmap -sV target"}}),
        json.dumps({"type": "complete", "data": {"status": "awaiting_approval"}}),
        json.dumps({"type": "tool_result", "data": {"tool": "kali_execute"}}),
        json.dumps({"type": "complete", "data": {"status": "ok"}}),
        json.dumps({"type": "agent_message", "data": {"content": "late"}}),   # post-terminal
    ])
    approves = [(t, m) for t, m in ws.sent if m.get("type") == "approve"]
    assert len(approves) == 1 and approves[0][1]["approved"] is True
    # ordering with ZERO slack: the approve left only AFTER the
    # awaiting_approval window was delivered (0.2s inter-frame gap)
    assert approves[0][0] >= ws.delivered["awaiting_approval"]
    assert summary["complete_statuses"] == ["awaiting_approval", "ok"]
    assert summary["pending_approvals_at_end"] == 0
    # post-terminal frame was read and classified late (drain window)
    fm = analyze.frames_metrics(out.parent / (out.name + ".frames.jsonl"))
    assert fm["late_frames"] == 1
    assert summary["invariants"]["terminal_closure"] is True


def test_approval_never_sent_when_window_never_opens(monkeypatch, tmp_path):
    """Negative: approval_needed with no complete(awaiting_approval) must
    produce ZERO approve sends — the decision stays queued (and the
    approval-pending-end gate catches it downstream)."""
    ws, summary, _ = _drive(monkeypatch, tmp_path, [
        json.dumps({"type": "started", "session_id": "testsid0002", "model": "glm-5.3"}),
        json.dumps({"type": "approval_needed", "data": {"prompt": "run nmap -sV target"}}),
        json.dumps({"type": "complete", "data": {"status": "ok"}}),   # window never opened
    ])
    assert not [m for _, m in ws.sent if m.get("type") == "approve"]
    assert summary["pending_approvals_at_end"] == 1


def test_busy_reject_self_heals_without_counting_as_error(monkeypatch, tmp_path):
    """Review r2 P1: the flushed approve lands in the sub-ms gap before
    turn_task completion — server answers the busy error frame. The retry
    must resend, and the SELF-HEALED reject must not land in error_count
    (t1-fulldepth gates on errors == 0)."""
    ws, summary, _ = _drive(monkeypatch, tmp_path, [
        json.dumps({"type": "started", "session_id": "testsid0003", "model": "glm-5.3"}),
        json.dumps({"type": "approval_needed", "data": {"prompt": "run nmap -sV target"}}),
        json.dumps({"type": "complete", "data": {"status": "awaiting_approval"}}),
        json.dumps({"type": "error", "data": {"message":
                    'A turn is already running — send {"type": "stop"} to cancel it first'}}),
        json.dumps({"type": "error", "data": {"message":
                    'A turn is already running — send {"type": "stop"} to cancel it first'}}),
        json.dumps({"type": "tool_result", "data": {"tool": "kali_execute"}}),
        json.dumps({"type": "complete", "data": {"status": "ok"}}),
    ])
    approves = [m for _, m in ws.sent if m.get("type") == "approve"]
    assert len(approves) == 2                      # flush + ONE retry (budget spent)
    assert summary["busy_rejects"] == 1            # only the self-healed one counted
    # the SECOND busy-reject exceeded the retry budget -> visible error, not silence
    assert summary["error_count"] == 1


def test_drain_window_is_observe_only(monkeypatch, tmp_path):
    """Review r2 P2: a late cost_update crossing the cap after the terminal
    frame must NOT provoke a stop — drain must not perturb the SUT or flip
    complete_statuses[-1]."""
    ws, summary, _ = _drive(monkeypatch, tmp_path, [
        json.dumps({"type": "started", "session_id": "testsid0004", "model": "glm-5.3"}),
        json.dumps({"type": "complete", "data": {"status": "ok"}}),
        json.dumps({"type": "cost_update", "data": {"total_cost_usd": 5.0,
                                                    "input_tokens": 100, "output_tokens": 10}}),
    ], cap=0.01)
    assert not [m for _, m in ws.sent if m.get("type") == "stop"]
    assert summary["watchdog"] is False
    assert summary["complete_statuses"] == ["ok"]


def test_audit_missing_is_a_hard_failure():
    """Codex r1 P1: metered cost_updates with no audit file used to skip the
    reconcile/completeness gates entirely — total audit loss still PASSed."""
    summary = {"status": None, "invariants": {"terminal_closure": True,
                                              "approval_closure": True,
                                              "no_complete_while_approval_open": True},
               "approvals": 0, "cost_updates": 3, "cost_usd_product": 0.5,
               "tokens_in": 1000, "tokens_cached": 900, "error_count": 0,
               "complete_statuses": ["ok"]}
    fm = {"statuses": ["ok"], "dup_pairs": 0, "late_frames": 0}
    gates = analyze.evaluate(summary, fm, "full", audit=None, hud=None)
    g = next(g for g in gates if g["gate"] == "audit-present")
    assert g["pass"] is False and g["severity"] == "hard"


def test_expect_chaos_hits_and_graceful_consumed():
    summary = {"status": None, "invariants": {"terminal_closure": True,
                                              "approval_closure": True,
                                              "no_complete_while_approval_open": True},
               "approvals": 0, "error_count": 0, "chaos": {"refusals": 2},
               "complete_statuses": ["ok"], "cost_updates": 0}
    fm = {"statuses": ["ok"], "dup_pairs": 0, "late_frames": 0}
    gates = analyze.evaluate(summary, fm, "full", audit=None, hud=None,
                             expect={"chaos_hits": ">=2", "graceful": "complete_or_stopped"})
    by_gate = {g["gate"]: g for g in gates}
    assert by_gate["expect-chaos-hits"]["pass"] is True
    assert by_gate["expect-graceful"]["pass"] is True
    # negative: proxy bypassed -> hits absent -> gate fails hard
    summary2 = dict(summary, chaos={})
    gates2 = analyze.evaluate(summary2, fm, "full", audit=None, hud=None,
                              expect={"chaos_hits": ">=1"})
    assert next(g for g in gates2 if g["gate"] == "expect-chaos-hits")["pass"] is False
    # negative: graceful must reject terminal error and empty statuses
    summary3 = dict(summary, complete_statuses=["error"])
    gates3 = analyze.evaluate(summary3, fm, "full", audit=None, hud=None,
                              expect={"graceful": "complete_or_stopped"})
    assert next(g for g in gates3 if g["gate"] == "expect-graceful")["pass"] is False
    gates4 = analyze.evaluate(dict(summary, complete_statuses=[]), fm, "full",
                              audit=None, hud=None, expect={"graceful": "complete_or_stopped"})
    assert next(g for g in gates4 if g["gate"] == "expect-graceful")["pass"] is False
    # approval-pending-end: a stranded queued decision is a hard failure
    summary5 = dict(summary, pending_approvals_at_end=1)
    gates5 = analyze.evaluate(summary5, fm, "full", audit=None, hud=None)
    assert next(g for g in gates5 if g["gate"] == "approval-pending-end")["pass"] is False


# ------------------------------------------------------------- verdicts ----

def _write_min_run(tmp_path, name, expect=None, extra_summary=None):
    d = tmp_path / name
    d.mkdir()
    (d / "scenarios").mkdir()
    summary = {
        "invariants": {"terminal_closure": True, "approval_closure": True,
                       "no_complete_while_approval_open": True},
        "approvals": 0, "cost_updates": 0, "complete_statuses": ["ok"],
        "session_id": None, "seconds": 100, "cost_usd_product": 1.0,
        "tokens_in": 10, "tokens_out": 10, "tokens_cached": 0}
    if extra_summary:
        summary.update(extra_summary)
    manifest = [{"name": "t", "summary": summary, "expect": expect}]
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def test_render_wires_expect_into_gates(monkeypatch, tmp_path):
    """Review r2: dropping expect=sc.get('expect') from the render->evaluate
    call reverted the wiring with no test failing — pin it."""
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    d = _write_min_run(tmp_path, "20260103-000000-full",
                       expect={"chaos_hits": ">=1"}, extra_summary={"chaos": {"refusals": 1}})
    out = analyze.render(d, "full", {"git": {"head": "x" * 40, "webapi_commit": "y", "branch": "b"}},
                         [("kali-containers", True, "")])
    assert any(g["gate"] == "expect-chaos-hits" and g["pass"] for g in out["gates"])


def test_cleanup_failure_flips_verdict_to_fail(monkeypatch, tmp_path):
    """Codex r1 P1: cleanup assertions rendered as WARN while the verdict
    stayed PASS."""
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    d = _write_min_run(tmp_path, "20260101-000000-full")
    out = analyze.render(d, "full", {"git": {"head": "x" * 40, "webapi_commit": "y", "branch": "b"}},
                         [("kali-containers", False, "ours-remaining=['clearwing-kali-1']")])
    assert out["overall"] == "FAIL"
    assert any(g["gate"] == "cleanup-kali-containers" and not g["pass"]
               for g in out["gates"])


def test_trend_breach_yields_regression_not_pass(monkeypatch, tmp_path):
    """Codex r1 P2: failed trend gates never reached the verdict — the
    SPEC's REGRESSION outcome did not exist in code."""
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    monkeypatch.setattr(analyze, "previous_full",
                        lambda run_dir: {"seconds": 100, "cost_usd_product": 1.0})
    d = _write_min_run(tmp_path, "20260102-000000-full")
    m = json.loads((d / "manifest.json").read_text())
    m[0]["summary"]["seconds"] = 300               # +200% >> ±30% band
    (d / "manifest.json").write_text(json.dumps(m))
    out = analyze.render(d, "full", {"git": {"head": "x" * 40, "webapi_commit": "y", "branch": "b"}},
                         [("kali-containers", True, "")])
    assert out["overall"] == "REGRESSION"
    assert any(g["gate"].startswith("trend-") and not g["pass"] for g in out["gates"])


# ------------------------------------------------------------- surgery ----

def test_surgical_config_puts_fallbacks_under_provider():
    """Codex r1 P2: text-append glued fallbacks onto the LAST top-level
    block. With a mapping after provider, the fallback must still land
    structurally under provider."""
    original = (
        "provider:\n"
        "  base_url: https://api.example.com/api/coding/paas/v4\n"
        "  model: glm-5.3\n"
        "  api_key: sk-test-1234567890\n"
        "scanning:\n"
        "  intensity: deep\n"
        "  exclude:\n"
        "    - 10.0.0.0/8\n"
    )
    new_text, chaos_primary = surgery.build_surgical_config(original, "http://127.0.0.1:8787")
    cfg = yaml.safe_load(new_text)
    assert chaos_primary == "http://127.0.0.1:8787/api/coding/paas/v4"
    assert cfg["provider"]["base_url"] == chaos_primary
    fb = cfg["provider"]["fallbacks"]
    assert isinstance(fb, list) and len(fb) == 1
    assert fb[0]["base_url"] == "https://api.example.com/api/coding/paas/v4"
    assert fb[0]["api_key"] == "sk-test-1234567890"
    # the trailing mapping survives intact, NOT hosting the fallback
    assert cfg["scanning"] == {"intensity": "deep", "exclude": ["10.0.0.0/8"]}
    assert "fallbacks" not in cfg["scanning"]


def test_surgical_config_refuses_local_proxy_and_missing_fields():
    with pytest.raises(RuntimeError):
        surgery.build_surgical_config(
            "provider:\n  base_url: http://127.0.0.1:8787/x\n  model: m\n  api_key: k\n",
            "http://127.0.0.1:8787")
    with pytest.raises(RuntimeError):          # loopback spelled differently
        surgery.build_surgical_config(
            "provider:\n  base_url: http://localhost:8787/x\n  model: m\n  api_key: k\n",
            "http://127.0.0.1:8787")
    with pytest.raises(RuntimeError):
        surgery.build_surgical_config("provider:\n  model: m\n", "http://127.0.0.1:8787")


def test_config_surgery_enter_exit_roundtrip(monkeypatch, tmp_path):
    """Review r2: the enter()/exit() integration (backup, atomic write,
    post-write self-verify, byte-identical restore) was untested — only the
    pure helper was. Hermetic: LIVE_CFG + Path.home redirected to tmp."""
    home = tmp_path / "home"
    cfgdir = home / ".clearwing"
    cfgdir.mkdir(parents=True)
    live = cfgdir / "config.yaml"
    original = ("provider:\n"
                "  base_url: https://api.example.com/v4\n"
                "  model: glm-5.3\n"
                "  api_key: sk-live-test-000000000000\n"
                "scanning:\n"
                "  intensity: deep\n")
    live.write_text(original)
    monkeypatch.setattr(surgery, "LIVE_CFG", live)
    monkeypatch.setattr(surgery.Path, "home", lambda: home)
    monkeypatch.setattr(surgery, "_sessions_running_8899", lambda: False)
    surg = surgery.ConfigSurgery(proxy_base="http://127.0.0.1:8787")
    try:
        surg.enter()
        assert surg.original_url == "https://api.example.com/v4"
        mid = yaml.safe_load(live.read_text())
        assert mid["provider"]["base_url"] == "http://127.0.0.1:8787/v4"
        assert mid["provider"]["fallbacks"][0]["base_url"] == "https://api.example.com/v4"
        assert mid["scanning"] == {"intensity": "deep"}
        baks = list(cfgdir.glob("config.yaml.bak-*"))
        assert len(baks) == 1 and (baks[0].stat().st_mode & 0o777) == 0o600
        assert baks[0].read_text() == original
    finally:
        report = surg.exit()
    assert report["restored"] and report["diff_clean"] and report["backup_deleted"]
    assert live.read_text() == original
    assert not list(cfgdir.glob("config.yaml.bak-*"))
    assert not (cfgdir / "config.yaml.cwe2e-tmp").exists()


def test_sessions_running_detection(monkeypatch):
    """Codex r1 P1: the old heuristic required the string 'true' alongside
    '"running"' — /api/sessions objects never carry a boolean, so the guard
    never fired. Review r2: non-200 (incl. 503 store-unavailable) and
    `paused` sessions must also block surgery (fail-closed)."""
    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _with(body, code=200):
        monkeypatch.setattr(surgery, "http_status", lambda url, timeout=3.0: code)
        monkeypatch.setattr(surgery.urllib.request, "urlopen",
                            lambda *a, **k: _Resp(body))

    _with(json.dumps([{"session_id": "s1", "status": "running"}]).encode())
    assert surgery._sessions_running_8899() is True
    _with(json.dumps([{"session_id": "s1", "status": "paused"}]).encode())
    assert surgery._sessions_running_8899() is True
    _with(json.dumps([{"session_id": "s1", "status": "completed"}]).encode())
    assert surgery._sessions_running_8899() is False
    _with(json.dumps([]).encode())
    assert surgery._sessions_running_8899() is False
    _with(b"<html>gateway error</html>")
    with pytest.raises(RuntimeError):
        surgery._sessions_running_8899()
    _with(b"[]", code=503)                     # store unavailable -> refuse
    with pytest.raises(RuntimeError):
        surgery._sessions_running_8899()
    _with(b"[]", code=-1)                      # unreachable -> refuse
    with pytest.raises(RuntimeError):
        surgery._sessions_running_8899()


# ------------------------------------------------- pure gate helpers ------

def test_evaluate_key_gate_branches():
    """Codex r1 P2 + review r2: the authoritative comparison is against the
    8899 process environ (a stale webui-8899.key mirror must not brick the
    run); the file mirror is only the fallback."""
    kf = b"webui-key-aaaaaaaaaaaaaaaaaaaa\n"
    # environ authoritative — match (newline-stripped both sides)
    name, ok, _ = runner.evaluate_key_gate(kf, "webui-key-aaaaaaaaaaaaaaaaaaaa", b"other\n")
    assert name == "key-matches-live-process" and ok is True
    # environ mismatch -> HARD fail even when the mirror matches
    name, ok, _ = runner.evaluate_key_gate(kf, "different-key-0000000000000", kf)
    assert name == "key-matches-live-process" and ok is False
    # environ unreadable -> mirror fallback
    name, ok, _ = runner.evaluate_key_gate(kf, None, kf)
    assert name == "key-files-match" and ok is True
    name, ok, _ = runner.evaluate_key_gate(kf, None, b"stale-mirror-00000000000\n")
    assert name == "key-files-match" and ok is False
    # nothing to compare against -> info only
    name, ok, _ = runner.evaluate_key_gate(kf, None, None)
    assert name == "key-files-info" and ok is True
    name, ok, _ = runner.evaluate_key_gate(None, "x", None)
    assert name == "key-files-info" and ok is True


def test_proc_freshness_branches():
    assert runner.proc_freshness(None, 100) is False       # unreadable -> fail closed
    assert runner.proc_freshness(100, None) is False
    assert runner.proc_freshness(50.0, 100) is False       # process predates HEAD
    assert runner.proc_freshness(99.0, 100) is True        # within 2s skew tolerance
    assert runner.proc_freshness(97.9, 100) is False       # beyond tolerance
    assert runner.proc_freshness(200.0, 100) is True


# ------------------------------------------- adjudication / export gate ---

def _adj_run_dir(tmp_path, name, verdict="FAIL", full=True):
    d = _write_min_run(tmp_path, name)
    if full:
        (d / "gates.json").write_text(json.dumps([
            {"gate": "cache-min", "pass": False, "severity": "hard", "detail": "cache=54.2%"}]))
        (d / "ledger.json").write_text(json.dumps({"cost_usd_product": 0.7, "seconds": 574}))
        (d / "report.md").write_text(f"# cw-e2e full — {name} — {verdict}\n")
    return d


class _Args:
    pass


def test_adjudicate_and_export_gate(tmp_path, capsys):
    """SPEC §9 hard rule made mechanical: export refuses anything that is
    not a FINAL (reviewed) adjudication — the 2026-09-18 #61 sequencing
    violation cannot recur."""
    fatal = _adj_run_dir(tmp_path, "20260101-175100-quick", full=False)  # FATAL 廢輪：無 report/gates
    (fatal / "cleanup.json").write_text("[]")                             # 僅有殘骸
    d = _adj_run_dir(tmp_path, "20260101-180000-full")
    a = _Args()
    a.run_dirs = [str(fatal), str(d)]
    a.finalize = False
    runner.cmd_adjudicate(a)
    adj = d / "adjudication.md"                     # lands in the LAST dir
    text = adj.read_text()
    assert "DRAFT -->" in text and fatal.name in text and "cache-min" in text
    # export blocked on DRAFT
    e = _Args()
    e.run_dir = str(d)
    e.out = None
    with pytest.raises(SystemExit):
        runner.cmd_export(e)
    # finalize refused while the review record is empty
    a2 = _Args()
    a2.run_dirs = [str(d)]
    a2.finalize = True
    with pytest.raises(SystemExit):
        runner.cmd_adjudicate(a2)
    assert "DRAFT -->" in adj.read_text()
    # review record filled but verdict line still empty -> finalize refuses
    adj.write_text(text.replace(
        "<!-- paste the four-lens review summary here; 觸發: 判定翻案/對外開單前/發版判定 -->",
        "四鏡複審完成：證據核實重算全數通過；對抗方法論確認無替代解釋；流程對照合規；交付一致。"))
    with pytest.raises(SystemExit):
        runner.cmd_adjudicate(a2)
    # verdict line present -> finalize stamps FINAL -> export quotes IT
    adj.write_text(adj.read_text().replace("verdict: \n", "verdict: PASS  # 翻案：門檻口徑\n", 1)
                   if "verdict: \n" in adj.read_text()
                   else adj.read_text().replace("verdict: ", "verdict: PASS  # 翻案\n_", 1))
    runner.cmd_adjudicate(a2)
    assert "adjudication-status: FINAL" in adj.read_text()
    runner.cmd_export(e)                            # no SystemExit anymore
    out = capsys.readouterr().out
    assert "**PASS**" in out and "machine verdict was FAIL" in out \
        and "cache-min" in out and str(d) in out


# --------------------------------------- cache per-call / degenerate -----

def test_cache_prefix_median_function():
    med = analyze.cache_prefix_median(
        [(10000, 9500, 50), (10000, 9500, 50), (12000, 11400, 60), (24000, 120, 50)])
    assert med == 95.0                       # growth call (24000) excluded, first excluded
    assert analyze.cache_prefix_median([(10000, 9500, 50)]) is None   # needs >=2 ratios
    assert analyze.cache_prefix_median([]) is None
    # auxiliary contexts (<5k tokens: summarizer/operator) never count
    med_aux = analyze.cache_prefix_median(
        [(10000, 9500, 50), (452, 0, 615), (10100, 9600, 50), (10200, 9700, 50)])
    assert med_aux == pytest.approx(95.07, abs=0.01)   # the 452-token call never drags
    low = analyze.cache_prefix_median(
        [(10000, 5000, 50), (10000, 5050, 50), (12000, 6000, 60)])
    assert low == 50.25                      # statistics.median([50.0, 50.5])


def _eval_with_audit(per_call, **summary_over):
    summary = {"status": None, "invariants": {"terminal_closure": True,
                                              "approval_closure": True,
                                              "no_complete_while_approval_open": True},
               "approvals": 0, "error_count": 0, "cost_updates": len(per_call),
               "complete_statuses": ["ok"], "tokens_in": sum(c[0] for c in per_call),
               "tokens_out": sum(c[2] for c in per_call),
               "tokens_cached": sum(c[1] for c in per_call), "cost_usd_product": 0.5}
    summary.update(summary_over)
    audit = {"llm_calls": len(per_call), "tokens_in": summary["tokens_in"],
             "tokens_out": summary["tokens_out"], "tokens_cached": summary["tokens_cached"],
             "real_cost_cache_aware": 0.5, "calls": per_call}
    fm = {"statuses": ["ok"], "dup_pairs": 0, "late_frames": 0}
    return analyze.evaluate(summary, fm, "full", audit=audit, hud=None)


def test_cache_prefix_median_gate_replaces_aggregate():
    """2026-09-18 round: the aggregate 54.2% FAILED a healthy run — the
    per-call caliber must PASS it while still catching true degradation."""
    healthy = [(24000, 22300, 400), (25000, 23400, 300), (25500, 24000, 300),
               (29800, 24500, 1400), (30000, 28400, 100)]
    by = {g["gate"]: g for g in _eval_with_audit(healthy)}
    assert by["cache-prefix-median"]["pass"] is True
    degraded = [(24000, 12000, 400), (25000, 12600, 300), (25500, 12800, 300)]
    by = {g["gate"]: g for g in _eval_with_audit(degraded)}
    assert by["cache-prefix-median"]["pass"] is False


def test_degenerate_output_detector():
    """t3-warm-chain-2 (2026-09-18): single 23.7k-in call, 0% cache, 3-token
    reply — must be a hard FAIL, not a silent pass."""
    by = {g["gate"]: g for g in _eval_with_audit([(23709, 0, 3)])}
    assert by["degenerate-output"]["pass"] is False
    # a >10k-in, 0-cache call answering in <10 tokens fails even amid a
    # healthy-looking session
    by = {g["gate"]: g for g in _eval_with_audit(
        [(15000, 14000, 300), (15500, 0, 6)])}
    assert by["degenerate-output"]["pass"] is False
    by = {g["gate"]: g for g in _eval_with_audit(
        [(20000, 19000, 300), (21000, 20000, 350)])}
    assert by["degenerate-output"]["pass"] is True
    # warm-chain-1/-3 shape: terse 3-token reply ON FULL CACHE = healthy
    by = {g["gate"]: g for g in _eval_with_audit([(23709, 22336, 3)])}
    assert by["degenerate-output"]["pass"] is True


def test_flag_max_range_gate():
    summary = {"status": None, "invariants": {"terminal_closure": True,
                                              "approval_closure": True,
                                              "no_complete_while_approval_open": True},
               "approvals": 0, "error_count": 0, "cost_updates": 0,
               "complete_statuses": ["ok"], "flag_faces": 14}
    fm = {"statuses": ["ok"], "dup_pairs": 0, "late_frames": 0}
    gates = analyze.evaluate(summary, fm, "full", audit=None, hud=None, flag_max=15)
    assert next(g for g in gates if g["gate"] == "flag-faces-max")["pass"] is True
    gates = analyze.evaluate(dict(summary, flag_faces=16), fm, "full",
                             audit=None, hud=None, flag_max=15)
    assert next(g for g in gates if g["gate"] == "flag-faces-max")["pass"] is False


def test_partial_run_trend_suppressed(monkeypatch, tmp_path):
    """G7: --only subset vs whole-tier baseline is apples-to-oranges — the
    2026-09-18 n=2 run took 2 spurious trend FAILs from it."""
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    monkeypatch.setattr(analyze, "previous_full",
                        lambda run_dir: {"seconds": 1282, "cost_usd_product": 1.018})
    d = _write_min_run(tmp_path, "20260101-000000-full-partial")
    m = json.loads((d / "manifest.json").read_text())
    m[0]["summary"]["seconds"] = 415
    (d / "manifest.json").write_text(json.dumps(m))
    out = analyze.render(d, "full", {"git": {"head": "x" * 40, "webapi_commit": "y", "branch": "b"}},
                         [("kali-containers", True, "")])
    trend_gates = [g for g in out["gates"] if g["severity"] == "trend"]
    assert len(trend_gates) == 1 and "suppressed" in trend_gates[0]["detail"]
    assert not any(g["gate"].startswith("trend-") for g in out["gates"])


# ------------------------------------------------- P3 state / ledger -----

def test_check_llm_profile_preflight(monkeypatch, tmp_path):
    """P3.4: a missing/bad CW_LLM_PROFILE must die at VERIFY with guidance,
    not at scenario 1 mid-run (20260918-175138 FATAL lesson)."""
    monkeypatch.delenv("CW_LLM_PROFILE", raising=False)
    with pytest.raises(SystemExit):
        runner.check_llm_profile()
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: {valid: yaml")
    monkeypatch.setenv("CW_LLM_PROFILE", str(bad))
    with pytest.raises(SystemExit):
        runner.check_llm_profile()
    good = tmp_path / "llm.yaml"
    good.write_text("zai:\n  api_key: sk-test-00000000000000000000\n")
    monkeypatch.setenv("CW_LLM_PROFILE", str(good))
    runner.check_llm_profile()                     # no raise
    nokey = tmp_path / "nokey.yaml"
    nokey.write_text("zai: {}\n")
    monkeypatch.setenv("CW_LLM_PROFILE", str(nokey))
    with pytest.raises(SystemExit):
        runner.check_llm_profile()


def test_cleanup_target_compare_and_aside_diff(monkeypatch, tmp_path):
    """P3.1 post-snapshot diff + P3.2 target-state hard gate."""
    home = tmp_path / "home"
    (home / ".clearwing").mkdir(parents=True)
    run_dir = tmp_path / "run"
    (run_dir / "state" / "aside-pre").mkdir(parents=True)
    (run_dir / "scenarios").mkdir()
    (run_dir / "state" / "aside-pre" / "memory.db").write_bytes(b"OLD-MEMORY")
    (home / ".clearwing" / "memory.db").write_bytes(b"NEW-MEMORY-1234")   # mutated during round
    (run_dir / "state" / "pre-state.json").write_text(json.dumps(
        {"target_snapshot": {"192.168.73.82": {"88": False, "445": True}}}))
    monkeypatch.setattr(runner, "Path", __import__("pathlib").Path)  # ensure same class
    monkeypatch.setattr(runner.Path, "home", lambda: home)
    monkeypatch.setattr(runner, "RESULTS", tmp_path)   # N2: never touch the real budget ledger
    monkeypatch.setattr(runner, "listening_pids", lambda port: [])
    monkeypatch.setattr(runner, "tcp_probe",
                        lambda host, port, timeout=3.0: True if port == 445 else False)
    out = {r[0]: r for r in runner.cleanup_run(run_dir, [], before_8899=None, sids=[])}
    assert out["memory-aside-diff-recorded"][1] is True
    assert "memory.db:10->15B" in out["memory-aside-diff-recorded"][2] \
        or "memory.db" in out["memory-aside-diff-recorded"][2]
    assert out["target-state-unchanged"][1] is True    # 88 False->False, 445 True->True
    assert (run_dir / "state" / "aside-post" / "memory.db").read_bytes() == b"NEW-MEMORY-1234"
    # a port flipping = hard-gate failure
    monkeypatch.setattr(runner, "tcp_probe",
                        lambda host, port, timeout=3.0: False)
    out = {r[0]: r for r in runner.cleanup_run(run_dir, [], before_8899=None, sids=[])}
    assert out["target-state-unchanged"][1] is False


def test_budget_ledger_cumulative(monkeypatch, tmp_path):
    """P3.3: cumulative spend ledger with $700 cap awareness."""
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    monkeypatch.setattr(runner, "listening_pids", lambda port: [])
    home = tmp_path / "home"
    monkeypatch.setattr(runner.Path, "home", lambda: home)
    for i, cost in enumerate((0.5, 0.7)):
        d = tmp_path / f"2026010{i}-000000-quick"
        (d / "state").mkdir(parents=True)
        (d / "scenarios").mkdir()
        (d / "manifest.json").write_text(json.dumps(
            [{"name": "t", "summary": {"cost_usd_product": cost}}]))
        runner.cleanup_run(d, [], before_8899=None, sids=[])
    led = json.loads((tmp_path / "budget-ledger.json").read_text())
    assert led["total_usd"] == 1.2 and len(led["runs"]) == 2
    led["runs"].append({"ts": "x", "run": "synthetic", "cost_usd": 699.0})
    led["total_usd"] = 700.2
    (tmp_path / "budget-ledger.json").write_text(json.dumps(led))
    d = tmp_path / "20260102-000000-quick"
    (d / "state").mkdir(parents=True)
    (d / "scenarios").mkdir()
    (d / "manifest.json").write_text(json.dumps([{"name": "t", "summary": {"cost_usd_product": 0.1}}]))
    out = {r[0]: r for r in runner.cleanup_run(d, [], before_8899=None, sids=[])}
    assert out["budget-ledger-updated"][1] is False      # over cap -> hard gate


def test_adjudication_marker_exactness(tmp_path):
    """Review N1: a review body QUOTING the marker string must not satisfy
    the substring check (fail-open hole), and a mangled DRAFT marker must
    make finalize die instead of logging a false success."""
    d = _adj_run_dir(tmp_path, "20260101-000000-full")
    a = _Args()
    a.run_dirs = [str(d)]
    a.finalize = False
    runner.cmd_adjudicate(a)
    adj = d / "adjudication.md"
    # fill review record WITH a quoted FINAL marker inside the body
    adj.write_text(adj.read_text().replace(
        "<!-- paste the four-lens review summary here; 觸發: 判定翻案/對外開單前/發版判定 -->",
        "複審完成。文中引用標記 <!-- adjudication-status: FINAL --> 僅為引用，狀態仍 DRAFT。"
        "四鏡結論：證據核實通過、方法論無替代解釋、流程合規、交付一致。"))
    e = _Args()
    e.run_dir = str(d)
    e.out = None
    with pytest.raises(SystemExit):          # quoted marker must NOT unlock export
        runner.cmd_export(e)
    # mangled DRAFT marker -> finalize dies (no false success)
    adj.write_text(adj.read_text().replace(
        "<!-- adjudication-status: DRAFT -->", "<!-- adjudication-status: DRAFT- -->"))
    a2 = _Args()
    a2.run_dirs = [str(d)]
    a2.finalize = True
    with pytest.raises(SystemExit):
        runner.cmd_adjudicate(a2)


def test_adjudicate_refuses_overwrite_of_reviewed(tmp_path):
    """SPEC §2.7 refuse-overwrite: regenerating a draft must not destroy a
    human-filled review record or a FINAL document."""
    d = _adj_run_dir(tmp_path, "20260101-000000-full")
    a = _Args()
    a.run_dirs = [str(d)]
    a.finalize = False
    runner.cmd_adjudicate(a)
    adj = d / "adjudication.md"
    adj.write_text(adj.read_text().replace(
        "<!-- paste the four-lens review summary here; 觸發: 判定翻案/對外開單前/發版判定 -->",
        "四鏡複審完成：證據核實／對抗方法論／流程對照／交付一致性全數通過，無翻案。"))
    with pytest.raises(SystemExit):          # review record present -> refuse regen
        runner.cmd_adjudicate(a)
    adj.write_text(adj.read_text().replace(
        "<!-- adjudication-status: DRAFT -->", "<!-- adjudication-status: FINAL -->"))
    with pytest.raises(SystemExit):          # FINAL -> refuse regen
        runner.cmd_adjudicate(a)


# --------------------------------------- Codex #67 r1 fixes -------------

def test_aside_copy_includes_wal(tmp_path):
    """P1-A: SQLite WAL mode — copying only memory.db snapshots a stale
    database; the -wal sidecar must be captured too."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "memory.db").write_bytes(b"db")
    (home / "memory.db-wal").write_bytes(b"WAL-PENDING-WRITES")
    dest = tmp_path / "aside"
    dest.mkdir()
    copied = runner._aside_copy(home, dest)
    assert "memory.db-wal" in copied
    assert (dest / "memory.db-wal").read_bytes() == b"WAL-PENDING-WRITES"


def test_probe_majority(monkeypatch):
    """P2-c: a single lost SYN must never set the recorded baseline."""
    seq = iter([False, True, True])
    monkeypatch.setattr(runner, "tcp_probe", lambda h, p, timeout=3.0: next(seq))
    assert runner.probe_majority("h", 88) is True
    seq2 = iter([False, False, True])
    monkeypatch.setattr(runner, "tcp_probe", lambda h, p, timeout=3.0: next(seq2))
    assert runner.probe_majority("h", 88) is False


def test_cache_prefix_median_per_context():
    """P2-e: an interleaved uncached operator call is NOT a main-prefix
    miss and must not reset the main line's growth baseline."""
    calls = [(10000, 9500, 50, "main"), (6000, 0, 60, "operator"),
             (10100, 9600, 50, "main"), (10200, 9700, 50, "main")]
    med = analyze.cache_prefix_median(calls)
    assert med == pytest.approx(95.07, abs=0.01)   # main-only ratios; operator's
    # single call is its context's first -> excluded entirely


def test_r3_manual_not_clobbered_by_reanalyze(monkeypatch, tmp_path):
    """P2-b: a half-filled r3-manual.md is human state — `cw-e2e analyze`
    re-render must not destroy it."""
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    d = _write_min_run(tmp_path, "20260101-000000-full",
                       extra_summary={"session_id": "abc12345"})
    ver = {"git": {"head": "x" * 40, "webapi_commit": "y", "branch": "b"}}
    analyze.render(d, "full", ver, [("k", True, "")])
    (d / "r3-manual.md").write_text("# HUMAN IN-PROGRESS REVIEW\n- finding A\n")
    analyze.render(d, "full", ver, [("k", True, "")])     # re-render
    assert "HUMAN IN-PROGRESS REVIEW" in (d / "r3-manual.md").read_text()


# ---------------------------------------------------- P4 R3 manual ------

def test_r3_manual_template_and_pending_flag(monkeypatch, tmp_path):
    """P4.1: Full runs generate the fact-check template; r3-done silences
    regeneration and adjudicate flags the pending state."""
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    d = _write_min_run(tmp_path, "20260101-000000-full",
                       extra_summary={"session_id": "abc12345"})
    analyze.render(d, "full", {"git": {"head": "x" * 40, "webapi_commit": "y", "branch": "b"}},
                   [("kali-containers", True, "")])
    r3 = d / "r3-manual.md"
    assert r3.exists() and "abc12345" in r3.read_text() and "r3-done" in r3.read_text()
    a = _Args()
    a.run_dirs = [str(d)]
    a.finalize = False
    runner.cmd_adjudicate(a)
    assert "R3 未完成" in (d / "adjudication.md").read_text()
    (d / "r3-done").write_text("")
    analyze.render(d, "full", {"git": {"head": "x" * 40, "webapi_commit": "y", "branch": "b"}},
                   [("kali-containers", True, "")])   # marker present -> no template regen
    a = _Args()
    a.run_dirs = [str(d)]
    a.finalize = False
    runner.cmd_adjudicate(a)
    assert "R3 未完成" not in (d / "adjudication.md").read_text()


# ------------------------------------------------------- process/HEAD -----
def test_proc_start_and_product_head_epochs():
    start = runner._proc_start_epoch(__import__("os").getpid())
    assert start is not None and abs(start - time.time()) < 120
    head = runner._product_head_epoch()
    assert head is not None and head > 0
