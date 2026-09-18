#!/usr/bin/env python3
"""cw-e2e analyze — three-source analysis, gate evaluation, baseline diff,
cost ledger, and report rendering for a run directory.

Three sources (frozen protocol): driver frame counters + frames.jsonl full
text + on-disk artifacts (audit.jsonl, session reports). Never trust a single
counter. Invariants over field shapes (SPEC §5).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml

E2E_ROOT = Path(__file__).resolve().parent
RESULTS = E2E_ROOT / "results"
SUITE = yaml.safe_load((E2E_ROOT / "suite.yaml").read_text())
PRICING = SUITE["pricing_glm53"]


def _log(msg: str) -> None:
    print(f"[cw-e2e-analyze {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------ frames ------

def frames_metrics(frames_path: Path) -> dict:
    types: dict[str, int] = {}
    statuses: list[str] = []
    agent_msgs: list[str] = []
    late = 0
    last_terminal_idx = None
    idx = 0
    for line in frames_path.read_text().splitlines():
        if line.startswith("#"):
            continue
        try:
            m = json.loads(line)
        except Exception:
            continue
        idx += 1
        t = m.get("type", "")
        types[t] = types.get(t, 0) + 1
        if t == "complete":
            st = (m.get("data") or {}).get("status") or "ok"
            statuses.append(st)
            if st in ("ok", "stopped", "error"):
                last_terminal_idx = idx
        elif last_terminal_idx is not None and t not in ("complete", "stopped"):
            late += 1
        elif t == "agent_message":
            c = (m.get("data") or {}).get("content")
            agent_msgs.append(c if isinstance(c, str) else "")
    dup = sum(1 for a, b in zip(agent_msgs, agent_msgs[1:]) if a and a == b)
    return {"frame_types": types, "statuses": statuses, "dup_pairs": dup,
            "late_frames": late, "frames": idx}


# ------------------------------------------------------------- audit ------

def audit_metrics(sid: str, home: Path) -> dict | None:
    f = Path(home).expanduser() / "audit" / sid / "audit.jsonl"
    if not f.exists():
        return None
    calls = tin = tout = tcached = 0
    costs = []
    for line in f.read_text().splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event_type") == "llm_call":
            calls += 1
            d = e.get("details") or {}
            tin += int(d.get("input_tokens") or 0)
            tout += int(d.get("output_tokens") or 0)
            tcached += int(d.get("cached_tokens") or 0)
            costs.append(float(d.get("cost_usd") or 0))
    real = (tin - tcached) * PRICING["input"] / 1e6 + tcached * PRICING["cached_input"] / 1e6 \
        + tout * PRICING["output"] / 1e6
    return {"llm_calls": calls, "tokens_in": tin, "tokens_out": tout, "tokens_cached": tcached,
            "audit_cost_sum": round(sum(costs), 4), "real_cost_cache_aware": round(real, 4),
            "no_cache_upper_bound": round((tin * PRICING["input"] + tout * PRICING["output"]) / 1e6, 4)}


# ------------------------------------------------------------- gates ------

def evaluate(summary: dict, fm: dict, tier_name: str, audit: dict | None,
             hud: dict | None, expect: dict | None = None,
             flag_baseline: int | None = None) -> list[dict]:
    """Gate evaluation. severity policy (SPEC §4, aligned after review):
    reconcile / audit-completeness / cache gates are HARD (they were 'ratio'
    before — breaches could not flip the verdict, which contradicted SPEC).
    'skip' severity never fails the run (skipped scenarios)."""
    g = []

    def add(name, ok, detail, sev="hard"):
        g.append({"gate": name, "pass": bool(ok), "severity": sev, "detail": detail})

    if summary.get("status") == "skipped":
        add("skipped", True, str(summary.get("reason", "skipped"))[:80], sev="skip")
        return g
    if summary.get("status") == "crashed":
        add("scenario-crashed", False, str(summary.get("error", ""))[:120], sev="hard")
        return g

    inv = summary.get("invariants", {})
    add("terminal-closure", inv.get("terminal_closure", False),
        f"statuses={fm['statuses'][-3:] if fm['statuses'] else []}")
    add("approval-closure", inv.get("approval_closure", True),
        f"approvals={summary.get('approvals')}")
    add("complete-while-approval-open", inv.get("no_complete_while_approval_open", True),
        f"count={summary.get('complete_while_approval_open')} (D1/#29 detector: terminal "
        f"complete arriving while an approved tool has not executed)")
    add("approval-pending-end", (summary.get("pending_approvals_at_end") or 0) == 0,
        f"pending={summary.get('pending_approvals_at_end')} (queued decisions never flushed — "
        "stranded approval window)")
    add("dup-pairs", fm["dup_pairs"] <= SUITE["thresholds"]["hard"]["dup_pairs_max"],
        f"dup_pairs={fm['dup_pairs']}")
    add("late-frames", fm["late_frames"] <= SUITE["thresholds"]["hard"]["late_frames_max"],
        f"late_frames={fm['late_frames']}")

    cache_pct = 100 * summary.get("tokens_cached", 0) / max(summary.get("tokens_in", 1), 1)
    if tier_name == "full" and summary.get("tokens_in", 0) > 100000:
        add("cache-nonzero", cache_pct > 1, f"cache={cache_pct:.1f}%")
        add("cache-min", cache_pct >= SUITE["thresholds"]["ratio"]["cache_min_pct"],
            f"cache={cache_pct:.1f}%")
    if summary.get("cost_updates"):
        if audit is None:
            # Metered LLM spend with NO audit trail used to skip both the
            # reconcile and completeness gates entirely — a total loss of
            # audit instrumentation still produced PASS (Codex r1).
            add("audit-present", False,
                f"cost_updates={summary['cost_updates']} but audit.jsonl missing — "
                "reconciliation impossible (hard criterion of this suite)")
        else:
            meter = summary.get("cost_usd_product") or 0
            real = audit["real_cost_cache_aware"]
            diff_pct = abs(meter - real) / max(real, 1e-9) * 100
            add("reconcile", diff_pct <= SUITE["thresholds"]["ratio"]["reconcile_max_pct"],
                f"meter=${meter} vs audit-recompute=${real} ({diff_pct:.2f}%)")
            missing = summary["cost_updates"] - audit["llm_calls"]
            add("audit-completeness", missing <= 0,
                f"cost_updates={summary['cost_updates']} vs audit_calls={audit['llm_calls']}"
                + (f" — MISSING {missing} llm_call(s) from audit" if missing > 0 else ""))

    if flag_baseline is not None and tier_name == "full":
        faces = summary.get("flag_faces")
        if faces is not None:
            add("flag-faces-baseline", faces <= flag_baseline,
                f"faces={faces} baseline={flag_baseline} (#35-family false-flag drift)")
    if hud and hud.get("type") == "hud":
        add("hud-render-match", hud.get("pass", False), str(hud))

    # ---- suite `expect:` consumption (was DEAD-KEY — review P0-5) ----
    if expect:
        for k, v in expect.items():
            if k == "approvals":
                want = int(str(v).lstrip(">="))
                add("expect-approvals", (summary.get("approvals") or 0) >= want,
                    f"approvals={summary.get('approvals')} want>={want}")
            elif k == "cancelled_turn":
                add("expect-cancelled-turn",
                    bool((summary.get("stopped") or {}).get("cancelled_turn")) is True,
                    f"stopped={summary.get('stopped')}")
            elif k == "watchdog":
                add("expect-watchdog-fired", summary.get("watchdog") is True,
                    f"watchdog={summary.get('watchdog')} (driver cost-cap stop path)")
            elif k == "complete_status":
                want = v if isinstance(v, list) else [v]
                got = summary.get("complete_statuses") or []
                add("expect-complete-status", bool(got) and got[-1] in want,
                    f"last={got[-1] if got else None} want={want}")
            elif k == "errors":
                add("expect-errors", (summary.get("error_count") or 0) == int(v),
                    f"errors={summary.get('error_count')} want={v}")
            elif k == "chaos_hits":
                # Injected-fault count: the proxy must actually have been in
                # the path — a provider bypassing the proxy (or failing
                # before touching it) otherwise passes (Codex r1).
                want = int(str(v).lstrip(">="))
                hits = (summary.get("chaos") or {}).get("refusals")
                add("expect-chaos-hits", hits is not None and hits >= want,
                    f"proxy_faulted={hits} want>={want}")
            elif k == "graceful":
                # watchdog-lowcap: after the driver's cost-cap stop, the run
                # must end complete_or_stopped — never a hard drop. The YAML
                # value gates ENFORCEMENT (graceful: false -> no gate).
                if v:
                    got = summary.get("complete_statuses") or []
                    add("expect-graceful", bool(got) and got[-1] in ("ok", "stopped"),
                        f"last={got[-1] if got else None} want=ok|stopped")
    return g


# ------------------------------------------------------ baseline diff -----

def previous_full(run_dir: Path) -> dict | None:
    cands = sorted(d for d in RESULTS.glob("*-full") if d.is_dir() and d != run_dir)
    if not cands:
        return None
    m = cands[-1] / "ledger.json"
    return json.loads(m.read_text()) if m.exists() else None


def trend_gate(cur: dict, prev: dict | None) -> list[dict]:
    if not prev:
        return [{"gate": "baseline", "pass": True, "severity": "trend",
                 "detail": "first run — baseline established"}]
    out = []
    band = SUITE["thresholds"]["trend"]["duration_band_pct"]
    for key in ("seconds", "cost_usd_product"):
        p = prev.get(key)
        c = cur.get(key)
        b = SUITE["thresholds"]["trend"].get(f"{key.split('_')[0]}_band_pct", band)
        if p and c:
            ok = abs(c - p) / max(p, 1e-9) * 100 <= b
            out.append({"gate": f"trend-{key}", "pass": bool(ok), "severity": "trend",
                        "detail": f"prev={p} cur={c} (band ±{b}%)"})
    return out


# ------------------------------------------------------------ render ------

def render(run_dir: Path, tier_name: str, ver: dict, cleanup: list) -> dict:
    manifest_p = run_dir / "manifest.json"
    scenarios = json.loads(manifest_p.read_text()) if manifest_p.exists() else []
    home_default = str(Path.home() / ".clearwing")
    if tier_name == "re-analyze":
        tier_name = run_dir.name.rsplit("-", 1)[-1].replace("-partial", "")
    if not cleanup:
        cj = run_dir / "cleanup.json"
        if cj.exists():
            cleanup = [tuple(x) for x in json.loads(cj.read_text())]
    flag_baseline = None
    try:
        flag_baseline = SUITE["tiers"]["full"].get("analysis", {}).get("flag_baseline")
    except (KeyError, AttributeError):
        pass
    all_gates: list[dict] = []
    rows = []
    ledger = {"seconds": 0, "cost_usd_product": 0.0, "tokens_in": 0, "tokens_out": 0}
    for sc in scenarios:
        name, s = sc["name"], sc["summary"]
        sid = s.get("session_id")
        fm = frames_metrics(run_dir / "scenarios" / f"{name}.frames.jsonl") \
            if (run_dir / "scenarios" / f"{name}.frames.jsonl").exists() else {"statuses": [], "dup_pairs": 0, "late_frames": 0}
        audit = audit_metrics(sid, s.get("home") or home_default) if sid else None
        if s.get("invariants") is not None:      # real WS run — full gate set
            gates = evaluate(s, fm, tier_name, audit, None,
                             expect=sc.get("expect"), flag_baseline=flag_baseline)
        elif s.get("status") in ("skipped", "crashed"):
            gates = evaluate(s, fm, tier_name, None, None)
        else:                                     # probe/hud/cli/pytest — pass/fail only
            gates = [{"gate": "probe", "pass": s.get("pass", s.get("status") != "skipped"),
                      "severity": "hard", "detail": str(s)[:120]}]
        all_gates.extend(gates)
        if s.get("seconds"):
            ledger["seconds"] += s["seconds"]
        ledger["cost_usd_product"] += s.get("cost_usd_product") or 0
        ledger["tokens_in"] += s.get("tokens_in") or 0
        ledger["tokens_out"] += s.get("tokens_out") or 0
        rows.append({
            "name": name, "sid": sid, "seconds": s.get("seconds"),
            "statuses": fm.get("statuses", [])[-3:] or s.get("complete_statuses", [])[-3:],
            "errors": s.get("error_count", 0), "cost": s.get("cost_usd_product"),
            "cache_pct": round(100 * s.get("tokens_cached", 0) / max(s.get("tokens_in", 1), 1), 1)
            if s.get("tokens_in") else None,
            "dup": fm.get("dup_pairs"), "late": fm.get("late_frames"),
            "memory": s.get("memory"), "flag_faces": s.get("flag_faces"),
            "audit": audit, "gates": [g for g in gates if not g["pass"]
                                      and g["severity"] != "skip"],
        })
    hud_row = next((r for r in rows if r["name"] == "warm-hud"), None)
    hud_json = run_dir / "scenarios" / "hud-proof.json"
    if hud_json.exists():
        h = json.loads(hud_json.read_text())
        all_gates.append({"gate": "hud-render-match", "pass": bool(h.get("pass")), "severity": "hard",
                          "detail": f"footer {h.get('hud_cost')}/{h.get('hud_tokens')} vs report "
                                    f"{h.get('report_cost')}/{h.get('report_tokens')}"})
        if hud_row and not h.get("pass"):
            hud_row["gates"].append({"gate": "hud-render", "pass": False, "detail": str(h)[:100]})
    # Cleanup assertions are gates, not decoration: surviving key-bearing
    # artifacts, suite containers, occupied ports, a changed 8899 process, or
    # a dead final health check used to render as WARN while the verdict
    # still said PASS (Codex r1).
    for name, ok, note in cleanup:
        if not ok:
            all_gates.append({"gate": f"cleanup-{name}", "pass": False,
                              "severity": "hard", "detail": str(note)[:100]})
    all_gates.extend(trend_gate(ledger, previous_full(run_dir)))
    # frame-type census: NEW frame types from product features become VISIBLE here
    census: dict[str, int] = {}
    for sc in scenarios:
        ft = (sc.get("summary") or {}).get("frame_types") or {}
        for k, v in ft.items():
            census[k] = census.get(k, 0) + int(v)
    (run_dir / "ledger.json").write_text(json.dumps(
        {**ledger, "no_cache_upper_bound": round(
            (ledger["tokens_in"] * PRICING["input"] + ledger["tokens_out"] * PRICING["output"]) / 1e6, 4)},
        indent=1))

    hard_fail = [g for g in all_gates if not g["pass"] and g["severity"] == "hard"]
    trend_fail = [g for g in all_gates if not g["pass"] and g["severity"] == "trend"]
    # SPEC §4 three-verdict scale: failed trend gates must surface as
    # REGRESSION — a ±30%-band breach rendered as clean PASS lost the
    # outcome class entirely (Codex r1).
    overall = "FAIL" if hard_fail else ("REGRESSION" if trend_fail else "PASS")
    partial = "-partial" in run_dir.name
    try:
        suite_sha = json.loads((run_dir / "state/pre-state.json").read_text()).get("suite_sha256", "?")
    except (OSError, ValueError):
        suite_sha = "?"
    verdict_line = {
        "quick": "quick PASS authorizes merge only (n=1, protocol-plane); "
                 "release needs Full PASS ×2 + deep-cold ≥1 (SPEC §4).",
        "full": "full PASS is one of the two consecutive passes required for release, "
                "plus a deep-cold sample (SPEC §4).",
    }.get(tier_name, "informational tier — no release authority.")
    if partial:
        verdict_line = "PARTIAL run (--only) — no tier-level authority; excluded from baselines."
    lines = [
        f"# cw-e2e {tier_name}{' (partial)' if partial else ''} — {run_dir.name} — {overall}",
        "",
        f"- git HEAD `{ver.get('git', {}).get('head', '?')[:12]}` · web-api.md `{ver.get('git', {}).get('webapi_commit', '?')}` · suite `{suite_sha}`",
        f"- ledger: product ${ledger['cost_usd_product']:.4f} · no-cache upper bound "
        f"${(ledger['tokens_in'] * PRICING['input'] + ledger['tokens_out'] * PRICING['output']) / 1e6:.2f}"
        f" · tokens in/out {ledger['tokens_in']:,}/{ledger['tokens_out']:,} · wall {ledger['seconds']}s",
        f"- verdict: **{overall}** — {verdict_line}",
        "",
        "| scenario | sid | s | statuses | err | $ | cache% | dup | late | flags | mem | failed gates |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['sid'] or '-'} | {r['seconds'] or '-'} | {r['statuses']} | "
            f"{r['errors']} | {r['cost'] if r['cost'] is not None else '-'} | {r['cache_pct'] if r['cache_pct'] is not None else '-'} | "
            f"{r['dup']} | {r['late']} | {r['flag_faces'] if r['flag_faces'] is not None else '-'} | "
            f"{r['memory'] or '-'} | "
            f"{'; '.join(g['gate'] + ':' + str(g['detail'])[:60] for g in r['gates']) or '—'} |")
    lines += ["", f"- frame-type census: `{json.dumps(census, ensure_ascii=False)}` "
              "(new/unknown types appearing here = protocol drift — see SPEC §10 checklist)", ""]
    lines += ["", "## gates", "",
              "| gate | sev | pass | detail |", "|---|---|---|---|"]
    for g in all_gates:
        lines.append(f"| {g['gate']} | {g['severity']} | {'✓' if g['pass'] else '✗'} | {str(g['detail'])[:100]} |")
    lines += ["", "## cleanup", ""]
    lines += [f"- {'PASS' if ok else 'WARN'} {name} {note}".rstrip() for name, ok, note in cleanup] or ["- (none)"]
    lines += ["", f"_generated {time.strftime('%F %T')} by cw-e2e analyze — three-source rule applies; "
              "judgements carry limits (n, warm/cold, scope)._"]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    (run_dir / "gates.json").write_text(json.dumps(all_gates, indent=1))
    _log(f"report: {run_dir / 'report.md'} — overall {overall} "
         f"({len(hard_fail)} hard failures, {sum(1 for g in all_gates if not g['pass'])} total failed)")
    return {"overall": overall, "gates": all_gates}
