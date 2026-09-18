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
    per_call: list[tuple[int, int, int]] = []      # (input, cached, output)
    for line in f.read_text().splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event_type") == "llm_call":
            calls += 1
            d = e.get("details") or {}
            ti = int(d.get("input_tokens") or 0)
            tc = int(d.get("cached_tokens") or 0)
            to = int(d.get("output_tokens") or 0)
            tin += ti
            tout += to
            tcached += tc
            per_call.append((ti, tc, to, e.get("agent") or "main"))
            costs.append(float(d.get("cost_usd") or 0))
    real = (tin - tcached) * PRICING["input"] / 1e6 + tcached * PRICING["cached_input"] / 1e6 \
        + tout * PRICING["output"] / 1e6
    return {"llm_calls": calls, "tokens_in": tin, "tokens_out": tout, "tokens_cached": tcached,
            "audit_cost_sum": round(sum(costs), 4), "real_cost_cache_aware": round(real, 4),
            "no_cache_upper_bound": round((tin * PRICING["input"] + tout * PRICING["output"]) / 1e6, 4),
            "calls": per_call}


def cache_prefix_median(per_call: list[tuple]) -> float | None:
    """Median cache ratio over STABLE-PREFIX calls, computed WITHIN each
    agent context. Contexts are tracked separately (Codex #67 r1 P2): an
    interleaved 6k uncached operator call must not count as a main-prefix
    miss, nor must it reset the main line's growth baseline. Growth calls
    (>20% vs the previous call of the SAME context) legitimately carry
    fresh content (scan blobs, compaction rewrites) and are excluded;
    each context's first call has no prefix and is excluded. Aggregate
    cache% is run-shape dominated (same build measured 54.2% and 98.6%)
    — the per-call caliber is the regression detector."""
    ratios = []
    prev_in: dict[str, int] = {}
    for row in per_call:
        ti, tc, agent = row[0], row[1], (row[3] if len(row) > 3 else "main")
        if ti < 5000:
            continue            # tiny auxiliary contexts (~450 tokens) have no
                                # shared prefix by design — 0% cache there is not
                                # degradation (n=2 replay lesson)
        prev = prev_in.get(agent)
        if prev is not None and ti <= prev * 1.2:
            ratios.append(100 * tc / max(ti, 1))
        prev_in[agent] = ti
    if len(ratios) < 2:
        return None
    import statistics
    return statistics.median(ratios)


# ------------------------------------------------------------- gates ------

def evaluate(summary: dict, fm: dict, tier_name: str, audit: dict | None,
             hud: dict | None, expect: dict | None = None,
             flag_max: int | None = None) -> list[dict]:
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
        # aggregate cache% stays as a liveness check only — the regression
        # detector is the per-call caliber below (four-lens review: same
        # build measured 54.2% and 98.6% aggregate)
        add("cache-nonzero", cache_pct > 1, f"cache={cache_pct:.1f}% (aggregate, shape-dependent)")
    if audit and audit.get("calls"):
        med = cache_prefix_median(audit["calls"])
        if med is not None:
            add("cache-prefix-median",
                med >= SUITE["thresholds"]["hard"].get("cache_prefix_median_min", 90),
                f"prefix-median={med:.1f}% want>="
                f"{SUITE['thresholds']['hard'].get('cache_prefix_median_min', 90)}% "
                f"(aggregate {cache_pct:.1f}% is informational)")
        # degenerate-output detector (t3-warm-chain-2: 23.7k-in, 0% cache,
        # 3-token reply — silently passed the old >100k-bound gates). Caliber
        # = CALL signature, not session totals: warm-chain-1/-3 legitimately
        # end with 3-token replies ON FULL CACHE (94%/99.8%) — terse-but-
        # cached is healthy recall behavior (review N3 replay lesson)
        tiny = [c for c in audit["calls"] if c[0] > 10000 and c[1] == 0 and c[2] < 10]
        add("degenerate-output", not tiny,
            f"degenerate_calls={len(tiny)} "
            "(a >10k-in, 0-cache call replying <10 tokens)")
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

    if flag_max is not None and tier_name == "full":
        faces = summary.get("flag_faces")
        if faces is not None:
            # baseline 9 measured same-build spread 8↔14 (2026-09-18) — a
            # hard <=9 gate flagged pure #35-family volatility; the range
            # keeps the drift ceiling without the noise
            add("flag-faces-max", faces <= flag_max,
                f"faces={faces} max={flag_max} (observed same-build spread 8-14)")
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
    flag_max = None
    try:
        flag_max = SUITE["tiers"]["full"].get("analysis", {}).get("flag_max", 15)
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
                             expect=sc.get("expect"), flag_max=flag_max)
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
    if "-partial" in run_dir.name:
        all_gates.append({"gate": "baseline", "pass": True, "severity": "trend",
                          "detail": "partial run (--only) — trend gates suppressed "
                          "(subset vs whole-tier baseline is not comparable)"})
    else:
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
    if hard_fail and "-partial" not in run_dir.name:
        # §9 trigger hint: a FAIL re-adjudicated to non-regression is exactly
        # the "判定翻案" review case, and any external posting needs the
        # review first — the adjudication/export layer enforces it.
        lines_hint = ("\n> §9 觸發：本 run 有 hard-gate FAIL。任何翻案判定或對外輸出前，"
                      "先 `cw-e2e adjudicate <run-dir>` 完成複審紀錄（export 會強制檢查）。\n")
    else:
        lines_hint = ""
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
        lines_hint,
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
    # P4.1 R3 manual fact-check template — suite.yaml's fact_check_samples
    # promised this since v1 but nothing consumed it (four-lens review G2).
    # Completion marker: `touch <run_dir>/r3-done`; adjudicate surfaces it.
    try:
        samples = int(SUITE["tiers"]["full"].get("analysis", {}).get("fact_check_samples", 0))
    except (KeyError, AttributeError, TypeError, ValueError):
        samples = 0
    r3p = run_dir / "r3-manual.md"
    # P2-b (Codex #67 r1): a half-filled manual is HUMAN state — re-running
    # `analyze` must not destroy it; only generate when absent entirely
    if samples and tier_name == "full" and not r3p.exists() and not (run_dir / "r3-done").exists():
        sids = [sc["summary"].get("session_id") for sc in scenarios
                if (sc["summary"] or {}).get("session_id")][:samples]
        r3 = [
            "# R3 人工抽核（Full）— " + run_dir.name, "",
            "> 完成抽核後 `touch " + str(run_dir / "r3-done") + "`；",
            "> adjudicate 對發版級宣稱（Full×2＋deep-cold）檢查此標記。", "",
            "## 待抽核報告（samples=" + str(samples) + "）", "",
        ]
        for sid in sids:
            r3.append(f"- sid `{sid}` → `~/.clearwing/results/sessions/{sid}/report.md`")
        r3 += [
            "", "## 抽核欄位（v3 fp-verification 紀律）", "",
            "1. 執行摘要數字 vs 工具活動表（tool calls/errors/cost 對拍）",
            "2. CVE/弱點聲稱的證據抽樣（宣稱 vs frames.jsonl 工具輸出）",
            "3. 權限/結果聲稱與卡點誠實性", "",
            "## 紀錄", "", "- （填寫後 touch r3-done）", "",
        ]
        (run_dir / "r3-manual.md").write_text("\n".join(r3))
    _log(f"report: {run_dir / 'report.md'} — overall {overall} "
         f"({len(hard_fail)} hard failures, {sum(1 for g in all_gates if not g['pass'])} total failed)")
    return {"overall": overall, "gates": all_gates}
