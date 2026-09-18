# Pathless start-frame/CLI base_url requests `/chat/completions` at the host root → upstream 404 kills the session

> Revision 2 (2026-09-17, post multi-agent review). An earlier draft framed this
> as a regression caused by PR #43's per-field merge leaking the config
> `adapter:` onto per-request endpoints. Git archaeology refuted that
> attribution — PR #43's `8d65939` added the guard that *prevents* exactly that
> leak, and the resolution+join code is byte-identical back to `3a8327e`.
> The framing below is the verified version.

## Environment
- HEAD `ac8f0ee` (PR #56), verified live 2026-09-17. Behavior on `790c8a9` /
  `7894280` / `3a8327e` **not live-tested** with a pathless base_url; code
  inspection shows the resolution+join path is byte-identical across all four
  refs, so this is most likely longstanding behavior, not a recent regression.
- config.yaml `provider:` has `adapter: openai`,
  `base_url: https://api.z.ai/api/coding/paas/v4`, `model: glm-5.3`
- WebUI start frame (or CLI `--base-url`) supplies a **pathless** base_url,
  e.g. `http://127.0.0.1:8787` (an OpenAI-compatible gateway / TCP proxy),
  with model pinned `glm-5.3`

## Symptom
Every LLM request is sent to `<base_url>/chat/completions` (host root).
Against an upstream that only serves the full path (api.z.ai →
`/api/coding/paas/v4/chat/completions`), the response is **HTTP 404 (nginx)**
and the call dies:

```
error: OpenAI-compatible fallback failed with HTTP 404: <html>...404 Not Found...nginx...
      (gave up after 1 retry)
complete: {"status":"error", ...}
```

404 is in the non-retriable class, so one wrong path kills the session in
<2 s even though the endpoint itself is healthy (the error-status session
report still lands — failure is clean, not a hang).

## Evidence (live, authorized LAB, 2026-09-17)
- Pathless `--base-url http://127.0.0.1:8787` → proxy log request lines are
  `POST /chat/completions HTTP/1.1`, task `complete(error)` in 0–2 s
  (`five-issues-retest-20260917/out/chaos-refuse{1,3}.{frames.jsonl,proxy.log}`).
- Same invocation with the full path
  `--base-url http://127.0.0.1:8787/api/coding/paas/v4` → request lines
  `POST /api/coding/paas/v4/chat/completions`, task completes ok in 29 s
  (`out/chaos-refuse1b.*`).
- Historical proxy logs from earlier verification rounds (fd5ac43 /
  3a8327e / 7894280) all show the full-path request line — consistent with
  those runs passing the full path in `--base-url` (their argv is unrecorded
  in any doc; the reports only say "帶混沌代理"). No code evidence exists
  that a pathless base_url ever auto-prefixed the z.ai path.

## Root cause
A pathless per-request base_url resolves with `endpoint.adapter` unset —
`resolve_llm_endpoint` explicitly scopes the config adapter to the config
block's own base_url (`clearwing/providers/env.py`:
"An adapter override describes the endpoint its config block named; it does
not follow a CLI/env base_url that replaced the configured one" —
`adapter = cfg_adapter if (cfg_adapter and base_url == cfg_base_url) else None`).
`_adapter_for_base_url`'s default branch then yields the plain `"openai"`
adapter (`clearwing/providers/manager.py:1041` → heuristic), and plain-HTTP
bases are served by `_openai_chat_http_fallback`'s
`urljoin(base_url + "/", "chat/completions")` (`clearwing/llm/native.py:1540-1543`,
byte-identical since at least `3a8327e`) — a bare host therefore drops to the
root path, and the non-retriable 404 terminates the session.

## Impact
- Any operator pointing a session at a pathless gateway/proxy base (split
  routing, chaos/replay proxies, corporate gateways) gets root-path 404s.
- Documented verification methodology that used a bare proxy host breaks;
  workaround (verified live): include the path in base_url.

## Suggested fix directions
1. Document/validate: require (or auto-complete) an API path on
   OpenAI-compatible base_urls, and emit an actionable configuration error on
   bare-host + 404 ("base_url carries no path and no dialect was resolved")
   instead of a bare upstream HTML 404.
2. Consider a path-inheritance option: when the CLI/start-frame base_url has
   no path but config's does, offer to reuse the config path prefix (opt-in;
   the current no-leak scoping is otherwise correct).
