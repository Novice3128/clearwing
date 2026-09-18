# Security report: unauthenticated read endpoints on the webui (0.0.0.0 default)

## Observation (2026-09-17/18, live on the member-deployed :8899)
These routes have NO `require_api_key` dependency while the webui is typically
started with `--host 0.0.0.0`:

- `GET /api/sessions` — session list
- `GET /api/sessions/{id}` — includes `vulnerabilities`, `exploit_results`
- `GET /api/metrics` — cost/usage metrics

(`clearwing/ui/web/app.py` — only `operate*` and `reports*` routes carry the
key dependency.) Any LAN host can therefore read scan results and findings
summaries without the web key. The key-gated paths are fine.

## Suggested fix
Either attach `require_api_key` to these routes, or bind loopback by default
(`--host 127.0.0.1`) and document the exposure explicitly for operators who
intentionally open it (e.g. behind a reverse proxy adding auth).

## Notes from our side
Our verification suite always binds its self-spawned instance to
`127.0.0.1` for exactly this reason (SPEC §2.5). Filing this so the default
deployment posture matches the product's own auth intent.

## Appendix (2026-09-18, from the verification-suite review)
Two further log-side findings on the live :8899 deployment:

1. **The access-log redact filter does not cover WebSocket accept lines.**
   `WebSocket /ws/agent?api_key=<plaintext> [accepted]` lines carry the key in
   clear (observed live), while `GET /?api_key=` lines are redacted. Since the
   browser path has no header option (browsers cannot set WS headers), this is
   the one place the key necessarily transits a query param — it should be
   redacted there too.
2. **The filter itself crashes** (`TypeError: cannot unpack non-iterable
   NoneType` in uvicorn's formatter, ~14 occurrences over two days) — a filter
   returning None breaks the logging pipeline; and the process's stdout has
   been pointing at a **deleted inode** since the 2026-09-17 21:29 restart
   (the on-disk `~/clearwing-webui-8899.log` stopped growing at 21:31), so
   current logs are invisible on disk. Recommend restarting with a correct
   redirect after fixing the filter.
