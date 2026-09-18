# Proposal: add a version header to `docs/web-api.md`

## Problem
`docs/web-api.md` changed 10 times in the 3 days to 2026-09-17 (12 commits
lifetime) and currently carries compatibility promises only in prose
("clients MUST ignore unrecognized frame types", "`status` is backward
compatible: absent = ok"). External consumers (our E2E driver lineage, any
third-party operator console) have no machine-checkable notion of which
contract generation they were built against — when an assertion breaks, the
first question ("did the contract move, or is my client wrong?") is currently
answered by git archaeology on both sides.

## Proposal
1. One header line at the top of `docs/web-api.md`, e.g.
   `contract-version: 2026-09-18.1` (date-based, bumped on any
   frame-shape/semantics change; editorial fixes don't bump).
2. Optionally expose the same string at `GET /api/health` (e.g.
   `{"status":"ok","contract":"2026-09-18.1"}`) so consumers can log what
   they talked to per session.
3. A short "what bumps the version" list (new frame type? no — clients must
   ignore unknown; renamed field? yes; changed field semantics e.g. cost
   totals scope? yes).

## Why it matters to us
Our verification suite records `git HEAD` + the `web-api.md` commit per run,
but a version string would let the suite fail with
"contract moved: expected X got Y" instead of a downstream metric silently
becoming incomparable (the exact failure mode behind the cost-metering
confusions of 2026-09-16/17).

## Effort estimate
Trivial: one doc line + optional health payload field + one test.
