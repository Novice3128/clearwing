# Proposal: structured fields on `approval_needed` frames (safety prerequisite)

## Problem
The operator-side approval gate (deny decisions for spraying / brute-force /
destructive ops) currently has to match against `approval_needed.data.prompt`
— a **human-readable description** with no wording-stability guarantee
(`docs/web-api.md` calls it exactly that). One rephrased sentence in the
product silently disables every external deny rule that keys on the text,
with no test turning red. Heuristics like "same prompt >2 times ⇒ deny"
inherit the same fragility.

This is not theoretical: our verification harness has carried regex deny
rules (`POLICY_DENY` / `DESTRUCTIVE_DENY`) across six live rounds
(2026-09-16 → 09-18), and they are the only automated guard between an
approval-happy agent and prohibited actions on the LAB targets.

## Proposal
1. Add structured fields to the `approval_needed` payload, e.g.
   `tool` (name), `args` (the tool-call arguments object), and optionally
   `risk_tags` (product-side classification such as `credential-use`,
   `network-spray`, `destructive`, `read-only`).
2. Keep `prompt` as the human-readable rendering; document that consumers
   MUST base decisions on the structured fields.
3. **Fail-closed guidance**: if a consumer's deny policy requires a field
   that is absent, the safe default is to deny (or surface prominently),
   not to approve.

## Why now
We are standardizing our E2E verification suite (`e2e/`, untracked in this
repo) and its approval gate is exactly this fragile regex layer. With
structured fields we can assert deny rules against stable inputs; without
them, every PR that touches approval copy is an untested hole in the
guardrail.

## Effort estimate
Small: payload construction in one place (`ui/web/app.py` approval frame
emission), doc section in `docs/web-api.md`, tests mirroring the existing
frame-contract tests.
