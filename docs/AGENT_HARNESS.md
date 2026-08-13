# Agent Execution Harness and Delivery Loop

## Purpose

This repository uses two related terms deliberately:

- **Harness engineering** is the design of the environment in which an agent
  works safely and repeatably: repository instructions, trusted boundaries,
  tools, permissions, tests, evidence, and stop conditions.
- **Loop engineering** is the operating cycle around that harness: discover a
  bounded task, implement it, independently verify the result, persist the
  evidence, and either finish, retry with concrete feedback, or escalate.

The goal is not autonomous financial decision-making. It is to make small
software changes reproducible and reviewable while preserving the application's
deterministic financial core.

## What is already the harness

| Concern | Repository mechanism |
| --- | --- |
| Trusted calculations | `analytics.py`, `scenario.py`, `forecast.py`, `tax_engine.py`, and `repository.py` |
| AI boundary | Validated `Intent` objects and the preview/confirm/audit flow in `ai_cfo.py` |
| Regression evidence | `tests/` with temporary SQLite databases and API tests |
| Multi-version validation | `.github/workflows/ci.yml` on Python 3.11–3.13 |
| Quality gate | `python scripts/check.py` locally and in CI |
| Persistent task evidence | GitHub issue/PR plus its acceptance checks and verification output |

`AGENTS.md` is the concise entry point for coding agents. This document is the
longer operating policy; `README.md` and `ARCHITECTURE.md` remain the product
and design sources of truth.

## Delivery loop

```text
Bounded issue / CI failure / explicit request
                  |
         task card + risk tier
                  |
     Builder in an isolated worktree
                  |
    focused tests -> full verification harness
                  |
        independent verifier checks
       acceptance criteria + diff + evidence
          /             |              \
      accepted       actionable fail    ambiguous/high-risk
         |                 |                  |
    reviewable PR   one concrete retry     human escalation
```

The builder and verifier must be separate contexts whenever practical. The
verifier initially reviews and tests; it does not silently repair the builder's
work. This avoids treating an agent's self-assessment as the only quality gate.

## Task card and completion contract

Use the following in an issue, pull request, or task description before work
starts:

```md
## Outcome
<observable user or system result>

## Scope
In: <files/behavior>
Out: <explicit non-goals>

## Risk tier
T0 / T1 / T2 / T3 (see below)

## Acceptance checks
- [ ] <behavioral condition>
- [ ] <regression/contract test>
- [ ] `python scripts/check.py`

## Loop limit and escalation
At most two implementation/repair attempts. Escalate if the requirement is
ambiguous, a gate still fails, or a T2/T3 boundary is encountered.
```

Completion evidence records the changed behavior, tests added or updated, the
exact verification command(s) and result, remaining risks, and any required
human review. A passing test without a stated acceptance condition is not a
complete task.

## Risk tiers

| Tier | Examples | Agent authority |
| --- | --- | --- |
| T0 | Documentation, isolated tests, CI ergonomics | May prepare a change and verification evidence. |
| T1 | Existing API behavior, deterministic engines, UI | May implement a bounded change with tests and independent verification. Human reviews the PR. |
| T2 | Tax datasets/rules, database schema or migrations, AI write contract, auth, dependencies | Prepare an evidence-backed proposal or patch; human review is required before merge. |
| T3 | Real/user database mutation, credentials, external financial action, production deploy, automatic merge | Do not execute autonomously. Escalate before acting. |

All work is at least T1 when it changes a financial result. If a task spans
tiers, use the highest tier.

## Verification commands

The canonical full gate is:

```bash
python scripts/check.py
```

It compiles source and tests, runs the suite with coverage, enforces the
existing total and AI-CFO coverage thresholds, and writes `coverage.xml` for
CI. Use a focused test during the inner loop, for example:

```bash
python -m pytest tests/test_ai_cfo.py -q
```

The current high-value regression corpus is intentionally domain-focused:

- `tests/test_ai_cfo.py`: intent normalization, preview/confirmation,
  idempotency, concurrency, audit trail, and retry behavior.
- `tests/test_tax_engine.py` and `data/tax/jp/*.json`: versioned tax rules.
- `tests/test_repository.py` and `tests/test_accounting_api.py`: ledger and
  double-entry persistence/API behavior.
- `tests/test_forecast.py`: deterministic cash-flow and life-event projection.

Do not replace these behavior checks with snapshots alone. Extend the focused
corpus whenever a production bug, ambiguity, or incorrect agent action is
found.

## Verifier checklist

The independent verifier answers these questions with evidence:

1. Does the diff satisfy each acceptance check and stay in scope?
2. Is numerical truth still inside deterministic code, with money represented
   as integer yen?
3. Are validation, confirmation, atomicity, idempotency, audit, and legacy DB
   compatibility preserved where relevant?
4. Does a new or changed test fail against the prior behavior?
5. Did the full harness pass? If not, is the failure classified and returned as
   an actionable retry item rather than hand-waved away?

## Stopping and learning

Stop and escalate rather than continuing a vague or unbounded loop when:

- requirements or acceptance checks conflict or are missing;
- two repair attempts fail to satisfy a gate;
- a change touches T2/T3 data or authority;
- the test cannot distinguish correct from incorrect financial behavior.

Turn the resolved failure into a regression test, a clarified task card, or a
concise `AGENTS.md` rule. That is how the harness becomes stronger over time.

## Next increments

This first increment intentionally keeps the toolchain small and reuses the
existing pytest/coverage foundation. Prioritize these only when their value is
clear:

1. Inject clocks into forecast, parser, and confirmation TTL paths, then add
   month/expiry boundary tests.
2. Add tax-data schema/boundary checks, public API contract checks, and
   wheel-install smoke tests as explicit critical gates.
3. Add formatter, linter, type checker, and a locked dependency workflow after
   selecting tools and fixing their baseline findings.
4. Automate worktree/PR orchestration only after the manual builder/verifier
   loop is working reliably. Human approval remains required for T2/T3 work.
