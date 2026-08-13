# Personal Finance OS — agent working agreement

## First read

- Read `README.md` for supported behavior and `ARCHITECTURE.md` for trust
  boundaries before changing domain behavior.
- Install development dependencies with `python -m pip install -e ".[dev]"`.
- Run the full verification harness with `python scripts/check.py` before
  declaring a task complete. During an iteration, run the smallest relevant
  `python -m pytest ...` selection first.

## Non-negotiable financial boundaries

- Treat `personal_finance.db` and any user-supplied database/export as private
  data. Do not mutate, delete, migrate, or use it as a test fixture. Tests must
  create databases under `tmp_path`.
- Keep money as integer yen. Do not introduce floating-point money arithmetic.
- The deterministic ledger, accounting, tax, analytics, scenario, and forecast
  layers are the source of numerical truth. An intent provider may normalize or
  explain requests; it must not calculate financial results or write directly
  to SQLite.
- Preserve the AI CFO preview, confirmation, idempotency, and audit boundary.
  Never broaden an AI write flow to update/delete/import data without an
  explicit design and human review.
- Tax-rule changes require a versioned data file, authoritative source links,
  boundary tests, and human review. Never silently change a historical rule.
- Do not add real financial integrations, credentials, background jobs, or
  automatic deployment/merge behavior without explicit approval.

## Small, evidence-driven loop

1. State the outcome, scope, risk tier, and observable acceptance checks.
2. Inspect the narrowest relevant code and tests; prefer a small reversible
   change over speculative refactoring.
3. Add or update a regression test when behavior changes.
4. Run focused tests, then `python scripts/check.py`.
5. Review the diff against the acceptance checks and report commands and
   results. Escalate after two failed repair attempts or whenever requirements,
   tax rules, schema migration, or real data handling are ambiguous.

## Code review rules

- Flag changes that move arithmetic, validation, or repository access into an
  LLM/intent-provider path.
- Flag changes that weaken confirmation tokens, transaction atomicity,
  idempotency, auditability, versioned tax evidence, or SQLite migration
  compatibility.
- For behavior changes, require a test that would fail on the previous code.
