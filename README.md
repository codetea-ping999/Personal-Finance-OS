# Personal Finance OS

**Ledger → Metrics → Decisions.**

Personal Finance OS is an explainable, deterministic personal-finance application inspired by the workflows of accounting, tax, and financial planning. The goal is not to let an LLM invent financial answers; the calculation engine produces the numbers and AI can later explain or orchestrate them.

## Phase 1 features

- SQLite personal ledger foundation: accounts + signed transactions
- Income, expense, cash-flow, assets, liabilities and net-worth aggregation
- Savings rate, emergency-fund months and a deterministic Financial Health Score
- Purchase What-if simulator with opportunity-cost calculation
- Pluggable progressive Tax Engine (no volatile tax rates hard-coded)
- FastAPI REST API
- Minimal browser dashboard
- Unit tests for ledger, analytics, scenarios and tax rules

## Architecture

```text
Browser / Future AI Agent
          |
       FastAPI
          |
  +-------+--------+
  |                |
Ledger        Decision Engines
(SQLite)      - Analytics
              - Scenario
              - Tax rules
```

The LLM layer is intentionally **not** part of the trusted calculation path.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest
uvicorn personal_finance_os.api:app --reload
```

Open `http://127.0.0.1:8000`.

## API examples

Create an account:

```bash
curl -X POST http://127.0.0.1:8000/api/accounts \
  -H "Content-Type: application/json" \
  -d '{"name":"Main Bank","account_type":"bank","opening_balance":4000000}'
```

Create a transaction (expenses are negative):

```bash
curl -X POST http://127.0.0.1:8000/api/transactions \
  -H "Content-Type: application/json" \
  -d '{"account_id":"ACCOUNT_ID","booked_on":"2026-08-12","amount":-300000,"kind":"expense","category":"computer","description":"Workstation"}'
```

Run a purchase scenario:

```bash
curl -X POST http://127.0.0.1:8000/api/scenarios/purchase \
  -H "Content-Type: application/json" \
  -d '{"price":300000,"horizon_months":60,"annual_return_rate":0.04}'
```

## Important accounting convention

For Phase 1, each transaction's `amount` is the signed change to the selected account balance. Income is positive and expenses are negative. Transfers and full double-entry bookkeeping are planned for the next accounting phase.

## Tax safety

Tax rules change by jurisdiction and tax year. This repository therefore ships the **calculation mechanism**, not current Japanese tax tables. A production Tax Engine should version rules by jurisdiction/year and source them from authoritative government publications, with human review for filing decisions.

## Roadmap

1. **Phase 1 – Finance Core**: ledger, metrics, scenarios, API/UI.
2. **Phase 2 – Accounting Core**: double-entry journal, CSV import, reconciliation, monthly P/L and B/S.
3. **Phase 3 – Japan Tax Rules**: versioned official tax datasets, deductions, estimates, evidence trail.
4. **Phase 4 – Financial Digital Twin**: recurring cash flows, life events, Monte Carlo / sensitivity analysis.
5. **Phase 5 – AI CFO**: natural-language queries and explanations over deterministic tools.

## Disclaimer

This software is for personal analysis and engineering experimentation. It is not tax, accounting, legal, or investment advice and is not a substitute for a licensed professional where professional judgment or filing responsibility is required.
