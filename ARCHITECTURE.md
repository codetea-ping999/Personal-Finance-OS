# Architecture

## Design principles

1. **Deterministic first** — money calculations must be reproducible.
2. **Explainability** — every dashboard number must trace back to ledger rows and explicit formulas.
3. **Versioned rules** — tax and regulatory logic is data/rules, not hidden LLM knowledge.
4. **AI outside the trust boundary** — AI may classify, explain and orchestrate; it does not own arithmetic truth.
5. **Local-first MVP** — SQLite minimizes operational cost and privacy exposure while the domain model stabilizes.

## Trust boundary

```text
Untrusted / probabilistic                 Trusted / deterministic
-------------------------                 -----------------------
Natural-language agent  ---- tool ---->   API validation
AI categorization                         Ledger
Explanations                              Accounting rules
                                         Tax rule versions
                                         Scenario formulas
```

## Phase 1 data model

### Account

- `id`
- `name`
- `account_type`: cash / bank / investment / liability
- `opening_balance`

### Transaction

- `id`
- `account_id`
- `booked_on`
- `amount` (signed account delta)
- `kind`: income / expense / transfer / adjustment
- `category`
- `description`

### Phase 2 journal

`journal_entries` stores one balanced transfer between two accounts:

- `booked_on`, `description`, `amount`
- `debit_account_id`, `credit_account_id`
- optional unique `external_id` for safe imports

Amounts are positive. Asset and expense accounts increase on the debit side;
income, liability and equity accounts increase on the credit side. The legacy
`transactions` table remains supported for backward compatibility during migration.

The monthly report endpoint derives income, expenses, assets, liabilities and
equity from journal rows and current account balances. It does not accept
client-supplied totals.

### Phase 3 salary tax

Tax rules are stored in `data/tax/jp/{tax_year}.json`, not in Python constants.
The tax engine loads exactly the requested year and fails explicitly when the
file is missing or invalid. Salary tax estimates include the rule version and
official source metadata so each result is traceable.

Phase 2 will introduce journals and postings so transfers, liabilities, receivables and accrual accounting are represented with proper double-entry invariants.

## Phase 4 digital twin

The forecast path is separate from historical ledger reporting:

```text
SQLite ledger balance as of forecast start
              +
recurring_cash_flows + life_events
              |
       ForecastEngine (deterministic)
              |
       monthly rows + annual summaries
              |
       FastAPI / browser dashboard
```

`recurring_cash_flows` stores positive monthly amounts with an income/expense
type and optional end date. `life_events` stores month-aligned one-time or
duration-based changes. `forecast_scenarios` stores reusable initial balance,
income growth, expense growth and annual return assumptions.

Forecasts default to 30 years and accept 1–50 years. The engine compounds
growth and return assumptions deterministically, applies all active events in
the same month, applies monthly return to the beginning-of-month balance, and
returns the assumptions used alongside the result so a projection can be
reproduced. Taxes, social insurance, investment product details and Monte
Carlo remain outside this phase.

## Phase 5 AI CFO

The AI CFO layer is an orchestration boundary, not a calculation engine:

```text
Japanese natural language
          |
  IntentProvider (local rules / mock / future structured LLM adapter)
          |
  Pydantic validated Intent
          |
  Existing deterministic services and repositories
          |
  Explainable result or write preview
```

`Intent` carries its type, normalized arguments, parser version, target period,
and data sources. The local provider has a deliberately limited Japanese
vocabulary and raises clarification instead of guessing. The `LLMIntentProvider`
interface can be implemented later without moving arithmetic or repository
access into the provider.

Read requests are exposed through `POST /api/ai/queries`. Supported reads are
summary/health, monthly report, account and transaction lists, forecast and
scenario comparison, purchase simulation, and Japanese salary-income tax.
Create requests return a preview. `POST /api/ai/confirm` hashes and validates
the one-time token, checks expiry, recomputes the ledger fingerprint, reserves
the action atomically, and invokes exactly one existing Repository create
method. The `ai_action_log` table stores the normalized Intent, parser version,
state, preview and ledger fingerprint, confirmation/execution timestamps,
result, and error. `GET /api/ai/audit` exposes the audit trail without exposing
the confirmation-token hash. AI v1 action execution records state history,
execution ownership, start time, retry count, and failure code. A confirmed
action is claimed atomically, and its domain insert plus `ai_action_log`
transition to `executed` commit in one SQLite transaction so a crashed attempt
can be retried without duplicating the ledger row.
