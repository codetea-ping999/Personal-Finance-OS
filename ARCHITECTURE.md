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

Phase 2 will introduce journals and postings so transfers, liabilities, receivables and accrual accounting are represented with proper double-entry invariants.
