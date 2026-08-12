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
- Double-entry journal entries with debit/credit invariants
- Journal CSV import with external-id deduplication
- Monthly income statement and balance-sheet summary API
- Versioned Japanese salary-income tax estimation for 2025 and 2026
- Financial Digital Twin: SQLite-backed recurring cash flows and life events
- Deterministic monthly forecasts for 1–50 years (30-year default)
- Saved forecast assumptions, sensitivity comparison, monthly graph and annual summaries

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

Register recurring cash flow and forecast it:

```bash
curl -X POST http://127.0.0.1:8000/api/recurring-cash-flows \
  -H "Content-Type: application/json" \
  -d '{"name":"Salary","flow_type":"income","amount":300000,"start_date":"2026-09-01"}'

curl -X POST http://127.0.0.1:8000/api/forecasts \
  -H "Content-Type: application/json" \
  -d '{"start_date":"2026-09-01","period_years":30,"scenario_name":"Base case"}'
```

Life events use `event_type: "one_time"` or `"recurring"`. A recurring event applies
from `start_date` for `duration_months`; `income_delta` and `expense_delta` are
integer-yen changes applied to each active month. Saved assumptions are managed
with `/api/forecast-scenarios`, and `/api/forecasts/compare` accepts multiple
named cases with optional assumption overrides.

Create a double-entry journal entry (amount is always positive):

```bash
curl -X POST http://127.0.0.1:8000/api/journal-entries \
  -H "Content-Type: application/json" \
  -d '{"booked_on":"2026-08-12","description":"Salary","debit_account_id":"BANK_ID","credit_account_id":"INCOME_ACCOUNT_ID","amount":400000}'
```

Monthly accounting report:

```bash
curl "http://127.0.0.1:8000/api/reports/monthly?year=2026&month=8"
```

Estimate Japanese salary income tax:

```bash
curl -X POST http://127.0.0.1:8000/api/tax/estimate \
  -H "Content-Type: application/json" \
  -d '{"tax_year":2026,"gross_salary":6000000,"social_insurance_premiums":900000}'
```

The response includes salary income deduction, basic deduction, taxable income,
income tax, reconstruction special tax, rule version, and official sources.
This is an estimate only; resident tax, social-insurance calculation and other
deductions are intentionally out of scope.

Journal CSV import accepts `booked_on,debit_account_id,credit_account_id,amount,description,external_id`.
Rows with an already-seen `external_id` are skipped, which makes retries safe.

口座明細との照合は、台帳残高との差額を確認できます。

```bash
curl -X POST http://127.0.0.1:8000/api/reconciliation \
  -H "Content-Type: application/json" \
  -d '{"account_id":"BANK_ID","statement_balance":4300000}'
```

## AI CFO v1

AI CFO は自然言語を限定的な日本語ローカル規則で構造化Intentへ正規化し、
既存の分析・予測・税計算・Repositoryを呼び出します。外部LLMや追加APIキーは
不要です。読み取りは結果とともに対象期間、データソース、パーサー/ルール
バージョンを返します。解釈できない依頼や曖昧な口座・日付・金額は推測せず、
`needs_clarification` を返します。
AI CFO v1の成功応答には `api_version: "1"` が含まれ、確認不要の曖昧さもHTTP 200
の会話結果として返ります。確認エラーは無効トークンが400、期限切れ・台帳変更・
二重確認が409で、`code`、`message`、`details`、`audit_id`、`action_id` を含みます。

```bash
curl -X POST http://127.0.0.1:8000/api/ai/queries \
  -H "Content-Type: application/json" \
  -d '{"text":"2026年8月の収支と純資産を教えて"}'
```

台帳を書き込むIntentは、必ずプレビューを返します。レスポンスの確認トークンを
明示的に送ると一度だけ実行されます。トークンは10分で失効し、プレビュー後に
台帳が変わった場合も拒否されます。確認済みの未実行操作は、プロセス中断後に
同じトークンで再開でき、ドメイン書込みと監査更新は同一SQLiteトランザクションで
確定します。監査APIは確認トークンのハッシュを返しません。

```bash
curl -X POST http://127.0.0.1:8000/api/ai/confirm \
  -H "Content-Type: application/json" \
  -d '{"confirmation_token":"PREVIEW_RESPONSE_TOKEN"}'

curl "http://127.0.0.1:8000/api/ai/audit?limit=100"
```

対応する作成操作は通常取引、借方/貸方仕訳、定期収支、ライフイベントです。
更新・削除・CSV一括取込はAI CFO v1のIntent対象外です。

## Important accounting convention

For Phase 1, each transaction's `amount` is the signed change to the selected account balance. Income is positive and expenses are negative. Transfers and full double-entry bookkeeping are planned for the next accounting phase.

## Tax safety

Tax rules change by jurisdiction and tax year. This repository therefore ships the **calculation mechanism**, not current Japanese tax tables. A production Tax Engine should version rules by jurisdiction/year and source them from authoritative government publications, with human review for filing decisions.

## Roadmap

1. **Phase 1 – Finance Core**: ledger, metrics, scenarios, API/UI.
2. **Phase 2 – Accounting Core**: double-entry journal, CSV import and monthly P/L and B/S (reconciliation follow-up).
3. **Phase 3 – Japan Tax Rules**: versioned official tax datasets, salary deductions, estimates, evidence trail.
4. **Phase 4 – Financial Digital Twin**: recurring cash flows, life events, deterministic forecasts and sensitivity analysis.
5. **Phase 5 – AI CFO**: natural-language queries, explanations, and confirmed writes over deterministic tools.

## Disclaimer

This software is for personal analysis and engineering experimentation. It is not tax, accounting, legal, or investment advice and is not a substitute for a licensed professional where professional judgment or filing responsibility is required.
