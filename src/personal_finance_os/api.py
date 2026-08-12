from __future__ import annotations

import os
import csv
import io
from dataclasses import asdict
from datetime import date

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .ai_cfo import AICFOService, AIActionError
from .ai_intents import IntentProvider
from .analytics import FinanceAnalyzer
from .forecast import DEFAULT_FORECAST_YEARS, ForecastEngine
from .models import AccountType, CashFlowType, LifeEventType, TransactionKind
from .repository import FinanceRepository
from .scenario import simulate_purchase
from .tax_engine import estimate_salary_tax


class AccountCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    account_type: AccountType
    opening_balance: int = 0


class TransactionCreate(BaseModel):
    account_id: str
    booked_on: date
    amount: int
    kind: TransactionKind
    category: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class PurchaseScenarioRequest(BaseModel):
    price: int = Field(gt=0)
    horizon_months: int = Field(default=60, gt=0, le=600)
    annual_return_rate: float = Field(default=0.04, gt=-1, le=1)


class JournalEntryCreate(BaseModel):
    booked_on: date
    description: str = Field(default="", max_length=500)
    debit_account_id: str
    credit_account_id: str
    amount: int = Field(gt=0)
    external_id: str | None = Field(default=None, max_length=200)


class ReconciliationRequest(BaseModel):
    account_id: str
    statement_balance: int


class SalaryTaxEstimateRequest(BaseModel):
    tax_year: int
    gross_salary: int
    social_insurance_premiums: int = 0


class AIQueryRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=5000)
    query: str | None = Field(default=None, min_length=1, max_length=5000)
    natural_language: str | None = Field(default=None, min_length=1, max_length=5000)
    input: str | None = Field(default=None, min_length=1, max_length=5000)

    def resolved_text(self) -> str:
        value = self.text or self.query or self.natural_language or self.input
        if not value or not value.strip():
            raise ValueError("text or query is required")
        return value.strip()


class AIConfirmRequest(BaseModel):
    # Empty values are validated by AICFOService so the v1 endpoint can return
    # the documented 400 error contract instead of FastAPI's generic 422 body.
    confirmation_token: str | None = Field(default=None, max_length=200)
    token: str | None = Field(default=None, max_length=200)

    def resolved_token(self) -> str:
        value = self.confirmation_token or self.token
        # Let AICFOService create the stable v1 error/audit contract for a
        # missing token instead of returning FastAPI's generic request error.
        return value or ""


class RecurringCashFlowCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    flow_type: CashFlowType
    amount: int = Field(gt=0)
    start_date: date
    end_date: date | None = None


class RecurringCashFlowUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    flow_type: CashFlowType | None = None
    amount: int | None = Field(default=None, gt=0)
    start_date: date | None = None
    end_date: date | None = None


class LifeEventCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    start_date: date
    duration_months: int = Field(default=1, gt=0, le=600)
    event_type: LifeEventType = LifeEventType.ONE_TIME
    income_delta: int = 0
    expense_delta: int = 0


class LifeEventUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    start_date: date | None = None
    duration_months: int | None = Field(default=None, gt=0, le=600)
    event_type: LifeEventType | None = None
    income_delta: int | None = None
    expense_delta: int | None = None


class ForecastScenarioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    initial_balance: int | None = None
    income_growth_rate: float = Field(default=0.0, gt=-1)
    expense_growth_rate: float = Field(default=0.0, gt=-1)
    annual_return_rate: float = Field(default=0.0, gt=-1)


class ForecastScenarioUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    initial_balance: int | None = None
    income_growth_rate: float | None = Field(default=None, gt=-1)
    expense_growth_rate: float | None = Field(default=None, gt=-1)
    annual_return_rate: float | None = Field(default=None, gt=-1)


class ForecastOverrides(BaseModel):
    initial_balance: int | None = None
    income_growth_rate: float | None = Field(default=None, gt=-1)
    expense_growth_rate: float | None = Field(default=None, gt=-1)
    annual_return_rate: float | None = Field(default=None, gt=-1)


class ForecastRequest(BaseModel):
    start_date: date | None = None
    period_years: int = Field(default=DEFAULT_FORECAST_YEARS, ge=1, le=50)
    years: int | None = Field(default=None, ge=1, le=50)
    scenario_name: str = Field(default="Base case", min_length=1, max_length=100)
    case_name: str | None = Field(default=None, min_length=1, max_length=100)
    initial_balance: int | None = None
    income_growth_rate: float | None = Field(default=None, gt=-1)
    expense_growth_rate: float | None = Field(default=None, gt=-1)
    annual_return_rate: float | None = Field(default=None, gt=-1)
    overrides: ForecastOverrides | None = None

    def resolved_years(self) -> int:
        return self.years if self.years is not None else self.period_years

    def resolved_name(self) -> str:
        return self.case_name or self.scenario_name

    def resolved_overrides(self) -> dict[str, int | float | None]:
        values = self.overrides.model_dump(exclude_none=True) if self.overrides else {}
        for key in ("initial_balance", "income_growth_rate", "expense_growth_rate", "annual_return_rate"):
            value = getattr(self, key)
            if value is not None:
                values[key] = value
        return values


class ForecastComparisonCase(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    scenario_name: str | None = Field(default=None, min_length=1, max_length=100)
    initial_balance: int | None = None
    income_growth_rate: float | None = Field(default=None, gt=-1)
    expense_growth_rate: float | None = Field(default=None, gt=-1)
    annual_return_rate: float | None = Field(default=None, gt=-1)
    overrides: ForecastOverrides | None = None

    def resolved_name(self, index: int) -> str:
        return self.name or self.scenario_name or f"Case {index + 1}"

    def resolved_overrides(self) -> dict[str, int | float | None]:
        values = self.overrides.model_dump(exclude_none=True) if self.overrides else {}
        for key in ("initial_balance", "income_growth_rate", "expense_growth_rate", "annual_return_rate"):
            value = getattr(self, key)
            if value is not None:
                values[key] = value
        return values


class ForecastComparisonRequest(BaseModel):
    start_date: date | None = None
    period_years: int = Field(default=DEFAULT_FORECAST_YEARS, ge=1, le=50)
    years: int | None = Field(default=None, ge=1, le=50)
    cases: list[ForecastComparisonCase] | None = Field(default=None, min_length=1, max_length=20)
    scenarios: list[ForecastComparisonCase] | None = Field(default=None, min_length=1, max_length=20)

    def resolved_years(self) -> int:
        return self.years if self.years is not None else self.period_years

    def resolved_cases(self) -> list[ForecastComparisonCase]:
        return self.cases or self.scenarios or []


def create_app(
    database: str | None = None,
    intent_provider: IntentProvider | None = None,
    ai_preview_ttl_seconds: int = 600,
) -> FastAPI:
    repository = FinanceRepository(database or os.getenv("PFOS_DATABASE", "personal_finance.db"))
    analyzer = FinanceAnalyzer(repository)
    forecast_engine = ForecastEngine(repository)
    ai_cfo = AICFOService(
        repository,
        analyzer=analyzer,
        forecast_engine=forecast_engine,
        provider=intent_provider,
        preview_ttl_seconds=ai_preview_ttl_seconds,
    )
    app = FastAPI(
        title="Personal Finance OS",
        version="0.1.0",
        description="Explainable personal ledger, analytics and deterministic scenario simulation.",
    )

    @app.get("/")
    def dashboard():
        return FileResponse(os.path.join(os.path.dirname(__file__), "web", "index.html"))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/accounts", status_code=201)
    def create_account(payload: AccountCreate):
        try:
            return asdict(repository.create_account(**payload.model_dump()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/accounts")
    def list_accounts():
        balances = repository.account_balances()
        return [asdict(account) | {"balance": balances[account.id]} for account in repository.list_accounts()]

    @app.post("/api/transactions", status_code=201)
    def create_transaction(payload: TransactionCreate):
        try:
            return asdict(repository.create_transaction(**payload.model_dump()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/transactions")
    def list_transactions(
        start: date | None = Query(default=None),
        end: date | None = Query(default=None),
    ):
        return [asdict(t) for t in repository.list_transactions(start=start, end=end)]

    @app.post("/api/recurring-cash-flows", status_code=201)
    @app.post("/api/forecast/recurring-cash-flows", status_code=201, include_in_schema=False)
    def create_recurring_cash_flow(payload: RecurringCashFlowCreate):
        try:
            return asdict(repository.create_recurring_cash_flow(**payload.model_dump()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/recurring-cash-flows")
    @app.get("/api/forecast/recurring-cash-flows", include_in_schema=False)
    def list_recurring_cash_flows():
        return [asdict(flow) for flow in repository.list_recurring_cash_flows()]

    @app.patch("/api/recurring-cash-flows/{flow_id}")
    @app.patch("/api/forecast/recurring-cash-flows/{flow_id}", include_in_schema=False)
    def update_recurring_cash_flow(flow_id: str, payload: RecurringCashFlowUpdate):
        try:
            return asdict(repository.update_recurring_cash_flow(flow_id, **payload.model_dump(exclude_unset=True)))
        except ValueError as exc:
            status = 404 if "does not exist" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @app.delete("/api/recurring-cash-flows/{flow_id}", status_code=204)
    @app.delete("/api/forecast/recurring-cash-flows/{flow_id}", status_code=204, include_in_schema=False)
    def delete_recurring_cash_flow(flow_id: str):
        try:
            repository.delete_recurring_cash_flow(flow_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/life-events", status_code=201)
    @app.post("/api/forecast/life-events", status_code=201, include_in_schema=False)
    def create_life_event(payload: LifeEventCreate):
        try:
            return asdict(repository.create_life_event(**payload.model_dump()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/life-events")
    @app.get("/api/forecast/life-events", include_in_schema=False)
    def list_life_events():
        return [asdict(event) for event in repository.list_life_events()]

    @app.patch("/api/life-events/{event_id}")
    @app.patch("/api/forecast/life-events/{event_id}", include_in_schema=False)
    def update_life_event(event_id: str, payload: LifeEventUpdate):
        try:
            return asdict(repository.update_life_event(event_id, **payload.model_dump(exclude_unset=True)))
        except ValueError as exc:
            status = 404 if "does not exist" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @app.delete("/api/life-events/{event_id}", status_code=204)
    @app.delete("/api/forecast/life-events/{event_id}", status_code=204, include_in_schema=False)
    def delete_life_event(event_id: str):
        try:
            repository.delete_life_event(event_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/forecast-scenarios", status_code=201)
    @app.post("/api/forecast/scenarios", status_code=201, include_in_schema=False)
    def create_forecast_scenario(payload: ForecastScenarioCreate):
        try:
            return asdict(repository.create_forecast_scenario(**payload.model_dump()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/forecast-scenarios")
    @app.get("/api/forecast/scenarios", include_in_schema=False)
    def list_forecast_scenarios():
        return [asdict(scenario) for scenario in repository.list_forecast_scenarios()]

    @app.patch("/api/forecast-scenarios/{scenario_id}")
    @app.patch("/api/forecast/scenarios/{scenario_id}", include_in_schema=False)
    def update_forecast_scenario(scenario_id: str, payload: ForecastScenarioUpdate):
        try:
            return asdict(repository.update_forecast_scenario(scenario_id, **payload.model_dump(exclude_unset=True)))
        except ValueError as exc:
            status = 404 if "does not exist" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @app.delete("/api/forecast-scenarios/{scenario_id}", status_code=204)
    @app.delete("/api/forecast/scenarios/{scenario_id}", status_code=204, include_in_schema=False)
    def delete_forecast_scenario(scenario_id: str):
        try:
            repository.delete_forecast_scenario(scenario_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def serialize_forecast(result):
        payload = asdict(result)
        payload["period"] = {
            "start_date": result.start_date,
            "end_date": result.end_date,
            "years": result.period_years,
            "months": result.period_months,
        }
        return payload

    @app.post("/api/forecasts")
    @app.post("/api/forecast", include_in_schema=False)
    def create_forecast(payload: ForecastRequest):
        try:
            result = forecast_engine.forecast(
                start_date=payload.start_date,
                period_years=payload.resolved_years(),
                scenario_name=payload.resolved_name(),
                overrides=payload.resolved_overrides(),
            )
            return serialize_forecast(result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/forecasts/compare")
    @app.post("/api/forecasts/sensitivity", include_in_schema=False)
    def compare_forecasts(payload: ForecastComparisonRequest):
        cases = payload.resolved_cases()
        if not cases:
            raise HTTPException(status_code=422, detail="cases must contain at least one scenario")
        try:
            results = [
                forecast_engine.forecast(
                    start_date=payload.start_date,
                    period_years=payload.resolved_years(),
                    scenario_name=case.resolved_name(index),
                    overrides=case.resolved_overrides(),
                )
                for index, case in enumerate(cases)
            ]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        serialized = [serialize_forecast(result) for result in results]
        return {
            "source": "forecast",
            "period_years": payload.resolved_years(),
            "start_date": results[0].start_date,
            "end_date": results[0].end_date,
            "cases": serialized,
            "scenarios": serialized,
        }

    @app.post("/api/journal-entries", status_code=201)
    def create_journal_entry(payload: JournalEntryCreate):
        try:
            return asdict(repository.create_journal_entry(**payload.model_dump()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/journal-entries")
    def list_journal_entries(start: date | None = Query(default=None), end: date | None = Query(default=None)):
        return [asdict(entry) for entry in repository.list_journal_entries(start, end)]

    @app.post("/api/import/journal")
    def import_journal(csv_text: str = Body(media_type="text/csv")):
        try:
            rows = list(csv.DictReader(io.StringIO(csv_text)))
            required = {"booked_on", "debit_account_id", "credit_account_id", "amount"}
            if not rows or not required.issubset(rows[0]):
                raise ValueError("CSV requires booked_on,debit_account_id,credit_account_id,amount columns")
            imported, skipped = repository.import_journal_csv(rows)
            return {"imported": imported, "skipped": skipped}
        except (ValueError, csv.Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reconciliation")
    def reconcile(payload: ReconciliationRequest):
        accounts = {account.id for account in repository.list_accounts()}
        if payload.account_id not in accounts:
            raise HTTPException(status_code=404, detail="account does not exist")
        ledger_balance = repository.account_balances()[payload.account_id]
        difference = payload.statement_balance - ledger_balance
        return {"account_id": payload.account_id, "ledger_balance": ledger_balance,
                "statement_balance": payload.statement_balance, "difference": difference,
                "status": "reconciled" if difference == 0 else "difference_found"}

    @app.post("/api/tax/estimate")
    def estimate_tax(payload: SalaryTaxEstimateRequest):
        if payload.tax_year < 1900 or payload.tax_year > 2100:
            raise HTTPException(status_code=400, detail="tax_year must be between 1900 and 2100")
        if payload.gross_salary < 0:
            raise HTTPException(status_code=400, detail="gross_salary must not be negative")
        if payload.social_insurance_premiums < 0:
            raise HTTPException(status_code=400, detail="social_insurance_premiums must not be negative")
        try:
            return asdict(estimate_salary_tax(**payload.model_dump()))
        except LookupError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/reports/monthly")
    def monthly_report(year: int = Query(ge=2000, le=2100), month: int = Query(ge=1, le=12)):
        from calendar import monthrange
        start = date(year, month, 1)
        end = date(year, month, monthrange(year, month)[1])
        accounts = {a.id: a for a in repository.list_accounts()}
        income = expense = 0
        for entry in repository.list_journal_entries(start, end):
            debit = accounts[entry.debit_account_id]
            credit = accounts[entry.credit_account_id]
            if debit.account_type == AccountType.EXPENSE: expense += entry.amount
            if credit.account_type == AccountType.INCOME: income += entry.amount
        balances = repository.account_balances()
        assets = sum(balances[a.id] for a in accounts.values() if a.account_type in {AccountType.CASH, AccountType.BANK, AccountType.INVESTMENT})
        liabilities = sum(balances[a.id] for a in accounts.values() if a.account_type == AccountType.LIABILITY)
        return {"year": year, "month": month, "income": income, "expenses": expense,
                "net_income": income - expense, "assets": assets, "liabilities": liabilities,
                "equity": assets - liabilities}

    @app.get("/api/summary")
    def summary(
        start: date | None = Query(default=None),
        end: date | None = Query(default=None),
    ):
        return asdict(analyzer.summarize(start=start, end=end))

    @app.post("/api/scenarios/purchase")
    def purchase_scenario(payload: PurchaseScenarioRequest):
        try:
            scenario = simulate_purchase(analyzer.summarize(), **payload.model_dump())
            return asdict(scenario)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/ai/queries")
    def ai_query(payload: AIQueryRequest):
        try:
            return ai_cfo.query(payload.resolved_text())
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_query_request",
                    "message": str(exc),
                    "details": {},
                    "audit_id": None,
                    "action_id": None,
                },
            ) from exc

    @app.post("/api/ai/confirm")
    def ai_confirm(payload: AIConfirmRequest):
        try:
            return ai_cfo.confirm(payload.resolved_token())
        except AIActionError as exc:
            detail: dict[str, object] = {
                "code": exc.code,
                "message": str(exc),
                "details": exc.details,
                "audit_id": exc.audit_id,
                "action_id": exc.action_id,
            }
            raise HTTPException(status_code=exc.status_code, detail=detail) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_confirmation_request",
                    "message": str(exc),
                    "details": {},
                    "audit_id": None,
                    "action_id": None,
                },
            ) from exc

    @app.get("/api/ai/audit")
    def ai_audit(limit: int = Query(default=100, ge=1, le=500)):
        return ai_cfo.audit(limit)

    return app


app = create_app()
