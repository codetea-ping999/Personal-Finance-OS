from __future__ import annotations

import os
from dataclasses import asdict
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .analytics import FinanceAnalyzer
from .models import AccountType, TransactionKind
from .repository import FinanceRepository
from .scenario import simulate_purchase


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


def create_app(database: str | None = None) -> FastAPI:
    repository = FinanceRepository(database or os.getenv("PFOS_DATABASE", "personal_finance.db"))
    analyzer = FinanceAnalyzer(repository)
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

    return app


app = create_app()
