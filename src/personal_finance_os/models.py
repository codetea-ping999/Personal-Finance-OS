from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class AccountType(StrEnum):
    CASH = "cash"
    BANK = "bank"
    INVESTMENT = "investment"
    LIABILITY = "liability"
    EQUITY = "equity"
    INCOME = "income"
    EXPENSE = "expense"


class TransactionKind(StrEnum):
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"
    ADJUSTMENT = "adjustment"


class CashFlowType(StrEnum):
    INCOME = "income"
    EXPENSE = "expense"


class LifeEventType(StrEnum):
    ONE_TIME = "one_time"
    RECURRING = "recurring"


@dataclass(frozen=True, slots=True)
class Account:
    id: str
    name: str
    account_type: AccountType
    opening_balance: int = 0


@dataclass(frozen=True, slots=True)
class Transaction:
    id: str
    account_id: str
    booked_on: date
    amount: int
    kind: TransactionKind
    category: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class JournalEntry:
    id: str
    booked_on: date
    description: str
    debit_account_id: str
    credit_account_id: str
    amount: int
    external_id: str | None = None


@dataclass(frozen=True, slots=True)
class RecurringCashFlow:
    id: str
    name: str
    flow_type: CashFlowType
    amount: int
    start_date: date
    end_date: date | None = None


@dataclass(frozen=True, slots=True)
class LifeEvent:
    id: str
    name: str
    start_date: date
    duration_months: int
    event_type: LifeEventType
    income_delta: int = 0
    expense_delta: int = 0


@dataclass(frozen=True, slots=True)
class ForecastScenario:
    id: str
    name: str
    initial_balance: int | None = None
    income_growth_rate: float = 0.0
    expense_growth_rate: float = 0.0
    annual_return_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class FinancialSummary:
    income: int
    expenses: int
    net_cash_flow: int
    savings_rate: float
    assets: int
    liabilities: int
    net_worth: int
    liquid_assets: int
    monthly_expenses: float
    monthly_net_cash_flow: float
    emergency_fund_months: float | None
    health_score: int


@dataclass(frozen=True, slots=True)
class PurchaseScenario:
    price: int
    horizon_months: int
    annual_return_rate: float
    baseline_future_assets: int
    purchase_future_assets: int
    opportunity_cost: int
    post_purchase_liquid_assets: int
    post_purchase_emergency_months: float | None
    affordability_score: int
    verdict: str


@dataclass(frozen=True, slots=True)
class ForecastMonth:
    month: date
    starting_balance: int
    income: int
    expenses: int
    net_cash_flow: int
    interest: int
    ending_balance: int


@dataclass(frozen=True, slots=True)
class ForecastYear:
    year: int
    income: int
    expenses: int
    net_cash_flow: int
    interest: int
    ending_balance: int


@dataclass(frozen=True, slots=True)
class ForecastResult:
    scenario_name: str
    source: str
    start_date: date
    end_date: date
    period_years: int
    period_months: int
    initial_balance: int
    assumptions: dict[str, int | float]
    monthly: list[ForecastMonth]
    annual: list[ForecastYear]
    ending_balance: int
    minimum_balance: int
