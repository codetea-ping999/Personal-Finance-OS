from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class AccountType(StrEnum):
    CASH = "cash"
    BANK = "bank"
    INVESTMENT = "investment"
    LIABILITY = "liability"


class TransactionKind(StrEnum):
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"
    ADJUSTMENT = "adjustment"


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
