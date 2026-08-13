from __future__ import annotations

from datetime import date

from .models import AccountType, FinancialSummary, TransactionKind
from .repository import FinanceRepository


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


class FinanceAnalyzer:
    def __init__(self, repository: FinanceRepository) -> None:
        self.repository = repository

    def summarize(self, start: date | None = None, end: date | None = None) -> FinancialSummary:
        transactions = self.repository.list_transactions(start=start, end=end)
        accounts = self.repository.list_accounts()
        balances = self.repository.account_balances()

        income = sum(t.amount for t in transactions if t.kind == TransactionKind.INCOME)
        expenses = -sum(t.amount for t in transactions if t.kind == TransactionKind.EXPENSE)
        net_cash_flow = income - expenses
        savings_rate = (net_cash_flow / income) if income > 0 else 0.0

        assets = 0
        liabilities = 0
        liquid_assets = 0
        for account in accounts:
            balance = balances.get(account.id, account.opening_balance)
            if account.account_type == AccountType.LIABILITY:
                liabilities += max(0, balance)
            else:
                assets += balance
                if account.account_type in {AccountType.CASH, AccountType.BANK}:
                    liquid_assets += balance

        net_worth = assets - liabilities
        period_months = self._period_months(transactions, start, end)
        monthly_expenses = (expenses / period_months) if period_months else 0.0
        monthly_net_cash_flow = (net_cash_flow / period_months) if period_months else 0.0
        emergency_months = liquid_assets / monthly_expenses if monthly_expenses > 0 else None
        health_score = self._health_score(
            savings_rate=savings_rate,
            emergency_months=emergency_months,
            liabilities=liabilities,
            assets=assets,
        )

        return FinancialSummary(
            income=income,
            expenses=expenses,
            net_cash_flow=net_cash_flow,
            savings_rate=round(savings_rate, 4),
            assets=assets,
            liabilities=liabilities,
            net_worth=net_worth,
            liquid_assets=liquid_assets,
            monthly_expenses=round(monthly_expenses, 2),
            monthly_net_cash_flow=round(monthly_net_cash_flow, 2),
            emergency_fund_months=None if emergency_months is None else round(emergency_months, 2),
            health_score=health_score,
        )

    @staticmethod
    def _period_months(transactions, start: date | None, end: date | None) -> float:
        if start is not None and end is not None:
            days = max(1, (end - start).days + 1)
            return max(1.0, days / 30.4375)
        if not transactions:
            return 0.0
        first = min(t.booked_on for t in transactions)
        last = max(t.booked_on for t in transactions)
        days = max(1, (last - first).days + 1)
        return max(1.0, days / 30.4375)

    @staticmethod
    def _health_score(
        savings_rate: float,
        emergency_months: float | None,
        liabilities: int,
        assets: int,
    ) -> int:
        savings_component = _clamp((savings_rate / 0.30) * 100) if savings_rate > 0 else 0
        emergency_component = (
            100.0
            if emergency_months is None and assets > 0
            else _clamp(((emergency_months or 0.0) / 6.0) * 100)
        )
        debt_ratio = liabilities / max(1, assets)
        debt_component = _clamp(100 - (debt_ratio * 100))
        net_worth_component = 100.0 if assets - liabilities >= 0 else 0.0

        weighted = (
            savings_component * 0.35
            + emergency_component * 0.30
            + debt_component * 0.20
            + net_worth_component * 0.15
        )
        return round(_clamp(weighted))
