from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
from math import isfinite

from .models import (
    CashFlowType,
    ForecastMonth,
    ForecastResult,
    ForecastScenario,
    ForecastYear,
    LifeEvent,
    LifeEventType,
    RecurringCashFlow,
)
from .repository import FinanceRepository


DEFAULT_FORECAST_YEARS = 30
MIN_FORECAST_YEARS = 1
MAX_FORECAST_YEARS = 50


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + (value.month - 1) + months
    year, month_zero_based = divmod(month_index, 12)
    return date(year, month_zero_based + 1, 1)


def _month_end(value: date) -> date:
    return date(value.year, value.month, monthrange(value.year, value.month)[1])


def _round_yen(value: float) -> int:
    # Forecast amounts are integer yen. This also makes negative balances
    # deterministic instead of depending on presentation-layer formatting.
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def future_value(principal: float, monthly_contribution: float, months: int, annual_rate: float) -> float:
    """Return a deterministic future value using monthly compounding."""
    if months <= 0:
        return principal
    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        return principal + monthly_contribution * months
    growth = (1 + monthly_rate) ** months
    return principal * growth + monthly_contribution * ((growth - 1) / monthly_rate)


class ForecastEngine:
    """Build reproducible monthly cash-flow projections from repository data."""

    def __init__(self, repository: FinanceRepository) -> None:
        self.repository = repository

    def forecast(
        self,
        start_date: date | None = None,
        period_years: int = DEFAULT_FORECAST_YEARS,
        scenario_name: str = "Base case",
        overrides: dict[str, int | float | None] | None = None,
    ) -> ForecastResult:
        if not MIN_FORECAST_YEARS <= period_years <= MAX_FORECAST_YEARS:
            raise ValueError(f"period_years must be between {MIN_FORECAST_YEARS} and {MAX_FORECAST_YEARS}")
        scenario_name = scenario_name.strip()
        if not scenario_name:
            raise ValueError("scenario_name must not be empty")

        start = _month_start(start_date or date.today())
        stored = self.repository.get_forecast_scenario_by_name(scenario_name)
        effective_overrides = dict(overrides or {})
        if effective_overrides.get("initial_balance") is None and (stored is None or stored.initial_balance is None):
            effective_overrides["initial_balance"] = self.repository.forecast_initial_balance(start - timedelta(days=1))
        assumptions = self._resolve_assumptions(stored, start, effective_overrides)
        months = period_years * 12
        recurring = self.repository.list_recurring_cash_flows()
        events = self.repository.list_life_events()

        balance = int(assumptions["initial_balance"])
        monthly: list[ForecastMonth] = []
        for month_index in range(months):
            month = _add_months(start, month_index)
            growth_income = (1 + float(assumptions["income_growth_rate"])) ** (month_index / 12)
            growth_expense = (1 + float(assumptions["expense_growth_rate"])) ** (month_index / 12)
            income = sum(
                _round_yen(flow.amount * growth_income)
                for flow in recurring
                if flow.flow_type == CashFlowType.INCOME and self._flow_active(flow, month)
            )
            expenses = sum(
                _round_yen(flow.amount * growth_expense)
                for flow in recurring
                if flow.flow_type == CashFlowType.EXPENSE and self._flow_active(flow, month)
            )
            for event in events:
                if self._event_active(event, month):
                    income += event.income_delta
                    expenses += event.expense_delta
            expenses = max(0, expenses)
            net_cash_flow = income - expenses
            interest = _round_yen(balance * float(assumptions["annual_return_rate"]) / 12)
            starting_balance = balance
            balance = starting_balance + net_cash_flow + interest
            monthly.append(
                ForecastMonth(
                    month=month,
                    starting_balance=starting_balance,
                    income=income,
                    expenses=expenses,
                    net_cash_flow=net_cash_flow,
                    interest=interest,
                    ending_balance=balance,
                )
            )

        annual = self._annualize(monthly)
        return ForecastResult(
            scenario_name=scenario_name,
            source="forecast",
            start_date=start,
            end_date=_month_end(monthly[-1].month),
            period_years=period_years,
            period_months=months,
            initial_balance=int(assumptions["initial_balance"]),
            assumptions=assumptions,
            monthly=monthly,
            annual=annual,
            ending_balance=balance,
            minimum_balance=min([int(assumptions["initial_balance"])] + [item.ending_balance for item in monthly]),
        )

    @staticmethod
    def _resolve_assumptions(
        stored: ForecastScenario | None,
        start: date,
        overrides: dict[str, int | float | None],
    ) -> dict[str, int | float]:
        defaults: dict[str, int | float] = {
            "initial_balance": 0,
            "income_growth_rate": 0.0,
            "expense_growth_rate": 0.0,
            "annual_return_rate": 0.0,
        }
        if stored is not None:
            defaults.update(
                initial_balance=stored.initial_balance if stored.initial_balance is not None else 0,
                income_growth_rate=stored.income_growth_rate,
                expense_growth_rate=stored.expense_growth_rate,
                annual_return_rate=stored.annual_return_rate,
            )
        for key in defaults:
            value = overrides.get(key)
            if value is not None:
                defaults[key] = int(value) if key == "initial_balance" else float(value)
        for key in ("income_growth_rate", "expense_growth_rate", "annual_return_rate"):
            value = float(defaults[key])
            if not isfinite(value) or value <= -1:
                raise ValueError(f"{key} must be finite and greater than -1")
        return defaults

    def forecast_with_defaults(
        self,
        start_date: date | None = None,
        period_years: int = DEFAULT_FORECAST_YEARS,
        scenario_name: str = "Base case",
        overrides: dict[str, int | float | None] | None = None,
    ) -> ForecastResult:
        """Same as forecast, using the current liquid ledger balance as the base."""
        start = _month_start(start_date or date.today())
        merged = dict(overrides or {})
        if merged.get("initial_balance") is None:
            merged["initial_balance"] = self.repository.forecast_initial_balance(start - timedelta(days=1))
        return self.forecast(start, period_years, scenario_name, merged)

    @staticmethod
    def _flow_active(flow: RecurringCashFlow, month: date) -> bool:
        return _month_start(flow.start_date) <= month and (
            flow.end_date is None or month <= _month_start(flow.end_date)
        )

    @staticmethod
    def _event_active(event: LifeEvent, month: date) -> bool:
        start = _month_start(event.start_date)
        if event.event_type == LifeEventType.ONE_TIME:
            return month == start
        return start <= month < _add_months(start, event.duration_months)

    @staticmethod
    def _annualize(monthly: list[ForecastMonth]) -> list[ForecastYear]:
        annual: list[ForecastYear] = []
        for year in sorted({item.month.year for item in monthly}):
            rows = [item for item in monthly if item.month.year == year]
            annual.append(
                ForecastYear(
                    year=year,
                    income=sum(item.income for item in rows),
                    expenses=sum(item.expenses for item in rows),
                    net_cash_flow=sum(item.net_cash_flow for item in rows),
                    interest=sum(item.interest for item in rows),
                    ending_balance=rows[-1].ending_balance,
                )
            )
        return annual
