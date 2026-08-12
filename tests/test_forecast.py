from datetime import date

from fastapi.testclient import TestClient

from personal_finance_os.api import create_app
from personal_finance_os.forecast import ForecastEngine
from personal_finance_os.models import AccountType, CashFlowType, LifeEventType
from personal_finance_os.repository import FinanceRepository


def test_forecast_expands_recurring_flows_and_events_at_month_boundaries(tmp_path):
    repo = FinanceRepository(tmp_path / "forecast.db")
    bank = repo.create_account("Bank", AccountType.BANK, 1_000_000)
    repo.create_recurring_cash_flow("salary", CashFlowType.INCOME, 300_000, date(2026, 1, 15))
    repo.create_recurring_cash_flow("rent", CashFlowType.EXPENSE, 100_000, date(2026, 1, 1), date(2026, 3, 31))
    repo.create_life_event("bonus", date(2026, 1, 20), 6, LifeEventType.ONE_TIME, income_delta=50_000)
    repo.create_life_event("childcare", date(2026, 2, 1), 2, LifeEventType.RECURRING, expense_delta=25_000)
    repo.create_life_event("overlap", date(2026, 2, 28), 1, LifeEventType.ONE_TIME, expense_delta=10_000)

    result = ForecastEngine(repo).forecast(date(2026, 1, 1), period_years=1)

    assert result.initial_balance == 1_000_000
    assert result.monthly[0].income == 350_000
    assert result.monthly[0].expenses == 100_000
    assert result.monthly[1].expenses == 135_000
    assert result.monthly[2].expenses == 125_000
    assert result.monthly[3].expenses == 0
    assert result.monthly[0].ending_balance == 1_250_000
    assert result.monthly[1].ending_balance == 1_415_000
    assert result.annual[0].income == sum(row.income for row in result.monthly)
    assert result.annual[0].expenses == sum(row.expenses for row in result.monthly)
    assert result.annual[0].net_cash_flow == sum(row.net_cash_flow for row in result.monthly)


def test_forecast_handles_zero_cashflow_negative_balance_and_period_limits(tmp_path):
    repo = FinanceRepository(tmp_path / "forecast.db")
    repo.create_account("Bank", AccountType.BANK, 0)
    repo.create_recurring_cash_flow("expense", CashFlowType.EXPENSE, 10_000, date(2026, 1, 1))

    result = ForecastEngine(repo).forecast(date(2026, 1, 1), period_years=1)
    long_result = ForecastEngine(repo).forecast(date(2026, 1, 1), period_years=50)

    assert result.monthly[0].income == 0
    assert result.monthly[0].expenses == 10_000
    assert result.monthly[0].ending_balance == -10_000
    assert result.minimum_balance == -120_000
    assert len(long_result.monthly) == 600
    assert len(long_result.annual) == 50


def test_saved_scenario_and_overrides_are_reflected_in_result(tmp_path):
    repo = FinanceRepository(tmp_path / "forecast.db")
    repo.create_recurring_cash_flow("salary", CashFlowType.INCOME, 100_000, date(2026, 1, 1))
    repo.create_forecast_scenario("growth", initial_balance=500_000, income_growth_rate=0.12, annual_return_rate=0.06)

    result = ForecastEngine(repo).forecast(
        date(2026, 1, 1), period_years=1, scenario_name="growth", overrides={"initial_balance": 1_000_000}
    )

    assert result.scenario_name == "growth"
    assert result.initial_balance == 1_000_000
    assert result.assumptions["income_growth_rate"] == 0.12
    assert result.assumptions["annual_return_rate"] == 0.06
    assert result.monthly[-1].income > result.monthly[0].income


def test_forecast_models_persist_update_delete_and_reload(tmp_path):
    database = tmp_path / "forecast.db"
    repo = FinanceRepository(database)
    flow = repo.create_recurring_cash_flow("old", CashFlowType.EXPENSE, 10_000, date(2026, 1, 1))
    event = repo.create_life_event("old event", date(2026, 1, 1))
    scenario = repo.create_forecast_scenario("old case")
    repo.update_recurring_cash_flow(flow.id, name="new", amount=20_000)
    repo.update_life_event(event.id, name="new event", expense_delta=3_000)
    repo.update_forecast_scenario(scenario.id, name="new case", annual_return_rate=0.02)

    reloaded = FinanceRepository(database)
    assert reloaded.get_recurring_cash_flow(flow.id).amount == 20_000
    assert reloaded.get_life_event(event.id).expense_delta == 3_000
    assert reloaded.get_forecast_scenario(scenario.id).annual_return_rate == 0.02
    reloaded.delete_recurring_cash_flow(flow.id)
    reloaded.delete_life_event(event.id)
    reloaded.delete_forecast_scenario(scenario.id)
    assert reloaded.list_recurring_cash_flows() == []
    assert reloaded.list_life_events() == []
    assert reloaded.list_forecast_scenarios() == []


def test_forecast_api_crud_forecast_and_comparison(tmp_path):
    client = TestClient(create_app(str(tmp_path / "forecast.db")))
    assert client.get("/").status_code == 200
    assert "Financial Digital Twin" in client.get("/").text
    flow = client.post(
        "/api/recurring-cash-flows",
        json={"name": "salary", "flow_type": "income", "amount": 300000, "start_date": "2026-01-01"},
    )
    assert flow.status_code == 201
    flow_id = flow.json()["id"]
    assert client.patch(f"/api/recurring-cash-flows/{flow_id}", json={"amount": 320000}).status_code == 200
    event = client.post(
        "/api/life-events",
        json={"name": "move", "start_date": "2026-02-01", "event_type": "one_time", "expense_delta": 200000},
    )
    assert event.status_code == 201
    scenario = client.post(
        "/api/forecast-scenarios", json={"name": "stress", "expense_growth_rate": 0.1}
    )
    assert scenario.status_code == 201

    forecast = client.post(
        "/api/forecasts",
        json={"start_date": "2026-01-01", "years": 1, "scenario_name": "stress"},
    )
    assert forecast.status_code == 200
    body = forecast.json()
    assert body["source"] == "forecast"
    assert body["period"]["months"] == 12
    assert body["assumptions"]["expense_growth_rate"] == 0.1
    assert len(body["monthly"]) == 12
    assert len(body["annual"]) == 1

    comparison = client.post(
        "/api/forecasts/compare",
        json={"start_date": "2026-01-01", "period_years": 1, "cases": [{"name": "base"}, {"name": "stress"}]},
    )
    assert comparison.status_code == 200
    assert len(comparison.json()["cases"]) == 2
    assert client.delete(f"/api/recurring-cash-flows/{flow_id}").status_code == 204
