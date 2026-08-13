from personal_finance_os.models import FinancialSummary
from personal_finance_os.scenario import simulate_purchase


def test_purchase_scenario_quantifies_opportunity_cost():
    summary = FinancialSummary(
        income=300_000,
        expenses=200_000,
        net_cash_flow=100_000,
        savings_rate=0.3333,
        assets=4_000_000,
        liabilities=0,
        net_worth=4_000_000,
        liquid_assets=2_000_000,
        monthly_expenses=200_000,
        monthly_net_cash_flow=100_000,
        emergency_fund_months=10,
        health_score=90,
    )

    result = simulate_purchase(summary, price=300_000, horizon_months=60, annual_return_rate=0.04)

    assert result.purchase_future_assets < result.baseline_future_assets
    assert result.opportunity_cost > 300_000
    assert result.post_purchase_liquid_assets == 1_700_000
    assert result.post_purchase_emergency_months == 8.5
    assert 0 <= result.affordability_score <= 100
