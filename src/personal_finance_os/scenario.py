from __future__ import annotations

from .models import FinancialSummary, PurchaseScenario
from .forecast import future_value


def simulate_purchase(
    summary: FinancialSummary,
    price: int,
    horizon_months: int = 60,
    annual_return_rate: float = 0.04,
) -> PurchaseScenario:
    if price <= 0:
        raise ValueError("price must be positive")
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive")
    if annual_return_rate <= -1:
        raise ValueError("annual_return_rate must be greater than -1")

    monthly_surplus = summary.monthly_net_cash_flow
    baseline = future_value(summary.assets, monthly_surplus, horizon_months, annual_return_rate)
    after_purchase_assets = max(0, summary.assets - price)
    purchase = future_value(after_purchase_assets, monthly_surplus, horizon_months, annual_return_rate)
    opportunity_cost = max(0, round(baseline - purchase))
    post_purchase_liquid = max(0, summary.liquid_assets - price)
    emergency_months = (
        post_purchase_liquid / summary.monthly_expenses
        if summary.monthly_expenses > 0
        else None
    )

    if emergency_months is None:
        liquidity_score = 100 if post_purchase_liquid >= 0 else 0
    else:
        liquidity_score = min(100, max(0, (emergency_months / 6) * 100))
    price_to_assets = price / max(1, summary.assets)
    concentration_score = max(0, 100 - price_to_assets * 100)
    affordability_score = round(liquidity_score * 0.7 + concentration_score * 0.3)

    if affordability_score >= 80:
        verdict = "comfortable"
    elif affordability_score >= 60:
        verdict = "manageable"
    elif affordability_score >= 40:
        verdict = "caution"
    else:
        verdict = "not_recommended"

    return PurchaseScenario(
        price=price,
        horizon_months=horizon_months,
        annual_return_rate=annual_return_rate,
        baseline_future_assets=round(baseline),
        purchase_future_assets=round(purchase),
        opportunity_cost=opportunity_cost,
        post_purchase_liquid_assets=post_purchase_liquid,
        post_purchase_emergency_months=None if emergency_months is None else round(emergency_months, 2),
        affordability_score=affordability_score,
        verdict=verdict,
    )
