from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaxBracket:
    upper_bound: int | None
    rate: float


@dataclass(frozen=True, slots=True)
class TaxRuleSet:
    name: str
    brackets: tuple[TaxBracket, ...]


def estimate_progressive_tax(taxable_income: int, rules: TaxRuleSet) -> int:
    """Estimate progressive tax from an explicitly supplied rule set.

    Personal Finance OS intentionally does not ship current tax rates in code.
    Rules change over time and should be versioned from authoritative sources.
    """
    if taxable_income <= 0:
        return 0
    remaining = taxable_income
    lower = 0
    tax = 0.0

    for bracket in rules.brackets:
        if bracket.rate < 0 or bracket.rate > 1:
            raise ValueError("tax bracket rate must be between 0 and 1")
        if bracket.upper_bound is None:
            tax += remaining * bracket.rate
            remaining = 0
            break
        width = bracket.upper_bound - lower
        if width <= 0:
            raise ValueError("tax brackets must be strictly increasing")
        taxable_here = min(remaining, width)
        tax += taxable_here * bracket.rate
        remaining -= taxable_here
        lower = bracket.upper_bound
        if remaining <= 0:
            break

    if remaining > 0:
        raise ValueError("tax rule set must include an open-ended final bracket")
    return round(tax)
