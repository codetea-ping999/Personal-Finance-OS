from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TaxBracket:
    upper_bound: int | None
    rate: float


@dataclass(frozen=True, slots=True)
class TaxRuleSet:
    name: str
    brackets: tuple[TaxBracket, ...]


@dataclass(frozen=True, slots=True)
class SalaryTaxEstimate:
    tax_year: int
    gross_salary: int
    employment_income_deduction: int
    employment_income: int
    social_insurance_premiums: int
    basic_deduction: int
    taxable_income: int
    income_tax: int
    reconstruction_special_tax: int
    total_tax: int
    rule_version: str
    sources: tuple[dict[str, Any], ...]


def _rules_path(tax_year: int, rules_dir: str | Path | None = None) -> Path:
    if rules_dir is not None:
        return Path(rules_dir) / f"{tax_year}.json"
    relative = Path("data") / "tax" / "jp" / f"{tax_year}.json"
    candidates = (
        Path(__file__).resolve().parents[2] / relative,
        Path(__file__).resolve().parents[1] / relative,
        Path.cwd() / relative,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def load_salary_tax_rules(tax_year: int, rules_dir: str | Path | None = None) -> dict[str, Any]:
    path = _rules_path(tax_year, rules_dir)
    if not path.exists():
        raise LookupError(f"unsupported tax year: {tax_year}")
    try:
        rules = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tax rule file: {path.name}") from exc
    _validate_salary_tax_rules(rules, tax_year)
    return rules


def _validate_salary_tax_rules(rules: dict[str, Any], tax_year: int) -> None:
    required = {"jurisdiction", "tax_year", "version", "effective_from", "employment_income_deduction",
                "basic_deductions", "income_tax_brackets", "reconstruction_special_tax_rate", "rounding", "sources"}
    if not isinstance(rules, dict) or not required.issubset(rules):
        raise ValueError("tax rule set is missing required fields")
    if rules["jurisdiction"] != "JP" or rules["tax_year"] != tax_year:
        raise ValueError("tax rule jurisdiction or year mismatch")
    brackets = rules["income_tax_brackets"]
    if not brackets or brackets[-1].get("upper_bound") is not None:
        raise ValueError("tax rule set must include an open-ended final bracket")
    previous = -1
    for bracket in brackets:
        upper = bracket.get("upper_bound")
        if upper is not None and upper <= previous:
            raise ValueError("income tax brackets must be strictly increasing")
        previous = upper if upper is not None else previous
        if not 0 <= float(bracket.get("rate", -1)) <= 1:
            raise ValueError("tax bracket rate must be between 0 and 1")
    for rows_key in ("employment_income_deduction", "basic_deductions"):
        rows = rules[rows_key]
        if not rows or rows[-1].get("max_salary", rows[-1].get("max_total_income", "missing")) is not None:
            raise ValueError(f"{rows_key} must include an open-ended final row")
    if not isinstance(rules["sources"], list) or not rules["sources"]:
        raise ValueError("tax rule set must include sources")


def _floor(value: Decimal | float | int) -> int:
    return int(Decimal(str(value)).to_integral_value(rounding=ROUND_FLOOR))


def _employment_income(gross_salary: int, rules: dict[str, Any]) -> int:
    for row in rules["employment_income_deduction"]:
        maximum = row.get("max_salary")
        if maximum is None or gross_salary <= maximum:
            kind = row["kind"]
            if kind == "fixed":
                return max(0, gross_salary - row["amount"])
            if kind == "income_fixed":
                return row["amount"]
            if kind == "salary_minus":
                return max(0, gross_salary - row["offset"])
            if kind == "rate_plus":
                deduction = _floor(Decimal(gross_salary) * Decimal(str(row["rate"])) + row["addition"])
                return max(0, gross_salary - deduction)
            raise ValueError(f"unsupported employment rule kind: {kind}")
    raise ValueError("employment income rules are incomplete")


def _basic_deduction(total_income: int, rules: dict[str, Any]) -> int:
    for row in rules["basic_deductions"]:
        maximum = row.get("max_total_income")
        if maximum is None or total_income <= maximum:
            return row["amount"]
    raise ValueError("basic deduction rules are incomplete")


def estimate_salary_tax(
    tax_year: int,
    gross_salary: int,
    social_insurance_premiums: int = 0,
    rules_dir: str | Path | None = None,
) -> SalaryTaxEstimate:
    if gross_salary < 0:
        raise ValueError("gross_salary must not be negative")
    if social_insurance_premiums < 0:
        raise ValueError("social_insurance_premiums must not be negative")
    rules = load_salary_tax_rules(tax_year, rules_dir)
    employment_income = _employment_income(gross_salary, rules)
    employment_deduction = gross_salary - employment_income
    basic_deduction = _basic_deduction(employment_income, rules)
    unit = rules["rounding"]["taxable_income_unit"]
    taxable_income = max(0, employment_income - social_insurance_premiums - basic_deduction)
    taxable_income = (taxable_income // unit) * unit

    income_tax = 0
    for bracket in rules["income_tax_brackets"]:
        if bracket["upper_bound"] is None or taxable_income <= bracket["upper_bound"]:
            income_tax = max(0, _floor(taxable_income * Decimal(str(bracket["rate"])) - bracket["deduction"]))
            break
    special_tax = _floor(Decimal(income_tax) * Decimal(str(rules["reconstruction_special_tax_rate"])))
    return SalaryTaxEstimate(
        tax_year=tax_year,
        gross_salary=gross_salary,
        employment_income_deduction=employment_deduction,
        employment_income=employment_income,
        social_insurance_premiums=social_insurance_premiums,
        basic_deduction=basic_deduction,
        taxable_income=taxable_income,
        income_tax=income_tax,
        reconstruction_special_tax=special_tax,
        total_tax=income_tax + special_tax,
        rule_version=rules["version"],
        sources=tuple(rules["sources"]),
    )


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
