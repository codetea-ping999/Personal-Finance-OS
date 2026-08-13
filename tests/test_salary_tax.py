import json

import pytest

from personal_finance_os.tax_engine import estimate_salary_tax, load_salary_tax_rules


def test_salary_tax_uses_versioned_rules_and_returns_provenance():
    result = estimate_salary_tax(2025, 6_000_000, 900_000)

    assert result.employment_income_deduction == 1_640_000
    assert result.employment_income == 4_360_000
    assert result.basic_deduction == 680_000
    assert result.taxable_income == 2_780_000
    assert result.income_tax == 180_500
    assert result.reconstruction_special_tax == 3_790
    assert result.total_tax == 184_290
    assert result.rule_version == "jp-income-tax-2025-v1"
    assert result.sources[0]["url"].startswith("https://www.nta.go.jp/")


def test_2026_employment_income_boundaries_and_zero_income():
    assert estimate_salary_tax(2026, 0).employment_income == 0
    assert estimate_salary_tax(2026, 690_000).employment_income == 0
    assert estimate_salary_tax(2026, 741_000).employment_income == 0
    assert estimate_salary_tax(2026, 741_001).employment_income == 1_001


def test_taxable_income_is_truncated_to_thousand_yen():
    result = estimate_salary_tax(2025, 5_000_000, 1_234)
    assert result.taxable_income % 1000 == 0


def test_unsupported_year_and_invalid_rule_file(tmp_path):
    with pytest.raises(LookupError):
        estimate_salary_tax(2099, 1_000_000)

    rule_path = tmp_path / "2025.json"
    rule_path.write_text(json.dumps({"tax_year": 2025}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_salary_tax_rules(2025, tmp_path)


def test_negative_salary_is_rejected():
    with pytest.raises(ValueError):
        estimate_salary_tax(2025, -1)
