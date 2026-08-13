from personal_finance_os.tax_engine import TaxBracket, TaxRuleSet, estimate_progressive_tax


def test_tax_engine_uses_versioned_external_rule_set():
    rules = TaxRuleSet(
        name="example-v1",
        brackets=(TaxBracket(1_000_000, 0.10), TaxBracket(None, 0.20)),
    )
    assert estimate_progressive_tax(1_500_000, rules) == 200_000
