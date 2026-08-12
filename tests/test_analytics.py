from datetime import date

from personal_finance_os.analytics import FinanceAnalyzer
from personal_finance_os.models import AccountType, TransactionKind
from personal_finance_os.repository import FinanceRepository


def test_financial_summary_is_explainable(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    bank = repo.create_account("Bank", AccountType.BANK, 1_000_000)
    repo.create_account("Loan", AccountType.LIABILITY, 200_000)
    repo.create_transaction(bank.id, date(2026, 8, 1), 300_000, TransactionKind.INCOME, "salary")
    repo.create_transaction(bank.id, date(2026, 8, 2), -100_000, TransactionKind.EXPENSE, "living")

    summary = FinanceAnalyzer(repo).summarize()

    assert summary.income == 300_000
    assert summary.expenses == 100_000
    assert summary.net_cash_flow == 200_000
    assert summary.monthly_net_cash_flow == 200_000
    assert summary.assets == 1_200_000
    assert summary.liabilities == 200_000
    assert summary.net_worth == 1_000_000
    assert 0 <= summary.health_score <= 100
