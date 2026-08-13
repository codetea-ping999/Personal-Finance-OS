from datetime import date

import pytest

from personal_finance_os.models import AccountType, TransactionKind
from personal_finance_os.repository import FinanceRepository


def test_repository_creates_accounts_and_transactions(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    account = repo.create_account("Main", AccountType.BANK, 100_000)
    repo.create_transaction(account.id, date(2026, 8, 1), 300_000, TransactionKind.INCOME, "salary")
    repo.create_transaction(account.id, date(2026, 8, 2), -70_000, TransactionKind.EXPENSE, "rent")

    assert repo.account_balances()[account.id] == 330_000
    assert len(repo.list_transactions()) == 2


def test_repository_rejects_wrong_sign(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    account = repo.create_account("Main", AccountType.BANK)
    with pytest.raises(ValueError):
        repo.create_transaction(account.id, date.today(), 500, TransactionKind.EXPENSE, "food")


def test_double_entry_balances_and_idempotent_csv_import(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    bank = repo.create_account("Bank", AccountType.BANK)
    income = repo.create_account("Salary", AccountType.INCOME)
    entry = repo.create_journal_entry(date(2026, 8, 1), "salary", bank.id, income.id, 300_000, "salary-1")

    assert repo.account_balances()[bank.id] == 300_000
    assert repo.account_balances()[income.id] == -300_000
    assert repo.list_journal_entries()[0] == entry
    assert repo.import_journal_csv([{
        "booked_on": "2026-08-01", "description": "salary", "debit_account_id": bank.id,
        "credit_account_id": income.id, "amount": "300000", "external_id": "salary-1",
    }]) == (0, 1)
