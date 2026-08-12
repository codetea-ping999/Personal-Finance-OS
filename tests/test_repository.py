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
