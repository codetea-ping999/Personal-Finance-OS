from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from uuid import uuid4

from .models import Account, AccountType, Transaction, TransactionKind


class FinanceRepository:
    def __init__(self, database: str | Path = "personal_finance.db") -> None:
        self.database = str(database)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    opening_balance INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                    booked_on TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_transactions_booked_on
                    ON transactions(booked_on);
                CREATE INDEX IF NOT EXISTS idx_transactions_account_id
                    ON transactions(account_id);
                """
            )

    def create_account(
        self,
        name: str,
        account_type: AccountType,
        opening_balance: int = 0,
    ) -> Account:
        account = Account(str(uuid4()), name.strip(), account_type, opening_balance)
        if not account.name:
            raise ValueError("account name must not be empty")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO accounts(id, name, account_type, opening_balance) VALUES (?, ?, ?, ?)",
                (account.id, account.name, account.account_type.value, account.opening_balance),
            )
        return account

    def list_accounts(self) -> list[Account]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, account_type, opening_balance FROM accounts ORDER BY created_at, name"
            ).fetchall()
        return [
            Account(
                id=row["id"],
                name=row["name"],
                account_type=AccountType(row["account_type"]),
                opening_balance=row["opening_balance"],
            )
            for row in rows
        ]

    def create_transaction(
        self,
        account_id: str,
        booked_on: date,
        amount: int,
        kind: TransactionKind,
        category: str,
        description: str = "",
    ) -> Transaction:
        if amount == 0:
            raise ValueError("transaction amount must not be zero")
        if kind == TransactionKind.INCOME and amount < 0:
            raise ValueError("income must use a positive amount")
        if kind == TransactionKind.EXPENSE and amount > 0:
            raise ValueError("expense must use a negative amount")
        category = category.strip()
        if not category:
            raise ValueError("category must not be empty")

        transaction = Transaction(
            id=str(uuid4()),
            account_id=account_id,
            booked_on=booked_on,
            amount=amount,
            kind=kind,
            category=category,
            description=description.strip(),
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO transactions(
                        id, account_id, booked_on, amount, kind, category, description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transaction.id,
                        transaction.account_id,
                        transaction.booked_on.isoformat(),
                        transaction.amount,
                        transaction.kind.value,
                        transaction.category,
                        transaction.description,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("account does not exist") from exc
        return transaction

    def list_transactions(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Transaction]:
        clauses: list[str] = []
        params: list[str] = []
        if start is not None:
            clauses.append("booked_on >= ?")
            params.append(start.isoformat())
        if end is not None:
            clauses.append("booked_on <= ?")
            params.append(end.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT id, account_id, booked_on, amount, kind, category, description "
            f"FROM transactions {where} ORDER BY booked_on, created_at"
        )
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            Transaction(
                id=row["id"],
                account_id=row["account_id"],
                booked_on=date.fromisoformat(row["booked_on"]),
                amount=row["amount"],
                kind=TransactionKind(row["kind"]),
                category=row["category"],
                description=row["description"],
            )
            for row in rows
        ]

    def account_balances(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT a.id,
                       a.opening_balance + COALESCE(SUM(t.amount), 0) AS balance
                  FROM accounts a
             LEFT JOIN transactions t ON t.account_id = a.id
              GROUP BY a.id, a.opening_balance
                """
            ).fetchall()
        return {row["id"]: int(row["balance"]) for row in rows}
