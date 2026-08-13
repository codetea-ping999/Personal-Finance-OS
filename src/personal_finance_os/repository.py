from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .models import (
    Account,
    AccountType,
    CashFlowType,
    ForecastScenario,
    JournalEntry,
    LifeEvent,
    LifeEventType,
    RecurringCashFlow,
    Transaction,
    TransactionKind,
)


class AIActionStateError(RuntimeError):
    """Raised when an AI action cannot be advanced from its stored state."""

    def __init__(self, action_id: str, state: str) -> None:
        super().__init__(f"AI action {action_id} is in state {state}")
        self.action_id = action_id
        self.state = state


class AILedgerChanged(RuntimeError):
    """Raised when a confirmed action no longer matches its preview ledger."""

    def __init__(self, action_id: str) -> None:
        super().__init__(f"AI action {action_id} ledger changed")
        self.action_id = action_id


class _ManagedSQLiteConnection(sqlite3.Connection):
    """Make ``with repository._connect()`` close connections on Windows too."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class FinanceRepository:
    def __init__(self, database: str | Path = "personal_finance.db") -> None:
        self.database = str(database)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30, factory=_ManagedSQLiteConnection)
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

                CREATE TABLE IF NOT EXISTS journal_entries (
                    id TEXT PRIMARY KEY,
                    booked_on TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    debit_account_id TEXT NOT NULL REFERENCES accounts(id),
                    credit_account_id TEXT NOT NULL REFERENCES accounts(id),
                    amount INTEGER NOT NULL CHECK(amount > 0),
                    external_id TEXT UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK(debit_account_id <> credit_account_id)
                );
                CREATE INDEX IF NOT EXISTS idx_journal_entries_booked_on
                    ON journal_entries(booked_on);

                CREATE TABLE IF NOT EXISTS recurring_cash_flows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    flow_type TEXT NOT NULL CHECK(flow_type IN ('income', 'expense')),
                    amount INTEGER NOT NULL CHECK(amount > 0),
                    start_date TEXT NOT NULL,
                    end_date TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK(end_date IS NULL OR end_date >= start_date)
                );
                CREATE INDEX IF NOT EXISTS idx_recurring_cash_flows_dates
                    ON recurring_cash_flows(start_date, end_date);

                CREATE TABLE IF NOT EXISTS life_events (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    duration_months INTEGER NOT NULL CHECK(duration_months > 0),
                    event_type TEXT NOT NULL CHECK(event_type IN ('one_time', 'recurring')),
                    income_delta INTEGER NOT NULL DEFAULT 0,
                    expense_delta INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_life_events_dates
                    ON life_events(start_date);

                CREATE TABLE IF NOT EXISTS forecast_scenarios (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    initial_balance INTEGER,
                    income_growth_rate REAL NOT NULL DEFAULT 0,
                    expense_growth_rate REAL NOT NULL DEFAULT 0,
                    annual_return_rate REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS ai_action_log (
                    id TEXT PRIMARY KEY,
                    raw_input TEXT NOT NULL,
                    intent_json TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    state TEXT NOT NULL,
                    confirmation_token_hash TEXT UNIQUE,
                    expires_at TEXT,
                    ledger_fingerprint TEXT,
                    preview_json TEXT,
                    confirmed_at TEXT,
                    executed_at TEXT,
                    result_json TEXT,
                    error TEXT,
                    rule_version TEXT NOT NULL DEFAULT 'unknown',
                    state_history_json TEXT NOT NULL DEFAULT '[]',
                    execution_state TEXT NOT NULL DEFAULT 'pending',
                    execution_started_at TEXT,
                    execution_owner TEXT,
                    resume_count INTEGER NOT NULL DEFAULT 0,
                    failure_code TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_ai_action_log_created_at
                    ON ai_action_log(created_at);
                CREATE INDEX IF NOT EXISTS idx_ai_action_log_state
                    ON ai_action_log(state);
                """
            )
            self._migrate_ai_action_log(connection)

    @staticmethod
    def _migrate_ai_action_log(connection: sqlite3.Connection) -> None:
        """Add only AI CFO columns; legacy ledger tables are never rewritten."""
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(ai_action_log)").fetchall()
        }
        additions = (
            ("rule_version", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("state_history_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("execution_state", "TEXT NOT NULL DEFAULT 'pending'"),
            ("execution_started_at", "TEXT"),
            ("execution_owner", "TEXT"),
            ("resume_count", "INTEGER NOT NULL DEFAULT 0"),
            ("failure_code", "TEXT"),
        )
        for name, definition in additions:
            if name not in columns:
                connection.execute(f"ALTER TABLE ai_action_log ADD COLUMN {name} {definition}")
        connection.execute(
            "UPDATE ai_action_log SET execution_state='executed' "
            "WHERE state='executed' AND execution_state='pending'"
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

    def account_balances(self, as_of: date | None = None) -> dict[str, int]:
        transaction_filter = "AND t.booked_on <= ?" if as_of is not None else ""
        journal_filter = "AND j.booked_on <= ?" if as_of is not None else ""
        params = [as_of.isoformat(), as_of.isoformat()] if as_of is not None else []
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT a.id,
                       a.opening_balance + COALESCE((
                           SELECT SUM(t.amount) FROM transactions t
                           WHERE t.account_id = a.id {transaction_filter}
                       ), 0)
                       + COALESCE((
                           SELECT SUM(CASE WHEN j.debit_account_id = a.id THEN j.amount ELSE -j.amount END)
                           FROM journal_entries j
                           WHERE (j.debit_account_id = a.id OR j.credit_account_id = a.id)
                             {journal_filter}
                       ), 0) AS balance
                  FROM accounts a
                """,
                params,
            ).fetchall()
        return {row["id"]: int(row["balance"]) for row in rows}

    def forecast_initial_balance(self, as_of: date | None = None) -> int:
        accounts = self.list_accounts()
        balances = self.account_balances(as_of=as_of)
        asset_types = {AccountType.CASH, AccountType.BANK, AccountType.INVESTMENT}
        return sum(balances.get(account.id, 0) for account in accounts if account.account_type in asset_types)

    def create_journal_entry(
        self, booked_on: date, description: str, debit_account_id: str,
        credit_account_id: str, amount: int, external_id: str | None = None,
    ) -> JournalEntry:
        if amount <= 0:
            raise ValueError("journal amount must be positive")
        if debit_account_id == credit_account_id:
            raise ValueError("debit and credit accounts must differ")
        entry = JournalEntry(str(uuid4()), booked_on, description.strip(), debit_account_id,
                             credit_account_id, amount, external_id)
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO journal_entries
                    (id, booked_on, description, debit_account_id, credit_account_id, amount, external_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (entry.id, entry.booked_on.isoformat(), entry.description,
                     entry.debit_account_id, entry.credit_account_id, entry.amount, entry.external_id),
                )
        except sqlite3.IntegrityError as exc:
            if external_id and "external_id" in str(exc):
                raise ValueError("external_id already imported") from exc
            raise ValueError("both accounts must exist and external_id must be unique") from exc
        return entry

    def list_journal_entries(self, start: date | None = None, end: date | None = None) -> list[JournalEntry]:
        clauses, params = [], []
        if start is not None:
            clauses.append("booked_on >= ?"); params.append(start.isoformat())
        if end is not None:
            clauses.append("booked_on <= ?"); params.append(end.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, booked_on, description, debit_account_id, credit_account_id, amount, external_id "
                f"FROM journal_entries {where} ORDER BY booked_on, created_at", params).fetchall()
        return [JournalEntry(row["id"], date.fromisoformat(row["booked_on"]), row["description"],
                             row["debit_account_id"], row["credit_account_id"], row["amount"], row["external_id"])
                for row in rows]

    def import_journal_csv(self, rows: list[dict[str, str]]) -> tuple[int, int]:
        imported = skipped = 0
        for row in rows:
            external_id = (row.get("external_id") or "").strip() or None
            try:
                self.create_journal_entry(date.fromisoformat(row["booked_on"]), row.get("description", ""),
                                          row["debit_account_id"], row["credit_account_id"],
                                          int(row["amount"]), external_id)
                imported += 1
            except ValueError as exc:
                if external_id and str(exc) == "external_id already imported":
                    skipped += 1
                else:
                    raise
        return imported, skipped

    @staticmethod
    def _clean_name(name: str) -> str:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name must not be empty")
        return cleaned

    @staticmethod
    def _validate_date_range(start_date: date, end_date: date | None) -> None:
        if end_date is not None and end_date < start_date:
            raise ValueError("end_date must be on or after start_date")

    def create_recurring_cash_flow(
        self,
        name: str,
        flow_type: CashFlowType,
        amount: int,
        start_date: date,
        end_date: date | None = None,
    ) -> RecurringCashFlow:
        name = self._clean_name(name)
        if amount <= 0:
            raise ValueError("amount must be positive")
        self._validate_date_range(start_date, end_date)
        flow = RecurringCashFlow(str(uuid4()), name, CashFlowType(flow_type), amount, start_date, end_date)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO recurring_cash_flows
                (id, name, flow_type, amount, start_date, end_date)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (flow.id, flow.name, flow.flow_type.value, flow.amount,
                 flow.start_date.isoformat(), flow.end_date.isoformat() if flow.end_date else None),
            )
        return flow

    def list_recurring_cash_flows(self) -> list[RecurringCashFlow]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, name, flow_type, amount, start_date, end_date
                FROM recurring_cash_flows ORDER BY start_date, created_at"""
            ).fetchall()
        return [
            RecurringCashFlow(
                row["id"], row["name"], CashFlowType(row["flow_type"]), row["amount"],
                date.fromisoformat(row["start_date"]),
                date.fromisoformat(row["end_date"]) if row["end_date"] else None,
            )
            for row in rows
        ]

    def get_recurring_cash_flow(self, flow_id: str) -> RecurringCashFlow:
        flow = next((item for item in self.list_recurring_cash_flows() if item.id == flow_id), None)
        if flow is None:
            raise ValueError("recurring cash flow does not exist")
        return flow

    def update_recurring_cash_flow(self, flow_id: str, **changes: object) -> RecurringCashFlow:
        current = self.get_recurring_cash_flow(flow_id)
        updated = RecurringCashFlow(
            id=current.id,
            name=self._clean_name(str(changes.get("name", current.name))),
            flow_type=CashFlowType(changes.get("flow_type", current.flow_type)),
            amount=int(changes.get("amount", current.amount)),
            start_date=changes.get("start_date", current.start_date),
            end_date=changes.get("end_date", current.end_date),
        )
        if updated.amount <= 0:
            raise ValueError("amount must be positive")
        if not isinstance(updated.start_date, date) or (updated.end_date is not None and not isinstance(updated.end_date, date)):
            raise ValueError("start_date and end_date must be dates")
        self._validate_date_range(updated.start_date, updated.end_date)
        with self._connect() as connection:
            connection.execute(
                """UPDATE recurring_cash_flows
                SET name=?, flow_type=?, amount=?, start_date=?, end_date=? WHERE id=?""",
                (updated.name, updated.flow_type.value, updated.amount, updated.start_date.isoformat(),
                 updated.end_date.isoformat() if updated.end_date else None, updated.id),
            )
        return updated

    def delete_recurring_cash_flow(self, flow_id: str) -> None:
        with self._connect() as connection:
            result = connection.execute("DELETE FROM recurring_cash_flows WHERE id=?", (flow_id,))
        if result.rowcount == 0:
            raise ValueError("recurring cash flow does not exist")

    def create_life_event(
        self,
        name: str,
        start_date: date,
        duration_months: int = 1,
        event_type: LifeEventType = LifeEventType.ONE_TIME,
        income_delta: int = 0,
        expense_delta: int = 0,
    ) -> LifeEvent:
        name = self._clean_name(name)
        if duration_months <= 0:
            raise ValueError("duration_months must be positive")
        event = LifeEvent(
            str(uuid4()), name, start_date, duration_months, LifeEventType(event_type),
            income_delta, expense_delta,
        )
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO life_events
                (id, name, start_date, duration_months, event_type, income_delta, expense_delta)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (event.id, event.name, event.start_date.isoformat(), event.duration_months,
                 event.event_type.value, event.income_delta, event.expense_delta),
            )
        return event

    def list_life_events(self) -> list[LifeEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, name, start_date, duration_months, event_type, income_delta, expense_delta
                FROM life_events ORDER BY start_date, created_at"""
            ).fetchall()
        return [
            LifeEvent(
                row["id"], row["name"], date.fromisoformat(row["start_date"]), row["duration_months"],
                LifeEventType(row["event_type"]), row["income_delta"], row["expense_delta"],
            )
            for row in rows
        ]

    def get_life_event(self, event_id: str) -> LifeEvent:
        event = next((item for item in self.list_life_events() if item.id == event_id), None)
        if event is None:
            raise ValueError("life event does not exist")
        return event

    def update_life_event(self, event_id: str, **changes: object) -> LifeEvent:
        current = self.get_life_event(event_id)
        updated = LifeEvent(
            id=current.id,
            name=self._clean_name(str(changes.get("name", current.name))),
            start_date=changes.get("start_date", current.start_date),
            duration_months=int(changes.get("duration_months", current.duration_months)),
            event_type=LifeEventType(changes.get("event_type", current.event_type)),
            income_delta=int(changes.get("income_delta", current.income_delta)),
            expense_delta=int(changes.get("expense_delta", current.expense_delta)),
        )
        if not isinstance(updated.start_date, date):
            raise ValueError("start_date must be a date")
        if updated.duration_months <= 0:
            raise ValueError("duration_months must be positive")
        with self._connect() as connection:
            connection.execute(
                """UPDATE life_events SET name=?, start_date=?, duration_months=?, event_type=?,
                income_delta=?, expense_delta=? WHERE id=?""",
                (updated.name, updated.start_date.isoformat(), updated.duration_months,
                 updated.event_type.value, updated.income_delta, updated.expense_delta, updated.id),
            )
        return updated

    def delete_life_event(self, event_id: str) -> None:
        with self._connect() as connection:
            result = connection.execute("DELETE FROM life_events WHERE id=?", (event_id,))
        if result.rowcount == 0:
            raise ValueError("life event does not exist")

    def create_forecast_scenario(
        self,
        name: str,
        initial_balance: int | None = None,
        income_growth_rate: float = 0.0,
        expense_growth_rate: float = 0.0,
        annual_return_rate: float = 0.0,
    ) -> ForecastScenario:
        name = self._clean_name(name)
        if initial_balance is not None and not isinstance(initial_balance, int):
            raise ValueError("initial_balance must be an integer")
        self._validate_rates(income_growth_rate, expense_growth_rate, annual_return_rate)
        scenario = ForecastScenario(
            str(uuid4()), name, initial_balance, float(income_growth_rate),
            float(expense_growth_rate), float(annual_return_rate),
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO forecast_scenarios
                    (id, name, initial_balance, income_growth_rate, expense_growth_rate, annual_return_rate)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (scenario.id, scenario.name, scenario.initial_balance, scenario.income_growth_rate,
                     scenario.expense_growth_rate, scenario.annual_return_rate),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("forecast scenario name already exists") from exc
        return scenario

    @staticmethod
    def _validate_rates(income_growth_rate: float, expense_growth_rate: float, annual_return_rate: float) -> None:
        if income_growth_rate <= -1 or expense_growth_rate <= -1 or annual_return_rate <= -1:
            raise ValueError("rates must be greater than -1")

    def list_forecast_scenarios(self) -> list[ForecastScenario]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, name, initial_balance, income_growth_rate, expense_growth_rate, annual_return_rate
                FROM forecast_scenarios ORDER BY created_at, name"""
            ).fetchall()
        return [
            ForecastScenario(
                row["id"], row["name"], row["initial_balance"], row["income_growth_rate"],
                row["expense_growth_rate"], row["annual_return_rate"],
            )
            for row in rows
        ]

    def get_forecast_scenario(self, scenario_id: str) -> ForecastScenario:
        scenario = next((item for item in self.list_forecast_scenarios() if item.id == scenario_id), None)
        if scenario is None:
            raise ValueError("forecast scenario does not exist")
        return scenario

    def get_forecast_scenario_by_name(self, name: str) -> ForecastScenario | None:
        return next((item for item in self.list_forecast_scenarios() if item.name == name), None)

    def update_forecast_scenario(self, scenario_id: str, **changes: object) -> ForecastScenario:
        current = self.get_forecast_scenario(scenario_id)
        updated = ForecastScenario(
            id=current.id,
            name=self._clean_name(str(changes.get("name", current.name))),
            initial_balance=changes.get("initial_balance", current.initial_balance),
            income_growth_rate=float(changes.get("income_growth_rate", current.income_growth_rate)),
            expense_growth_rate=float(changes.get("expense_growth_rate", current.expense_growth_rate)),
            annual_return_rate=float(changes.get("annual_return_rate", current.annual_return_rate)),
        )
        self._validate_rates(updated.income_growth_rate, updated.expense_growth_rate, updated.annual_return_rate)
        with self._connect() as connection:
            try:
                connection.execute(
                    """UPDATE forecast_scenarios SET name=?, initial_balance=?, income_growth_rate=?,
                    expense_growth_rate=?, annual_return_rate=?, updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                    (updated.name, updated.initial_balance, updated.income_growth_rate, updated.expense_growth_rate,
                     updated.annual_return_rate, updated.id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("forecast scenario name already exists") from exc
        return updated

    def delete_forecast_scenario(self, scenario_id: str) -> None:
        with self._connect() as connection:
            result = connection.execute("DELETE FROM forecast_scenarios WHERE id=?", (scenario_id,))
        if result.rowcount == 0:
            raise ValueError("forecast scenario does not exist")

    @staticmethod
    def _ledger_fingerprint_for_connection(connection: sqlite3.Connection) -> str:
        tables = {
            "accounts": connection.execute(
                "SELECT id, name, account_type, opening_balance FROM accounts ORDER BY id"
            ).fetchall(),
            "transactions": connection.execute(
                "SELECT id, account_id, booked_on, amount, kind, category, description FROM transactions ORDER BY id"
            ).fetchall(),
            "journal_entries": connection.execute(
                "SELECT id, booked_on, description, debit_account_id, credit_account_id, amount, external_id "
                "FROM journal_entries ORDER BY id"
            ).fetchall(),
            "recurring_cash_flows": connection.execute(
                "SELECT id, name, flow_type, amount, start_date, end_date FROM recurring_cash_flows ORDER BY id"
            ).fetchall(),
            "life_events": connection.execute(
                "SELECT id, name, start_date, duration_months, event_type, income_delta, expense_delta "
                "FROM life_events ORDER BY id"
            ).fetchall(),
            "forecast_scenarios": connection.execute(
                "SELECT id, name, initial_balance, income_growth_rate, expense_growth_rate, annual_return_rate "
                "FROM forecast_scenarios ORDER BY id"
            ).fetchall(),
        }
        payload = {table: [dict(row) for row in rows] for table, rows in tables.items()}
        return sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def ledger_fingerprint(self) -> str:
        """Return a stable digest of state that can affect an AI write.

        The audit table is intentionally excluded.  Creating or reading an AI
        action must not invalidate its own preview; ledger rows and account
        definitions must.
        """
        with self._connect() as connection:
            return self._ledger_fingerprint_for_connection(connection)

    @staticmethod
    def _history(history_json: object, event: str, at: str, **details: object) -> str:
        try:
            history = json.loads(str(history_json or "[]"))
        except json.JSONDecodeError:
            history = []
        if not isinstance(history, list):
            history = []
        history.append({"event": event, "state": event, "at": at, **details})
        return json.dumps(history, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def create_ai_action_log(
        self,
        *,
        raw_input: str,
        intent_json: str,
        parser_version: str,
        state: str,
        confirmation_token_hash: str | None = None,
        expires_at: str | None = None,
        ledger_fingerprint: str | None = None,
        preview_json: str | None = None,
        result_json: str | None = None,
        error: str | None = None,
        rule_version: str = "unknown",
        failure_code: str | None = None,
    ) -> str:
        action_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        execution_state = "executed" if state == "executed" else "failed" if state == "failed" else "pending"
        history = self._history("[]", state, now)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO ai_action_log(
                    id, raw_input, intent_json, parser_version, state,
                    confirmation_token_hash, expires_at, ledger_fingerprint,
                    preview_json, result_json, error, rule_version,
                    state_history_json, execution_state, failure_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    action_id, raw_input, intent_json, parser_version, state,
                    confirmation_token_hash, expires_at, ledger_fingerprint,
                    preview_json, result_json, error, rule_version,
                    history, execution_state, failure_code,
                ),
            )
        return action_id

    def get_ai_action_log(self, action_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM ai_action_log WHERE id=?", (action_id,)).fetchone()
        return dict(row) if row is not None else None

    def find_ai_action_by_token_hash(self, token_hash: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ai_action_log WHERE confirmation_token_hash=?",
                (token_hash,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_ai_action_logs(self, limit: int = 100) -> list[dict[str, object]]:
        limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ai_action_log ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_ai_action_log(
        self,
        action_id: str,
        *,
        state: str | None = None,
        confirmed_at: str | None = None,
        executed_at: str | None = None,
        result_json: str | None = None,
        error: str | None = None,
        execution_state: str | None = None,
        execution_started_at: str | None = None,
        failure_code: str | None = None,
    ) -> bool:
        with self._connect() as connection:
            current = connection.execute(
                "SELECT state, state_history_json FROM ai_action_log WHERE id=?",
                (action_id,),
            ).fetchone()
            if current is None:
                return False
            changes: list[str] = []
            values: list[object] = []
            now = datetime.now(timezone.utc).isoformat()
            if state is not None:
                changes.append("state=?")
                values.append(state)
                if state != current["state"]:
                    changes.append("state_history_json=?")
                    values.append(self._history(current["state_history_json"], state, now))
            for column, value in (
                ("confirmed_at", confirmed_at),
                ("executed_at", executed_at),
                ("result_json", result_json),
                ("error", error),
                ("execution_state", execution_state),
                ("execution_started_at", execution_started_at),
                ("failure_code", failure_code),
            ):
                if value is not None:
                    changes.append(f"{column}=?")
                    values.append(value)
            if not changes:
                return False
            values.append(action_id)
            cursor = connection.execute(
                f"UPDATE ai_action_log SET {', '.join(changes)} WHERE id=?",
                values,
            )
        return cursor.rowcount == 1

    def confirm_ai_action(self, action_id: str, confirmed_at: str) -> bool:
        """Atomically reserve a preview for one confirmation attempt."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, state_history_json FROM ai_action_log WHERE id=?",
                (action_id,),
            ).fetchone()
            if row is None or row["state"] != "previewed":
                connection.rollback()
                return False
            history = self._history(row["state_history_json"], "confirmed", confirmed_at)
            cursor = connection.execute(
                "UPDATE ai_action_log SET state='confirmed', confirmed_at=?, "
                "execution_state='pending', execution_owner=NULL, error=NULL, failure_code=NULL, state_history_json=? "
                "WHERE id=? AND state='previewed'",
                (confirmed_at, history, action_id),
            )
            connection.commit()
        return cursor.rowcount == 1

    def begin_ai_action_execution(
        self,
        action_id: str,
        started_at: str,
        *,
        owner_id: str | None = None,
    ) -> str | None:
        """Claim an execution attempt; a new process can reclaim an abandoned one."""
        owner_id = owner_id or uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, resume_count, state_history_json "
                "FROM ai_action_log WHERE id=?",
                (action_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            if row["state"] != "confirmed":
                connection.rollback()
                return False
            attempt = int(row["resume_count"] or 0) + 1
            history = self._history(row["state_history_json"], "execution_started", started_at, attempt=attempt)
            connection.execute(
                "UPDATE ai_action_log SET execution_state='running', execution_started_at=?, "
                "execution_owner=?, resume_count=?, error=NULL, failure_code=NULL, state_history_json=? "
                "WHERE id=? AND state='confirmed'",
                (started_at, owner_id, attempt, history, action_id),
            )
            connection.commit()
        return owner_id

    def mark_ai_action_failed(self, action_id: str, error: str, failure_code: str) -> bool:
        """Keep the public state confirmed so a failed attempt can be retried."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_history_json FROM ai_action_log WHERE id=? AND state='confirmed'",
                (action_id,),
            ).fetchone()
            if row is None:
                return False
            now = datetime.now(timezone.utc).isoformat()
            history = self._history(row["state_history_json"], "execution_failed", now, code=failure_code)
            cursor = connection.execute(
                "UPDATE ai_action_log SET execution_state='failed', execution_owner=NULL, error=?, failure_code=?, "
                "state_history_json=? WHERE id=? AND state='confirmed'",
                (error, failure_code, history, action_id),
            )
        return cursor.rowcount == 1

    def reject_ai_action(self, action_id: str, error: str, failure_code: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_history_json FROM ai_action_log WHERE id=? AND state='confirmed'",
                (action_id,),
            ).fetchone()
            if row is None:
                return False
            now = datetime.now(timezone.utc).isoformat()
            history = self._history(row["state_history_json"], "rejected", now, code=failure_code)
            cursor = connection.execute(
                "UPDATE ai_action_log SET state='rejected', execution_state='rejected', execution_owner=NULL, "
                "error=?, failure_code=?, state_history_json=? WHERE id=? AND state='confirmed'",
                (error, failure_code, history, action_id),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _value(value: object) -> object:
        return getattr(value, "value", value)

    def _execute_ai_operation(self, connection: sqlite3.Connection, operation: str, arguments: dict[str, Any]) -> object:
        """Write one AI-supported operation on the caller's open transaction."""
        if operation == "create_transaction":
            account_id = str(arguments.get("account_id") or "")
            amount = int(arguments["amount"])
            kind = TransactionKind(self._value(arguments["kind"]))
            if kind == TransactionKind.INCOME:
                signed_amount = amount
            elif kind == TransactionKind.EXPENSE:
                signed_amount = -amount
            else:
                raise ValueError("AI CFO v1 supports income and expense transactions only")
            if amount <= 0:
                raise ValueError("transaction amount must be positive")
            category = str(arguments["category"]).strip()
            if not category:
                raise ValueError("category must not be empty")
            transaction = Transaction(
                id=str(uuid4()), account_id=account_id, booked_on=arguments["booked_on"],
                amount=signed_amount, kind=kind, category=category,
                description=str(arguments.get("description", "")).strip(),
            )
            try:
                connection.execute(
                    "INSERT INTO transactions(id, account_id, booked_on, amount, kind, category, description) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (transaction.id, transaction.account_id, transaction.booked_on.isoformat(), transaction.amount,
                     transaction.kind.value, transaction.category, transaction.description),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("account does not exist") from exc
            return transaction

        if operation == "create_journal_entry":
            debit_account_id = str(arguments.get("debit_account_id") or "")
            credit_account_id = str(arguments.get("credit_account_id") or "")
            amount = int(arguments["amount"])
            if amount <= 0:
                raise ValueError("journal amount must be positive")
            if debit_account_id == credit_account_id:
                raise ValueError("debit and credit accounts must differ")
            entry = JournalEntry(
                str(uuid4()), arguments["booked_on"], str(arguments.get("description", "")).strip(),
                debit_account_id, credit_account_id, amount, None,
            )
            try:
                connection.execute(
                    "INSERT INTO journal_entries(id, booked_on, description, debit_account_id, credit_account_id, amount, external_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (entry.id, entry.booked_on.isoformat(), entry.description, entry.debit_account_id,
                     entry.credit_account_id, entry.amount, entry.external_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("both accounts must exist") from exc
            return entry

        if operation == "create_recurring_cash_flow":
            name = str(arguments["name"]).strip()
            amount = int(arguments["amount"])
            start_date = arguments["start_date"]
            end_date = arguments.get("end_date")
            flow_type = CashFlowType(self._value(arguments["flow_type"]))
            if not name:
                raise ValueError("name must not be empty")
            if amount <= 0:
                raise ValueError("amount must be positive")
            if end_date is not None and end_date < start_date:
                raise ValueError("end_date must be on or after start_date")
            flow = RecurringCashFlow(str(uuid4()), name, flow_type, amount, start_date, end_date)
            connection.execute(
                "INSERT INTO recurring_cash_flows(id, name, flow_type, amount, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?)",
                (flow.id, flow.name, flow.flow_type.value, flow.amount, flow.start_date.isoformat(),
                 flow.end_date.isoformat() if flow.end_date else None),
            )
            return flow

        if operation == "create_life_event":
            name = str(arguments["name"]).strip()
            duration_months = int(arguments["duration_months"])
            if not name:
                raise ValueError("name must not be empty")
            if duration_months <= 0:
                raise ValueError("duration_months must be positive")
            event = LifeEvent(
                str(uuid4()), name, arguments["start_date"], duration_months,
                LifeEventType(self._value(arguments.get("event_type", LifeEventType.ONE_TIME))),
                int(arguments.get("income_delta", 0)), int(arguments.get("expense_delta", 0)),
            )
            connection.execute(
                "INSERT INTO life_events(id, name, start_date, duration_months, event_type, income_delta, expense_delta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event.id, event.name, event.start_date.isoformat(), event.duration_months, event.event_type.value,
                 event.income_delta, event.expense_delta),
            )
            return event

        raise ValueError(f"unsupported AI operation: {operation}")

    def execute_ai_action(
        self,
        action_id: str,
        operation: str,
        arguments: dict[str, Any],
        result_serializer: Callable[[object], str],
        executed_at: str,
        execution_owner: str,
        expected_ledger_fingerprint: str | None = None,
    ) -> object:
        """Atomically write the domain row and mark the action executed."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, execution_state, execution_owner, state_history_json FROM ai_action_log WHERE id=?",
                (action_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise AIActionStateError(action_id, "missing")
            if row["state"] != "confirmed":
                connection.rollback()
                raise AIActionStateError(action_id, str(row["state"]))
            if row["execution_state"] != "running" or row["execution_owner"] != execution_owner:
                connection.rollback()
                raise AIActionStateError(action_id, "claimed_by_other")
            if (
                expected_ledger_fingerprint is not None
                and expected_ledger_fingerprint != self._ledger_fingerprint_for_connection(connection)
            ):
                connection.rollback()
                raise AILedgerChanged(action_id)
            result = self._execute_ai_operation(connection, operation, arguments)
            result_json = result_serializer(result)
            history = self._history(row["state_history_json"], "executed", executed_at)
            connection.execute(
                "UPDATE ai_action_log SET state='executed', execution_state='executed', executed_at=?, "
                "result_json=?, error=NULL, failure_code=NULL, execution_owner=NULL, state_history_json=? "
                "WHERE id=? AND state='confirmed' AND execution_owner=?",
                (executed_at, result_json, history, action_id, execution_owner),
            )
            connection.commit()
        return result
