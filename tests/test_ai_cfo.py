from concurrent.futures import ThreadPoolExecutor
from datetime import date
import sqlite3

import pytest

from fastapi.testclient import TestClient

from personal_finance_os.ai_intents import (
    ClarificationRequired,
    Intent,
    LocalJapaneseIntentProvider,
    MockIntentProvider,
)
from personal_finance_os.ai_cfo import AICFOService, AIActionError, EmptyArguments
from personal_finance_os import ai_cfo as ai_cfo_module
from personal_finance_os.api import create_app
from personal_finance_os.models import AccountType, TransactionKind
from personal_finance_os.repository import FinanceRepository


def test_local_japanese_provider_normalizes_read_and_write_intents():
    provider = LocalJapaneseIntentProvider()

    summary = provider.parse("2026年8月の収支と純資産を教えて")
    assert summary.type == "summary"
    assert summary.arguments["start"] == date(2026, 8, 1)
    assert summary.period.end == date(2026, 8, 31)

    transaction = provider.parse("2026年8月12日に食費をメイン口座から3000円支出として登録")
    assert transaction.type == "create_transaction"
    assert transaction.arguments["amount"] == 3000
    assert transaction.arguments["kind"] == TransactionKind.EXPENSE.value

    tax = provider.parse("2025年の年収600万円、社会保険料90万円の給与所得税を試算")
    assert tax.type == "salary_tax"
    assert tax.arguments == {
        "tax_year": 2025,
        "gross_salary": 6_000_000,
        "social_insurance_premiums": 900_000,
    }


def test_ai_read_result_is_deterministic_and_explainable(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    account = client.post(
        "/api/accounts",
        json={"name": "Main", "account_type": "bank", "opening_balance": 100_000},
    ).json()
    client.post(
        "/api/transactions",
        json={
            "account_id": account["id"],
            "booked_on": "2026-08-02",
            "amount": -30_000,
            "kind": "expense",
            "category": "food",
        },
    )

    response = client.post("/api/ai/queries", json={"query": "2026年8月の収支を教えて"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "executed"
    assert body["result"]["expenses"] == 30_000
    assert body["period"] == {"start": "2026-08-01", "end": "2026-08-31"}
    assert body["data_sources"]
    assert body["parser_version"] == "jp-local-rules-v1"
    assert body["explanation"]["rule_version"] == "analytics-v1"


def test_ai_write_requires_confirmation_and_is_single_use(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    account = client.post("/api/accounts", json={"name": "Main", "account_type": "bank"}).json()

    preview = client.post(
        "/api/ai/queries",
        json={"text": "2026年8月12日に食費をMain口座から3000円支出として登録"},
    )
    body = preview.json()
    assert body["status"] == "previewed"
    assert client.get("/api/transactions").json() == []
    assert body["preview"]["will_change"]["amount"] == -3000

    confirmed = client.post("/api/ai/confirm", json={"confirmation_token": body["confirmation_token"]})
    assert confirmed.status_code == 200
    assert confirmed.json()["result"]["amount"] == -3000
    assert len(client.get("/api/transactions").json()) == 1

    repeated = client.post("/api/ai/confirm", json={"confirmation_token": body["confirmation_token"]})
    assert repeated.status_code == 409
    assert len(client.get("/api/transactions").json()) == 1


def test_ai_write_rejects_ledger_change_after_preview(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    account = repo.create_account("Main", AccountType.BANK)
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    preview = client.post(
        "/api/ai/queries",
        json={"text": "2026年8月12日に食費をMain口座から3000円支出として登録"},
    ).json()

    repo.create_transaction(account.id, date(2026, 8, 1), -100, TransactionKind.EXPENSE, "other")
    response = client.post("/api/ai/confirm", json={"confirmation_token": preview["confirmation_token"]})
    assert response.status_code == 409
    assert "ledger changed" in response.json()["detail"]["message"]
    assert len(repo.list_transactions()) == 1
    assert client.get("/api/ai/audit").json()[0]["state"] == "rejected"


def test_ai_supports_recurring_and_life_event_writes(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    for text, endpoint, expected in (
        (
            "2026年9月1日から給与30万円を毎月登録",
            "/api/recurring-cash-flows",
            "給与",
        ),
        (
            "2026年9月1日にライフイベント「出産」を登録、12か月、支出増5万円",
            "/api/life-events",
            "出産",
        ),
    ):
        preview = client.post("/api/ai/queries", json={"text": text}).json()
        assert preview["status"] == "previewed"
        confirmed = client.post("/api/ai/confirm", json={"confirmation_token": preview["confirmation_token"]})
        assert confirmed.status_code == 200
        assert confirmed.json()["result"]["name"] == expected
    assert len(client.get("/api/recurring-cash-flows").json()) == 1
    assert len(client.get("/api/life-events").json()) == 1


def test_ai_can_use_a_mock_intent_provider_without_external_services(tmp_path):
    provider = MockIntentProvider(
        Intent(type="account_list", arguments={}, parser_version="mock-v1", data_sources=["accounts"])
    )
    client = TestClient(create_app(str(tmp_path / "finance.db"), intent_provider=provider))
    response = client.post("/api/ai/queries", json={"text": "anything"})
    assert response.status_code == 200
    assert response.json()["intent"]["parser_version"] == "mock-v1"


@pytest.mark.parametrize(
    ("text", "intent_type"),
    (
        ("2026年8月の収支を教えて", "summary"),
        ("2026年8月の月次レポートを見せて", "monthly_report"),
        ("口座一覧を見せて", "account_list"),
        ("2026年8月の取引明細を見せて", "transaction_list"),
        ("2026年8月から30年の将来予測を教えて", "forecast"),
        ("基本ケースと転職ケースを比較", "forecast_compare"),
        ("500万円の購入シミュレーションをして", "purchase_simulation"),
        ("2025年の年収600万円の給与所得税を試算", "salary_tax"),
        ("2026年8月12日に食費をMain口座から3,000円支出として登録", "create_transaction"),
        ("2026年8月12日、借方:食費口座、貸方:Main口座、金額:3,000円で登録", "create_journal_entry"),
        ("2026年9月1日から給与30万円を毎月登録", "create_recurring_cash_flow"),
        ("2026年9月1日にライフイベント「出産」を登録、12か月、支出増5万円", "create_life_event"),
    ),
)
def test_japanese_intent_golden_table(text, intent_type):
    intent = LocalJapaneseIntentProvider().parse(text)
    assert intent.type == intent_type
    assert intent.parser_version == "jp-local-rules-v1"
    assert intent.intent_contract_version == "intent-v1"


@pytest.mark.parametrize(
    ("text", "expected_date", "expected_amount"),
    (
        ("2026-08-12に食費をMain口座から3,000円支出として登録", date(2026, 8, 12), 3_000),
        ("２０２６年８月１２日に食費をMain口座から３千円支出として登録", date(2026, 8, 12), 3_000),
        ("2026/08/12に給与をMain口座から30万円収入として登録", date(2026, 8, 12), 300_000),
    ),
)
def test_japanese_parser_normalizes_date_digits_commas_and_units(text, expected_date, expected_amount):
    arguments = LocalJapaneseIntentProvider().parse(text).arguments
    assert arguments["booked_on"] == expected_date
    assert arguments["amount"] == expected_amount


@pytest.mark.parametrize(
    ("text", "code"),
    (
        ("2026年2月30日に食費をMain口座から3,000円支出として登録", "invalid_date"),
        ("2026年8月12日に食費をMain口座から-3,000円支出として登録", "negative_amount"),
        ("2026年8月12日に食費をMain口座から3,000支出として登録", "amount_unit_required"),
        ("2026年8月12日に食費をMain口座から支出として登録", "missing_required_field"),
    ),
)
def test_japanese_parser_returns_stable_error_codes(text, code):
    with pytest.raises(ClarificationRequired) as raised:
        LocalJapaneseIntentProvider().parse(text)
    assert raised.value.code == code


def test_ai_v1_clarification_and_confirm_error_contract(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))

    clarification = client.post("/api/ai/queries", json={"text": "これは対応外の依頼"})
    assert clarification.status_code == 200
    assert clarification.json()["api_version"] == "1"
    assert clarification.json()["status"] == "needs_clarification"
    assert clarification.json()["code"] == "unsupported_intent"

    missing = client.post("/api/ai/confirm", json={})
    assert missing.status_code == 400
    detail = missing.json()["detail"]
    assert set(("code", "message", "details", "audit_id", "action_id")) <= set(detail)
    assert detail["code"] == "confirmation_token_required"
    assert detail["audit_id"]

    invalid = client.post("/api/ai/confirm", json={"confirmation_token": "not-a-token"})
    assert invalid.status_code == 400
    invalid_detail = invalid.json()["detail"]
    assert invalid_detail["code"] == "invalid_confirmation_token"
    assert invalid_detail["action_id"] is None
    assert invalid_detail["audit_id"]


def test_ai_action_is_atomic_retryable_and_auditable_after_injected_failure(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    account = repo.create_account("Main", AccountType.BANK)
    service = AICFOService(repo)
    preview = service.query("2026年8月12日に食費をMain口座から3000円支出として登録")

    original = repo._execute_ai_operation
    attempts = {"count": 0}

    def fail_once(connection, operation, arguments):
        if attempts["count"] == 0:
            attempts["count"] += 1
            connection.execute(
                "INSERT INTO transactions(id, account_id, booked_on, amount, kind, category) "
                "VALUES ('rolled-back', ?, '2026-08-12', -1, 'expense', 'test')",
                (account.id,),
            )
            raise RuntimeError("injected mid-transaction failure")
        return original(connection, operation, arguments)

    repo._execute_ai_operation = fail_once
    with pytest.raises(AIActionError) as raised:
        service.confirm(preview["confirmation_token"])
    assert raised.value.status_code == 500
    assert raised.value.code == "execution_failed"
    assert repo.list_transactions() == []
    failed = repo.get_ai_action_log(preview["action_id"])
    assert failed["state"] == "confirmed"
    assert failed["execution_state"] == "failed"
    assert failed["failure_code"] == "execution_failed"
    assert failed["resume_count"] == 1

    repo._execute_ai_operation = original
    result = service.confirm(preview["confirmation_token"])
    assert result["api_version"] == "1"
    assert len(repo.list_transactions()) == 1
    audit = service.audit()[0]
    assert "confirmation_token_hash" not in audit
    assert audit["state"] == "executed"
    assert audit["resume_count"] == 2
    assert [item["event"] for item in audit["state_history"]] == [
        "previewed", "confirmed", "execution_started", "execution_failed", "execution_started", "executed"
    ]
    assert audit["intent"]["type"] == "create_transaction"
    assert audit["parser_version"] == "jp-local-rules-v1"
    assert audit["rule_version"] == "deterministic-service-v1"


def test_same_confirmation_token_allows_only_one_write_across_services(tmp_path):
    db = tmp_path / "finance.db"
    repo = FinanceRepository(db)
    repo.create_account("Main", AccountType.BANK)
    preview_service = AICFOService(repo)
    preview = preview_service.query("2026年8月12日に食費をMain口座から3000円支出として登録")
    services = [AICFOService(FinanceRepository(db)) for _ in range(6)]

    def confirm(service):
        try:
            return service.confirm(preview["confirmation_token"])
        except AIActionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(services)) as pool:
        results = list(pool.map(confirm, services))
    assert sum(isinstance(result, dict) and result["status"] == "executed" for result in results) == 1
    assert len(repo.list_transactions()) == 1


def test_process_interrupt_before_audit_commit_can_resume_without_duplicate_write(tmp_path):
    db = tmp_path / "interrupt.db"
    repo = FinanceRepository(db)
    repo.create_account("Main", AccountType.BANK)
    service = AICFOService(repo)
    preview = service.query("2026年8月12日に食費をMain口座から3000円支出として登録")
    original = repo._execute_ai_operation

    def interrupt(connection, operation, arguments):
        connection.execute(
            "INSERT INTO transactions(id, account_id, booked_on, amount, kind, category) "
            "VALUES ('interrupted', ?, '2026-08-12', -1, 'expense', 'test')",
            (repo.list_accounts()[0].id,),
        )
        raise KeyboardInterrupt

    repo._execute_ai_operation = interrupt
    with pytest.raises(KeyboardInterrupt):
        service.confirm(preview["confirmation_token"])
    assert repo.list_transactions() == []
    assert repo.get_ai_action_log(preview["action_id"])["execution_state"] == "running"

    repo._execute_ai_operation = original
    resumed = AICFOService(FinanceRepository(db)).confirm(preview["confirmation_token"])
    assert resumed["status"] == "executed"
    assert len(repo.list_transactions()) == 1


def test_legacy_database_gets_ai_tables_without_changing_ledger(tmp_path):
    db = tmp_path / "legacy.db"
    connection = sqlite3.connect(db)
    try:
        connection.executescript(
            """
            CREATE TABLE accounts (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, account_type TEXT NOT NULL,
                opening_balance INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE transactions (
                id TEXT PRIMARY KEY, account_id TEXT NOT NULL, booked_on TEXT NOT NULL,
                amount INTEGER NOT NULL, kind TEXT NOT NULL, category TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE journal_entries (
                id TEXT PRIMARY KEY, booked_on TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                debit_account_id TEXT NOT NULL, credit_account_id TEXT NOT NULL, amount INTEGER NOT NULL,
                external_id TEXT UNIQUE, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO accounts(id, name, account_type, opening_balance)
            VALUES ('legacy-account', 'Legacy', 'bank', 12345);
            """
        )
    finally:
        connection.close()
    repo = FinanceRepository(db)
    assert [(account.id, account.name, account.opening_balance) for account in repo.list_accounts()] == [
        ("legacy-account", "Legacy", 12345)
    ]
    with repo._connect() as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "ai_action_log" in tables
    created = repo.create_transaction(
        "legacy-account", date(2026, 8, 1), -100, TransactionKind.EXPENSE, "food"
    )
    assert created.account_id == "legacy-account"


def test_ai_service_executes_all_read_intents_and_keeps_explanations(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    repo.create_account("Main", AccountType.BANK, opening_balance=100_000)
    provider = LocalJapaneseIntentProvider()
    service = AICFOService(repo, provider=MockIntentProvider(intents=[
        provider.parse("2026年8月の収支を教えて"),
        provider.parse("2026年8月の月次レポートを見せて"),
        provider.parse("口座一覧を見せて"),
        provider.parse("2026年8月の取引明細を見せて"),
        provider.parse("2026年8月から30年の将来予測を教えて"),
        provider.parse("基本ケースと転職ケースを比較"),
        provider.parse("500万円の購入シミュレーションをして"),
        provider.parse("2025年の年収600万円の給与所得税を試算"),
    ]))
    for _ in range(8):
        response = service.query("固定テスト入力")
        assert response["status"] == "executed"
        assert response["api_version"] == "1"
        assert response["explanation"]["parser_version"] == "jp-local-rules-v1"
        assert response["explanation"]["rule_version"]


def test_ai_service_writes_journal_and_direct_execution_compatibility(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    repo.create_account("食費", AccountType.EXPENSE)
    repo.create_account("Main", AccountType.BANK)
    service = AICFOService(repo)
    preview = service.query("2026年8月12日、借方:食費口座、貸方:Main口座、金額:3,000円で登録")
    assert preview["status"] == "previewed"
    assert service.confirm(preview["confirmation_token"])["status"] == "executed"
    assert len(repo.list_journal_entries()) == 1

    # Keep the internal helper compatible for local integrations that used it
    # before the atomic action path was introduced.
    transaction_intent = Intent(type="create_transaction", arguments={
        "account_name": "Main", "booked_on": date(2026, 8, 13), "amount": 10,
        "kind": "income", "category": "test", "description": "",
    })
    recurring_intent = Intent(type="create_recurring_cash_flow", arguments={
        "name": "Salary", "flow_type": "income", "amount": 10,
        "start_date": date(2026, 9, 1), "end_date": None,
    })
    life_intent = Intent(type="create_life_event", arguments={
        "name": "Move", "start_date": date(2026, 10, 1), "duration_months": 1,
        "event_type": "one_time", "income_delta": 0, "expense_delta": 10,
    })
    for intent in (transaction_intent, recurring_intent, life_intent):
        validated = service._validate_intent(intent)
        service._execute(intent.type, validated)
    assert len(repo.list_transactions()) == 1
    assert len(repo.list_recurring_cash_flows()) == 1
    assert len(repo.list_life_events()) == 1


@pytest.mark.parametrize(
    ("text", "code"),
    (
        ("", "empty_input"),
        ("2026年13月の収支を教えて", "invalid_period"),
        ("月次レポートを見せて", "missing_required_field"),
        ("ケースを比較", "missing_required_field"),
        ("基本ケースとケースを比較", "ambiguous_value"),
        ("給与所得税を試算", "missing_required_field"),
        ("2025年の社会保険料90万円の給与所得税を試算", "missing_required_field"),
        ("2026年8月12日にMain口座から3000円取引として登録", "ambiguous_value"),
        ("2026年8月12日に食費から3000円支出として登録", "missing_required_field"),
        ("2026年9月1日から30万円を毎月登録", "ambiguous_value"),
        ("2026年9月1日にライフイベントを登録", "missing_required_field"),
        ("購入シミュレーションをして", "missing_required_field"),
    ),
)
def test_japanese_parser_required_fields_and_ambiguity_codes(text, code):
    with pytest.raises(ClarificationRequired) as raised:
        LocalJapaneseIntentProvider().parse(text)
    assert raised.value.code == code


def test_summary_without_period_keeps_period_optional_in_v1():
    intent = LocalJapaneseIntentProvider().parse("収支を教えて")
    assert intent.type == "summary"
    assert intent.period is None


def test_confirm_expiry_and_abandoned_execution_is_recoverable(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    repo.create_account("Main", AccountType.BANK)
    service = AICFOService(repo)

    expired = service.query("2026年8月12日に食費をMain口座から3000円支出として登録")
    with repo._connect() as connection:
        connection.execute(
            "UPDATE ai_action_log SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (expired["action_id"],),
        )
    with pytest.raises(AIActionError) as expired_error:
        service.confirm(expired["confirmation_token"])
    assert expired_error.value.status_code == 409
    assert expired_error.value.code == "confirmation_expired"

    running = service.query("2026年8月13日に食費をMain口座から3000円支出として登録")
    assert repo.confirm_ai_action(running["action_id"], "2026-08-13T00:00:00+00:00")
    assert repo.begin_ai_action_execution(running["action_id"], "2026-08-13T00:00:01+00:00")
    recovered = service.confirm(running["confirmation_token"])
    assert recovered["status"] == "executed"
    assert repo.get_ai_action_log(running["action_id"])["resume_count"] == 2


def test_ai_internal_validation_error_paths_are_explicit_and_audited(tmp_path):
    repo = FinanceRepository(tmp_path / "finance.db")
    repo.create_account("Main", AccountType.BANK)
    with pytest.raises(ValueError):
        AICFOService(repo, preview_ttl_seconds=0)

    bad_provider = MockIntentProvider(handler=lambda _: (_ for _ in ()).throw(ValueError("provider failed")))
    bad_service = AICFOService(repo, provider=bad_provider)
    assert bad_service.query("bad")["code"] == "parser_error"

    unsupported = AICFOService(repo, provider=MockIntentProvider(Intent(type="unknown", arguments={})))
    assert unsupported.query("bad")["code"] == "invalid_intent_arguments"
    invalid_arguments = AICFOService(
        repo, provider=MockIntentProvider(Intent(type="create_transaction", arguments={}))
    )
    assert invalid_arguments.query("bad")["code"] == "invalid_intent_arguments"
    with pytest.raises(ValueError):
        unsupported._read("unknown", EmptyArguments())
    with pytest.raises(ValueError):
        unsupported._execute("unknown", EmptyArguments())

    duplicate_repo = FinanceRepository(tmp_path / "duplicate.db")
    duplicate_repo.create_account("Main", AccountType.BANK)
    duplicate_repo.create_account("Main口座", AccountType.BANK)
    duplicate_service = AICFOService(duplicate_repo)
    duplicate = duplicate_service.query("2026年8月12日に食費をMain口座から3000円支出として登録")
    assert duplicate["code"] == "account_ambiguous"

    journal_same_account = AICFOService(
        repo,
        provider=MockIntentProvider(Intent(type="create_journal_entry", arguments={
            "booked_on": date(2026, 8, 1), "description": "", "debit_account_name": "Main",
            "credit_account_name": "Main", "amount": 10,
        })),
    )
    assert journal_same_account.query("bad")["code"] == "invalid_intent_arguments"

    recurring_invalid = AICFOService(
        repo,
        provider=MockIntentProvider(Intent(type="create_recurring_cash_flow", arguments={
            "name": "bad", "flow_type": "income", "amount": 10,
            "start_date": date(2026, 9, 2), "end_date": date(2026, 9, 1),
        })),
    )
    assert recurring_invalid.query("bad")["code"] == "invalid_intent_arguments"

    invalid_stored = AICFOService(repo)
    preview = invalid_stored.query("2026年8月12日に食費をMain口座から3000円支出として登録")
    with repo._connect() as connection:
        connection.execute("UPDATE ai_action_log SET intent_json='{}' WHERE id=?", (preview["action_id"],))
    with pytest.raises(AIActionError) as stored_error:
        invalid_stored.confirm(preview["confirmation_token"])
    assert stored_error.value.code == "invalid_intent_arguments"

    malformed = repo.create_ai_action_log(
        raw_input="malformed", intent_json="{", parser_version="test", state="failed", error="bad"
    )
    malformed_audit = invalid_stored.audit()
    assert next(item for item in malformed_audit if item["id"] == malformed)["intent"] == "{"
    assert ai_cfo_module.AICFOService._parse_timestamp("not-a-timestamp") is None
    assert ai_cfo_module.AICFOService._parse_timestamp("2026-08-13T00:00:00") is not None
