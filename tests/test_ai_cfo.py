from datetime import date

from fastapi.testclient import TestClient

from personal_finance_os.ai_intents import (
    Intent,
    LocalJapaneseIntentProvider,
    MockIntentProvider,
)
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
