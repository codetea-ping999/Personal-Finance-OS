from fastapi.testclient import TestClient

from personal_finance_os.api import create_app


def test_journal_api_and_monthly_report(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    bank = client.post("/api/accounts", json={"name": "Bank", "account_type": "bank"}).json()
    salary = client.post("/api/accounts", json={"name": "Salary", "account_type": "income"}).json()
    response = client.post("/api/journal-entries", json={
        "booked_on": "2026-08-01", "description": "salary",
        "debit_account_id": bank["id"], "credit_account_id": salary["id"], "amount": 300000,
    })
    assert response.status_code == 201
    report = client.get("/api/reports/monthly?year=2026&month=8")
    assert report.json()["income"] == 300000
    assert report.json()["net_income"] == 300000


def test_csv_import_endpoint(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    bank = client.post("/api/accounts", json={"name": "Bank", "account_type": "bank"}).json()
    expense = client.post("/api/accounts", json={"name": "Food", "account_type": "expense"}).json()
    csv_text = "booked_on,debit_account_id,credit_account_id,amount,description,external_id\n2026-08-02,{},{},5000,food,food-1\n".format(expense["id"], bank["id"])
    response = client.post("/api/import/journal", content=csv_text, headers={"content-type": "text/csv"})
    assert response.status_code == 200
    assert response.json() == {"imported": 1, "skipped": 0}


def test_reconciliation_reports_difference(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    bank = client.post("/api/accounts", json={"name": "Bank", "account_type": "bank", "opening_balance": 1000}).json()
    response = client.post("/api/reconciliation", json={"account_id": bank["id"], "statement_balance": 900})
    assert response.json()["difference"] == -100
    assert response.json()["status"] == "difference_found"


def test_statement_import_matching_and_transaction_confirmation(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    bank = client.post("/api/accounts", json={"name": "Bank", "account_type": "bank"}).json()
    transaction = client.post("/api/transactions", json={
        "account_id": bank["id"], "booked_on": "2026-08-02", "amount": -5000,
        "kind": "expense", "category": "food", "description": "lunch",
    }).json()
    payload = {"account_id": bank["id"], "mapping": {"date": "date", "amount": "amount", "description": "memo", "balance": "balance"}, "csv_text": "date,amount,memo,balance\n2026/08/02,-5000,lunch,95000\n2026/08/03,10000,salary,105000\n"}
    imported = client.post("/api/statement-imports", json=payload)
    assert imported.status_code == 201
    assert imported.json()["imported"] == 2
    reconciliation = client.get(f"/api/statement-accounts/{bank['id']}/reconciliation").json()
    assert reconciliation["statement_balance"] == 105000
    assert reconciliation["difference"] == 110000
    assert client.post("/api/statement-imports", json=payload).json()["skipped"] == 2
    lines = client.get("/api/statement-lines", params={"account_id": bank["id"]}).json()
    candidates = client.get(f"/api/statement-lines/{lines[0]['id']}/candidates").json()["candidates"]
    assert candidates[0]["id"] == transaction["id"]
    assert client.post(f"/api/statement-lines/{lines[0]['id']}/match", json={"matched_type": "transaction", "matched_id": transaction["id"]}).status_code == 200
    created = client.post(f"/api/statement-lines/{lines[1]['id']}/transaction", json={"kind": "income", "category": "salary"})
    assert created.status_code == 201
    assert created.json()["amount"] == 10000


def test_statement_import_deposit_withdrawal_and_transfer(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    first = client.post("/api/accounts", json={"name": "First", "account_type": "bank"}).json()
    second = client.post("/api/accounts", json={"name": "Second", "account_type": "cash"}).json()
    for account, csv_text in ((first, "date,in,out\n2026-08-04,0,3000\n"), (second, "date,in,out\n2026-08-05,3000,0\n")):
        response = client.post("/api/statement-imports", json={"account_id": account["id"], "mapping": {"date": "date", "deposit": "in", "withdrawal": "out"}, "csv_text": csv_text})
        assert response.status_code == 201
    lines = client.get("/api/statement-lines").json()
    outgoing = next(line for line in lines if line["amount"] < 0)
    candidates = client.get(f"/api/statement-lines/{outgoing['id']}/transfer-candidates").json()["candidates"]
    result = client.post(f"/api/statement-lines/{outgoing['id']}/transfer", json={"other_line_id": candidates[0]["id"]})
    assert result.status_code == 201
    assert result.json()["amount"] == 3000


def test_statement_import_rejects_bad_rows_without_saving(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    bank = client.post("/api/accounts", json={"name": "Bank", "account_type": "bank"}).json()
    response = client.post("/api/statement-imports", json={"account_id": bank["id"], "mapping": {"date": "date", "amount": "amount"}, "csv_text": "date,amount\n2026-08-01,100\nbad,-20\n"})
    assert response.status_code == 400
    assert client.get("/api/statement-lines").json() == []


def test_salary_tax_api_returns_breakdown_and_sources(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    response = client.post("/api/tax/estimate", json={
        "tax_year": 2025, "gross_salary": 6000000, "social_insurance_premiums": 900000,
    })
    assert response.status_code == 200
    assert response.json()["total_tax"] == 184290
    assert response.json()["rule_version"] == "jp-income-tax-2025-v1"
    assert response.json()["sources"]


def test_salary_tax_api_rejects_invalid_and_unsupported_years(tmp_path):
    client = TestClient(create_app(str(tmp_path / "finance.db")))
    invalid = client.post("/api/tax/estimate", json={"tax_year": 2025, "gross_salary": -1})
    unsupported = client.post("/api/tax/estimate", json={"tax_year": 2099, "gross_salary": 1000000})
    assert invalid.status_code == 400
    assert unsupported.status_code == 422
