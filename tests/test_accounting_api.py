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
