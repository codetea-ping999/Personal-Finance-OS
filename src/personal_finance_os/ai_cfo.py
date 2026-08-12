from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .ai_intents import (
    ClarificationRequired,
    Intent,
    IntentProvider,
    IntentType,
    LocalJapaneseIntentProvider,
)
from .analytics import FinanceAnalyzer
from .forecast import ForecastEngine
from .models import AccountType, CashFlowType, LifeEventType, TransactionKind
from .repository import FinanceRepository
from .scenario import simulate_purchase
from .tax_engine import estimate_salary_tax


class SummaryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date | None = None
    end: date | None = None


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MonthlyReportArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date
    end: date


class TransactionListArguments(SummaryArguments):
    pass


class ForecastArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date | None = None
    period_years: int = Field(default=30, ge=1, le=50)
    scenario_name: str = Field(default="Base case", min_length=1, max_length=100)
    overrides: dict[str, int | float | None] = Field(default_factory=dict)


class ForecastCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    overrides: dict[str, int | float | None] = Field(default_factory=dict)


class ForecastCompareArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date | None = None
    period_years: int = Field(default=30, ge=1, le=50)
    cases: list[ForecastCase] = Field(min_length=1, max_length=20)


class PurchaseArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    price: int = Field(gt=0)
    horizon_months: int = Field(default=60, gt=0, le=600)
    annual_return_rate: float = Field(default=0.04, gt=-1, le=1)


class SalaryTaxArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tax_year: int = Field(ge=1900, le=2100)
    gross_salary: int = Field(ge=0)
    social_insurance_premiums: int = Field(default=0, ge=0)


class CreateTransactionArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_name: str = Field(min_length=1, max_length=100)
    account_id: str | None = None
    booked_on: date
    amount: int = Field(gt=0)
    kind: TransactionKind
    category: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class CreateJournalArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booked_on: date
    description: str = Field(default="", max_length=500)
    debit_account_name: str = Field(min_length=1, max_length=100)
    credit_account_name: str = Field(min_length=1, max_length=100)
    debit_account_id: str | None = None
    credit_account_id: str | None = None
    amount: int = Field(gt=0)


class CreateRecurringArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    flow_type: CashFlowType
    amount: int = Field(gt=0)
    start_date: date
    end_date: date | None = None


class CreateLifeEventArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    start_date: date
    duration_months: int = Field(gt=0, le=600)
    event_type: LifeEventType = LifeEventType.ONE_TIME
    income_delta: int = 0
    expense_delta: int = 0


ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    IntentType.SUMMARY: SummaryArguments,
    IntentType.MONTHLY_REPORT: MonthlyReportArguments,
    IntentType.ACCOUNT_LIST: EmptyArguments,
    IntentType.TRANSACTION_LIST: TransactionListArguments,
    IntentType.FORECAST: ForecastArguments,
    IntentType.FORECAST_COMPARE: ForecastCompareArguments,
    IntentType.PURCHASE_SIMULATION: PurchaseArguments,
    IntentType.SALARY_TAX: SalaryTaxArguments,
    IntentType.CREATE_TRANSACTION: CreateTransactionArguments,
    IntentType.CREATE_JOURNAL_ENTRY: CreateJournalArguments,
    IntentType.CREATE_RECURRING_CASH_FLOW: CreateRecurringArguments,
    IntentType.CREATE_LIFE_EVENT: CreateLifeEventArguments,
}

READ_INTENTS = {
    IntentType.SUMMARY,
    IntentType.MONTHLY_REPORT,
    IntentType.ACCOUNT_LIST,
    IntentType.TRANSACTION_LIST,
    IntentType.FORECAST,
    IntentType.FORECAST_COMPARE,
    IntentType.PURCHASE_SIMULATION,
    IntentType.SALARY_TAX,
}
WRITE_INTENTS = set(ARGUMENT_MODELS) - READ_INTENTS


class AIActionError(ValueError):
    def __init__(self, message: str, status_code: int = 400, *, action_id: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.action_id = action_id


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return value


def _dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _account_key(value: str) -> str:
    return "".join(value.casefold().split()).replace("口座", "")


class AICFOService:
    """Orchestrates validated intents over deterministic finance services."""

    def __init__(
        self,
        repository: FinanceRepository,
        analyzer: FinanceAnalyzer | None = None,
        forecast_engine: ForecastEngine | None = None,
        provider: IntentProvider | None = None,
        preview_ttl_seconds: int = 600,
    ) -> None:
        if preview_ttl_seconds <= 0:
            raise ValueError("preview_ttl_seconds must be positive")
        self.repository = repository
        self.analyzer = analyzer or FinanceAnalyzer(repository)
        self.forecast_engine = forecast_engine or ForecastEngine(repository)
        self.provider = provider or LocalJapaneseIntentProvider()
        self.preview_ttl_seconds = preview_ttl_seconds

    def query(self, raw_input: str) -> dict[str, Any]:
        raw_input = raw_input.strip()
        try:
            intent = self.provider.parse(raw_input)
        except ClarificationRequired as exc:
            audit_id = self._log_failure(raw_input, getattr(self.provider, "parser_version", "unknown"), str(exc))
            return {
                "status": "needs_clarification",
                "needs_clarification": True,
                "questions": exc.questions,
                "reason": exc.reason,
                "audit_id": audit_id,
            }
        except (ValueError, TypeError) as exc:
            audit_id = self._log_failure(raw_input, getattr(self.provider, "parser_version", "unknown"), str(exc))
            return {
                "status": "needs_clarification",
                "needs_clarification": True,
                "questions": ["依頼を対応する定型表現で明示してください"],
                "reason": str(exc),
                "audit_id": audit_id,
            }

        try:
            validated = self._validate_intent(intent)
            if intent.type in WRITE_INTENTS:
                return self._preview(raw_input, intent, validated)
            result = self._read(intent.type, validated)
        except (ValidationError, ValueError, KeyError, LookupError) as exc:
            audit_id = self._log_failure(raw_input, intent.parser_version, self._error_text(exc), intent)
            return {
                "status": "needs_clarification",
                "needs_clarification": True,
                "questions": [self._error_text(exc)],
                "reason": self._error_text(exc),
                "audit_id": audit_id,
            }

        explanation = {
            "period": self._result_period(intent, result, validated),
            "data_sources": intent.data_sources,
            "parser_version": intent.parser_version,
            "rule_version": self._rule_version(intent.type, result),
        }
        result_json = _dumps(result)
        audit_id = self.repository.create_ai_action_log(
            raw_input=raw_input,
            intent_json=_dumps(intent),
            parser_version=intent.parser_version,
            state="executed",
            result_json=result_json,
        )
        return {
            "status": "executed",
            "needs_clarification": False,
            "audit_id": audit_id,
            "intent": _jsonable(intent),
            "result": _jsonable(result),
            "explanation": explanation,
            "period": explanation["period"],
            "data_sources": intent.data_sources,
            "parser_version": intent.parser_version,
            "rule_version": explanation["rule_version"],
        }

    def confirm(self, confirmation_token: str) -> dict[str, Any]:
        if not confirmation_token or not confirmation_token.strip():
            raise AIActionError("confirmation token is required", 400)
        token_hash = sha256(confirmation_token.encode("utf-8")).hexdigest()
        action = self.repository.find_ai_action_by_token_hash(token_hash)
        if action is None:
            raise AIActionError("invalid confirmation token", 400)
        action_id = str(action["id"])
        state = str(action["state"])
        if state != "previewed":
            raise AIActionError(
                f"action cannot be confirmed from state: {state}",
                409,
                action_id=action_id,
            )
        expires_at = self._parse_timestamp(action.get("expires_at"))
        if expires_at is None or _now() >= expires_at:
            self.repository.update_ai_action_log(
                action_id, state="expired", error="confirmation token expired"
            )
            raise AIActionError("confirmation token expired", 409, action_id=action_id)

        expected_fingerprint = str(action.get("ledger_fingerprint") or "")
        if expected_fingerprint != self.repository.ledger_fingerprint():
            error = "ledger changed after preview; confirmation rejected"
            self.repository.update_ai_action_log(action_id, state="rejected", error=error)
            raise AIActionError(error, 409, action_id=action_id)

        confirmed_at = _now().isoformat()
        if not self.repository.confirm_ai_action(action_id, confirmed_at):
            current = self.repository.get_ai_action_log(action_id) or {}
            raise AIActionError(
                f"action cannot be confirmed from state: {current.get('state', 'unknown')}",
                409,
                action_id=action_id,
            )

        # Re-check after the atomic reservation so a concurrent ledger write
        # cannot slip between validation and execution.
        if expected_fingerprint != self.repository.ledger_fingerprint():
            error = "ledger changed after preview; confirmation rejected"
            self.repository.update_ai_action_log(action_id, state="rejected", error=error)
            raise AIActionError(error, 409, action_id=action_id)

        try:
            intent = Intent.model_validate(json.loads(str(action["intent_json"])))
            validated = self._validate_intent(intent)
            result = self._execute(intent.type, validated)
        except (ValidationError, ValueError, KeyError, LookupError) as exc:
            error = self._error_text(exc)
            self.repository.update_ai_action_log(action_id, state="failed", error=error)
            raise AIActionError(error, 400, action_id=action_id) from exc

        result_json = _dumps(result)
        self.repository.update_ai_action_log(
            action_id,
            state="executed",
            executed_at=_now().isoformat(),
            result_json=result_json,
        )
        return {
            "status": "executed",
            "action_id": action_id,
            "audit_id": action_id,
            "intent": _jsonable(intent),
            "result": _jsonable(result),
        }

    def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in self.repository.list_ai_action_logs(limit):
            record = dict(row)
            record.pop("confirmation_token_hash", None)
            for field in ("intent_json", "preview_json", "result_json"):
                value = record.get(field)
                if value:
                    try:
                        record[field.removesuffix("_json")] = json.loads(str(value))
                    except json.JSONDecodeError:
                        record[field.removesuffix("_json")] = value
                record.pop(field, None)
            records.append(record)
        return records

    def _validate_intent(self, intent: Intent) -> BaseModel:
        if intent.type not in ARGUMENT_MODELS:
            raise ValueError(f"unsupported intent type: {intent.type}")
        model = ARGUMENT_MODELS[intent.type]
        return model.model_validate(intent.arguments)

    def _preview(self, raw_input: str, intent: Intent, validated: BaseModel) -> dict[str, Any]:
        effective_intent, preview = self._prepare_write(intent, validated)
        token = f"{uuid4().hex}{uuid4().hex}"
        expires_at = _now() + timedelta(seconds=self.preview_ttl_seconds)
        action_id = self.repository.create_ai_action_log(
            raw_input=raw_input,
            intent_json=_dumps(effective_intent),
            parser_version=effective_intent.parser_version,
            state="previewed",
            confirmation_token_hash=sha256(token.encode("utf-8")).hexdigest(),
            expires_at=expires_at.isoformat(),
            ledger_fingerprint=self.repository.ledger_fingerprint(),
            preview_json=_dumps(preview),
        )
        return {
            "status": "previewed",
            "needs_clarification": False,
            "action_id": action_id,
            "audit_id": action_id,
            "confirmation_token": token,
            "expires_at": expires_at.isoformat(),
            "intent": _jsonable(effective_intent),
            "preview": _jsonable(preview),
            "explanation": {
                "data_sources": effective_intent.data_sources,
                "parser_version": effective_intent.parser_version,
                "rule_version": "deterministic-service-v1",
            },
        }

    def _prepare_write(self, intent: Intent, validated: BaseModel) -> tuple[Intent, dict[str, Any]]:
        arguments = validated.model_dump(mode="python")
        if intent.type == IntentType.CREATE_TRANSACTION:
            values = validated
            assert isinstance(values, CreateTransactionArguments)
            if values.kind not in {TransactionKind.INCOME, TransactionKind.EXPENSE}:
                raise ValueError("AI CFO v1 supports income and expense transactions only")
            account = self._resolve_account(values.account_name)
            arguments["account_id"] = account.id
            signed_amount = values.amount if values.kind == TransactionKind.INCOME else -values.amount
            return intent.model_copy(update={"arguments": arguments}), {
                "operation": intent.type,
                "table": "transactions",
                "will_change": {
                    "account_id": account.id,
                    "account_name": account.name,
                    "booked_on": values.booked_on,
                    "amount": signed_amount,
                    "kind": values.kind.value,
                    "category": values.category,
                    "description": values.description,
                },
            }
        if intent.type == IntentType.CREATE_JOURNAL_ENTRY:
            values = validated
            assert isinstance(values, CreateJournalArguments)
            debit = self._resolve_account(values.debit_account_name)
            credit = self._resolve_account(values.credit_account_name)
            if debit.id == credit.id:
                raise ValueError("debit and credit accounts must differ")
            arguments["debit_account_id"] = debit.id
            arguments["credit_account_id"] = credit.id
            return intent.model_copy(update={"arguments": arguments}), {
                "operation": intent.type,
                "table": "journal_entries",
                "will_change": {
                    "booked_on": values.booked_on,
                    "description": values.description,
                    "debit_account_id": debit.id,
                    "debit_account_name": debit.name,
                    "credit_account_id": credit.id,
                    "credit_account_name": credit.name,
                    "amount": values.amount,
                },
            }
        if intent.type == IntentType.CREATE_RECURRING_CASH_FLOW:
            values = validated
            assert isinstance(values, CreateRecurringArguments)
            if values.end_date is not None and values.end_date < values.start_date:
                raise ValueError("end_date must be on or after start_date")
            return intent, {
                "operation": intent.type,
                "table": "recurring_cash_flows",
                "will_change": arguments,
            }
        if intent.type == IntentType.CREATE_LIFE_EVENT:
            return intent, {
                "operation": intent.type,
                "table": "life_events",
                "will_change": arguments,
            }
        raise ValueError(f"unsupported write intent: {intent.type}")

    def _execute(self, intent_type: str, validated: BaseModel) -> Any:
        if intent_type == IntentType.CREATE_TRANSACTION:
            values = validated
            assert isinstance(values, CreateTransactionArguments)
            if values.kind == TransactionKind.INCOME:
                amount = values.amount
            elif values.kind == TransactionKind.EXPENSE:
                amount = -values.amount
            else:
                raise ValueError("AI CFO v1 supports income and expense transactions only")
            return self.repository.create_transaction(
                account_id=values.account_id or self._resolve_account(values.account_name).id,
                booked_on=values.booked_on,
                amount=amount,
                kind=values.kind,
                category=values.category,
                description=values.description,
            )
        if intent_type == IntentType.CREATE_JOURNAL_ENTRY:
            values = validated
            assert isinstance(values, CreateJournalArguments)
            return self.repository.create_journal_entry(
                booked_on=values.booked_on,
                description=values.description,
                debit_account_id=values.debit_account_id or self._resolve_account(values.debit_account_name).id,
                credit_account_id=values.credit_account_id or self._resolve_account(values.credit_account_name).id,
                amount=values.amount,
            )
        if intent_type == IntentType.CREATE_RECURRING_CASH_FLOW:
            values = validated
            assert isinstance(values, CreateRecurringArguments)
            return self.repository.create_recurring_cash_flow(**values.model_dump())
        if intent_type == IntentType.CREATE_LIFE_EVENT:
            values = validated
            assert isinstance(values, CreateLifeEventArguments)
            return self.repository.create_life_event(**values.model_dump())
        raise ValueError(f"unsupported write intent: {intent_type}")

    def _read(self, intent_type: str, values: BaseModel) -> Any:
        if intent_type == IntentType.SUMMARY:
            assert isinstance(values, SummaryArguments)
            return self.analyzer.summarize(start=values.start, end=values.end)
        if intent_type == IntentType.MONTHLY_REPORT:
            assert isinstance(values, MonthlyReportArguments)
            return self._monthly_report(values.start, values.end)
        if intent_type == IntentType.ACCOUNT_LIST:
            balances = self.repository.account_balances()
            return [asdict(account) | {"balance": balances.get(account.id, 0)} for account in self.repository.list_accounts()]
        if intent_type == IntentType.TRANSACTION_LIST:
            assert isinstance(values, TransactionListArguments)
            accounts = {account.id: account.name for account in self.repository.list_accounts()}
            return [
                asdict(item) | {"account_name": accounts.get(item.account_id)}
                for item in self.repository.list_transactions(values.start, values.end)
            ]
        if intent_type == IntentType.FORECAST:
            assert isinstance(values, ForecastArguments)
            return self.forecast_engine.forecast(
                start_date=values.start_date,
                period_years=values.period_years,
                scenario_name=values.scenario_name,
                overrides=values.overrides,
            )
        if intent_type == IntentType.FORECAST_COMPARE:
            assert isinstance(values, ForecastCompareArguments)
            results = [
                self.forecast_engine.forecast(
                    start_date=values.start_date,
                    period_years=values.period_years,
                    scenario_name=item.name,
                    overrides=item.overrides,
                )
                for item in values.cases
            ]
            return {
                "source": "forecast",
                "period_years": values.period_years,
                "cases": results,
            }
        if intent_type == IntentType.PURCHASE_SIMULATION:
            assert isinstance(values, PurchaseArguments)
            return simulate_purchase(self.analyzer.summarize(), **values.model_dump())
        if intent_type == IntentType.SALARY_TAX:
            assert isinstance(values, SalaryTaxArguments)
            return estimate_salary_tax(**values.model_dump())
        raise ValueError(f"unsupported read intent: {intent_type}")

    def _monthly_report(self, start: date, end: date) -> dict[str, Any]:
        accounts = {account.id: account for account in self.repository.list_accounts()}
        income = expenses = 0
        for entry in self.repository.list_journal_entries(start, end):
            debit = accounts[entry.debit_account_id]
            credit = accounts[entry.credit_account_id]
            if debit.account_type == AccountType.EXPENSE:
                expenses += entry.amount
            if credit.account_type == AccountType.INCOME:
                income += entry.amount
        balances = self.repository.account_balances()
        assets = sum(
            balances[account.id]
            for account in accounts.values()
            if account.account_type in {AccountType.CASH, AccountType.BANK, AccountType.INVESTMENT}
        )
        liabilities = sum(
            balances[account.id]
            for account in accounts.values()
            if account.account_type == AccountType.LIABILITY
        )
        return {
            "year": start.year,
            "month": start.month,
            "income": income,
            "expenses": expenses,
            "net_income": income - expenses,
            "assets": assets,
            "liabilities": liabilities,
            "equity": assets - liabilities,
        }

    @staticmethod
    def _rule_version(intent_type: str, result: Any) -> str:
        if isinstance(result, dict) and result.get("rule_version"):
            return str(result["rule_version"])
        return {
            IntentType.SUMMARY: "analytics-v1",
            IntentType.MONTHLY_REPORT: "monthly-report-v1",
            IntentType.ACCOUNT_LIST: "ledger-read-v1",
            IntentType.TRANSACTION_LIST: "ledger-read-v1",
            IntentType.FORECAST: "forecast-engine-v1",
            IntentType.FORECAST_COMPARE: "forecast-engine-v1",
            IntentType.PURCHASE_SIMULATION: "purchase-scenario-v1",
        }.get(intent_type, "deterministic-service-v1")

    @staticmethod
    def _result_period(intent: Intent, result: Any, values: BaseModel) -> dict[str, Any] | None:
        if intent.type == IntentType.SALARY_TAX and isinstance(values, SalaryTaxArguments):
            return {"tax_year": values.tax_year, "start": f"{values.tax_year}-01-01", "end": f"{values.tax_year}-12-31"}
        if intent.type == IntentType.FORECAST and hasattr(result, "start_date") and hasattr(result, "end_date"):
            return {"start": _jsonable(result.start_date), "end": _jsonable(result.end_date)}
        if intent.type == IntentType.FORECAST_COMPARE and isinstance(result, dict) and result.get("cases"):
            first = result["cases"][0]
            if hasattr(first, "start_date") and hasattr(first, "end_date"):
                return {"start": _jsonable(first.start_date), "end": _jsonable(first.end_date)}
        if intent.period is not None:
            return _jsonable(intent.period)
        if intent.type == IntentType.PURCHASE_SIMULATION and isinstance(values, PurchaseArguments):
            return {"horizon_months": values.horizon_months}
        return None

    def _resolve_account(self, name: str):
        wanted = _account_key(name)
        accounts = self.repository.list_accounts()
        matches = [account for account in accounts if _account_key(account.name) == wanted]
        if not matches:
            raise ClarificationRequired(
                f"口座が見つかりません: {name}",
                [f"口座名を次のいずれかから明示してください: {', '.join(account.name for account in accounts)}"],
            )
        if len(matches) > 1:
            raise ClarificationRequired(
                f"口座名が曖昧です: {name}",
                [f"対象口座を明示してください: {', '.join(account.name for account in matches)}"],
            )
        return matches[0]

    def _log_failure(self, raw_input: str, parser_version: str, error: str, intent: Intent | None = None) -> str:
        return self.repository.create_ai_action_log(
            raw_input=raw_input,
            intent_json=_dumps(intent) if intent is not None else "{}",
            parser_version=parser_version,
            state="failed",
            error=error,
        )

    @staticmethod
    def _error_text(error: Exception) -> str:
        if isinstance(error, ValidationError):
            return "; ".join(str(item.get("msg", "invalid intent")) for item in error.errors())
        return str(error)

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
