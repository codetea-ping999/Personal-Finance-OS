from __future__ import annotations

"""Structured intent contracts and the offline Japanese intent provider.

The provider deliberately understands a small, explicit vocabulary.  It never
falls back to fuzzy interpretation: requests with missing or ambiguous values
raise :class:`ClarificationRequired` and are kept outside the trusted write
path.
"""

from abc import ABC, abstractmethod
from calendar import monthrange
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
import re
from typing import Any, Callable, Iterable
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .forecast import DEFAULT_FORECAST_YEARS
from .models import CashFlowType, LifeEventType, TransactionKind


PARSER_VERSION = "jp-local-rules-v1"


class IntentType(StrEnum):
    SUMMARY = "summary"
    MONTHLY_REPORT = "monthly_report"
    ACCOUNT_LIST = "account_list"
    TRANSACTION_LIST = "transaction_list"
    FORECAST = "forecast"
    FORECAST_COMPARE = "forecast_compare"
    PURCHASE_SIMULATION = "purchase_simulation"
    SALARY_TAX = "salary_tax"
    CREATE_TRANSACTION = "create_transaction"
    CREATE_JOURNAL_ENTRY = "create_journal_entry"
    CREATE_RECURRING_CASH_FLOW = "create_recurring_cash_flow"
    CREATE_LIFE_EVENT = "create_life_event"


class IntentPeriod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def validate_order(self) -> "IntentPeriod":
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("period end must be on or after period start")
        return self


class Intent(BaseModel):
    """The only object allowed to cross from a provider into the AI CFO layer."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    parser_version: str = PARSER_VERSION
    period: IntentPeriod | None = None
    data_sources: list[str] = Field(default_factory=list)

    @property
    def intent_type(self) -> str:
        return self.type

    @property
    def normalized_arguments(self) -> dict[str, Any]:
        return self.arguments


class ClarificationRequired(ValueError):
    """Raised when a provider cannot safely normalize a request."""

    def __init__(self, reason: str, questions: Iterable[str] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.questions = list(questions)


class IntentProvider(ABC):
    """Provider boundary for local rules today and an LLM tomorrow."""

    parser_version = PARSER_VERSION

    @abstractmethod
    def parse(self, text: str) -> Intent:
        raise NotImplementedError


class LLMIntentProvider(IntentProvider):
    """Future structured-output LLM adapter.

    An implementation may call an LLM, but it must return this module's
    validated ``Intent`` and must not execute repository operations itself.
    """

    @abstractmethod
    def request_structured_intent(self, text: str) -> Intent:
        raise NotImplementedError

    def parse(self, text: str) -> Intent:
        intent = self.request_structured_intent(text)
        if not isinstance(intent, Intent):
            raise ValueError("LLM provider must return a validated Intent")
        return intent


class MockIntentProvider(IntentProvider):
    """Deterministic provider for tests and local integrations."""

    def __init__(
        self,
        intents: Iterable[Intent] | Intent | None = None,
        handler: Callable[[str], Intent] | None = None,
    ) -> None:
        if isinstance(intents, Intent):
            intents = (intents,)
        self._intents = iter(intents or ())
        self._handler = handler

    def parse(self, text: str) -> Intent:
        if self._handler is not None:
            return self._handler(text)
        try:
            return next(self._intents)
        except StopIteration as exc:
            raise ClarificationRequired(
                "mock provider has no intent",
                ["テスト用Intentを1件以上指定してください"],
            ) from exc


def _normalized_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _parse_date(value: str, label: str = "日付") -> date:
    value = value.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value):
            year, month, day = (int(part) for part in value.split("-"))
        elif re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", value):
            year, month, day = (int(part) for part in value.split("/"))
        else:
            match = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日?", value)
            if not match:
                raise ValueError
            year, month, day = (int(item) for item in match.groups())
        return date(year, month, day)
    except (TypeError, ValueError) as exc:
        raise ClarificationRequired(
            f"{label}を解釈できません: {value}",
            [f"{label}をYYYY-MM-DD形式で指定してください"],
        ) from exc


def _parse_month_period(text: str) -> IntentPeriod | None:
    match = re.search(r"(\d{4})年(\d{1,2})月", text)
    if not match:
        match = re.search(r"(\d{4})[-/](\d{1,2})(?!\d|[-/]\d)", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        try:
            start = date(year, month, 1)
            end = date(year, month, monthrange(year, month)[1])
            return IntentPeriod(start=start, end=end)
        except ValueError as exc:
            raise ClarificationRequired(
                "対象月を解釈できません",
                ["対象月をYYYY年MM月形式で指定してください"],
            ) from exc
    if "今月" in text:
        today = date.today()
        start = date(today.year, today.month, 1)
        end = date(today.year, today.month, monthrange(today.year, today.month)[1])
        return IntentPeriod(start=start, end=end)
    return None


def _parse_explicit_date(text: str, required: bool = False) -> date | None:
    patterns = (
        r"\d{4}-\d{1,2}-\d{1,2}",
        r"\d{4}/\d{1,2}/\d{1,2}",
        r"\d{4}年\d{1,2}月\d{1,2}日?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _parse_date(match.group(0))
    if required:
        raise ClarificationRequired(
            "書込みの日付がありません",
            ["書込み日をYYYY-MM-DD形式で指定してください"],
        )
    return None


def _parse_amount(text: str, *, label: str = "金額", required: bool = True) -> int | None:
    # A sign is deliberately captured separately so negative amounts are not
    # silently converted into a valid positive write.
    pattern = r"(?P<sign>[+\-−]?)\s*(?P<number>[\d,]+(?:\.\d+)?)\s*(?P<unit>万円|万|千円|千|円)?"
    matches = list(re.finditer(pattern, text))
    if not matches:
        if required:
            raise ClarificationRequired(
                f"{label}がありません",
                [f"{label}を整数の円または万円で指定してください"],
            )
        return None
    # Prefer an amount immediately followed by a currency marker. This avoids
    # interpreting years, months, or account names as money.
    marked = [m for m in matches if m.group("unit")]
    match = marked[0] if marked else None
    if match is None:
        if required:
            raise ClarificationRequired(
                f"{label}の単位がありません",
                [f"{label}に円、千円、または万円を付けてください"],
            )
        return None
    if match.group("sign") in {"-", "−"}:
        raise ClarificationRequired(
            f"{label}は負数にできません",
            [f"{label}は正数で指定してください"],
        )
    try:
        number = Decimal(match.group("number").replace(",", ""))
    except InvalidOperation as exc:
        raise ClarificationRequired(f"{label}を解釈できません") from exc
    multiplier = {None: 1, "円": 1, "千": 1_000, "千円": 1_000, "万": 10_000, "万円": 10_000}
    amount = number * multiplier[match.group("unit")]
    if amount != amount.to_integral_value() or amount <= 0:
        raise ClarificationRequired(
            f"{label}は正の整数円で指定してください",
            [f"{label}を整数の円または万円で指定してください"],
        )
    return int(amount)


def _parse_rate(text: str, default: float | None = None) -> float | None:
    match = re.search(r"(?:年利|利率|リターン|利回り)\s*([+-]?\d+(?:\.\d+)?)\s*%", text)
    if not match:
        return default
    rate = float(match.group(1)) / 100
    if rate <= -1:
        raise ClarificationRequired("利率が範囲外です", ["利率は-100%より大きく指定してください"])
    return rate


def _parse_duration_months(text: str, default: int | None = None) -> int | None:
    matches = re.findall(r"(\d+)\s*(?:年|年間)", text)
    if matches and ("予測" in text or "シミュレーション" in text or "比較" in text):
        return int(matches[-1]) * 12
    match = re.search(r"(\d+)\s*(?:か月|ヶ月|ヵ月|month|months)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default


def _source_for(intent_type: str, arguments: dict[str, Any]) -> list[str]:
    mapping = {
        IntentType.SUMMARY: ["transactions", "accounts", "analytics.FinanceAnalyzer"],
        IntentType.MONTHLY_REPORT: ["journal_entries", "accounts", "monthly_report"],
        IntentType.ACCOUNT_LIST: ["accounts"],
        IntentType.TRANSACTION_LIST: ["transactions", "accounts"],
        IntentType.FORECAST: ["accounts", "transactions", "recurring_cash_flows", "life_events", "forecast.ForecastEngine"],
        IntentType.FORECAST_COMPARE: ["accounts", "transactions", "recurring_cash_flows", "life_events", "forecast.ForecastEngine"],
        IntentType.PURCHASE_SIMULATION: ["transactions", "accounts", "analytics.FinanceAnalyzer", "scenario.simulate_purchase"],
        IntentType.SALARY_TAX: ["tax_rules.jp", "tax_engine.estimate_salary_tax"],
        IntentType.CREATE_TRANSACTION: ["transactions", "repository.create_transaction"],
        IntentType.CREATE_JOURNAL_ENTRY: ["journal_entries", "repository.create_journal_entry"],
        IntentType.CREATE_RECURRING_CASH_FLOW: ["recurring_cash_flows", "repository.create_recurring_cash_flow"],
        IntentType.CREATE_LIFE_EVENT: ["life_events", "repository.create_life_event"],
    }
    return mapping.get(intent_type, [])


class LocalJapaneseIntentProvider(IntentProvider):
    """Limited, offline Japanese parser for explicit v1 request forms."""

    parser_version = PARSER_VERSION

    def parse(self, text: str) -> Intent:
        text = _normalized_text(text)
        if not text:
            raise ClarificationRequired("入力が空です", ["自然言語の依頼を入力してください"])

        # Write intents are checked first because they often contain words like
        # 収支 or 給料 that also occur in read questions.
        if "借方" in text and "貸方" in text:
            return self._journal(text)
        if any(word in text for word in ("ライフイベント", "ライフ・イベント")):
            return self._life_event(text)
        if "毎月" in text and any(word in text for word in ("登録", "追加", "記録")):
            return self._recurring(text)
        if any(word in text for word in ("登録", "追加", "記録")) and any(
            word in text for word in ("支出", "出金", "収入", "入金", "給料", "取引")
        ):
            return self._transaction(text)

        if "税" in text and any(word in text for word in ("給与", "年収", "所得")):
            return self._salary_tax(text)
        if "購入" in text and any(word in text for word in ("シミュレーション", "シュミレーション", "試算", "買う", "買")):
            return self._purchase(text)
        if "比較" in text and any(word in text for word in ("予測", "シナリオ", "ケース")):
            return self._forecast_compare(text)
        if any(word in text for word in ("将来予測", "資産予測", "フォーキャスト")):
            return self._forecast(text)
        if any(word in text for word in ("月次レポート", "月間レポート", "月次報告")):
            return self._monthly_report(text)
        if "口座" in text and any(word in text for word in ("一覧", "リスト", "残高", "教えて", "見せて")):
            return self._account_list(text)
        if "取引" in text and any(word in text for word in ("一覧", "明細", "履歴", "リスト", "教えて", "見せて")):
            return self._transaction_list(text)
        if any(word in text for word in ("収支", "資産", "純資産", "健康スコア", "サマリー")):
            return self._summary(text)

        raise ClarificationRequired(
            "対応するIntentを特定できません",
            ["収支、月次レポート、口座一覧、取引一覧、将来予測、購入シミュレーション、給与所得税のいずれかを指定してください"],
        )

    def _intent(self, intent_type: str, arguments: dict[str, Any], period: IntentPeriod | None = None) -> Intent:
        return Intent(
            type=intent_type,
            arguments=arguments,
            parser_version=self.parser_version,
            period=period,
            data_sources=_source_for(intent_type, arguments),
        )

    def _summary(self, text: str) -> Intent:
        period = _parse_month_period(text)
        return self._intent(IntentType.SUMMARY, {"start": period.start if period else None, "end": period.end if period else None}, period)

    def _monthly_report(self, text: str) -> Intent:
        period = _parse_month_period(text)
        if period is None:
            raise ClarificationRequired("月次レポートの対象月がありません", ["対象月をYYYY年MM月形式で指定してください"])
        return self._intent(IntentType.MONTHLY_REPORT, {"start": period.start, "end": period.end}, period)

    def _account_list(self, text: str) -> Intent:
        return self._intent(IntentType.ACCOUNT_LIST, {})

    def _transaction_list(self, text: str) -> Intent:
        period = _parse_month_period(text)
        arguments = {"start": period.start if period else None, "end": period.end if period else None}
        return self._intent(IntentType.TRANSACTION_LIST, arguments, period)

    def _forecast(self, text: str) -> Intent:
        years = _parse_duration_months(text, DEFAULT_FORECAST_YEARS * 12)
        scenario_match = re.search(r"(?:シナリオ|ケース)\s*[「『]?([^」』、,]+)", text)
        scenario_name = scenario_match.group(1).split("の", 1)[0].strip(" 「『」』") if scenario_match else "Base case"
        period = _parse_month_period(text)
        start = _parse_explicit_date(text, required=False)
        if start:
            period = IntentPeriod(start=start)
        arguments = {"start_date": period.start if period else None, "period_years": max(1, years // 12), "scenario_name": scenario_name, "overrides": {}}
        return self._intent(IntentType.FORECAST, arguments, period)

    def _forecast_compare(self, text: str) -> Intent:
        years = _parse_duration_months(text, DEFAULT_FORECAST_YEARS * 12)
        period = _parse_month_period(text)
        start = _parse_explicit_date(text, required=False)
        if start:
            period = IntentPeriod(start=start)
        # Explicit separator forms are intentional: "基本ケースと転職ケース".
        match = re.search(
            r"([^\s、,。]+?ケース)\s*と\s*([^\s、,。]+?ケース)\s*(?:を|の)?(?:将来予測を)?(?:比較|比べ)(?:して)?",
            text,
        )
        if not match:
            raise ClarificationRequired("比較するケースがありません", ["比較するケースを「AとBを比較」の形式で指定してください"])
        names = [item.rsplit("の", 1)[-1].strip(" 「『」』") for item in match.groups()]
        if any(not item or item in {"ケース", "シナリオ"} for item in names):
            raise ClarificationRequired("比較するケース名が曖昧です", ["比較するケース名を明示してください"])
        arguments = {"start_date": period.start if period else None, "period_years": max(1, years // 12), "cases": [{"name": name, "overrides": {}} for name in names]}
        return self._intent(IntentType.FORECAST_COMPARE, arguments, period)

    def _purchase(self, text: str) -> Intent:
        price = _parse_amount(text, label="購入価格")
        horizon = _parse_duration_months(text, 60) or 60
        rate = _parse_rate(text, 0.04)
        return self._intent(IntentType.PURCHASE_SIMULATION, {"price": price, "horizon_months": horizon, "annual_return_rate": rate})

    def _salary_tax(self, text: str) -> Intent:
        year_match = re.search(r"(\d{4})年", text)
        if not year_match:
            raise ClarificationRequired("税年度がありません", ["税年度をYYYY年形式で指定してください"])
        amounts = re.findall(r"[+-]?\s*\d+(?:\.\d+)?\s*(?:万円|万|千円|千|円)", text)
        if not amounts:
            raise ClarificationRequired("給与額がありません", ["年収を正数の円または万円で指定してください"])
        salary = _parse_amount(amounts[0], label="年収")
        social = 0
        if any(word in text for word in ("社会保険", "社保")):
            if len(amounts) < 2:
                raise ClarificationRequired("社会保険料がありません", ["社会保険料を正数の円または万円で指定してください"])
            social = _parse_amount(amounts[1], label="社会保険料") or 0
        tax_year = int(year_match.group(1))
        period = IntentPeriod(start=date(tax_year, 1, 1), end=date(tax_year, 12, 31))
        return self._intent(
            IntentType.SALARY_TAX,
            {"tax_year": tax_year, "gross_salary": salary, "social_insurance_premiums": social},
            period,
        )

    def _transaction(self, text: str) -> Intent:
        booked_on = _parse_explicit_date(text, required=True)
        amount = _parse_amount(text)
        if any(word in text for word in ("支出", "出金", "使った", "支払")):
            kind = TransactionKind.EXPENSE.value
        elif any(word in text for word in ("収入", "入金", "給料", "受け取", "income")):
            kind = TransactionKind.INCOME.value
        elif any(word in text for word in ("expense", "支出")):
            kind = TransactionKind.EXPENSE.value
        else:
            raise ClarificationRequired("取引種別が曖昧です", ["収入または支出を明示してください"])
        account = self._extract_account(text, ("口座", "から", "に"))
        category_match = re.search(
            r"(?:カテゴリ|カテゴリー|科目|項目)\s*[：:=]?\s*([^、,。\s]+?)(?=を|として|登録|追加|記録|$)",
            text,
        )
        category = category_match.group(1).strip() if category_match else None
        if category is None:
            for candidate in ("食費", "家賃", "交通費", "光熱費", "給与", "給料"):
                if candidate in text:
                    category = candidate
                    break
        if not category:
            raise ClarificationRequired("カテゴリがありません", ["取引カテゴリを指定してください"])
        description_match = re.search(r"(?:摘要|説明|メモ)\s*[：:]?\s*([^。]+)", text)
        description = description_match.group(1).strip() if description_match else ""
        return self._intent(IntentType.CREATE_TRANSACTION, {
            "account_name": account, "booked_on": booked_on, "amount": amount,
            "kind": kind, "category": category, "description": description,
        })

    def _journal(self, text: str) -> Intent:
        booked_on = _parse_explicit_date(text, required=True)
        amount = _parse_amount(text)
        debit = self._extract_label_value(text, "借方")
        credit = self._extract_label_value(text, "貸方")
        if not debit or not credit:
            raise ClarificationRequired("借方または貸方口座がありません", ["借方と貸方の口座名を指定してください"])
        description_match = re.search(r"(?:摘要|説明)\s*[：:=]?\s*([^、。]+)", text)
        description = description_match.group(1).strip() if description_match else ""
        description = re.sub(r"(?:として)?\s*(?:登録|追加|記録)$", "", description).strip().removesuffix("を").strip()
        return self._intent(IntentType.CREATE_JOURNAL_ENTRY, {
            "booked_on": booked_on, "description": description,
            "debit_account_name": debit, "credit_account_name": credit, "amount": amount,
        })

    def _recurring(self, text: str) -> Intent:
        start_date = _parse_explicit_date(text, required=True)
        amount = _parse_amount(text)
        if any(word in text for word in ("支出", "出費", "費用")):
            flow_type = CashFlowType.EXPENSE.value
        elif any(word in text for word in ("収入", "給与", "給料", "入金")):
            flow_type = CashFlowType.INCOME.value
        else:
            raise ClarificationRequired("定期収支の種別が曖昧です", ["収入または支出を明示してください"])
        name_match = re.search(r"(?:定期収支|毎月)\s*[「『]?([^、,。]+?)(?:」』)?\s*(?:を|として|が)?\s*(?:毎月|登録|追加)", text)
        name = name_match.group(1).strip(" 「『」』") if name_match else None
        if not name or name in {"収入", "支出"}:
            name_match = re.search(r"(?:給与|給料|家賃|保険|サブスク|賃料)", text)
            name = name_match.group(0) if name_match else None
        if not name:
            raise ClarificationRequired("定期収支名がありません", ["定期収支の名前を指定してください"])
        end_date = _parse_explicit_date(text[text.find("まで"):] if "まで" in text else "", required=False) if "まで" in text else None
        return self._intent(IntentType.CREATE_RECURRING_CASH_FLOW, {
            "name": name, "flow_type": flow_type, "amount": amount,
            "start_date": start_date, "end_date": end_date,
        })

    def _life_event(self, text: str) -> Intent:
        start_date = _parse_explicit_date(text, required=True)
        name_match = re.search(r"(?:ライフイベント|ライフ・イベント)\s*[「『]?[：:=]?\s*([^、,。]+?)(?:」』)?(?:を|として)", text)
        if not name_match:
            name_match = re.search(r"(?:ライフイベント|ライフ・イベント)\s*[：:=]?\s*([^、,。]+)", text)
        name = name_match.group(1).strip(" 「『」』") if name_match else None
        if not name:
            raise ClarificationRequired("ライフイベント名がありません", ["ライフイベント名を指定してください"])
        duration = _parse_duration_months(text, 1) or 1
        event_type = LifeEventType.RECURRING.value if any(
            word in text for word in ("毎月", "継続", "期間", "recurring", "定期")
        ) else LifeEventType.ONE_TIME.value
        income = _parse_labeled_amount(text, ("収入増", "収入増加", "収入"), 0)
        expense = _parse_labeled_amount(text, ("支出増", "支出増加", "支出", "費用"), 0)
        return self._intent(IntentType.CREATE_LIFE_EVENT, {
            "name": name, "start_date": start_date, "duration_months": duration,
            "event_type": event_type, "income_delta": income, "expense_delta": expense,
        })

    @staticmethod
    def _extract_label_value(text: str, label: str) -> str | None:
        match = re.search(
            rf"{re.escape(label)}\s*[：:=]?\s*([^、,。]+?)(?=\s*(?:借方|貸方|金額|摘要|説明)|[、,。]|$)",
            text,
        )
        return match.group(1).strip() if match else None

    def _extract_account(self, text: str, labels: tuple[str, ...]) -> str:
        for pattern in (
            r"(?:を|の)\s*([^、,。]+?(?:口座|アカウント))\s*(?:から|に|へ)",
            r"(?:^|[、,])\s*([^、,。]+?(?:口座|アカウント))\s*(?:から|に|へ)",
            r"(?:を|の)\s*([^、,。\s]+)\s*(?:から|に|へ)",
        ):
            before_account = re.search(pattern, text)
            if before_account:
                return before_account.group(1).strip()
        explicit = re.search(r"(?:口座|アカウント)\s*[：:=]?\s*([^、,。]+)", text)
        if explicit:
            return explicit.group(1).strip().removesuffix("から").removesuffix("に").strip()
        for label in labels:
            match = re.search(rf"([^、,。\s]+){re.escape(label)}", text)
            if match:
                return match.group(1).strip()
        raise ClarificationRequired("口座名がありません", ["対象口座名を明示してください"])


# Friendly aliases for callers that prefer the shorter names.
JapaneseRuleIntentProvider = LocalJapaneseIntentProvider
LocalRuleIntentProvider = LocalJapaneseIntentProvider
FutureLLMIntentProvider = LLMIntentProvider
IntentKind = IntentType


def _parse_labeled_amount(text: str, labels: tuple[str, ...], default: int) -> int:
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*(?:[：:=]|は|が)?\s*([+\-−]?\s*\d+(?:\.\d+)?\s*(?:万円|万|千円|千|円))", text)
        if match:
            amount = _parse_amount(match.group(1), label=label)
            return amount or default
    return default
