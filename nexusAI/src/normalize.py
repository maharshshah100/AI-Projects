"""
Normalization: maps whatever labels a real-world statement uses
("Revenue" vs "Total Sales" vs "Net Revenue") onto one canonical internal
schema, so the analysis engine always works with consistent field names
regardless of how the source file happens to be labeled.

Two-tier matching, cheapest first:
  1. Exact/alias match against a hand-curated dictionary of common
     real-world label variants (fast, zero false positives for anything
     we've seen before).
  2. Fuzzy match (rapidfuzz) as a fallback for labels that are close to
     a known alias but not identical (e.g. a typo, or a variant we
     haven't explicitly listed) -- with a high similarity threshold so
     it doesn't silently misclassify something it shouldn't.

Anything that matches neither is kept as "unmapped" rather than guessed
at -- an unmapped line item is reported, not silently dropped or forced
into the wrong bucket.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from ingest import RawStatement

FUZZY_THRESHOLD = 85  # 0-100 scale; conservative on purpose

# canonical_field -> list of known real-world label variants (lowercase)
ALIASES: dict[str, list[str]] = {
    # --- Income Statement ---
    "revenue": ["revenue", "total revenue", "net revenue", "total sales", "net sales", "sales"],
    "cogs": ["cost of goods sold", "cogs", "cost of sales", "cost of revenue"],
    "gross_profit": ["gross profit", "gross income"],
    "sga": ["selling, general & administrative", "sg&a", "selling general and administrative",
            "general and administrative expenses"],
    "rnd": ["research & development", "research and development", "r&d"],
    "marketing": ["marketing", "marketing expenses", "advertising"],
    "total_opex": ["total operating expenses", "operating expenses", "total opex"],
    "operating_income": ["operating income", "operating income (ebit)", "ebit"],
    "interest_expense": ["interest expense", "interest expenses", "net interest expense"],
    "pretax_income": ["pre-tax income", "pretax income", "income before taxes",
                       "earnings before tax"],
    "tax_expense": ["income tax expense", "tax expense", "provision for income taxes",
                     "income tax expense / (benefit)", "income tax expense (benefit)",
                     "income tax (benefit) expense", "tax provision (benefit)"],
    "net_income": ["net income", "net earnings", "net profit", "profit for the year",
                    "net income / (loss)", "net income (loss)", "net income/(loss)",
                    "net loss", "net income (loss) attributable to shareholders"],

    # --- Balance Sheet ---
    "cash": ["cash and cash equivalents", "cash & cash equivalents", "cash"],
    "accounts_receivable": ["accounts receivable", "trade receivables", "receivables"],
    "inventory": ["inventory", "inventories"],
    "prepaid_expenses": ["prepaid expenses", "prepaid expense"],
    "total_current_assets": ["total current assets"],
    "ppe_net": ["property, plant & equipment (net)", "property plant and equipment",
                "ppe, net", "net ppe"],
    "intangible_assets": ["intangible assets", "goodwill and intangible assets"],
    "total_noncurrent_assets": ["total non-current assets", "total noncurrent assets",
                                 "total long-term assets"],
    "total_assets": ["total assets"],
    "accounts_payable": ["accounts payable", "trade payables"],
    "short_term_debt": ["short-term debt", "short term debt", "current portion of debt"],
    "accrued_liabilities": ["accrued liabilities", "accrued expenses"],
    "total_current_liabilities": ["total current liabilities"],
    "long_term_debt": ["long-term debt", "long term debt", "non-current debt"],
    "deferred_tax_liabilities": ["deferred tax liabilities"],
    "total_noncurrent_liabilities": ["total non-current liabilities", "total noncurrent liabilities",
                                      "total long-term liabilities"],
    "total_liabilities": ["total liabilities"],
    "common_stock": ["common stock", "share capital", "paid-in capital"],
    "retained_earnings": ["retained earnings", "accumulated earnings"],
    "total_equity": ["total shareholders' equity", "total shareholders equity",
                      "total equity", "total stockholders' equity"],

    # --- Cash Flow ---
    "da": ["depreciation & amortization", "depreciation and amortization", "d&a"],
    "change_ar": ["change in accounts receivable", "increase/decrease in receivables"],
    "change_inventory": ["change in inventory", "increase/decrease in inventory"],
    "change_ap": ["change in accounts payable", "increase/decrease in payables"],
    "cfo": ["operating cash flow", "operating cash flows", "cash flow from operating activities", "cash from operating activities",
            "net cash from operating activities", "cash flow from operations",
            "net cash from operations", "net cash provided by operating activities",
            "net cash provided by operating activities / operating cash flow"],
    "capex": ["capital expenditures", "capex", "purchases of property and equipment"],
    "cfi": ["investing cash flow", "cash from investing activities", "net cash from investing activities",
            "net cash used in investing", "net cash used in investing activities"],
    "debt_issuance": ["net debt issuance / (repayment)", "net debt issuance",
                       "proceeds from (repayment of) debt"],
    "dividends_paid": ["dividends paid", "cash dividends paid"],
    "cff": ["financing cash flow", "cash from financing activities", "net cash from financing activities",
            "net cash used in financing activities", "net cash provided by financing activities",
            "net cash from financing", "net cash used in financing"],
    "net_change_cash": ["net change in cash", "net increase (decrease) in cash", "net increase in cash and cash equivalents", "net change in cash and cash equivalents", "increase (decrease) in cash and cash equivalents"],
    "beginning_cash": ["beginning cash balance", "cash at beginning of period", "cash and cash equivalents at beginning of period"],
    "ending_cash": ["ending cash balance", "cash at end of period", "cash and cash equivalents at end of period"],
}

# Rows we deliberately never map -- they're derived display metrics already
# computed by the source file (percentages, checks), not raw line items the
# analysis engine should treat as inputs. Re-deriving our own ratios from
# raw line items is more reliable than trusting a percent column that could
# be formatted/rounded differently across source files.
IGNORE_PATTERNS = ["margin %", "balance check", "growth %"]

# Build the reverse lookup once: alias string -> canonical field
_ALIAS_TO_FIELD: dict[str, str] = {}
for field_name, variants in ALIASES.items():
    for v in variants:
        _ALIAS_TO_FIELD[v] = field_name


@dataclass
class NormalizedStatement:
    sheet_name: str
    periods: list[str]
    unit_hint: str | None = None
    fields: dict[str, dict[str, float]] = field(default_factory=dict)   # canonical_field -> {period: value}
    unmapped: list[tuple[str, dict[str, float]]] = field(default_factory=list)
    match_log: list[tuple[str, str, str, float]] = field(default_factory=list)  # (raw_label, matched_field, method, score)


def _should_ignore(label_lower: str) -> bool:
    return any(pat in label_lower for pat in IGNORE_PATTERNS)


def normalize_statement(stmt: RawStatement) -> NormalizedStatement:
    result = NormalizedStatement(sheet_name=stmt.sheet_name, periods=stmt.periods, unit_hint=stmt.unit_hint)

    for raw_label, values in stmt.rows:
        label_lower = raw_label.strip().lower()

        if _should_ignore(label_lower):
            continue

        # Tier 1: exact alias match
        if label_lower in _ALIAS_TO_FIELD:
            field_name = _ALIAS_TO_FIELD[label_lower]
            result.fields[field_name] = values
            result.match_log.append((raw_label, field_name, "exact", 100.0))
            continue

        # Tier 2: fuzzy match against all known aliases
        match = process.extractOne(
            label_lower, _ALIAS_TO_FIELD.keys(), scorer=fuzz.token_sort_ratio
        )
        if match is not None and match[1] >= FUZZY_THRESHOLD:
            matched_alias, score, _ = match
            field_name = _ALIAS_TO_FIELD[matched_alias]
            result.fields[field_name] = values
            result.match_log.append((raw_label, field_name, "fuzzy", score))
            continue

        # No confident match -- report it, don't guess
        result.unmapped.append((raw_label, values))
        result.match_log.append((raw_label, "UNMAPPED", "none", 0.0))

    return result


def normalize_workbook(raw_statements: dict[str, RawStatement]) -> dict[str, NormalizedStatement]:
    return {name: normalize_statement(stmt) for name, stmt in raw_statements.items()}


if __name__ == "__main__":
    from ingest import ingest_workbook

    raw = ingest_workbook("data/sample/Nimbus_Retail_Financial_Statements.xlsx")
    normalized = normalize_workbook(raw)

    for sheet_name, norm in normalized.items():
        print(f"\n=== {sheet_name} ===")
        print(f"Mapped fields: {len(norm.fields)}  |  Unmapped: {len(norm.unmapped)}")
        for raw_label, field_name, method, score in norm.match_log:
            marker = "OK" if field_name != "UNMAPPED" else "!!"
            print(f"  [{marker}] {raw_label!r:45s} -> {field_name:28s} ({method}, {score:.0f})")
