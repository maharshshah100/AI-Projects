"""
Ingestion: reads an uploaded financial statement workbook (Excel/CSV) and
extracts every labeled data row per sheet, with its values across whatever
period columns exist (e.g. FY2023, FY2024, FY2025).

Deliberately dumb and literal at this stage -- it does NOT try to guess
which row means "revenue" vs "cost of goods sold" (that's normalize.py's
job). This module's only responsibility is: turn a messy spreadsheet layout
into a clean list of (label, {period: value}) pairs, robust to title rows,
section headers, blank rows, and notes -- without assuming a fixed row
layout, since a real uploaded file won't match our sample's exact rows.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

import pandas as pd

# A period column header looks like "FY2023", "2023", "FY23", "Q1 2024", etc.
PERIOD_PATTERN = re.compile(
    r"^(FY\s?\d{2,4}|Q[1-4]\s?\d{2,4}|\d{4})$", re.IGNORECASE
)


@dataclass
class RawStatement:
    sheet_name: str
    periods: list[str]                          # e.g. ["FY2023", "FY2024", "FY2025"]
    rows: list[tuple[str, dict[str, float]]] = field(default_factory=list)
    unit_hint: str | None = None                 # e.g. "USD thousands", parsed if present


def _find_header_row(raw: pd.DataFrame) -> tuple[int, list[int], list[str]] | None:
    """
    Scans the top of the sheet for the row that contains period labels
    (FY2023, FY2024, ...) in multiple columns -- that's the header row.
    Returns (header_row_index, value_column_indices, period_labels) or None
    if no such row is found in the first 15 rows (covers title/subtitle/
    note rows above the real table, however many there are).
    """
    max_scan = min(15, len(raw))
    for row_idx in range(max_scan):
        row = raw.iloc[row_idx]
        matches = []
        for col_idx, val in enumerate(row):
            if isinstance(val, str) and PERIOD_PATTERN.match(val.strip()):
                matches.append((col_idx, val.strip()))
        if len(matches) >= 2:  # need at least 2 periods to call this a header row
            cols = [c for c, _ in matches]
            labels = [v for _, v in matches]
            return row_idx, cols, labels
    return None


def _find_unit_hint(raw: pd.DataFrame, header_row: int) -> str | None:
    """Looks for something like '(in USD thousands)' near the header row."""
    for row_idx in range(max(0, header_row - 1), header_row + 1):
        for val in raw.iloc[row_idx]:
            if isinstance(val, str) and "(in" in val.lower():
                return val.strip("() ")
    return None


def parse_sheet(raw: pd.DataFrame, sheet_name: str) -> RawStatement | None:
    header_info = _find_header_row(raw)
    if header_info is None:
        return None  # not a statement sheet we can parse (e.g. a notes-only tab)

    header_row, value_cols, periods = header_info
    unit_hint = _find_unit_hint(raw, header_row)
    label_col = 0  # by convention (and by every real statement layout) labels sit in column A

    stmt = RawStatement(sheet_name=sheet_name, periods=periods, unit_hint=unit_hint)

    for row_idx in range(header_row + 1, len(raw)):
        row = raw.iloc[row_idx]
        label = row.iloc[label_col]
        if not isinstance(label, str) or not label.strip():
            continue
        clean_label = label.strip()

        # Skip rows that are clearly not data rows: notes/disclaimers
        # (long free-text) or section headers (label present, but every
        # value cell is empty).
        values = {}
        any_numeric = False
        for col_idx, period in zip(value_cols, periods):
            v = row.iloc[col_idx]
            if isinstance(v, (int, float)) and not pd.isna(v):
                values[period] = float(v)
                any_numeric = True
        if not any_numeric:
            continue  # section header row, e.g. "Current Assets" with no values
        if len(clean_label) > 120:
            continue  # almost certainly a note/disclaimer, not a line item

        stmt.rows.append((clean_label, values))

    return stmt


def ingest_workbook(path: str) -> dict[str, RawStatement]:
    """
    Reads every sheet in the workbook and returns a dict of
    {sheet_name: RawStatement} for every sheet that looks like a financial
    statement (has a detectable period header row). Sheets that don't
    (e.g. a cover page) are silently skipped, not treated as errors --
    a real uploaded workbook may have extra tabs we don't care about.
    """
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    results = {}
    for name, raw in sheets.items():
        stmt = parse_sheet(raw, name)
        if stmt is not None and stmt.rows:
            results[name] = stmt
    return results


def ingest_csv(path: str, sheet_name: str = "Sheet1") -> dict[str, RawStatement]:
    """
    Same idea as ingest_workbook, for a single CSV file (one statement
    per file). Reads with Python's csv module and pads ragged rows
    manually rather than pd.read_csv directly -- a real financial-
    statement CSV export is very likely to have title/note rows with
    fewer columns than the data rows below them, and pandas' default C
    parser raises a hard ParserError on that ragged shape instead of
    handling it gracefully (found this the hard way testing against a
    real-looking CSV export, not a hypothetical edge case).
    """
    import csv as csv_module
    with open(path, newline="", encoding="utf-8") as f:
        raw_rows = list(csv_module.reader(f))
    max_cols = max((len(row) for row in raw_rows), default=0)
    padded_rows = [row + [""] * (max_cols - len(row)) for row in raw_rows]
    raw = pd.DataFrame(padded_rows).replace("", pd.NA)
    # Numeric-looking strings need converting back to numbers for parse_sheet
    # to detect them as data values (csv.reader returns everything as str).
    for col in raw.columns:
        converted = pd.to_numeric(raw[col], errors="coerce")
        # Only replace with the numeric version where conversion actually
        # succeeded -- keep original strings (labels) where it didn't,
        # since errors="ignore" was removed in pandas 3.x.
        raw[col] = converted.where(converted.notna(), raw[col])

    stmt = parse_sheet(raw, sheet_name)
    return {sheet_name: stmt} if stmt is not None and stmt.rows else {}


if __name__ == "__main__":
    results = ingest_workbook("data/sample/Nimbus_Retail_Financial_Statements.xlsx")
    for sheet_name, stmt in results.items():
        print(f"\n=== {sheet_name} ===")
        print(f"Periods: {stmt.periods}  |  Unit: {stmt.unit_hint}")
        print(f"Rows parsed: {len(stmt.rows)}")
        for label, values in stmt.rows[:5]:
            print(f"  {label!r}: {values}")
        print("  ...")


def detect_statement_type(raw_text_top: str, filename: str = "") -> str | None:
    """
    Decides whether a file is an Income Statement, Balance Sheet or Cash
    Flow Statement from its title text and filename -- users won't name
    uploads the way our samples are named. Returns the canonical sheet
    name used throughout the pipeline, or None if undetectable.
    """
    t = (raw_text_top + " " + filename).lower()
    if "cash flow" in t or "cashflow" in t or "cash_flow" in t:
        return "Cash Flow Statement"
    if "balance sheet" in t or "balance_sheet" in t or "financial position" in t:
        return "Balance Sheet"
    if ("income statement" in t or "income_statement" in t or "profit and loss" in t
            or "profit & loss" in t or "p&l" in t or "statement of operations" in t):
        return "Income Statement"
    return None
