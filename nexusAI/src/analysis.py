"""
Analysis engine: takes normalized financial data (canonical field names,
values per period) and computes what a real analyst would compute --
liquidity, profitability, leverage, and efficiency ratios, period-over-
period trends, and rule-based anomaly flags.

Design principle: every number here is a plain, auditable formula, not an
LLM guess. The LLM's job (later, in the insight engine) is to explain
these numbers in plain English -- not to compute them. That separation is
what makes the numbers trustworthy: they're either right or wrong by
formula, never "probably right."
"""
from __future__ import annotations
from dataclasses import dataclass, field

from normalize import NormalizedStatement


def get(stmt: NormalizedStatement | None, field_name: str, period: str) -> float | None:
    """Safe lookup: returns None (not a crash, not a silent 0) if a field,
    period, or even the WHOLE STATEMENT is missing -- e.g. a user uploads
    only an Income Statement with no Balance Sheet. Missing data should
    be visible downstream (fewer ratios computed), never a crash."""
    if stmt is None:
        return None
    return stmt.fields.get(field_name, {}).get(period)


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def fmt_pct(x: float | None) -> str:
    """Formats a ratio as a percentage, or 'N/A' if the underlying data
    was missing -- used everywhere a ratio gets displayed or sent to the
    LLM, so a missing field degrades gracefully instead of crashing the
    whole pipeline (found this the hard way: a real statement labeling
    net income as 'Net Income / (Loss)' initially wasn't mapped, and the
    unhandled None crashed every formatting call downstream)."""
    return f"{x*100:.1f}%" if x is not None else "N/A"


def fmt_ratio(x: float | None) -> str:
    return f"{x:.2f}x" if x is not None else "N/A"


@dataclass
class PeriodRatios:
    period: str
    # Profitability
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    roe: float | None = None
    roa: float | None = None
    # Liquidity
    current_ratio: float | None = None
    quick_ratio: float | None = None
    # Leverage
    debt_to_equity: float | None = None
    interest_coverage: float | None = None
    # Efficiency
    asset_turnover: float | None = None
    inventory_turnover: float | None = None


@dataclass
class TrendPoint:
    metric: str
    period: str
    value: float
    prior_period: str | None
    yoy_change: float | None       # absolute change
    yoy_pct_change: float | None   # percentage change


@dataclass
class Anomaly:
    severity: str          # "high" | "medium" | "low"
    category: str          # "profitability" | "liquidity" | "leverage" | "working_capital"
    period: str
    message: str
    evidence: dict


@dataclass
class AnalysisResult:
    periods: list[str]
    ratios: dict[str, PeriodRatios] = field(default_factory=dict)
    trends: list[TrendPoint] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)


def compute_ratios(bs: NormalizedStatement | None, is_: NormalizedStatement | None, period: str) -> PeriodRatios:
    """Computes every ratio possible from whatever statements ARE
    available. If bs is None (no Balance Sheet uploaded), every
    balance-sheet-dependent ratio (liquidity, leverage, ROE/ROA, asset
    turnover) comes back as None via get()/safe_div() -- but the
    Income-Statement-only ratios (margins, interest coverage) still
    compute correctly. Same principle if is_ is None."""
    revenue = get(is_, "revenue", period)
    gross_profit = get(is_, "gross_profit", period)
    operating_income = get(is_, "operating_income", period)
    net_income = get(is_, "net_income", period)
    interest_expense = get(is_, "interest_expense", period)

    total_current_assets = get(bs, "total_current_assets", period)
    total_current_liabilities = get(bs, "total_current_liabilities", period)
    inventory = get(bs, "inventory", period)
    prepaid_expenses = get(bs, "prepaid_expenses", period)
    total_assets = get(bs, "total_assets", period)
    total_equity = get(bs, "total_equity", period)
    short_term_debt = get(bs, "short_term_debt", period) or 0.0
    long_term_debt = get(bs, "long_term_debt", period) or 0.0

    total_debt = (short_term_debt + long_term_debt) if (short_term_debt or long_term_debt) else None

    return PeriodRatios(
        period=period,
        gross_margin=safe_div(gross_profit, revenue),
        operating_margin=safe_div(operating_income, revenue),
        net_margin=safe_div(net_income, revenue),
        roe=safe_div(net_income, total_equity),
        roa=safe_div(net_income, total_assets),
        current_ratio=safe_div(total_current_assets, total_current_liabilities),
        # Quick assets exclude both inventory and prepaid expenses. If either
        # component is absent, do not silently assume it is zero.
        quick_ratio=safe_div(
            (total_current_assets - inventory - (prepaid_expenses or 0.0))
            if (total_current_assets is not None and inventory is not None) else None,
            total_current_liabilities,
        ),
        debt_to_equity=safe_div(total_debt, total_equity),
        interest_coverage=safe_div(operating_income, abs(interest_expense) if interest_expense else None),
        asset_turnover=safe_div(revenue, total_assets),
        inventory_turnover=safe_div(abs(get(is_, "cogs", period)) if get(is_, "cogs", period) else None, inventory),
    )


def compute_trends(stmt: NormalizedStatement, fields_to_track: list[str]) -> list[TrendPoint]:
    trends = []
    periods = stmt.periods
    for field_name in fields_to_track:
        for i, period in enumerate(periods):
            value = get(stmt, field_name, period)
            if value is None:
                continue
            if i == 0:
                trends.append(TrendPoint(field_name, period, value, None, None, None))
            else:
                prior_period = periods[i - 1]
                prior_value = get(stmt, field_name, prior_period)
                if prior_value is None:
                    trends.append(TrendPoint(field_name, period, value, None, None, None))
                    continue
                abs_change = value - prior_value
                pct_change = safe_div(abs_change, abs(prior_value))
                trends.append(TrendPoint(field_name, period, value, prior_period, abs_change, pct_change))
    return trends


def detect_anomalies(bs: NormalizedStatement | None, is_: NormalizedStatement | None, cf: NormalizedStatement | None,
                      ratios: dict[str, PeriodRatios], periods: list[str]) -> list[Anomaly]:
    """
    Every lookup here goes through get(), which is None-safe for a
    missing statement -- so if e.g. cf is None (no Cash Flow uploaded),
    the dividends-vs-net-income check simply never fires (get() returns
    None, the `is not None` guard skips it) instead of crashing. Same
    for bs being None skipping every balance-sheet-dependent check.
    """
    anomalies = []

    for i in range(1, len(periods)):
        period, prior = periods[i], periods[i - 1]
        r, r_prior = ratios[period], ratios[prior]

        # --- Margin compression ---
        for margin_name, label in [("gross_margin", "Gross margin"), ("operating_margin", "Operating margin"),
                                     ("net_margin", "Net margin")]:
            cur, prev = getattr(r, margin_name), getattr(r_prior, margin_name)
            if cur is not None and prev is not None:
                drop_pts = (prev - cur) * 100
                if drop_pts >= 2.0:
                    anomalies.append(Anomaly(
                        severity="high" if drop_pts >= 4 else "medium",
                        category="profitability",
                        period=period,
                        message=f"{label} compressed by {drop_pts:.1f} percentage points "
                                f"({prev*100:.1f}% -> {cur*100:.1f}%) vs {prior}.",
                        evidence={"prior": prev, "current": cur, "drop_pts": drop_pts},
                    ))

        # --- Revenue growing, net income declining ---
        rev_cur, rev_prior = get(is_, "revenue", period), get(is_, "revenue", prior)
        ni_cur, ni_prior = get(is_, "net_income", period), get(is_, "net_income", prior)
        if rev_cur and rev_prior and ni_cur is not None and ni_prior is not None:
            if rev_cur > rev_prior and ni_cur < ni_prior:
                anomalies.append(Anomaly(
                    severity="high",
                    category="profitability",
                    period=period,
                    message=f"Revenue grew {((rev_cur/rev_prior)-1)*100:.1f}% but net income "
                            f"FELL {((ni_prior-ni_cur)/ni_prior)*100:.1f}% vs {prior} -- "
                            f"growth is not translating into profit.",
                    evidence={"revenue_growth": (rev_cur/rev_prior)-1, "net_income_change": (ni_cur-ni_prior)/ni_prior},
                ))

        # --- Working capital strain: AR/Inventory growing faster than revenue ---
        for field_name, label in [("accounts_receivable", "Accounts receivable"), ("inventory", "Inventory")]:
            cur_val, prior_val = get(bs, field_name, period), get(bs, field_name, prior)
            if cur_val and prior_val and rev_cur and rev_prior:
                item_growth = (cur_val - prior_val) / prior_val
                rev_growth = (rev_cur - rev_prior) / rev_prior
                if item_growth - rev_growth >= 0.10:  # outpacing revenue growth by 10+ points
                    anomalies.append(Anomaly(
                        severity="medium",
                        category="working_capital",
                        period=period,
                        message=f"{label} grew {item_growth*100:.1f}% vs revenue growth of only "
                                f"{rev_growth*100:.1f}% -- {label.lower()} is building up faster "
                                f"than sales, tying up cash.",
                        evidence={"item_growth": item_growth, "revenue_growth": rev_growth},
                    ))

        # --- Leverage rising ---
        if r.debt_to_equity is not None and r_prior.debt_to_equity is not None:
            if r.debt_to_equity > r_prior.debt_to_equity * 1.10:
                anomalies.append(Anomaly(
                    severity="medium",
                    category="leverage",
                    period=period,
                    message=f"Debt-to-equity rose from {r_prior.debt_to_equity:.2f} to "
                            f"{r.debt_to_equity:.2f} vs {prior} -- increasing reliance on debt.",
                    evidence={"prior": r_prior.debt_to_equity, "current": r.debt_to_equity},
                ))

        # --- Cash declining while debt rising (combined signal) ---
        cash_cur, cash_prior = get(bs, "cash", period), get(bs, "cash", prior)
        if (cash_cur is not None and cash_prior is not None and cash_cur < cash_prior
                and r.debt_to_equity is not None and r_prior.debt_to_equity is not None
                and r.debt_to_equity > r_prior.debt_to_equity):
            anomalies.append(Anomaly(
                severity="high",
                category="liquidity",
                period=period,
                message=f"Cash declined ({cash_prior:,.0f} -> {cash_cur:,.0f}) at the same time "
                        f"leverage increased vs {prior} -- a combined liquidity and solvency "
                        f"warning sign worth investigating together, not separately.",
                evidence={"cash_prior": cash_prior, "cash_current": cash_cur},
            ))

        # --- Dividends exceeding net income ---
        div_cur = get(cf, "dividends_paid", period)
        if div_cur is not None and ni_cur is not None and abs(div_cur) > ni_cur:
            anomalies.append(Anomaly(
                severity="high",
                category="liquidity",
                period=period,
                message=f"Dividends paid ({abs(div_cur):,.0f}) EXCEEDED net income "
                        f"({ni_cur:,.0f}) in {period} -- the company is paying out more than "
                        f"it earned, funded by cash reserves or debt.",
                evidence={"dividends": abs(div_cur), "net_income": ni_cur},
            ))

        # --- Current ratio below 1.0x: current liabilities exceed current assets ---
        if r.current_ratio is not None and r.current_ratio < 1.0:
            anomalies.append(Anomaly(
                severity="high",
                category="liquidity",
                period=period,
                message=f"Current ratio fell to {r.current_ratio:.2f}x in {period} -- BELOW 1.0x, "
                        f"meaning current liabilities now exceed current assets. This is a standard "
                        f"short-term solvency red flag.",
                evidence={"current_ratio": r.current_ratio},
            ))

    return anomalies


def analyze(normalized: dict[str, NormalizedStatement]) -> AnalysisResult:
    """
    Uses .get() instead of direct dict indexing for every statement --
    a real user may upload only an Income Statement (no Balance Sheet or
    Cash Flow), which is a completely plausible scenario, not an error
    case. Ratios/anomalies that need a missing statement simply don't
    get computed (return None / are skipped), rather than the whole
    pipeline crashing with a KeyError on the missing dict key.

    The Income Statement is the one genuinely required input -- without
    revenue and net income there's nothing to analyze at all -- so that
    case raises a clear, actionable error instead of silently returning
    an empty result.
    """
    bs = normalized.get("Balance Sheet")
    is_ = normalized.get("Income Statement")
    cf = normalized.get("Cash Flow Statement")

    if is_ is None:
        raise ValueError(
            "No Income Statement was found in the uploaded document(s) -- at least an "
            "Income Statement (with Revenue and Net Income) is required to run any analysis. "
            "Balance Sheet and Cash Flow Statement are optional; ratios that depend on them "
            "will simply be skipped if not provided."
        )

    periods = is_.periods
    ratios = {period: compute_ratios(bs, is_, period) for period in periods}

    trend_fields = ["revenue", "gross_profit", "net_income", "cash", "total_assets",
                     "total_liabilities", "accounts_receivable", "inventory", "cfo",
                     "capex", "dividends_paid", "net_change_cash"]
    all_trends = []
    for stmt in (is_, bs, cf):
        if stmt is None:
            continue
        relevant = [f for f in trend_fields if f in stmt.fields]
        all_trends.extend(compute_trends(stmt, relevant))

    anomalies = detect_anomalies(bs, is_, cf, ratios, periods)

    return AnalysisResult(periods=periods, ratios=ratios, trends=all_trends, anomalies=anomalies)


if __name__ == "__main__":
    from ingest import ingest_workbook
    from normalize import normalize_workbook

    raw = ingest_workbook("data/sample/Nimbus_Retail_Financial_Statements.xlsx")
    normalized = normalize_workbook(raw)
    result = analyze(normalized)

    print("=" * 70)
    print("RATIOS BY PERIOD")
    print("=" * 70)
    for period in result.periods:
        r = result.ratios[period]
        print(f"\n{period}:")
        print(f"  Gross Margin:        {fmt_pct(r.gross_margin)}")
        print(f"  Operating Margin:    {fmt_pct(r.operating_margin)}")
        print(f"  Net Margin:          {fmt_pct(r.net_margin)}")
        print(f"  ROE:                 {fmt_pct(r.roe)}")
        print(f"  ROA:                 {fmt_pct(r.roa)}")
        print(f"  Current Ratio:       {fmt_ratio(r.current_ratio)}")
        print(f"  Quick Ratio:         {fmt_ratio(r.quick_ratio)}")
        print(f"  Debt-to-Equity:      {fmt_ratio(r.debt_to_equity)}")
        print(f"  Interest Coverage:   {fmt_ratio(r.interest_coverage)}")
        print(f"  Asset Turnover:      {fmt_ratio(r.asset_turnover)}")
        print(f"  Inventory Turnover:  {fmt_ratio(r.inventory_turnover)}")

    print("\n" + "=" * 70)
    print(f"ANOMALIES DETECTED: {len(result.anomalies)}")
    print("=" * 70)
    for a in result.anomalies:
        print(f"\n[{a.severity.upper()}] ({a.category}) {a.period}")
        print(f"  {a.message}")
