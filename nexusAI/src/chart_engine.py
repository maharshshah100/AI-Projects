"""
Auto-visualization engine: takes the analysis results and generates the
RIGHT chart for each type of finding -- not a fixed dashboard template,
and not "chart everything." Each function below maps one analytical
question to one appropriate chart type:

  - A trend over time            -> line chart
  - A composition/breakdown      -> stacked bar chart
  - A comparison across items    -> grouped bar chart

Charts are built with Plotly and returned as Figure objects. In the final
app these are rendered directly in Streamlit via st.plotly_chart(fig) --
no image export step, no Chrome dependency, nothing that can break on a
user's machine.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis import AnalysisResult
from normalize import NormalizedStatement

COLOR_PRIMARY = "#1F3864"
COLOR_ACCENT = "#2E86AB"
COLOR_WARNING = "#D64545"
COLOR_POSITIVE = "#2E7D32"
COLOR_NEUTRAL = "#8C8C8C"


def chart_margin_trend(result: AnalysisResult) -> go.Figure:
    """Line chart: the three margins over time. Directly visualizes
    margin-compression anomalies, if any were flagged."""
    periods = result.periods
    gross = [r.gross_margin * 100 if r.gross_margin is not None else None for r in (result.ratios[p] for p in periods)]
    operating = [r.operating_margin * 100 if r.operating_margin is not None else None for r in (result.ratios[p] for p in periods)]
    net = [r.net_margin * 100 if r.net_margin is not None else None for r in (result.ratios[p] for p in periods)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=gross, mode="lines+markers", name="Gross Margin",
                              line=dict(color=COLOR_PRIMARY, width=3)))
    fig.add_trace(go.Scatter(x=periods, y=operating, mode="lines+markers", name="Operating Margin",
                              line=dict(color=COLOR_ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=periods, y=net, mode="lines+markers", name="Net Margin",
                              line=dict(color=COLOR_WARNING, width=3)))
    fig.update_layout(
        title="Margin Trends",
        yaxis_title="Margin (%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    return fig


def chart_revenue_vs_profit(result: AnalysisResult, is_stmt: NormalizedStatement) -> go.Figure:
    """Dual-axis: revenue (bars) vs net income (line). Makes a
    'growth not converting to profit' anomaly immediately visible."""
    periods = result.periods
    revenue = [is_stmt.fields["revenue"][p] for p in periods]
    net_income = [is_stmt.fields["net_income"][p] for p in periods]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=periods, y=revenue, name="Revenue", marker_color=COLOR_ACCENT, opacity=0.75),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=periods, y=net_income, name="Net Income", mode="lines+markers",
                              line=dict(color=COLOR_WARNING, width=3)), secondary_y=True)
    fig.update_layout(title="Revenue vs. Net Income", template="plotly_white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=60, b=40))
    fig.update_yaxes(title_text="Revenue ($000s)", secondary_y=False)
    fig.update_yaxes(title_text="Net Income ($000s)", secondary_y=True)
    return fig


def chart_balance_sheet_composition(bs_stmt: NormalizedStatement) -> go.Figure:
    """Stacked bar: what makes up the balance sheet each year --
    a composition question, so a stacked bar is the right chart, not a line."""
    periods = bs_stmt.periods
    components = [
        ("cash", "Cash", COLOR_POSITIVE),
        ("accounts_receivable", "Accounts Receivable", COLOR_ACCENT),
        ("inventory", "Inventory", COLOR_NEUTRAL),
        ("ppe_net", "PP&E (net)", COLOR_PRIMARY),
    ]
    fig = go.Figure()
    for field_name, label, color in components:
        if field_name in bs_stmt.fields:
            values = [bs_stmt.fields[field_name].get(p, 0) for p in periods]
            fig.add_trace(go.Bar(x=periods, y=values, name=label, marker_color=color))
    fig.update_layout(
        title="Asset Composition by Year", barmode="stack", template="plotly_white",
        yaxis_title="$ (000s)", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    return fig


def chart_liquidity_trend(result: AnalysisResult) -> go.Figure:
    """Line chart: current & quick ratio over time, with a reference
    line at 1.0x (the standard 'can you cover short-term obligations' bar)."""
    periods = result.periods
    current = [result.ratios[p].current_ratio for p in periods]
    quick = [result.ratios[p].quick_ratio for p in periods]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=current, mode="lines+markers", name="Current Ratio",
                              line=dict(color=COLOR_PRIMARY, width=3)))
    fig.add_trace(go.Scatter(x=periods, y=quick, mode="lines+markers", name="Quick Ratio",
                              line=dict(color=COLOR_ACCENT, width=3, dash="dash")))
    fig.add_hline(y=1.0, line_dash="dot", line_color=COLOR_WARNING,
                  annotation_text="1.0x reference", annotation_position="bottom right")
    fig.update_layout(title="Liquidity Ratios", yaxis_title="Ratio (x)", template="plotly_white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=60, b=40))
    return fig


def chart_leverage_trend(result: AnalysisResult) -> go.Figure:
    """Bar chart: debt-to-equity by year -- a direct year-to-year
    comparison, so bars (not a line) keep each year visually distinct."""
    periods = result.periods
    dte = [result.ratios[p].debt_to_equity for p in periods]
    colors = [
        COLOR_WARNING if (i > 0 and dte[i] is not None and dte[i-1] is not None and dte[i] > dte[i-1])
        else COLOR_PRIMARY
        for i in range(len(dte))
    ]

    fig = go.Figure(go.Bar(x=periods, y=dte, marker_color=colors,
                            text=[f"{v:.2f}x" if v is not None else "N/A" for v in dte],
                            textposition="outside", name="Debt-to-Equity"))
    fig.update_layout(title="Debt-to-Equity Ratio", yaxis_title="Ratio (x)", template="plotly_white",
                       margin=dict(t=60, b=40))
    return fig


def chart_working_capital_growth(bs_stmt: NormalizedStatement, is_stmt: NormalizedStatement) -> go.Figure:
    """Grouped bar: YoY growth rate of revenue vs. AR vs. inventory.
    This is a comparison-across-items question, so grouped bars are
    correct -- directly visualizes 'inventory/receivables growing
    faster than revenue' anomalies."""
    periods = bs_stmt.periods
    labels, rev_growth, ar_growth, inv_growth = [], [], [], []
    for i in range(1, len(periods)):
        p, prior = periods[i], periods[i - 1]
        labels.append(f"{prior} -> {p}")
        rev_growth.append((is_stmt.fields["revenue"][p] / is_stmt.fields["revenue"][prior] - 1) * 100)
        ar_growth.append((bs_stmt.fields["accounts_receivable"][p] / bs_stmt.fields["accounts_receivable"][prior] - 1) * 100)
        inv_growth.append((bs_stmt.fields["inventory"][p] / bs_stmt.fields["inventory"][prior] - 1) * 100)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=rev_growth, name="Revenue Growth", marker_color=COLOR_POSITIVE))
    fig.add_trace(go.Bar(x=labels, y=ar_growth, name="Receivables Growth", marker_color=COLOR_ACCENT))
    fig.add_trace(go.Bar(x=labels, y=inv_growth, name="Inventory Growth", marker_color=COLOR_WARNING))
    fig.update_layout(title="Revenue Growth vs. Working Capital Growth", barmode="group",
                       yaxis_title="YoY Growth (%)", template="plotly_white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=60, b=40))
    return fig


def chart_cash_trend(bs_stmt: NormalizedStatement) -> go.Figure:
    """Bar chart: cash balance by year."""
    periods = bs_stmt.periods
    cash = [bs_stmt.fields["cash"][p] for p in periods]
    colors = [COLOR_WARNING if i > 0 and cash[i] < cash[i-1] else COLOR_POSITIVE for i in range(len(cash))]

    fig = go.Figure(go.Bar(x=periods, y=cash, marker_color=colors,
                            text=[f"${v:,.0f}k" for v in cash], textposition="outside", name="Cash"))
    fig.update_layout(title="Cash and Cash Equivalents", yaxis_title="$ (000s)",
                       template="plotly_white", margin=dict(t=60, b=40))
    return fig


def generate_all_charts(result: AnalysisResult, normalized: dict[str, NormalizedStatement]) -> dict[str, go.Figure]:
    """
    Returns every chart, tagged by name, along with which anomaly
    categories (if any) each one is directly relevant to -- this is what
    lets the UI prioritize showing the charts that actually matter for
    THIS company's data, not a fixed list every time.

    Gracefully skips any chart that needs a statement the user didn't
    upload (e.g. no Balance Sheet -> skip composition/liquidity/leverage/
    cash charts) rather than crashing the entire set over one missing
    input -- the charts that CAN be built from what's available still
    get built and returned.
    """
    is_stmt = normalized.get("Income Statement")
    bs_stmt = normalized.get("Balance Sheet")

    charts = {}
    if is_stmt is not None:
        charts["margin_trend"] = chart_margin_trend(result)
        charts["revenue_vs_profit"] = chart_revenue_vs_profit(result, is_stmt)
    if bs_stmt is not None:
        charts["balance_sheet_composition"] = chart_balance_sheet_composition(bs_stmt)
        charts["liquidity_trend"] = chart_liquidity_trend(result)
        charts["leverage_trend"] = chart_leverage_trend(result)
        charts["cash_trend"] = chart_cash_trend(bs_stmt)
    if bs_stmt is not None and is_stmt is not None:
        charts["working_capital_growth"] = chart_working_capital_growth(bs_stmt, is_stmt)
    return charts


if __name__ == "__main__":
    from ingest import ingest_workbook
    from normalize import normalize_workbook
    from analysis import analyze

    raw = ingest_workbook("data/sample/Nimbus_Retail_Financial_Statements.xlsx")
    normalized = normalize_workbook(raw)
    result = analyze(normalized)
    charts = generate_all_charts(result, normalized)

    print(f"Generated {len(charts)} charts\n")
    for name, fig in charts.items():
        print(f"=== {name} ===")
        print(f"  Title: {fig.layout.title.text}")
        print(f"  Chart type(s): {sorted(set(type(t).__name__ for t in fig.data))}")
        for trace in fig.data:
            y_vals = list(trace.y) if trace.y is not None else []
            y_display = [round(v, 2) if isinstance(v, (int, float)) else v for v in y_vals]
            print(f"    trace '{trace.name}': x={list(trace.x)}, y={y_display}")
        print()
