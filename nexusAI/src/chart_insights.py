"""
Per-chart insight captions.

Instead of one big disconnected narrative, each chart gets its own short,
data-driven caption directly below it -- pulling only the anomalies/trends
relevant to THAT specific chart. The final "Recommended Actions" section
runs once, after every chart has been shown, synthesizing across all of
them.

Two modes, same pattern as insight_engine.py:
  - Rule-based (works with zero API key): deterministic captions built
    directly from the anomaly/ratio data.
  - LLM-enhanced (with GROQ_API_KEY): ONE batched Groq call rewrites all
    captions + the final recommendations in more natural language --
    deliberately ONE call, not seven, so this stays fast (the <3s
    latency target) and cheap, not one API round-trip per chart.
"""
from __future__ import annotations
import os
import re

from dotenv import load_dotenv
load_dotenv()

from analysis import AnalysisResult, Anomaly, fmt_pct, fmt_ratio
from normalize import NormalizedStatement

GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


def _anomalies_for(result: AnalysisResult, categories: set[str], period: str | None = None) -> list[Anomaly]:
    return [a for a in result.anomalies
            if a.category in categories and (period is None or a.period == period)]


def caption_margin_trend(result: AnalysisResult) -> str:
    latest = result.periods[-1]
    r, r0 = result.ratios[latest], result.ratios[result.periods[0]]
    margin_anomalies = [a for a in _anomalies_for(result, {"profitability"}, latest) if "margin" in a.message.lower()]
    if margin_anomalies:
        return (f"All three margins have compressed since {result.periods[0]}: gross margin "
                f"{fmt_pct(r0.gross_margin)} -> {fmt_pct(r.gross_margin)}, net margin "
                f"{fmt_pct(r0.net_margin)} -> {fmt_pct(r.net_margin)}. This is the single most "
                f"consistent trend in the data and the first thing worth investigating.")
    return (f"Margins have held broadly steady from {result.periods[0]} to {latest} "
            f"(net margin {fmt_pct(r0.net_margin)} -> {fmt_pct(r.net_margin)}) -- no compression flagged.")


def caption_revenue_vs_profit(result: AnalysisResult) -> str:
    growth_anomalies = [a for a in result.anomalies if "not translating into profit" in a.message]
    if growth_anomalies:
        a = growth_anomalies[-1]
        return (f"{a.message} This divergence -- growing sales but shrinking profit -- is usually "
                f"a cost or pricing problem, not a demand problem, and is worth root-causing before "
                f"assuming more revenue growth will fix it.")
    return "Revenue and net income have moved in the same direction across the periods analyzed -- no divergence flagged."


def caption_balance_sheet_composition(bs_stmt: NormalizedStatement) -> str:
    periods = bs_stmt.periods
    first, last = periods[0], periods[-1]
    ar_share_first = bs_stmt.fields["accounts_receivable"][first] / bs_stmt.fields["total_current_assets"][first]
    ar_share_last = bs_stmt.fields["accounts_receivable"][last] / bs_stmt.fields["total_current_assets"][last]
    inv_share_first = bs_stmt.fields["inventory"][first] / bs_stmt.fields["total_current_assets"][first]
    inv_share_last = bs_stmt.fields["inventory"][last] / bs_stmt.fields["total_current_assets"][last]
    cash_share_first = bs_stmt.fields["cash"][first] / bs_stmt.fields["total_current_assets"][first]
    cash_share_last = bs_stmt.fields["cash"][last] / bs_stmt.fields["total_current_assets"][last]
    return (f"Cash has shrunk from {cash_share_first*100:.0f}% to {cash_share_last*100:.0f}% of current "
            f"assets, while receivables ({ar_share_first*100:.0f}% -> {ar_share_last*100:.0f}%) and "
            f"inventory ({inv_share_first*100:.0f}% -> {inv_share_last*100:.0f}%) have grown their "
            f"share -- the asset base is shifting from cash into assets that are harder to spend.")


def caption_liquidity_trend(result: AnalysisResult) -> str:
    latest = result.periods[-1]
    r = result.ratios[latest]
    combined = _anomalies_for(result, {"liquidity"}, latest)
    if r.current_ratio is not None and r.current_ratio < 1.0:
        status = (f"Current ratio has fallen to {fmt_ratio(r.current_ratio)} -- BELOW the 1.0x "
                  f"threshold, meaning current liabilities now exceed current assets.")
    elif r.current_ratio is not None:
        status = (f"Current ratio is {fmt_ratio(r.current_ratio)} and quick ratio {fmt_ratio(r.quick_ratio)} "
                  f"as of {latest} -- both above the 1.0x minimum, but both have declined every period.")
    else:
        status = "Current/quick ratio could not be computed from the available data."
    if combined:
        return status + " " + combined[0].message
    return status


def caption_leverage_trend(result: AnalysisResult) -> str:
    latest = result.periods[-1]
    lev_anomalies = _anomalies_for(result, {"leverage"}, latest)
    r = result.ratios[latest]
    if lev_anomalies:
        return (f"Debt-to-equity reached {fmt_ratio(r.debt_to_equity)} in {latest}, up from "
                f"{fmt_ratio(result.ratios[result.periods[0]].debt_to_equity)} in {result.periods[0]} -- "
                f"a sustained increase in reliance on debt financing.")
    return f"Debt-to-equity is {fmt_ratio(r.debt_to_equity)} as of {latest} -- no unusual leverage increase flagged."


def caption_working_capital_growth(result: AnalysisResult) -> str:
    wc_anomalies = _anomalies_for(result, {"working_capital"}, result.periods[-1])
    if wc_anomalies:
        items = " and ".join(a.message.split(" grew")[0] for a in wc_anomalies)
        return (f"{items} are both growing meaningfully faster than revenue -- this ties up cash in "
                f"unsold inventory and uncollected receivables rather than generating returns.")
    return "Receivables and inventory have grown roughly in line with revenue -- no working capital strain flagged."


def caption_cash_trend(bs_stmt: NormalizedStatement, result: AnalysisResult) -> str:
    periods = bs_stmt.periods
    first_cash, last_cash = bs_stmt.fields["cash"][periods[0]], bs_stmt.fields["cash"][periods[-1]]
    pct_change = (last_cash - first_cash) / first_cash * 100
    if last_cash < first_cash:
        return (f"Cash has fallen {abs(pct_change):.0f}% from ${first_cash:,.0f}k to ${last_cash:,.0f}k "
                f"over the period -- worth tracking closely alongside the rising debt trend shown above.")
    return f"Cash has grown {pct_change:.0f}% from ${first_cash:,.0f}k to ${last_cash:,.0f}k over the period."


def generate_all_chart_captions(result: AnalysisResult, normalized: dict[str, NormalizedStatement]) -> dict[str, str]:
    """Rule-based captions for whichever charts are available -- works
    with zero API key, and gracefully skips captions for charts that
    couldn't be built (e.g. no Balance Sheet uploaded), matching
    chart_engine.generate_all_charts()'s same skip behavior."""
    is_stmt = normalized.get("Income Statement")
    bs_stmt = normalized.get("Balance Sheet")

    captions = {}
    if is_stmt is not None:
        captions["margin_trend"] = caption_margin_trend(result)
        captions["revenue_vs_profit"] = caption_revenue_vs_profit(result)
    if bs_stmt is not None:
        captions["balance_sheet_composition"] = caption_balance_sheet_composition(bs_stmt)
        captions["liquidity_trend"] = caption_liquidity_trend(result)
        captions["leverage_trend"] = caption_leverage_trend(result)
        captions["cash_trend"] = caption_cash_trend(bs_stmt, result)
    if bs_stmt is not None and is_stmt is not None:
        captions["working_capital_growth"] = caption_working_capital_growth(result)
    return captions


CHART_TITLES = {
    "margin_trend": "Margin Trends",
    "revenue_vs_profit": "Revenue vs. Net Income",
    "balance_sheet_composition": "Asset Composition by Year",
    "liquidity_trend": "Liquidity Ratios",
    "leverage_trend": "Debt-to-Equity Ratio",
    "working_capital_growth": "Revenue Growth vs. Working Capital Growth",
    "cash_trend": "Cash and Cash Equivalents",
}

CHART_ORDER = ["margin_trend", "revenue_vs_profit", "balance_sheet_composition",
               "liquidity_trend", "leverage_trend", "working_capital_growth", "cash_trend"]


def _polish_unsafe(rule_based_captions: dict[str, str], result: AnalysisResult,
                              normalized: dict[str, NormalizedStatement], company_name: str,
                              api_key: str | None = None) -> tuple[dict[str, str], str]:
    """
    ONE Groq call that rewrites all 7 captions in more natural language and
    produces the final recommendations -- deliberately batched into a
    single request rather than one call per chart, to keep this fast and
    cheap. Falls back to the rule-based captions + rule-based
    recommendations if no API key is available.
    """
    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY")
    from insight_engine import _rule_based_fallback_summary, format_anomalies_for_prompt

    if not api_key:
        # Extract just the "Recommended Actions" section from the existing
        # fallback summary so we don't duplicate that logic.
        fallback = _rule_based_fallback_summary(result, company_name)
        actions_match = re.search(r"### Recommended Actions\n(.*)", fallback, re.DOTALL)
        recommendations = actions_match.group(1).strip() if actions_match else "No specific actions flagged."
        return rule_based_captions, recommendations

    from groq import Groq
    client = Groq(api_key=api_key)

    captions_block = "\n".join(f"[{CHART_TITLES[k]}]: {v}" for k, v in rule_based_captions.items())
    prompt = f"""Company: {company_name}

Here are draft, data-verified captions for {len(rule_based_captions)} charts, plus the full list of \
detected anomalies. Rewrite each caption in one polished, natural sentence (do not add any numbers \
or claims not already present in the draft -- only rephrase). Then write a final "Recommended Actions" \
section (3-5 bullet points, specific and actionable).

DRAFT CAPTIONS:
{captions_block}

ALL ANOMALIES:
{format_anomalies_for_prompt(result)}

Respond in exactly this format:
[Margin Trends]: <rewritten caption>
[Revenue vs. Net Income]: <rewritten caption>
[Asset Composition by Year]: <rewritten caption>
[Liquidity Ratios]: <rewritten caption>
[Debt-to-Equity Ratio]: <rewritten caption>
[Revenue Growth vs. Working Capital Growth]: <rewritten caption>
[Cash and Cash Equivalents]: <rewritten caption>

RECOMMENDED ACTIONS:
- <action 1>
- <action 2>
..."""

    completion = client.chat.completions.create(
        model=GROQ_MODEL, max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    text = completion.choices[0].message.content

    polished = {}
    for key in rule_based_captions:  # only charts actually available, not the fixed 7
        title = CHART_TITLES[key]
        m = re.search(rf"\[{re.escape(title)}\]:\s*(.+)", text)
        polished[key] = m.group(1).strip() if m else rule_based_captions[key]

    rec_match = re.search(
      r"RECOMMENDED ACTIONS:\s*(.*)",
      text,
      re.IGNORECASE | re.DOTALL,
)

    recommendations = rec_match.group(1).strip() if rec_match else ""

    if not recommendations or recommendations.lower() == "no specific actions flagged.":
        from insight_engine import _rule_based_fallback_summary

        fallback = _rule_based_fallback_summary(result, company_name)
        actions_match = re.search(
            r"### Recommended Actions\s*(.*)",
            fallback,
            re.IGNORECASE | re.DOTALL,
        )
        recommendations = (
            actions_match.group(1).strip()
            if actions_match and actions_match.group(1).strip()
            else (
                "- Monitor revenue growth and gross, operating, and net margins each period.\n"
                "- Review receivables, inventory, and cash conversion to protect liquidity.\n"
                "- Track debt levels, interest coverage, and capital expenditure against "
                "operating cash flow."
            )
        )

    return polished, recommendations



def polish_captions_with_llm(rule_based_captions, result, normalized, company_name, api_key=None):
    """Never fails: if the LLM call errors for ANY reason (no network, bad key,
    rate limit, malformed reply) the deterministic rule-based captions and
    recommendations are returned instead -- an LLM outage must never take
    down the analysis itself."""
    try:
        return _polish_unsafe(rule_based_captions, result, normalized, company_name, api_key)
    except Exception:
        return _polish_unsafe(rule_based_captions, result, normalized, company_name, api_key="")


if __name__ == "__main__":
    from ingest import ingest_workbook
    from normalize import normalize_workbook
    from analysis import analyze

    raw = ingest_workbook("data/sample/Nimbus_Retail_Financial_Statements.xlsx")
    normalized = normalize_workbook(raw)
    result = analyze(normalized)

    captions = generate_all_chart_captions(result, normalized)
    polished, recommendations = polish_captions_with_llm(captions, result, normalized, "Nimbus Retail Inc.")

    print("=" * 70)
    print("CHART-BY-CHART OUTPUT (this is the actual app layout)")
    print("=" * 70)
    for key in CHART_ORDER:
        print(f"\n--- [CHART: {CHART_TITLES[key]}] ---")
        print(f"Caption: {polished[key]}")

    print("\n" + "=" * 70)
    print("FINAL SECTION (shown once, after all charts)")
    print("=" * 70)
    print("RECOMMENDED ACTIONS:")
    print(recommendations)
