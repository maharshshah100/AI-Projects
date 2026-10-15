"""
Insight engine: takes the ALREADY-COMPUTED ratios, trends, and anomalies
from analysis.py and asks an LLM (Groq, Llama 3.3 70B) to write a
plain-English, stakeholder-ready narrative from them.

CRITICAL DESIGN PRINCIPLE: the LLM never computes anything here. Every
number in the prompt is already correct, by formula, before the LLM ever
sees it. The LLM's only job is explaining and prioritizing numbers that
are already known to be right -- not deriving new ones. This is what
keeps the narrative trustworthy: it can't hallucinate a ratio that
doesn't exist, because it's never asked to calculate one.
"""
from __future__ import annotations
import os

from dotenv import load_dotenv
load_dotenv()

from analysis import AnalysisResult, fmt_pct, fmt_ratio
from normalize import NormalizedStatement

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a financial analyst writing a summary for a company's leadership \
team (CFO, board members) who are busy and want the point quickly, not a lecture.

You will be given a company's computed financial ratios, year-over-year trends, and a \
list of anomalies already detected by a separate rules-based system.

Rules:
1. Use ONLY the numbers given to you. Do not calculate new ratios, estimate figures \
that weren't provided, or reference typical industry benchmarks unless explicitly given.
2. Every claim must trace back to a specific number you were given -- write like an \
analyst citing evidence, not like someone speculating.
3. Structure your response in exactly these sections, in this order:
   - Executive Summary (2-3 sentences, the single most important takeaway)
   - Key Strengths (bullet points, only include genuine strengths -- don't invent \
positives to "balance" the tone if the data doesn't support them)
   - Areas of Concern (bullet points, ordered most severe first, using the severity \
levels you were given)
   - Recommended Actions (specific, actionable bullet points a CFO could actually act on \
this quarter -- not generic advice like "monitor closely")
4. Be direct. If the numbers show a real problem, say so plainly -- do not soften a \
declining net income into "an opportunity for optimization."
5. Keep the whole response under 400 words. Executives want the signal, not the noise."""


def format_ratios_for_prompt(result: AnalysisResult) -> str:
    lines = []
    for period in result.periods:
        r = result.ratios[period]
        lines.append(
            f"{period}: Gross margin {fmt_pct(r.gross_margin)}, "
            f"Operating margin {fmt_pct(r.operating_margin)}, "
            f"Net margin {fmt_pct(r.net_margin)}, ROE {fmt_pct(r.roe)}, ROA {fmt_pct(r.roa)}, "
            f"Current ratio {fmt_ratio(r.current_ratio)}, Quick ratio {fmt_ratio(r.quick_ratio)}, "
            f"Debt-to-equity {fmt_ratio(r.debt_to_equity)}, Interest coverage {fmt_ratio(r.interest_coverage)}, "
            f"Asset turnover {fmt_ratio(r.asset_turnover)}, Inventory turnover {fmt_ratio(r.inventory_turnover)}"
        )
    return "\n".join(lines)


def format_anomalies_for_prompt(result: AnalysisResult) -> str:
    if not result.anomalies:
        return "No anomalies were flagged by the rules-based detection system."
    lines = []
    for a in result.anomalies:
        lines.append(f"[{a.severity.upper()}] ({a.category}, {a.period}) {a.message}")
    return "\n".join(lines)


def format_key_figures(is_stmt: NormalizedStatement | None, bs_stmt: NormalizedStatement | None) -> str:
    """Builds whichever key-figure lines are actually available. A
    missing statement (e.g. no Balance Sheet uploaded) just means fewer
    lines in the prompt, not a crash -- same principle as analysis.py."""
    if is_stmt is None:
        return "(No Income Statement data available.)"
    periods = is_stmt.periods
    lines = []

    def line_if_present(stmt, field_name, label, prefix="$", suffix="k"):
        if stmt is not None and field_name in stmt.fields:
            vals = ", ".join(f"{p}={prefix}{stmt.fields[field_name].get(p, 0):,.0f}{suffix}"
                              for p in periods if p in stmt.fields[field_name])
            if vals:
                lines.append(f"{label}: {vals}")

    line_if_present(is_stmt, "revenue", "Revenue")
    line_if_present(is_stmt, "net_income", "Net Income")
    line_if_present(bs_stmt, "cash", "Cash")

    if bs_stmt is not None and "short_term_debt" in bs_stmt.fields and "long_term_debt" in bs_stmt.fields:
        vals = ", ".join(
            f"{p}=${(bs_stmt.fields['short_term_debt'].get(p, 0) + bs_stmt.fields['long_term_debt'].get(p, 0)):,.0f}k"
            for p in periods)
        lines.append(f"Total Debt: {vals}")

    if not lines:
        return "(No key figures could be computed from the available data.)"
    return "\n".join(lines)


def build_prompt(result: AnalysisResult, normalized: dict[str, NormalizedStatement], company_name: str) -> str:
    is_stmt = normalized.get("Income Statement")
    bs_stmt = normalized.get("Balance Sheet")
    missing_note = ""
    if bs_stmt is None:
        missing_note = "\n(Note: no Balance Sheet was provided -- balance-sheet-dependent ratios like liquidity and leverage are not available for this analysis.)"
    return f"""Company: {company_name}
Periods analyzed: {', '.join(result.periods)}{missing_note}

KEY FIGURES:
{format_key_figures(is_stmt, bs_stmt)}

COMPUTED RATIOS BY PERIOD:
{format_ratios_for_prompt(result)}

ANOMALIES DETECTED (by a separate rules-based system, already verified against the data):
{format_anomalies_for_prompt(result)}

Write the stakeholder summary now, following the exact structure and rules you were given."""


def generate_insight(result: AnalysisResult, normalized: dict[str, NormalizedStatement],
                      company_name: str = "the company", api_key: str | None = None,
                      stream: bool = False):
    """
    Returns the narrative text (str) if stream=False, or a generator
    yielding text chunks if stream=True (for the UI's streaming display
    in Phase 6).
    """
    prompt = build_prompt(result, normalized, company_name)
    api_key = api_key or os.environ.get("GROQ_API_KEY")

    if not api_key:
        fallback = _rule_based_fallback_summary(result, company_name)
        if stream:
            def _gen():
                yield fallback
            return _gen()
        return fallback

    from groq import Groq
    client = Groq(api_key=api_key)

    if stream:
        def _stream_gen():
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=700,
                stream=True,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _stream_gen()

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def _rule_based_fallback_summary(result: AnalysisResult, company_name: str) -> str:
    """
    No API key available -- rather than showing nothing, produce a
    genuinely useful (if less polished) summary directly from the
    anomalies, so the tool is never a dead end. Clearly labeled as a
    fallback, not pretending to be the LLM's writing.

    Includes rule-based strengths and recommended actions too, not just
    a raw findings dump -- the LLM version writes these as prose with
    more nuance, but the *substance* (what's good, what to do about each
    problem) should already be present even without a live API key.
    """
    lines = [
        f"[Rule-based fallback summary -- no GROQ_API_KEY set, so this wasn't written by "
        f"the LLM. Get a free key at console.groq.com for the full narrative version.]\n",
        f"## {company_name} -- Financial Summary ({result.periods[0]} to {result.periods[-1]})\n",
    ]
    high = [a for a in result.anomalies if a.severity == "high"]
    medium = [a for a in result.anomalies if a.severity == "medium"]
    latest = result.periods[-1]
    r = result.ratios[latest]

    lines.append(f"**{len(result.anomalies)} findings detected** ({len(high)} high severity, "
                 f"{len(medium)} medium severity).\n")

    # --- Strengths: genuine, rule-based, not invented for balance ---
    strengths = []
    if r.current_ratio is not None and r.current_ratio >= 1.5:
        strengths.append(f"Current ratio of {fmt_ratio(r.current_ratio)} is still comfortably above the "
                          f"1.0x minimum, despite the declining trend.")
    if r.interest_coverage is not None and r.interest_coverage >= 3.0:
        strengths.append(f"Interest coverage of {fmt_ratio(r.interest_coverage)} indicates operating income "
                          f"still comfortably covers interest obligations.")
    revenue_trends = [t for t in result.trends if t.metric == "revenue" and t.yoy_pct_change is not None]
    if revenue_trends and all(t.yoy_pct_change > 0 for t in revenue_trends):
        strengths.append("Revenue has grown every year in the period analyzed, indicating the "
                          "top-line/demand side of the business is not the immediate concern.")
    if strengths:
        lines.append("### Key Strengths")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    if high:
        lines.append("### High-Severity Findings")
        for a in high:
            lines.append(f"- **[{a.period}]** {a.message}")
        lines.append("")
    if medium:
        lines.append("### Medium-Severity Findings")
        for a in medium:
            lines.append(f"- **[{a.period}]** {a.message}")
        lines.append("")

    # --- Recommended actions: mapped directly from anomaly categories ---
    categories_present = {a.category for a in result.anomalies}
    actions = []
    if "profitability" in categories_present:
        actions.append("Investigate the specific drivers of margin compression (input costs, pricing, "
                        "mix) before the next reporting period -- this is the largest single issue by "
                        "count and severity.")
    if "liquidity" in categories_present:
        actions.append("Reassess the dividend payout policy given it has exceeded net income -- "
                        "continuing this funded by cash/debt is not sustainable indefinitely.")
    if "leverage" in categories_present:
        actions.append("Slow the pace of new debt issuance until operating cash flow improves, "
                        "to avoid compounding the leverage trend.")
    if "working_capital" in categories_present:
        actions.append("Review credit terms extended to customers and inventory ordering policy -- "
                        "both are growing faster than the sales they're meant to support.")
    if actions:
        lines.append("### Recommended Actions")
        for act in actions:
            lines.append(f"- {act}")

    return "\n".join(lines)


if __name__ == "__main__":
    from ingest import ingest_workbook
    from normalize import normalize_workbook
    from analysis import analyze

    raw = ingest_workbook("data/sample/Nimbus_Retail_Financial_Statements.xlsx")
    normalized = normalize_workbook(raw)
    result = analyze(normalized)

    print("=" * 70)
    print("EXACT PROMPT THAT WOULD BE SENT TO GROQ")
    print("=" * 70)
    prompt = build_prompt(result, normalized, "Nimbus Retail Inc.")
    print(prompt)

    print("\n" + "=" * 70)
    print("GENERATED SUMMARY (fallback, since no GROQ_API_KEY in this sandbox)")
    print("=" * 70)
    summary = generate_insight(result, normalized, company_name="Nimbus Retail Inc.")
    print(summary)
