"""
Chat engine -- the conversational layer.

Every question is routed down the fastest path that can answer it correctly:

  1. EXACT FIGURE  ("what was revenue in FY2024?", "net margin?")
     Pure lookup in the already-computed data. No LLM, no network:
     instant, and cannot be "almost right".
  2. YOUR DOCUMENTS  (open-ended: "why did margins fall?", "what does
     management plan for the dividend?")
     The LLM answers using ONLY the company's computed analysis plus the
     most relevant passages retrieved from uploaded notes.
  3. THE WEB  (not answerable from the documents: "what's a healthy
     debt-to-equity ratio for retail?")
     Tavily search -> LLM synthesis, always labeled as external.
  4. LIMITED MODE  (no LLM key / network problem)
     Shows the most relevant material found instead of failing.

Replies are streaming iterators of text so the UI can show words as they
are generated (perceived latency ~ time to first token).
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Iterator

from dotenv import load_dotenv
load_dotenv()

from analysis import fmt_pct, fmt_ratio
from normalize import ALIASES
from insight_engine import format_ratios_for_prompt, format_anomalies_for_prompt
from web_search import web_search, format_web_context

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _model() -> str:
    return os.environ.get("GROQ_MODEL", DEFAULT_MODEL)


@dataclass
class ChatReply:
    source: str                      # "exact" | "documents" | "web" | "general" | "limited"
    stream: Iterator[str]
    citations: list[dict] = field(default_factory=list)
    note: str | None = None


# ----------------------------------------------------------------------------
# 1. Exact-figure lookup
# ----------------------------------------------------------------------------
RATIO_KEYWORDS = [
    ("gross margin", "gross_margin", "pct"), ("operating margin", "operating_margin", "pct"),
    ("ebit margin", "operating_margin", "pct"), ("net margin", "net_margin", "pct"),
    ("profit margin", "net_margin", "pct"), ("return on equity", "roe", "pct"),
    ("roe", "roe", "pct"), ("return on assets", "roa", "pct"), ("roa", "roa", "pct"),
    ("current ratio", "current_ratio", "x"), ("quick ratio", "quick_ratio", "x"),
    ("acid test", "quick_ratio", "x"), ("debt to equity", "debt_to_equity", "x"),
    ("interest coverage", "interest_coverage", "x"), ("asset turnover", "asset_turnover", "x"),
    ("inventory turnover", "inventory_turnover", "x"),
]
RATIO_LABELS = {
    "gross_margin": "Gross margin", "operating_margin": "Operating margin", "net_margin": "Net margin",
    "roe": "Return on equity", "roa": "Return on assets", "current_ratio": "Current ratio",
    "quick_ratio": "Quick ratio", "debt_to_equity": "Debt-to-equity",
    "interest_coverage": "Interest coverage", "asset_turnover": "Asset turnover",
    "inventory_turnover": "Inventory turnover",
}
RATIO_STATEMENT = {  # which uploaded statement a missing ratio depends on
    "current_ratio": "Balance Sheet", "quick_ratio": "Balance Sheet", "debt_to_equity": "Balance Sheet",
    "roe": "Balance Sheet", "roa": "Balance Sheet", "asset_turnover": "Balance Sheet",
    "inventory_turnover": "Balance Sheet",
}

# Words that mean "this needs reasoning/benchmarks, not a lookup"
OPEN_ENDED = re.compile(
    r"\b(why|how come|explain|should|recommend|suggest|advice|good|healthy|ideal|benchmark|typical|normal|"
    r"industry|average|compare|comparison|improve|risk|risks|define|meaning|mean|concern|worried|worry|"
    r"outlook|plan|strategy|reason|cause|driver|drivers)\b|what (is|are|was|were) (a|an)\b")

_alias_to_field = sorted(
    ((a.replace("-", " "), f) for f, variants in ALIASES.items() for a in variants),
    key=lambda x: -len(x[0]),
)


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().replace("-", " ")).strip()


def _find_periods(q: str, periods: list[str]) -> list[str]:
    qn = re.sub(r"\s+", "", q.lower())
    hit = [p for p in periods if p.lower().replace(" ", "") in qn]
    if hit:
        return hit
    years = re.findall(r"\b(20\d{2})\b", q)
    if years:
        hit = [p for p in periods if any(y in p for y in years)]
        if hit:
            return hit
    ql = q.lower()
    if re.search(r"\b(last year|previous|prior|before)\b", ql) and len(periods) >= 2:
        return [periods[-2]]
    if re.search(r"\b(latest|current|this year|most recent|now|last quarter)\b", ql):
        return [periods[-1]]
    return []


def _unit_suffix(stmt) -> str:
    hint = (stmt.unit_hint or "").strip()
    return f" ({hint})" if hint else ""


def try_exact_answer(question: str, session, last_topic: tuple | None = None):
    """Returns (markdown_text, citations, topic) or None if this isn't a lookup question."""
    if session.result is None or not session.normalized:
        return None
    ql = _norm_q(question)
    if OPEN_ENDED.search(ql):
        return None

    periods = session.result.periods
    topic = None

    # ratio?
    for kw, attr, kind in RATIO_KEYWORDS:
        if re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", ql):
            topic = ("ratio", attr, kind)
            break
    # line item?
    if topic is None:
        for alias, fld in _alias_to_field:
            if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", ql):
                if any(fld in st.fields for st in session.normalized.values()):
                    topic = ("field", fld, None)
                    break
    # follow-up ("and last year?", "what about FY2024?")
    if topic is None and last_topic and re.search(
            r"\b(and|what about|how about|then|same)\b|^(in|for|during)\b", ql) and _find_periods(question, periods):
        topic = last_topic
    if topic is None:
        return None

    chosen = _find_periods(question, periods) or periods

    if topic[0] == "ratio":
        attr, kind = topic[1], topic[2]
        label = RATIO_LABELS[attr]
        vals = {p: getattr(session.result.ratios[p], attr) for p in chosen}
        if all(v is None for v in vals.values()):
            need = RATIO_STATEMENT.get(attr)
            why = f" It needs a {need}, which wasn't uploaded." if need and need not in session.normalized else \
                  " The required line items weren't found in your documents."
            return (f"I can't compute **{label}** from the uploaded data.{why}", [], topic)
        fmt = fmt_pct if kind == "pct" else fmt_ratio
        body = " · ".join(f"{p}: **{fmt(v)}**" for p, v in vals.items())
        text = f"**{label}** — {body}"
        series = [v for v in vals.values() if v is not None]
        if len(chosen) > 1 and len(series) >= 2:
            delta = series[-1] - series[0]
            unit = "pts" if kind == "pct" else "x"
            amt = f"{delta*100:+.1f} pts" if kind == "pct" else f"{delta:+.2f}x"
            text += f"\n\nChange from {chosen[0]} to {chosen[-1]}: **{amt}**."
        cites = [{"label": "Computed ratio", "detail": "calculated from your uploaded statements"}]
        return text, cites, topic

    fld = topic[1]
    stmt_name, stmt = next((n, st) for n, st in session.normalized.items() if fld in st.fields)
    label = ALIASES[fld][0].capitalize()
    data = stmt.fields[fld]
    vals = {p: data[p] for p in chosen if p in data}
    if not vals:
        return (f"**{label}** isn't reported for {', '.join(chosen)} in your {stmt_name}.", [], topic)
    body = " · ".join(f"{p}: **{v:,.0f}**" for p, v in vals.items())
    text = f"**{label}**{_unit_suffix(stmt)} — {body}"
    items = list(vals.items())
    if len(items) >= 2 and items[0][1]:
        first, last = items[0][1], items[-1][1]
        pct = (last - first) / abs(first) * 100
        text += f"\n\nChange from {items[0][0]} to {items[-1][0]}: **{pct:+.1f}%**."
    cites = [{"label": "Exact figure", "detail": f"{session.statement_files.get(stmt_name, stmt_name)} → {stmt_name}"}]
    return text, cites, topic


# ----------------------------------------------------------------------------
# Context builders
# ----------------------------------------------------------------------------
def _cash_flow_control_totals(session) -> str:
    """Build explicit, deterministic cash-flow bridges from normalized statement rows.

    This prevents the LLM from treating net change in cash as operating cash flow.
    Source cash-flow line items are signed as reported (cash outflows are usually negative).
    """
    st = session.normalized.get("Cash Flow Statement")
    if st is None:
        return "Cash-flow bridge: unavailable; no Cash Flow Statement was uploaded."

    fields = st.fields
    periods = session.result.periods if session.result else list(
        dict.fromkeys(p for values in fields.values() for p in values)
    )
    def value(field_name, period):
        raw = fields.get(field_name, {}).get(period)
        return float(raw) if raw is not None else None

    lines = [
        "CASH-FLOW DEFINITIONS (authoritative; do not interchange these metrics):",
        "- Operating cash flow (OCF) is the reported 'Operating Cash Flow' / 'Cash from Operating Activities' line (normalized field: cfo).",
        "- Net change in cash is a separate total (normalized field: net_change_cash), generally OCF + investing cash flow + financing cash flow.",
        "- Cash after capex and dividends is a derived measure: OCF - absolute capex outflow - absolute dividends paid. It is NOT OCF and is NOT net change in cash.",
        "- Do not label net change in cash, free cash flow, or cash after capex/dividends as operating cash flow.",
        "PERIOD-BY-PERIOD CASH-FLOW FIGURES (USD as reported; missing means not supplied):",
    ]
    for period in periods:
        cfo = value("cfo", period)
        capex = value("capex", period)
        dividends = value("dividends_paid", period)
        cfi = value("cfi", period)
        cff = value("cff", period)
        net_change = value("net_change_cash", period)
        def fmt(v):
            return "not supplied" if v is None else f"${v:,.0f}"
        lines.append(
            f"{period}: operating cash flow (cfo)={fmt(cfo)}; capex as reported={fmt(capex)}; "
            f"dividends paid as reported={fmt(dividends)}; investing cash flow (cfi)={fmt(cfi)}; "
            f"financing cash flow (cff)={fmt(cff)}; reported net change in cash={fmt(net_change)}."
        )
        if cfo is not None and capex is not None and dividends is not None:
            after_capex_dividends = cfo - abs(capex) - abs(dividends)
            lines.append(
                f"{period} deterministic calculation: OCF less absolute capex and dividends = "
                f"${cfo:,.0f} - ${abs(capex):,.0f} - ${abs(dividends):,.0f} = ${after_capex_dividends:,.0f}. "
                "This is cash after capex and dividends, not operating cash flow."
            )
        if cfo is not None and cfi is not None and cff is not None and net_change is not None:
            calculated_change = cfo + cfi + cff
            delta = calculated_change - net_change
            lines.append(
                f"{period} cash-flow reconciliation: OCF + CFI + CFF = ${calculated_change:,.0f}; "
                f"reported net change in cash = ${net_change:,.0f}; difference = ${delta:,.0f}."
            )
    return "\n".join(lines)


def _deterministic_trend_context(session) -> str:
    """Compute period changes from normalized source values, never from the LLM."""
    if not session.result:
        return "Deterministic trend summary unavailable: analysis has not run."
    periods = session.result.periods
    income = session.normalized.get("Income Statement")
    cashflow = session.normalized.get("Cash Flow Statement")
    balance = session.normalized.get("Balance Sheet")
    def getv(stmt, field, period):
        if stmt is None:
            return None
        value = stmt.fields.get(field, {}).get(period)
        return float(value) if value is not None else None
    def money(v):
        return "not supplied" if v is None else f"${v:,.0f}"
    def growth(first, last):
        if first is None or last is None or first == 0:
            return "not calculable from supplied values"
        return f"{(last / abs(first) - 1) * 100:+.1f}% overall"
    rows = ["DETERMINISTIC TRENDS (computed from normalized source values):"]
    metrics = [
        ("Revenue", income, "revenue"),
        ("Net income", income, "net_income"),
        ("Operating cash flow (OCF)", cashflow, "cfo"),
        ("Net change in cash", cashflow, "net_change_cash"),
        ("Accounts receivable", balance, "accounts_receivable"),
        ("Inventory", balance, "inventory"),
    ]
    for label, stmt, field in metrics:
        values = [getv(stmt, field, period) for period in periods]
        available = [(period, value) for period, value in zip(periods, values) if value is not None]
        if not available:
            rows.append(f"- {label}: not supplied.")
            continue
        series = ", ".join(f"{period}={money(value)}" for period, value in available)
        rows.append(f"- {label}: {series}.")
        if len(available) >= 2:
            first_period, first_value = available[0]
            last_period, last_value = available[-1]
            rows.append(f"  {label} change from {first_period} to {last_period}: {growth(first_value, last_value)}.")
            for (prev_period, prev), (cur_period, cur) in zip(available, available[1:]):
                if prev != 0:
                    rows.append(f"  YoY {cur_period} vs {prev_period}: {(cur / abs(prev) - 1) * 100:+.1f}%.")
    if cashflow is not None:
        rows.append("DETERMINISTIC CASH AFTER CAPEX AND DIVIDENDS (OCF - absolute capex - absolute dividends; distinct from net change in cash):")
        for period in periods:
            cfo = getv(cashflow, "cfo", period)
            capex = getv(cashflow, "capex", period)
            dividends = getv(cashflow, "dividends_paid", period)
            if cfo is not None and capex is not None and dividends is not None:
                rows.append(f"- {period}: {money(cfo)} - {money(abs(capex))} - {money(abs(dividends))} = {money(cfo-abs(capex)-abs(dividends))}.")
            else:
                rows.append(f"- {period}: not calculable; OCF, capex, or dividends missing.")
    return "\n".join(rows)


def analysis_context(session) -> str:
    if session.result is None:
        return "No financial statements have been uploaded."
    lines = [f"Company: {session.company_name}", f"Periods: {', '.join(session.result.periods)}", ""]
    lines.append("LINE ITEMS (as reported):")
    for stmt_name, st in session.normalized.items():
        unit = (st.unit_hint or "").strip()
        lines.append(f"-- {stmt_name} {f'({unit})' if unit else ''}")
        for fld, vals in st.fields.items():
            lines.append(f"   {ALIASES[fld][0]}: " + ", ".join(f"{p}={v:,.0f}" for p, v in vals.items()))
    lines += ["", _deterministic_trend_context(session), "", _cash_flow_control_totals(session), "",
              "COMPUTED RATIOS:", format_ratios_for_prompt(session.result), "", "FLAGGED FINDINGS:",
              format_anomalies_for_prompt(session.result)]
    return "\n".join(lines)


def doc_context(session, question: str, top_k: int = 3):
    if session.retriever is None:
        return "", []
    try:
        hits = session.retriever.search(question, top_k=top_k)
    except Exception:
        return "", []
    text = "\n\n".join(f"[D{i}] ({h['source']} — {h['section']})\n{h['text']}" for i, h in enumerate(hits, 1))
    cites = [{"label": f"D{i}", "detail": f"{h['source']} — {h['section']}"} for i, h in enumerate(hits, 1)]
    return text, cites


GROUNDED_SYSTEM = """You are FinSight, a financial analysis assistant talking with a company's finance team.
You are given CONTEXT: the company's own figures, computed ratios, flagged findings, and excerpts from its documents.

Rules:
1. Use the structured LINE ITEMS and CASH-FLOW DEFINITIONS in CONTEXT as the source of truth. Never override them with a retrieved excerpt.
2. Keep financial metrics distinct. Operating cash flow is the reported cfo field; net change in cash is the separate net_change_cash field. Never call net change in cash "operating cash flow".
3. Use the deterministic cash-flow calculations supplied in CONTEXT. Do not recalculate them unless checking arithmetic, and never invent missing values.
4. A retrieved document excerpt is not proof that a number belongs to a different metric. Cite [D1], [D2] only for claims actually supported by those excerpts; cite the computed analysis as "the analysis".
5. If the CONTEXT lacks a figure, say it was not supplied. Do not fill gaps by guessing or use outside knowledge.
6. Advice may be suggested, but label targets or thresholds as recommendations, not as facts from the documents. Be concise and use clear Markdown."""

WEB_SYSTEM = """You are FinSight, a financial analysis assistant. The user's question could not be answered from their own
documents, so you are given WEB RESULTS (external) and a short summary of the company's own numbers.

Rules:
1. Base general claims on the WEB RESULTS and cite them like [W1]. Never present web information as coming from their documents.
2. Where helpful, relate it to the company's own numbers (given) and offer 2-3 concrete options or suggestions.
3. If the web results don't help, say so plainly. Under 200 words. Start by noting this is outside their documents."""

GENERAL_SYSTEM = """You are FinSight, a financial analysis assistant. The user's question isn't covered by their uploaded documents
and live web search is not enabled. Answer briefly from general financial knowledge, say clearly that this is general
knowledge (not from their documents and not web-verified), and relate it to the company's numbers where helpful.
Under 150 words."""


def _company_brief(session) -> str:
    if session.result is None:
        return "(no company figures uploaded)"
    p = session.result.periods[-1]
    r = session.result.ratios[p]
    top = [a.message for a in session.result.anomalies if a.severity == "high"][:3]
    return (f"{session.company_name}, latest period {p}: gross margin {fmt_pct(r.gross_margin)}, net margin "
            f"{fmt_pct(r.net_margin)}, current ratio {fmt_ratio(r.current_ratio)}, debt-to-equity "
            f"{fmt_ratio(r.debt_to_equity)}. Key concerns: " + (" | ".join(top) if top else "none flagged."))


# ----------------------------------------------------------------------------
# LLM plumbing
# ----------------------------------------------------------------------------
def _stream_llm(system: str, user: str, history: list[dict], max_tokens: int = 1400) -> Iterator[str]:
    """Stream visible answer text and surface provider errors instead of silently returning empty output."""
    try:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        formatting_rules = """FORMATTING RULES:
- Use standard Markdown only; use normal hyphens and ordinary punctuation.
- Never put asterisks between digits or words.
- Do not put backticks inside bold formatting.
- Preserve source figures exactly; distinguish source figures from suggestions.
- If the context supplies deterministic calculations, use them and do not contradict them.
- If evidence is missing, say it is not supplied instead of inventing it.
"""
        msgs = [{"role":"system", "content": system + "\n\n" + formatting_rules}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]
                 if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str)]
        msgs.append({"role": "user", "content": user})
        model = _model()
        kwargs = {"model": model, "messages": msgs, "temperature": 0.2, "stream": True}
        if "gpt-oss" in model.lower():
            # Groq reasoning models use max_completion_tokens; reasoning_effort
            # reserves less of the budget for hidden reasoning and more for answer text.
            kwargs["max_completion_tokens"] = max(1600, max_tokens)
            kwargs["reasoning_effort"] = "low"
        else:
            kwargs["max_tokens"] = max_tokens
        completion = client.chat.completions.create(**kwargs)
        yielded = False
        for chunk in completion:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if isinstance(content, str) and content:
                yielded = True
                yield content
        if not yielded:
            yield ("The AI provider returned no visible answer text. Check the terminal for provider details, "
                   "then verify GROQ_MODEL and the Groq API/model availability in your .env file.")
    except Exception as exc:
        # Keep the app usable while making the underlying cause visible.
        detail = str(exc).strip().replace("\n", " ")[:500]
        yield f"AI request failed ({type(exc).__name__}): {detail or 'No error details returned by provider.'}"


def _peek(stream: Iterator[str], probe_chars: int = 18):
    """Reads just enough of a stream to see how it starts, so we can decide
    routing (e.g. NOT_IN_DOCUMENTS) before showing anything to the user."""
    buf, it = "", iter(stream)
    for c in it:
        buf += c
        if len(buf) >= probe_chars:
            break

    def rejoin():
        if buf:
            yield buf
        yield from it
    return buf, rejoin()


def _once(text: str) -> Iterator[str]:
    yield text


class ChatEngine:
    def __init__(self, session):
        self.session = session
        self.last_topic = None

    @staticmethod
    def has_llm() -> bool:
        return bool(os.environ.get("GROQ_API_KEY"))

    def ask(self, question: str, history: list[dict] | None = None) -> ChatReply:
        history = history or []
        question = question.strip()
        if not question:
            return ChatReply("limited", _once("Ask me anything about your uploaded documents."))

        # 1) exact figure -- instant, no LLM
        try:
            exact = try_exact_answer(question, self.session, self.last_topic)
        except Exception:
            exact = None
        if exact:
            text, cites, topic = exact
            self.last_topic = topic
            return ChatReply("exact", _once(text), cites)

        # 2) grounded in the user's documents
        a_ctx = analysis_context(self.session)
        d_ctx, d_cites = doc_context(self.session, question)
        context = a_ctx + ("\n\nDOCUMENT EXCERPTS:\n" + d_ctx if d_ctx else "")

        if self.has_llm():
            try:
                stream = _stream_llm(GROUNDED_SYSTEM, f"CONTEXT:\n{context}\n\nQUESTION: {question}", history)
                head, stream = _peek(stream)
                if head.strip() and not head.strip().upper().startswith("NOT_IN"):
                    cites = [{"label": "Analysis", "detail": "ratios & figures computed from your statements"}] + d_cites
                    return ChatReply("documents", stream, cites)
                if not head.strip():
                    return self._limited(question, d_ctx, d_cites,
                                         "The model returned no visible text; showing available analysis instead.")
            except Exception as e:
                return self._limited(question, d_ctx, d_cites, f"AI answers are temporarily unavailable ({type(e).__name__}).")
        else:
            return self._limited(question, d_ctx, d_cites,
                                 "AI answer generation isn't configured (add GROQ_API_KEY) — showing the most relevant material instead.")

        # 3) not in the documents -> web
        return self._web(question, history)

    def _web(self, question: str, history: list[dict]) -> ChatReply:
        brief = _company_brief(self.session)
        res = web_search(question)
        if res["ok"] and res["results"]:
            ctx = format_web_context(res["results"])
            cites = [{"label": f"W{i}", "detail": f"{r['title']} — {r['url']}"} for i, r in enumerate(res["results"], 1)]
            try:
                stream = _stream_llm(WEB_SYSTEM, f"COMPANY NUMBERS:\n{brief}\n\nWEB RESULTS:\n{ctx}\n\nQUESTION: {question}", history)
                return ChatReply("web", stream, cites, note="This isn't in your documents — answered from web sources.")
            except Exception:
                snippets = "\n\n".join(f"**{r['title']}**\n{r['content'][:300]}…\n{r['url']}" for r in res["results"])
                return ChatReply("web", _once("This isn't in your documents. Top web results:\n\n" + snippets), cites)
        # no web search available -> general knowledge, clearly labeled
        reason = "Web search isn't enabled (add TAVILY_API_KEY)" if (res["error"] or "").startswith("TAVILY_API_KEY") \
            else "Web search failed just now"
        try:
            stream = _stream_llm(GENERAL_SYSTEM, f"COMPANY NUMBERS:\n{brief}\n\nQUESTION: {question}", history, max_tokens=350)
            return ChatReply("general", stream, [], note=f"{reason}. This is general knowledge, not from your documents.")
        except Exception as e:
            return ChatReply("limited", _once(f"This isn't covered by your uploaded documents. {reason}, and AI answers are unavailable."), [])

    def _limited(self, question, d_ctx, d_cites, note) -> ChatReply:
        parts = []
        if d_ctx:
            parts.append("**Most relevant passages from your documents:**\n\n" + d_ctx[:1100])
        if self.session.result and self.session.result.anomalies:
            high = [a.message for a in self.session.result.anomalies if a.severity == "high"][:3]
            if high:
                parts.append("**Key findings from the analysis:**\n" + "\n".join(f"- {m}" for m in high))
        if not parts:
            parts.append("I couldn't find anything relevant in the uploaded documents.")
        return ChatReply("limited", _once("\n\n".join(parts)), d_cites, note=note)
