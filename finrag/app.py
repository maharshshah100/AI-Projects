"""
app.py — FinRAG Streamlit frontend.

An interactive, deployable UI over the real pipeline:

ingest -> decompose -> retrieval -> rerank -> gate -> web fallback -> generate

The user-facing interface focuses on clear, evidence-grounded answers.
Pipeline traces and retrieval diagnostics remain available internally
through the returned pipeline result for development and evaluation.
"""

import os
import re
import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from dotenv import load_dotenv

from decompose import decompose_rule_based, KNOWN_COMPANIES
from retrieval import HybridRetriever
from rerank import heuristic_rerank
from gate import check_confidence, format_citation

from web_search import (
    search_web,
    format_web_source,
    get_tavily_api_key,
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinRAG",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Structured revenue data for charting
# ---------------------------------------------------------------------------

REVENUE_SERIES = {
    "AAPL": {
        "label": "Apple",
        "unit": "$B",
        "years": [2016, 2019, 2020, 2021, 2022, 2023, 2024],
        "values": [215.64, 260.17, 274.52, 365.82, 394.33, 383.29, 391.04],
    },

    "MSFT": {
        "label": "Microsoft",
        "unit": "$B",
        "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "values": [
            91.15,
            96.57,
            110.36,
            125.84,
            143.01,
            168.09,
            198.27,
            211.91,
            245.12,
        ],
    },

    "AMZN": {
        "label": "Amazon",
        "unit": "$B",
        "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "values": [
            135.99,
            177.87,
            232.89,
            280.52,
            386.06,
            469.82,
            513.98,
            574.78,
            637.96,
        ],
    },

    "GOOGL": {
        "label": "Alphabet",
        "unit": "$B",
        "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "values": [
            90.27,
            110.86,
            136.82,
            161.86,
            182.53,
            257.64,
            282.84,
            307.39,
            350.02,
        ],
    },

    "META": {
        "label": "Meta",
        "unit": "$B",
        "years": [2023, 2024],
        "values": [134.90, 164.50],
    },

    "TSLA": {
        "label": "Tesla",
        "unit": "$B",
        "years": [2023, 2024],
        "values": [96.77, 97.69],
    },

    "NVDA": {
        "label": "Nvidia",
        "unit": "$B (FY)",
        "years": [2023, 2024],
        "values": [26.97, 60.92],
    },

    "NFLX": {
        "label": "Netflix",
        "unit": "$B",
        "years": [2020, 2021, 2022, 2023, 2024],
        "values": [25.0, 29.7, 31.6, 33.72, 39.0],
    },

    "PYPL": {
        "label": "PayPal",
        "unit": "$B",
        "years": [2021, 2024],
        "values": [25.4, 31.79],
    },

    "V": {
        "label": "Visa",
        "unit": "$B (FY)",
        "years": [2023, 2024],
        "values": [32.65, 35.93],
    },

    "JPM": {
        "label": "JPMorgan Chase",
        "unit": "$B (net revenue)",
        "years": [2023],
        "values": [158.1],
    },

    "GS": {
        "label": "Goldman Sachs",
        "unit": "$B (net revenue, approx.)",
        "years": [2021, 2025],
        "values": [59.34, 58.28],
    },
}


BANK_TICKERS = {"JPM", "GS"}


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_retriever():
    return HybridRetriever(
        "data/processed/chunks.json"
    )


@st.cache_data
def load_chunks_meta():
    with open(
        "data/processed/chunks.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------

def get_groq_api_key():
    """
    Load Groq API key from local .env or Streamlit Secrets.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if api_key:
            return api_key

    except (KeyError, FileNotFoundError):
        pass

    return None


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def detect_intent(query: str) -> str:

    q = query.lower()

    superlative_words = [
        "highest",
        "most",
        "best",
        "fastest",
        "top",
        "leading",
        "biggest",
        "largest",
        "which company",
        "which one",
    ]

    if any(
        word in q
        for word in superlative_words
    ):
        return "superlative"

    if re.search(
        r"\bcompare\b|\bvs\.?\b|\bversus\b",
        q,
    ):
        return "comparison"

    trend_words = [
        "how has",
        "how have",
        "how did",
        "changed",
        "change",
        "trend",
        "trends",
        "growth",
        "grew",
        "increased",
        "decreased",
        "declined",
        "risen",
        "rose",
        "fallen",
        "fall",
        "over the years",
        "over time",
        "recently",
        "trajectory",
        "historical",
        "history",
    ]

    if any(
        word in q
        for word in trend_words
    ):
        return "trend"

    return "lookup"


# ---------------------------------------------------------------------------
# Intent-specific instructions
# ---------------------------------------------------------------------------

INTENT_INSTRUCTIONS = {

    "superlative": (
        "This is a RANKING question. Identify the company that best "
        "satisfies the user's criterion. Give the winning company, the "
        "relevant figure, and a brief evidence-supported explanation. "
        "Do not unnecessarily list every other company."
    ),

    "comparison": (
        "This is a COMPARISON question. Give a direct synthesized "
        "comparison between the requested companies. Explain who is ahead, "
        "the magnitude of the difference, and the most important "
        "evidence-supported reason for the difference. Do not simply provide "
        "two disconnected company summaries."
    ),

    "trend": (
        "This is a TREND-ANALYSIS question. The user wants more than a "
        "single number. Synthesize the available evidence and explain the "
        "trend clearly. When the evidence supports it, include: the starting "
        "value, ending value, absolute change, percentage change, direction "
        "and magnitude of the trend, the main evidence-supported driver, "
        "and a concise overall takeaway. Give enough context that the user "
        "can understand the trend without immediately needing a follow-up "
        "question. Do not invent causes or explanations that are not "
        "supported by the retrieved evidence."
    ),

    "lookup": (
        "This is a factual lookup question. Answer the question directly, "
        "but provide enough context to make the answer useful. If the "
        "retrieved evidence contains closely related supporting information "
        "that helps explain the answer, include it briefly."
    ),
}


# ---------------------------------------------------------------------------
# Grounded generation prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """
You are FinRAG, a grounded financial research assistant.

Your job is to answer the user's question using ONLY the numbered source
excerpts provided below.

Never invent a financial figure, percentage, date, company fact, business
driver, or conclusion that is not supported by the sources.

QUESTION TYPE:
{intent}

{intent_instructions}


==================================================
ANSWER STYLE
==================================================

Give the user a useful, well-explained answer rather than the shortest
possible answer.

The answer should feel like a professional financial research assistant
explaining the evidence to the user.

For analytical questions, do not merely repeat numbers. Explain what the
numbers mean and what the overall trend indicates.

However, do not add unsupported speculation just to make the answer sound
more sophisticated.


==================================================
FORMATTING REQUIREMENTS
==================================================

- Start directly with the answer.
- Use a short bold heading when it improves readability.
- Use clean bullet points for multiple observations.
- Use normal Markdown.
- Use $ before financial values when the source uses dollars.
- Write values naturally, for example:
  $26.97 billion
  $60.92 billion
  126%
- Put spaces between words and numbers.
- Never generate malformed formatting such as:
  billionin
  FY2024to
  ∗∗
- Never output raw HTML.
- Never output SVG.
- Never output XML.
- Never output code unless the user explicitly asks for code.
- Do not create unnecessary sections.
- Do not repeat the same conclusion multiple times.


==================================================
TABLES, COMPARISONS, AND NUMERICAL ANALYSIS
==================================================

When a question requests a comparison across multiple companies, years, or
metrics, prefer a readable Markdown table if it makes the answer clearer.

For a multi-company request:
- Address every company explicitly named by the user.
- Do not silently omit a company because evidence is incomplete.
- If a value cannot be verified from the supplied sources, write "Not
  available in retrieved evidence" for that value and explain the gap.
  Do not treat a missing local chunk as proof that a company never reported
  the figure; use any relevant web-search evidence supplied in the context too.
- Separate reported figures from calculated figures.
- When calculating change, use ending value minus starting value.
- When calculating percentage growth, use (ending - starting) / starting * 100.
- Keep units consistent and label them clearly. For company revenue comparisons,
  prefer one common unit (usually USD billions) for every company and year.
  Never label a billion-dollar change as millions, or mix $M and $B without
  explicitly converting and labeling the values.
- If ranking companies, rank only comparable, supported values and state
  which metric determines each ranking. Check the units and arithmetic before
  presenting the final order.
- If reporting periods differ, explain the fiscal-year-end difference when
  the supplied evidence establishes it.
- After the table, summarize the most important findings and the overall
  takeaway; do not merely repeat the table row by row.
- Never fill missing values from memory, charts, or unsupported assumptions.

For trend questions spanning several years, use a year-by-year table when
multiple data points are available, then describe the direction, major
changes, and any drivers explicitly supported by the sources.

==================================================
CITATIONS
==================================================

Every factual claim derived from the retrieved evidence must include an
inline citation.

Use:

[1]

or:

[1][2]

Do not create citations that do not correspond to the numbered sources.

Do not add a separate "Sources" or "Sources cited" section.

The inline citations are sufficient.


==================================================
CRITICAL EVIDENCE RULE
==================================================

The sources were retrieved using similarity/keyword retrieval, reranking,
or web search.

A source can mention the correct company while still failing to answer the
specific question.

Before answering, verify that the retrieved evidence actually supports the
claim.

If the evidence does not support the requested claim, explicitly say so.

Never answer from a loosely related source.


==================================================
TIME PERIOD RULE
==================================================

If the user asks about a particular year, period, or level of detail that
the sources do not contain:

- Say that the requested evidence is not available.
- Do not silently substitute another year.
- Do not extrapolate.
- Do not guess.


==================================================
TREND ANALYSIS RULE
==================================================

For trend questions, provide a synthesized explanation.

When supported by the sources, explain:

- where the metric started,
- where it ended,
- how much it changed in absolute terms,
- how much it changed in percentage terms,
- whether the trend represents acceleration, deceleration, stability,
  or decline,
- the main evidence-supported driver,
- and the overall takeaway.

Do not manufacture a driver simply because it is commonly associated with
the company or industry.

If the source explicitly says that AI demand, data-center demand, pricing,
volume, acquisitions, product launches, or another factor affected the
result, you may explain that factor and cite it.

If the source does not explain the driver, simply say that the retrieved
evidence does not establish the cause.


==================================================
WEB SOURCE RULE
==================================================

Web sources are evidence retrieved by Tavily.

Do not claim that a web source is an SEC filing unless the source itself
clearly identifies it as one.

Treat web evidence with the same citation discipline as SEC evidence.


==================================================
FINANCIAL COMPARABILITY RULE
==================================================

If a source explains that a bank reports net revenue rather than gross
sales, and the user compares that bank with a non-bank company, explicitly
explain the comparability issue.

Do not treat those figures as directly equivalent without qualification.


==================================================
FUTURE INFORMATION RULE
==================================================

Do NOT automatically provide a future outlook.

Only discuss future expectations when:

1. the user explicitly asks about the future, OR
2. the retrieved sources contain clear forward-looking evidence such as
   management guidance, stated expectations, planned investments, forecasts,
   or explicit future plans.

Do not infer future performance from historical growth alone.

Do not append a generic "Future outlook" section to historical questions.


==================================================
ANSWER LENGTH
==================================================

For a simple factual lookup:
- usually 1-3 useful bullets.

For a comparison:
- usually 3-5 useful bullets.

For a trend or analytical question:
- usually 4-7 useful bullets or a short paragraph followed by bullets.

The goal is to answer the user's question comprehensively enough that the
user does not immediately need several follow-up questions.

Do not add filler merely to increase length.


==================================================
SOURCES
==================================================

{context}
"""


# ---------------------------------------------------------------------------
# Web context formatting
# ---------------------------------------------------------------------------

def format_context_source(
    index,
    source,
):
    """
    Format either SEC evidence or web evidence for Groq.
    """

    if source.get(
        "filing_type"
    ) == "Web":

        title = source.get(
            "section",
            "Web source",
        )

        url = source.get(
            "source_url",
            "",
        )

        source_label = title

        if url:
            source_label += (
                f" — {url}"
            )

    else:

        source_label = format_citation(
            source
        )

    return (
        f"[{index}] Source: "
        f"{source_label}\n"
        f"{source.get('text', '')}"
    )


# ---------------------------------------------------------------------------
# Generated answer cleanup
# ---------------------------------------------------------------------------

def clean_generated_answer(
    answer: str,
) -> str:
    """
    Clean common formatting artifacts produced by the LLM.

    This function does not change the financial meaning of the answer.
    It only removes malformed markup and spacing artifacts.
    """

    if not answer:
        return ""

    text = answer

    # -------------------------------------------------------
    # Remove accidental fenced SVG / HTML / XML blocks
    # -------------------------------------------------------

    text = re.sub(
        r"```(?:svg|html|xml)?\s*.*?```",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # -------------------------------------------------------
    # Remove raw SVG blocks
    # -------------------------------------------------------

    text = re.sub(
        r"<svg\b.*?</svg>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # -------------------------------------------------------
    # Remove raw HTML/XML tags
    # -------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    # -------------------------------------------------------
    # Fix malformed Markdown emphasis
    # -------------------------------------------------------

    text = text.replace(
        "∗∗",
        "**",
    )

    # -------------------------------------------------------
    # Fix common missing-space artifacts
    # -------------------------------------------------------

    text = re.sub(
        r"(\d)(billion|million|trillion)",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(billion|million|trillion)(in|to)",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(FY\d{4})\s*\*{0,2}to\*{0,2}",
        r"\1 to ",
        text,
        flags=re.IGNORECASE,
    )

    # -------------------------------------------------------
    # Normalize excessive whitespace
    # -------------------------------------------------------

    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Main RAG + Web fallback pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    api_key: str,
    chat_history=None,
):

    chat_history = chat_history or []
    retrieval_query = query

    # Resolve short and metric-only follow-ups before retrieval. In particular,
    # questions such as "What was the percentage growth, and how does it compare
    # with the previous fiscal year?" contain no company name, so searching them
    # alone can return no useful evidence.
    query_lower = query.lower()
    followup_markers = (
        "what about", "and in ", "how about", "why", "explain that",
        "more details", "what caused", "compare them", "compare with",
        "compared with", "compared to", "how does it compare",
        "which one", "same for", "in that case", "previous fiscal year",
        "previous year", "percentage growth", "absolute growth",
        "what was the growth", "how much did it grow", "those figures",
        "that growth", "that figure", "the percentage", "how does that",
    )
    looks_like_followup = (
        len(query.split()) <= 8
        or any(marker in query_lower for marker in followup_markers)
    )

    if chat_history and looks_like_followup:
        previous_user = next(
            (item.get("content", "") for item in reversed(chat_history)
             if item.get("role") == "user"),
            "",
        )
        previous_answer = next(
            (item.get("content", "") for item in reversed(chat_history)
             if item.get("role") == "assistant"),
            "",
        )
        if previous_user:
            retrieval_query = (
                f"Previous user question: {previous_user}\n"
                f"Previous assistant answer: {previous_answer[:1800]}\n"
                f"Current follow-up question: {query}\n"
                "Resolve references in the follow-up using the prior context. "
                "Retrieve evidence for any additional years or calculations requested."
            )

    retriever = load_retriever()

    sub_queries = decompose_rule_based(
        retrieval_query
    )

    trace = {
        "sub_queries": [],
        "retrieved_chunks": [],
        "gate_results": [],
        "web_sources": [],
    }

    all_context = []
    missing_sub_queries = []

    # -------------------------------------------------------
    # STEP 1 — LOCAL SEC RAG
    # -------------------------------------------------------

    for sq in sub_queries:

        # Search local SEC corpus.
        #
        # If there is a company filter, restrict retrieval
        # to that company.
        #
        # If there is no company filter, search the complete
        # corpus.

        if sq.company_filter is not None:

            first_pass = retriever.search(
                sq.text,
                top_k=5,
                company_filter=sq.company_filter,
            )

        else:

            first_pass = retriever.search(
                sq.text,
                top_k=5,
            )

        reranked = heuristic_rerank(
            sq.text,
            first_pass,
            top_k=3,
        )

        gate = check_confidence(
            reranked
        )

        trace["sub_queries"].append(
            sq
        )

        trace["retrieved_chunks"].append(
            (
                sq.company_filter or "unscoped",
                reranked,
            )
        )

        trace["gate_results"].append(
            (
                sq.company_filter or "unscoped",
                gate,
            )
        )

        # ---------------------------------------------------
        # LOCAL EVIDENCE IS STRONG
        # ---------------------------------------------------

        if gate.confident:

            all_context.extend(
                reranked
            )
        else:
            missing_sub_queries.append(sq)

    # -------------------------------------------------------
    # STEP 2 — TAVILY FALLBACK
    # -------------------------------------------------------

    #
    # If local SEC retrieval did not produce any confident
    # evidence, use Tavily.
    #

    missing_company_evidence = any(
        not gate_result.confident
        for _, gate_result in trace["gate_results"]
    )
    should_try_web = (
        not all_context
        or (len(sub_queries) > 1 and missing_company_evidence)
    )

    if should_try_web:
        # Search each sub-question that lacked confident local evidence.
        # This is important for comparisons: a broad multi-company web query
        # can return results for only one company and leave the rest missing.
        web_results = []
        seen_urls = set()
        targeted_queries = [sq.text for sq in missing_sub_queries]
        if not targeted_queries:
            targeted_queries = [query]

        for targeted_query in targeted_queries:
            try:
                # Add a light filing-focused hint so the fallback is more likely
                # to find original annual reports rather than generic summaries.
                filing_query = (
                    f"{targeted_query} SEC 10-K annual report reported figures"
                )
                results = search_web(filing_query, max_results=3) or []
            except Exception as exc:
                trace.setdefault("web_errors", []).append(
                    f"Web search failed for sub-query: {targeted_query}: {exc}"
                )
                results = []

            for result in results:
                url = (result.get("url") or "").strip()
                identity = url or (
                    (result.get("title") or "") + "|" +
                    (result.get("text") or "")[:120]
                )
                if identity in seen_urls:
                    continue
                seen_urls.add(identity)
                web_results.append(result)

        trace["web_sources"] = web_results

        for result in web_results:
            all_context.append(
                {
                    "text": result.get("text", ""),
                    "company": result.get("title", "Web source"),
                    "filing_type": "Web",
                    "fiscal_year": "N/A",
                    "filed_date": "N/A",
                    "section": result.get("title", "Web result"),
                    "source_url": result.get("url", ""),
                }
            )

        if not web_results and not all_context:

            tavily_key = get_tavily_api_key()

            if tavily_key:
                reason = (
                    "Local SEC retrieval did not provide enough evidence, "
                    "and Tavily returned no usable web sources."
                )
            else:
                reason = (
                    "Local SEC retrieval did not provide enough evidence, "
                    "and TAVILY_API_KEY is not configured."
                )

            return {
                "declined": True,
                "reason": reason,
                "trace": trace,
            }

    # -------------------------------------------------------
    # STEP 3 — BUILD GROQ CONTEXT
    # -------------------------------------------------------

    context_block = "\n\n".join(
        format_context_source(
            i + 1,
            source,
        )
        for i, source in enumerate(
            all_context
        )
    )

    # -------------------------------------------------------
    # STEP 4 — DETECT QUESTION INTENT
    # -------------------------------------------------------

    intent = detect_intent(
        query
    )

    # -------------------------------------------------------
    # STEP 5 — BUILD SYSTEM PROMPT
    # -------------------------------------------------------

    system_prompt = (
        SYSTEM_PROMPT_TEMPLATE.format(
            intent=intent,
            intent_instructions=(
                INTENT_INSTRUCTIONS[
                    intent
                ]
            ),
            context=context_block,
        )
    )

    # -------------------------------------------------------
    # STEP 6 — GROQ GENERATION
    # -------------------------------------------------------

    import groq

    client = groq.Groq(
        api_key=api_key
    )

    response = (
        client.chat.completions.create(
            model="openai/gpt-oss-120b",

            # Longer analytical answers need room for complete tables,
            # rankings, explanations, and citations.
            max_tokens=3500,

            messages=(
                [{
                    "role": "system",
                    "content": system_prompt,
                }]
                + [
                    {"role": item.get("role"), "content": item.get("content", "")}
                    for item in chat_history[-6:]
                    if item.get("role") in ("user", "assistant")
                ]
                + [{"role": "user", "content": query}]
            ),
        )
    )

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    answer_text = choice.message.content or ""

    answer_text = clean_generated_answer(
        answer_text
    )

    # -------------------------------------------------------
    # STEP 7 — RETURN RESULT
    # -------------------------------------------------------

    return {
        "declined": False,

        "answer": answer_text,

        "intent": intent,

        "finish_reason": finish_reason,
        "output_truncated": finish_reason == "length",

        "trace": trace,

        "sources": all_context,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.title(
        "📊 FinRAG"
    )

    st.caption(
        "Advanced RAG over real SEC filings "
        "& web evidence"
    )

    # -------------------------------------------------------
    # API STATUS
    # -------------------------------------------------------

    api_key = get_groq_api_key()

    tavily_key = get_tavily_api_key()

    if api_key:

        st.success(
            "Groq API key loaded"
        )

    else:

        st.error(
            "GROQ_API_KEY is not configured "
            "in .env or Streamlit Secrets."
        )

    if tavily_key:

        st.success(
            "Tavily web-search key loaded"
        )

    else:

        st.warning(
            "TAVILY_API_KEY is not configured. "
            "Web fallback will be unavailable."
        )

    st.divider()

    # -------------------------------------------------------
    # CORPUS METRICS
    # -------------------------------------------------------

    chunks = load_chunks_meta()

    companies_covered = sorted(
        set(
            c["ticker"]
            for c in chunks
        )
    )

    st.metric(
        "Companies covered",
        len(companies_covered),
    )

    st.metric(
        "Indexed chunks",
        len(chunks),
    )

    with st.expander(
        "Companies in corpus"
    ):

        for ticker in companies_covered:

            bank_flag = (
                " 🏦 (bank — net revenue)"
                if ticker in BANK_TICKERS
                else ""
            )

            company_label = (
                REVENUE_SERIES
                .get(
                    ticker,
                    {},
                )
                .get(
                    "label",
                    ticker,
                )
            )

            st.write(
                f"**{company_label}** "
                f"({ticker})"
                f"{bank_flag}"
            )

    st.divider()

    st.caption(
        "Built on: hybrid TF-IDF + BM25 retrieval, "
        "query decomposition, confidence-gated generation, "
        "Tavily web fallback, and citation tracing."
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

tab_chat, tab_explore, tab_about = st.tabs(
    [
        "💬 Ask",
        "📈 Revenue Explorer",
        "ℹ️ About this project",
    ]
)


# ===========================================================================
# ASK TAB
# ===========================================================================

with tab_chat:

    st.subheader(
        "Ask a question about any covered company"
    )

    if "finrag_chat_history" not in st.session_state:
        st.session_state["finrag_chat_history"] = []

    for item in st.session_state["finrag_chat_history"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    example_cols = st.columns(
        4
    )

    examples = [
        "Which company grew the highest and by what rate last year?",
        "Compare Visa's and PayPal's recent revenue growth",
        "How has Nvidia's revenue changed recently?",
        "Compare JPMorgan Chase's revenue to Apple's — are they even comparable?",
    ]

    for col, example in zip(
        example_cols,
        examples,
    ):

        if col.button(
            example,
            use_container_width=True,
        ):

            st.session_state[
                "query_input"
            ] = example

    query = st.text_input(
        "Your question",

        key="query_input",

        placeholder=(
            "e.g. What have Microsoft's sales "
            "looked like over the last several years?"
        ),
    )

    if st.button(
        "Ask",
        type="primary",
    ):

        if not api_key:

            st.error(
                "GROQ_API_KEY is not configured "
                "in .env or Streamlit Secrets."
            )

        elif not query:

            st.warning(
                "Type a question."
            )

        else:

            with st.spinner(
                "Decomposing → SEC retrieval → "
                "re-ranking → confidence gate → "
                "web fallback if needed → generating..."
            ):

                result = run_pipeline(
                    query,
                    api_key,
                    chat_history=st.session_state["finrag_chat_history"],
                )

            # ---------------------------------------------------
            # DECLINED
            # ---------------------------------------------------

            if result["declined"]:

                st.warning(
                    f"**Could not answer confidently:** "
                    f"{result['reason']}"
                )

            # ---------------------------------------------------
            # USER-FACING ANSWER
            # ---------------------------------------------------

            else:

                st.markdown(
                    result["answer"]
                )

                if result.get("output_truncated"):
                    st.warning(
                        "The model reached its output-token limit before it "
                        "finished the response. Some table rows, rankings, "
                        "explanations, or citations may be missing. Try asking "
                        "for the analysis in two parts, or reduce the scope."
                    )

                st.session_state["finrag_chat_history"].append(
                    {"role": "user", "content": query}
                )
                st.session_state["finrag_chat_history"].append(
                    {"role": "assistant", "content": result["answer"]}
                )

    if st.button("Clear conversation"):
        st.session_state["finrag_chat_history"] = []
        st.rerun()


# ===========================================================================
# REVENUE EXPLORER TAB
# ===========================================================================

with tab_explore:

    st.subheader(
        "Revenue trends across covered companies"
    )

    selected = st.multiselect(
        "Select companies to plot",

        options=list(
            REVENUE_SERIES.keys()
        ),

        default=[
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
        ],

        format_func=lambda ticker:
            REVENUE_SERIES[
                ticker
            ][
                "label"
            ],
    )

    if selected:

        fig = go.Figure()

        for ticker in selected:

            series = (
                REVENUE_SERIES[
                    ticker
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=series[
                        "years"
                    ],

                    y=series[
                        "values"
                    ],

                    mode="lines+markers",

                    name=(
                        f"{series['label']} "
                        f"({series['unit']})"
                    ),
                )
            )

        fig.update_layout(
            xaxis_title="Year",

            yaxis_title="Revenue",

            height=500,

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        if any(
            ticker in BANK_TICKERS
            for ticker in selected
        ):

            st.info(
                "🏦 JPMorgan Chase and Goldman Sachs "
                "report net revenue (after interest expense), "
                "not gross sales like the other companies here. "
                "Their figures aren't directly comparable to "
                "the rest on this chart without accounting "
                "for that."
            )

    st.divider()

    st.caption(
        "Table view"
    )

    rows = []

    for ticker in selected:

        series = (
            REVENUE_SERIES[
                ticker
            ]
        )

        for year, value in zip(
            series["years"],
            series["values"],
        ):

            rows.append(
                {
                    "Company": series[
                        "label"
                    ],

                    "Year": year,

                    "Revenue": (
                        f"{value} "
                        f"{series['unit']}"
                    ),
                }
            )

    if rows:

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


# ===========================================================================
# ABOUT TAB
# ===========================================================================

with tab_about:

    st.markdown(
        """
### What this is

An advanced retrieval-augmented generation system over real financial
filing data, with a web-search fallback when the local evidence is
insufficient.

The goal is to demonstrate practical RAG engineering skills in a portfolio
project, rather than simply wrapping an LLM in a chatbot.

### Pipeline

1. **Query decomposition**
   - Multi-company questions are split into company-scoped sub-queries.

2. **Hybrid retrieval**
   - TF-IDF and BM25 retrieval are combined.

3. **Re-ranking**
   - Retrieved evidence is rescored using financial-specific signals
     such as number, year, and sentence overlap.

4. **Confidence gating**
   - Weak local evidence is not blindly passed to the model.

5. **Web fallback**
   - Tavily searches the web when local SEC evidence is insufficient.

6. **Grounded generation**
   - Groq synthesizes the answer from the retrieved evidence.

7. **Citation tracing**
   - The answer is accompanied by the sources used by the system.

### Important design principle

The LLM is NOT responsible for searching the web.

Tavily retrieves web evidence.

The Groq model receives that evidence and generates the final answer.

This reduces the chance of the model answering financial questions from
memory when reliable evidence is required.

### Known limitations

- The Revenue Explorer currently uses the structured revenue series
  defined in the application.
- The local SEC corpus currently covers 12 major companies.
- Web-search coverage depends on Tavily's available sources.
- Bank revenue (JPMorgan and Goldman Sachs) uses different accounting
  treatment than product companies and is flagged when comparisons
  involve those companies.
"""
    )