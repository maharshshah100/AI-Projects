"""
decompose.py — Query decomposition for multi-hop questions.

A complex question like "compare Apple's and Microsoft's revenue growth"
needs to become two separate, company-scoped retrievals — searching for both
companies at once dilutes the query and (as retrieval.py's own test run
demonstrated) can surface the wrong company's chunk for a single-entity
question.

Two decomposition strategies are provided:
  1. `decompose_rule_based()` — deterministic, offline, no API calls. Detects
     known company names/tickers in the query and splits accordingly. This is
     what runs by default in this sandbox (no live LLM access here).
  2. `decompose_llm()` — the real production version. Uses the Groq API
     to handle genuinely complex, open-ended decomposition an offline
     regex can't (e.g. "how did the two companies' AI strategies differ" has
     no explicit metric to split on). Requires GROQ_API_KEY and network
     — run this version yourself locally.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

KNOWN_COMPANIES = {
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "amazon": "AMZN", "amzn": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "googl": "GOOGL",
    "meta": "META", "facebook": "META",
    "tesla": "TSLA", "tsla": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA",
    "netflix": "NFLX", "nflx": "NFLX",
    "paypal": "PYPL", "pypl": "PYPL",
    "visa": "V",
    "jpmorgan": "JPM", "jpmorgan chase": "JPM", "jp morgan": "JPM", "jpm": "JPM",
    "goldman sachs": "GS", "goldman": "GS",
}

CANONICAL_NAME = {
    "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Alphabet",
    "META": "Meta", "TSLA": "Tesla", "NVDA": "Nvidia", "NFLX": "Netflix",
    "PYPL": "PayPal", "V": "Visa", "JPM": "JPMorgan Chase", "GS": "Goldman Sachs",
}


@dataclass
class SubQuery:
    text: str
    company_filter: Optional[str]  # ticker to restrict retrieval to, or None for unscoped


def decompose_rule_based(query: str) -> List[SubQuery]:
    """
    Offline, deterministic decomposition: detect which known companies are
    named in the query. If exactly one, scope the whole query to it. If two
    or more, split into one company-scoped sub-query per company so retrieval
    for each doesn't get diluted or cross-contaminated by the other.

    Special case: a generic superlative question ("which company grew the
    most?") that names NO specific company is still a valid, answerable
    question -- it just means "compare all covered companies" implicitly.
    Without this, the pipeline would wrongly decline a perfectly good
    question just because it didn't happen to name anyone.
    """
    q_lower = query.lower()
    mentioned = []
    for name, ticker in KNOWN_COMPANIES.items():
        if re.search(rf"\b{name}\b", q_lower):
            if ticker not in mentioned:
                mentioned.append(ticker)

    if not mentioned:
        superlative_cues = ["which company", "which one", "highest", "most", "best", "fastest",
                             "top", "leading", "biggest", "largest"]
        if any(cue in q_lower for cue in superlative_cues):
            all_tickers = sorted(set(KNOWN_COMPANIES.values()))
            return [SubQuery(text=f"{CANONICAL_NAME.get(t, t)} {query}", company_filter=t) for t in all_tickers]
        return [SubQuery(text=query, company_filter=None)]

    if len(mentioned) == 1:
        return [SubQuery(text=query, company_filter=mentioned[0])]

    # Multi-company: build one sub-query per company, keeping the rest of the
    # question intact so retrieval still matches on topic (revenue, risk, etc.)
    sub_queries = []
    for ticker in mentioned:
        company_name = CANONICAL_NAME.get(ticker, ticker)
        sub_queries.append(SubQuery(text=f"{company_name} {query}", company_filter=ticker))
    return sub_queries


LLM_DECOMPOSE_PROMPT = """You are decomposing a financial research question into independent \
sub-questions that can each be answered by retrieving from ONE company's filing at a time.

Rules:
- If the question only concerns one company, return it unchanged as a single sub-question.
- If it compares multiple companies, split into one sub-question per company, each \
fully self-contained (repeat any shared context like the metric or time period).
- If the question requires multiple distinct facts about the SAME company (e.g. \
"what was revenue and what was R&D spend"), split those into separate sub-questions too \
-- each sub-question should target exactly one retrievable fact.
- Return ONLY a JSON list of objects: [{{"text": "...", "company_filter": "AAPL"|"MSFT"|null}}]

Question: {question}
"""


def decompose_llm(query: str, api_key: str = None) -> List[SubQuery]:
    """
    Production version — requires `pip install groq` and network access,
    neither available in this build sandbox. Run this locally with your own
    GROQ_API_KEY (free tier available at console.groq.com) once you've
    cloned the project.
    """
    from groq import Groq  # noqa: local import — only needed when this path actually runs
    import json

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=500,
        messages=[{"role": "user", "content": LLM_DECOMPOSE_PROMPT.format(question=query)}],
    )
    parsed = json.loads(response.choices[0].message.content)
    return [SubQuery(text=p["text"], company_filter=p.get("company_filter")) for p in parsed]


if __name__ == "__main__":
    test_queries = [
        "How did Apple's revenue growth compare to Microsoft's over the last two fiscal years?",
        "What was Apple's Services gross margin in 2024?",
        "Compare Apple and Microsoft's approach to AI risk in their filings",
        "What were Microsoft's three operating segments?",
    ]
    for q in test_queries:
        print(f"\nQUERY: {q}")
        for sq in decompose_rule_based(q):
            print(f"  -> [{sq.company_filter or 'unscoped'}] {sq.text}")
