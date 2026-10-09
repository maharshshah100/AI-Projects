"""
generate.py — Answer synthesis, grounded strictly in retrieved context.

This is the one piece of the pipeline that genuinely requires a live LLM call
and therefore can't be executed inside this build sandbox (no network). The
code below is real, complete, and meant to be run locally with your own
GROQ_API_KEY. A `mock_generate()` fallback is also provided purely so
the rest of the pipeline (decompose -> retrieve -> rerank -> gate -> format)
can be exercised end-to-end offline, for development and demoing the
plumbing -- it is NOT a substitute for real generation quality and should
never be mistaken for one.
"""
from typing import List, Dict
from gate import format_context_block, format_citation

SYSTEM_PROMPT = """You are a financial research assistant. You answer questions ONLY using \
the numbered source excerpts provided below. Follow these rules strictly:

1. Every factual claim in your answer must be traceable to one of the numbered sources.
   Cite it inline like this: "Apple's total net sales were $391,035 million [1]."
2. If the provided sources do not contain enough information to answer confidently, say so \
explicitly -- do not fill the gap with general knowledge or a plausible-sounding guess.
3. When comparing two companies, be precise about which figure came from which company's \
filing -- never blend or average numbers from different sources unless asked to.
4. Keep the answer concise and directly responsive to the question asked.

SOURCES:
{context}
"""


def generate_llm(question: str, context_chunks: List[Dict], api_key: str = None) -> str:
    """
    Production version. Requires `pip install groq` and network access --
    run this locally. Groq has a genuinely free tier (fast Llama inference,
    no payment required to start) -- get a key at console.groq.com.
    """
    from groq import Groq  # noqa: local import, only needed when this path runs

    client = Groq(api_key=api_key)
    context = format_context_block(context_chunks)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def mock_generate(question: str, context_chunks: List[Dict]) -> str:
    """
    Offline stand-in so the full pipeline can be run and demoed without an
    API key. This does NOT synthesize a real answer -- it deterministically
    extracts the single most relevant sentence from the top chunk and cites
    it, so you can verify the plumbing (retrieval -> rerank -> gate ->
    context -> citation) works end-to-end before spending API credits on
    real generation.
    """
    if not context_chunks:
        return "I don't have enough information in the filings to answer that."

    q_tokens = set(question.lower().split())

    def line_score(line):
        has_dollar = "$" in line
        keyword_hits = sum(1 for t in q_tokens if len(t) > 3 and t.strip("?.,'s") in line.lower())
        digit_count = sum(ch.isdigit() for ch in line)
        return (has_dollar, keyword_hits, digit_count)

    best_line, best_chunk = None, None
    best_score = (False, -1, -1)
    for chunk in context_chunks:
        for line in chunk["text"].split("\n"):
            if not line.strip():
                continue
            score = line_score(line)
            if score > best_score:
                best_score = score
                best_line, best_chunk = line, chunk

    if best_line is None:
        best_line, best_chunk = context_chunks[0]["text"][:200], context_chunks[0]

    return (
        f"[MOCK GENERATION -- not a real synthesized answer, offline plumbing test only]\n\n"
        f"Most relevant retrieved line: \"{best_line.strip()}\"\n\n"
        f"Source: {format_citation(best_chunk)}"
    )


if __name__ == "__main__":
    from retrieval import HybridRetriever
    from rerank import heuristic_rerank
    from decompose import decompose_rule_based
    from gate import check_confidence

    retriever = HybridRetriever("data/processed/chunks.json")
    question = "What was Apple's total net sales in fiscal year 2024?"

    sub_queries = decompose_rule_based(question)
    sq = sub_queries[0]
    first_pass = retriever.search(sq.text, top_k=5, company_filter=sq.company_filter)
    reranked = heuristic_rerank(sq.text, first_pass, top_k=3)
    gate_result = check_confidence(reranked)

    print(f"QUESTION: {question}")
    print(f"Gate: confident={gate_result.confident}\n")

    if gate_result.confident:
        answer = mock_generate(question, reranked)
        print(answer)
    else:
        print(f"Declined: {gate_result.reason}")
