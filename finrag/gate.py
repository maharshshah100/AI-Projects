"""
gate.py — Confidence gating and citation formatting.

Two responsibilities, kept deliberately separate from generation:

1. Confidence gate: decide, from retrieval scores ALONE (no LLM call needed,
   so this is fully testable offline), whether the retrieved evidence is
   strong enough to answer from -- or whether the honest response is "I
   don't have strong enough evidence for that."

2. Citation formatting: attach exact source metadata (company, filing,
   section, URL) to every retrieved chunk so a downstream answer can cite
   precisely where each fact came from, not just "according to the filing."
"""
from dataclasses import dataclass
from typing import List, Dict


# Tuned against this corpus's own score distribution (see retrieval.py's test
# run): a fused RRF score below this means neither ranker found a strong
# match -- both TF-IDF cosine similarity and BM25 were weak, not just one.
CONFIDENCE_THRESHOLD = 0.015

# RRF alone is RANK-based: with a small corpus it always returns a "best
# available" match even when nothing is truly relevant. A minimum raw
# similarity floor catches total retrieval misses (near-zero overlap).
#
# IMPORTANT, empirically discovered limitation: raw TF-IDF similarity CANNOT
# reliably separate a legitimate question from a nonsense one once a real
# company name appears in both. Measured directly on this corpus: "How many
# unicorns does Tesla have in its stable?" scores 0.218 against Tesla's own
# data -- HIGHER than the legitimate question "What have Microsoft's sales
# looked like over the last several years?" scores against Microsoft's own
# data (0.077). TF-IDF has no real concept understanding; it just sees that
# "Tesla" matches "Tesla" and scores accordingly, blind to "unicorns" making
# the rest of the question nonsensical.
#
# Conclusion: a numeric threshold is the wrong tool for catching THAT failure
# mode -- it requires real semantic judgment, which only the LLM actually
# has. So this floor is deliberately set low, to catch only genuine retrieval
# misses (near-zero overlap, e.g. an off-topic query against the wrong
# company entirely), while the generation prompt is responsible for catching
# "the right company was mentioned, but this specific claim isn't actually
# supported by the sources" -- see generate.py / app.py's system prompt.
MIN_RAW_SIMILARITY = 0.04


@dataclass
class GateResult:
    confident: bool
    reason: str
    top_score: float


def check_confidence(results: List[Dict]) -> GateResult:
    if not results:
        return GateResult(confident=False, reason="No chunks retrieved at all.", top_score=0.0)

    top_score = results[0]["fused_score"]
    # Check the BEST raw similarity across all returned chunks, not just
    # results[0] -- results[0] is picked by fused/rerank score, which is a
    # different ranking than raw similarity, so the single strongest semantic
    # match can legitimately sit at rank #2 or #3. Gating on results[0] alone
    # was a real bug: it rejected a query where the correct answer chunk was
    # retrieved, just not ranked first by the (unrelated) fusion metric.
    best_tfidf = max((r["tfidf_score"] for r in results), default=0.0)

    if top_score < CONFIDENCE_THRESHOLD:
        return GateResult(
            confident=False,
            reason=f"Top fused score ({top_score:.4f}) is below threshold "
                   f"({CONFIDENCE_THRESHOLD}).",
            top_score=top_score,
        )
    if best_tfidf < MIN_RAW_SIMILARITY:
        return GateResult(
            confident=False,
            reason=f"Best raw similarity across retrieved chunks ({best_tfidf:.3f}) is below "
                   f"the floor ({MIN_RAW_SIMILARITY}) -- none of the candidates is a genuinely "
                   f"strong match, not just the top-ranked one. Note: this still can't catch a "
                   f"fabricated sub-claim embedded in an otherwise on-topic question -- that's "
                   f"what faithfulness scoring in the eval framework is specifically for.",
            top_score=top_score,
        )
    return GateResult(confident=True, reason="Retrieval confidence acceptable.", top_score=top_score)


def format_citation(chunk: Dict) -> str:
    return (f"{chunk['company']} {chunk['filing_type']} "
            f"(FY{chunk['fiscal_year']}, filed {chunk['filed_date']}), "
            f"\u201c{chunk['section']}\u201d \u2014 {chunk['source_url']}")


def format_context_block(chunks: List[Dict]) -> str:
    """Builds the exact context block that would be handed to the LLM for
    generation -- each chunk tagged with a citation index the model is
    instructed to reference inline."""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] Source: {format_citation(c)}\n{c['text']}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    from retrieval import HybridRetriever
    from decompose import decompose_rule_based

    retriever = HybridRetriever("data/processed/chunks.json")

    test_cases = [
        "What was Apple's Services gross margin in fiscal year 2024?",
        "What was Apple's revenue from operations on the moon?",  # should fail the gate
        "How did Apple's revenue compare to Microsoft's over the last two years?",
    ]

    for query in test_cases:
        print(f"\n{'='*70}\nQUERY: {query}\n{'='*70}")
        sub_queries = decompose_rule_based(query)
        for sq in sub_queries:
            results = retriever.search(sq.text, top_k=3, company_filter=sq.company_filter)
            gate = check_confidence(results)
            print(f"\n  Sub-query [{sq.company_filter or 'unscoped'}]: {sq.text}")
            print(f"  Confident: {gate.confident} (top_score={gate.top_score:.4f}) -- {gate.reason}")
            if gate.confident:
                print(f"  Top source: {format_citation(results[0])}")
