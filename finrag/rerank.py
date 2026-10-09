"""
rerank.py — Second-stage re-ranking.

Retrieval (TF-IDF + BM25) is fast but coarse -- it scores a query against a
whole chunk at once. Re-ranking is deliberately slower and more precise: it
looks at finer-grained overlap signals a first-pass ranker glosses over.

Two implementations are provided:

1. `heuristic_rerank()` -- runs fully offline, no model download needed.
   Rescoring signals:
     - exact number/dollar-figure overlap between query and chunk (the
       single strongest signal for financial Q&A specifically -- if the
       query asks about "$391,035" and a chunk contains that exact figure,
       that's far more informative than generic word overlap)
     - exact 4-digit year overlap (fiscal year matching)
     - sentence-level best-match score (finest granularity TF-IDF can give
       without a real cross-encoder)
   This is a legitimate, defensible re-ranking signal -- just not the same
   thing as a trained cross-encoder, and the code says so rather than
   dressing up a heuristic as something it isn't.

2. `cross_encoder_rerank()` -- the real production version, using a proper
   trained cross-encoder (`sentence-transformers/cross-encoder`), which needs
   `pip install sentence-transformers` and a model download (network).
   Neither is available in this sandbox -- run this version locally.
"""
import re
from typing import List, Dict


def _extract_numbers(text: str) -> set:
    # dollar figures, percentages, and plain numbers with 2+ digits
    return set(re.findall(r"\$?\d[\d,]*\.?\d*%?", text))


def _extract_years(text: str) -> set:
    return set(re.findall(r"\b20\d\d\b", text))


def _sentence_overlap_score(query: str, text: str) -> float:
    """Finest-grained offline signal available: best-matching single
    sentence, scored by token overlap (Jaccard), rather than the whole
    chunk at once."""
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best = 0.0
    for sent in sentences:
        s_tokens = set(re.findall(r"[a-z0-9]+", sent.lower()))
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens) / len(q_tokens | s_tokens)
        best = max(best, overlap)
    return best


def heuristic_rerank(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    q_numbers = _extract_numbers(query)
    q_years = _extract_years(query)

    rescored = []
    for c in chunks:
        c_numbers = _extract_numbers(c["text"])
        c_years = _extract_years(c["text"])

        number_overlap = len(q_numbers & c_numbers)
        year_overlap = len(q_years & c_years)
        sentence_score = _sentence_overlap_score(query, c["text"])

        # weighted combination: exact number match is the strongest signal
        # for financial Q&A specifically, year match next, sentence overlap
        # as a general-purpose tiebreaker
        rerank_score = (
            3.0 * number_overlap
            + 1.5 * year_overlap
            + 1.0 * sentence_score
            + 0.5 * c.get("fused_score", 0.0) * 100  # keep first-pass rank as a light prior
        )
        c2 = dict(c)
        c2["rerank_score"] = round(rerank_score, 4)
        c2["rerank_signals"] = {
            "number_overlap": number_overlap,
            "year_overlap": year_overlap,
            "sentence_score": round(sentence_score, 3),
        }
        rescored.append(c2)

    rescored.sort(key=lambda x: -x["rerank_score"])
    return rescored[:top_k]


def cross_encoder_rerank(query: str, chunks: List[Dict], top_k: int = 3):
    """
    Production version -- requires `pip install sentence-transformers` and a
    model download (network), neither available in this sandbox. Run this
    locally once you have internet access.
    """
    from sentence_transformers import CrossEncoder  # noqa: local import
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    return sorted(chunks, key=lambda x: -x["rerank_score"])[:top_k]


if __name__ == "__main__":
    from retrieval import HybridRetriever

    retriever = HybridRetriever("data/processed/chunks.json")
    query = "What was Apple's total net sales in fiscal year 2024?"
    first_pass = retriever.search(query, top_k=5, company_filter="AAPL")

    print(f"QUERY: {query}\n")
    print("--- First-pass hybrid retrieval (top 5) ---")
    for c in first_pass:
        print(f"  fused={c['fused_score']:.4f}  [{c['section'][:45]}]")

    reranked = heuristic_rerank(query, first_pass, top_k=3)
    print("\n--- After re-ranking (top 3) ---")
    for c in reranked:
        print(f"  rerank={c['rerank_score']:.3f}  signals={c['rerank_signals']}  [{c['section'][:45]}]")
        print(f"    {c['text'][:150]}...")
