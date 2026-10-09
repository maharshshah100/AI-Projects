"""
retrieval_eval.py — Measures retrieval quality against the golden dataset.

Two standard IR metrics, computed honestly (no LLM needed, fully offline):

  - recall@k: for each example with a known correct chunk, did retrieval
    return it anywhere in the top k results? Answers "did we find it at all."
  - MRR (Mean Reciprocal Rank): if found, how high up was it? A correct
    chunk buried at rank 5 is worse than one at rank 1, even though both
    count the same for recall@k -- MRR captures that difference.

Examples with should_answer=False and empty expected_chunk_ids are excluded
from these two metrics (there's nothing to "retrieve correctly" for a
question that shouldn't be answered at all) -- they're scored separately by
gate_eval.py instead.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import HybridRetriever
from decompose import decompose_rule_based
from golden_dataset import GOLDEN_DATASET


def evaluate_retrieval(retriever: HybridRetriever, k: int = 5):
    results = []
    for ex in GOLDEN_DATASET:
        if not ex.expected_chunk_ids:
            continue  # nothing to measure recall/MRR against

        # Use the REAL decomposition logic, not a simplified stand-in --
        # an earlier version of this eval called search() once with
        # ex.company_filter directly, which doesn't match what the actual
        # pipeline does for a query with no named company (it fans out to
        # ALL companies via decompose_rule_based's superlative fallback).
        # That mismatch caused a false failure on the superlative example
        # below before this fix -- the retrieval was never actually broken,
        # the eval was testing something the real pipeline doesn't do.
        sub_queries = decompose_rule_based(ex.query)
        retrieved_ids = []
        for sq in sub_queries:
            if sq.company_filter is None and len(sub_queries) == 1:
                hits = retriever.search(sq.text, top_k=k, company_filter=None)
            else:
                hits = retriever.search(sq.text, top_k=k, company_filter=sq.company_filter)
            retrieved_ids.extend(h["chunk_id"] for h in hits)

        hit = any(cid in retrieved_ids for cid in ex.expected_chunk_ids)
        rank = None
        for i, cid in enumerate(retrieved_ids, 1):
            if cid in ex.expected_chunk_ids:
                rank = i
                break

        results.append({
            "query": ex.query,
            "category": ex.category,
            "hit": hit,
            "rank": rank,
            "reciprocal_rank": (1.0 / rank) if rank else 0.0,
            "expected": ex.expected_chunk_ids,
            "retrieved": retrieved_ids,
            "n_sub_queries": len(sub_queries),
            "effective_pool_size": len(retrieved_ids),
        })

    n = len(results)
    recall_at_k = sum(r["hit"] for r in results) / n if n else 0.0
    mrr = sum(r["reciprocal_rank"] for r in results) / n if n else 0.0

    return {
        "n_examples": n,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "k": k,
        "per_example": results,
    }


if __name__ == "__main__":
    retriever = HybridRetriever("../data/processed/chunks.json")
    report = evaluate_retrieval(retriever, k=5)

    print(f"=== RETRIEVAL EVALUATION (n={report['n_examples']}, k={report['k']}) ===")
    print(f"Recall@{report['k']}: {report['recall_at_k']:.1%}")
    print(f"MRR:      {report['mrr']:.3f}")
    print()
    print("Per-example detail:")
    for r in report["per_example"]:
        status = f"HIT  (rank {r['rank']})" if r["hit"] else "MISS"
        pool_note = f" [pool={r['effective_pool_size']}, {r['n_sub_queries']} sub-q]" if r["n_sub_queries"] > 1 else ""
        print(f"  [{status:14s}] [{r['category']:10s}] {r['query'][:55]}{pool_note}")
        if not r["hit"]:
            print(f"      expected: {r['expected']}, got: {r['retrieved'][:5]}")
