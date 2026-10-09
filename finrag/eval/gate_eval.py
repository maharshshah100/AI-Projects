"""
gate_eval.py — Evaluates the confidence gate's decline/answer DECISION itself,
separately from retrieval quality.

This is a different question from retrieval_eval.py: not "did we find the
right chunk" but "did the gate correctly decide whether to answer at all."
Scored as a binary classifier would be -- precision, recall, and a confusion
matrix on the should_answer label from the golden dataset.

Why this matters as its own metric: a system that always answers has perfect
recall on real questions but zero ability to say "I don't know" -- exactly
the overconfident failure mode this whole project exists to avoid. A system
that always declines has perfect precision but is useless. Neither number
alone tells you if the gate is actually good; you need both together.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import HybridRetriever
from decompose import decompose_rule_based
from rerank import heuristic_rerank
from gate import check_confidence
from golden_dataset import GOLDEN_DATASET


def evaluate_gate(retriever: HybridRetriever):
    rows = []
    for ex in GOLDEN_DATASET:
        sub_queries = decompose_rule_based(ex.query)
        any_confident = False
        for sq in sub_queries:
            first_pass = retriever.search(sq.text, top_k=5, company_filter=sq.company_filter)
            reranked = heuristic_rerank(sq.text, first_pass, top_k=3)
            gate = check_confidence(reranked)
            if gate.confident:
                any_confident = True

        predicted_answer = any_confident
        actual_should_answer = ex.should_answer

        if predicted_answer and actual_should_answer:
            outcome = "true_positive"    # correctly chose to answer
        elif not predicted_answer and not actual_should_answer:
            outcome = "true_negative"    # correctly declined
        elif predicted_answer and not actual_should_answer:
            outcome = "false_positive"   # WRONG: answered when it shouldn't have (overconfident)
        else:
            outcome = "false_negative"   # WRONG: declined when it could have answered (overcautious)

        rows.append({"query": ex.query, "category": ex.category, "outcome": outcome,
                      "predicted_answer": predicted_answer, "should_answer": actual_should_answer})

    tp = sum(r["outcome"] == "true_positive" for r in rows)
    tn = sum(r["outcome"] == "true_negative" for r in rows)
    fp = sum(r["outcome"] == "false_positive" for r in rows)
    fn = sum(r["outcome"] == "false_negative" for r in rows)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "rows": rows,
    }


if __name__ == "__main__":
    retriever = HybridRetriever("../data/processed/chunks.json")
    report = evaluate_gate(retriever)

    print("=== CONFIDENCE GATE EVALUATION ===")
    print(f"Accuracy:  {report['accuracy']:.1%}")
    print(f"Precision: {report['precision']:.1%}  (of answers given, how many should have been answered)")
    print(f"Recall:    {report['recall']:.1%}  (of answerable questions, how many did it actually answer)")
    print(f"\nConfusion matrix: TP={report['tp']} TN={report['tn']} FP={report['fp']} FN={report['fn']}")
    print("\nFalse positives (answered when it should have declined -- the dangerous failure mode):")
    for r in report["rows"]:
        if r["outcome"] == "false_positive":
            print(f"  ⚠️  {r['query']}")
    print("\nFalse negatives (declined when it could have answered -- the overcautious failure mode):")
    for r in report["rows"]:
        if r["outcome"] == "false_negative":
            print(f"  ⚠️  {r['query']}")
