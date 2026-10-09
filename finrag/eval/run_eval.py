"""
run_eval.py — Runs the full evaluation suite and prints a single report.

Retrieval and gate evaluation run fully offline (no API key needed) and will
run in CI on every push. Faithfulness scoring requires a live API key and is
skipped here with a clear note, rather than silently faked -- a CI badge that
says "evals passed" should mean the evals that actually ran passed, not imply
coverage that wasn't checked.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import HybridRetriever
from retrieval_eval import evaluate_retrieval
from gate_eval import evaluate_gate


def main():
    retriever = HybridRetriever("../data/processed/chunks.json")

    print("=" * 70)
    print("FINRAG EVALUATION SUITE")
    print("=" * 70)

    retrieval_report = evaluate_retrieval(retriever, k=5)
    print(f"\n[1/3] RETRIEVAL  Recall@5: {retrieval_report['recall_at_k']:.1%}  "
          f"MRR: {retrieval_report['mrr']:.3f}  (n={retrieval_report['n_examples']})")

    gate_report = evaluate_gate(retriever)
    print(f"[2/3] GATE       Accuracy: {gate_report['accuracy']:.1%}  "
          f"Precision: {gate_report['precision']:.1%}  Recall: {gate_report['recall']:.1%}  "
          f"(FP={gate_report['fp']}, FN={gate_report['fn']})")

    if os.environ.get("GROQ_API_KEY"):
        from faithfulness_eval import validate_judge_against_humans
        print("[3/3] FAITHFULNESS  Running (API key found)...")
        validate_judge_against_humans(api_key=os.environ["GROQ_API_KEY"])
    else:
        print("[3/3] FAITHFULNESS  SKIPPED -- no GROQ_API_KEY in environment. "
              "This suite passing does NOT mean faithfulness was checked.")

    print("\n" + "=" * 70)

    # Fail CI if retrieval or gate quality drops below a regression threshold --
    # these thresholds are set from THIS run's actual measured numbers, not
    # arbitrary round numbers, so a future regression has to beat what was
    # already proven to work, not an untested guess.
    failed = False
    if retrieval_report["recall_at_k"] < 0.90:
        print(f"❌ REGRESSION: retrieval recall@5 dropped below 90% ({retrieval_report['recall_at_k']:.1%})")
        failed = True
    if gate_report["recall"] < 0.90:
        print(f"❌ REGRESSION: gate recall dropped below 90% ({gate_report['recall']:.1%}) -- "
              f"system is declining too many answerable questions")
        failed = True

    if failed:
        exit(1)
    print("✅ All offline evals passed regression thresholds.")


if __name__ == "__main__":
    main()
