"""
Full eval harness (Week 4). Reports:
  - Overall accuracy (refusal decision correct AND, for non-refusals,
    retrieval landed on a source from the expected tool)
  - Per-category breakdown (how-to, factual, troubleshooting, comparative,
    cross-tool, refusal, config)
  - Code-block integrity rate (from the chunker)
  - Retrieval-only latency (no LLM call needed to measure this)

Run: python3 eval/run_eval.py
"""
import glob
import json
import sys
import time

sys.path.insert(0, "src")
from retrieve import HybridRetriever  # noqa: E402
from generate import answer_question  # noqa: E402
from chunker import chunk_document, naive_chunk_document, code_block_integrity_rate  # noqa: E402


def tool_matches(expected_tool, got_sources) -> bool:
    if expected_tool is None:
        return True
    got_tools = {s["tool"] for s in got_sources}
    if expected_tool.startswith("any_of:"):
        allowed = set(expected_tool.replace("any_of:", "").split(","))
        return bool(got_tools & allowed)
    return expected_tool in got_tools


def run_retrieval_eval():
    golden = json.load(open("eval/golden_set.json"))
    retriever = HybridRetriever()

    results = []
    latencies = []
    for item in golden:
        t0 = time.time()
        res = answer_question(item["question"], retriever)
        latencies.append((time.time() - t0) * 1000)

        refuse_correct = res["refused"] == item["expected_refuse"]
        tool_correct = True
        source_correct = True
        if not item["expected_refuse"]:
            tool_correct = tool_matches(item["expected_tool"], res["sources"])
            if item["expected_source_contains"]:
                source_correct = any(
                    item["expected_source_contains"] in s["file"] for s in res["sources"]
                )
        row_correct = refuse_correct and tool_correct and source_correct
        results.append({
            **item,
            "got_refuse": res["refused"],
            "top_score": res["top_score"],
            "correct": row_correct,
        })

    return results, latencies


def print_breakdown(results, key):
    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(r[key], []).append(r)
    print(f"\nBy {key}:")
    for g, items in sorted(groups.items()):
        n_correct = sum(i["correct"] for i in items)
        print(f"  {g:<16} {n_correct}/{len(items)} = {100*n_correct/len(items):.1f}%")


def main():
    print("=" * 70)
    print("PART 1: Chunker comparison (code-aware vs naive)")
    print("=" * 70)
    smart, naive = [], []
    doc_files = (
        glob.glob("data/raw/kubernetes/*.md")
        + glob.glob("data/raw/docker/*.md")
        + glob.glob("data/raw/terraform/*.mdx")
    )
    for path in doc_files:
        text = open(path, encoding="utf-8").read()
        fname = path.split("/")[-1]
        smart += chunk_document(text, fname)
        naive += naive_chunk_document(text, fname)
    print(f"Docs: {len(doc_files)}")
    print(f"Smart chunker: {len(smart)} chunks, code integrity = {code_block_integrity_rate(smart)*100:.1f}%")
    print(f"Naive chunker: {len(naive)} chunks, code integrity = {code_block_integrity_rate(naive)*100:.1f}%")

    print("\n" + "=" * 70)
    print("PART 2: Retrieval + refusal accuracy on golden set")
    print("=" * 70)
    results, latencies = run_retrieval_eval()
    total_correct = sum(r["correct"] for r in results)
    print(f"\nOverall: {total_correct}/{len(results)} = {100*total_correct/len(results):.1f}%")
    print_breakdown(results, "category")

    print(f"\nRetrieval latency: avg={sum(latencies)/len(latencies):.1f}ms, "
          f"p95={sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")

    print("\n" + "=" * 70)
    print("WRONG ANSWERS (for the honest section of the README)")
    print("=" * 70)
    wrong = [r for r in results if not r["correct"]]
    if not wrong:
        print("(none)")
    for r in wrong:
        print(f"  Q: {r['question']}")
        print(f"     expected_refuse={r['expected_refuse']} got_refuse={r['got_refuse']} "
              f"score={r['top_score']:.3f} category={r['category']}")

    with open("eval/eval_report.json", "w") as f:
        json.dump({
            "chunker": {
                "smart_chunks": len(smart),
                "naive_chunks": len(naive),
                "smart_code_integrity": code_block_integrity_rate(smart),
                "naive_code_integrity": code_block_integrity_rate(naive),
            },
            "retrieval": {
                "total": len(results),
                "correct": total_correct,
                "accuracy": total_correct / len(results),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            },
            "wrong_answers": wrong,
        }, f, indent=2)
    print("\nWrote eval/eval_report.json")


if __name__ == "__main__":
    main()
