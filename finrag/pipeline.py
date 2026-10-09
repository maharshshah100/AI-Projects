"""
pipeline.py — FinRAG end-to-end pipeline.

Flow:

question
    ↓
query decomposition
    ↓
local SEC retrieval
    ↓
reranking
    ↓
confidence gate
    ↓
if insufficient → Tavily web search
    ↓
Groq grounded generation
    ↓
answer + citations
"""

import argparse

from decompose import decompose_rule_based
from retrieval import HybridRetriever
from rerank import heuristic_rerank
from gate import check_confidence, format_citation
from generate import mock_generate, generate_llm

from web_search import search_web, format_web_source


def answer_question(
    question: str,
    retriever: HybridRetriever,
    mode: str = "mock",
    api_key: str = None,
    use_web_fallback: bool = True,
):

    sub_queries = decompose_rule_based(question)

    print(
        f"\nDecomposed into {len(sub_queries)} "
        f"sub-quer{'y' if len(sub_queries) == 1 else 'ies'}:"
    )

    for sq in sub_queries:

        print(
            f"  - [{sq.company_filter or 'unscoped'}] "
            f"{sq.text}"
        )

    sub_answers = []

    for sq in sub_queries:

        # -------------------------------------------------------
        # STEP 1 — Local SEC RAG
        # -------------------------------------------------------

        first_pass = retriever.search(
            sq.text,
            top_k=5,
            company_filter=sq.company_filter,
        )

        reranked = heuristic_rerank(
            sq.text,
            first_pass,
            top_k=3,
        )

        gate_result = check_confidence(reranked)

        print(
            f"\n  Local SEC gate "
            f"[{sq.company_filter or 'unscoped'}]: "
            f"confident={gate_result.confident} "
            f"(score={gate_result.top_score:.4f})"
        )

        # -------------------------------------------------------
        # STEP 2 — LOCAL EVIDENCE IS GOOD
        # -------------------------------------------------------

        if gate_result.confident:

            context_chunks = reranked

            evidence_type = "SEC filing"

        # -------------------------------------------------------
        # STEP 3 — LOCAL EVIDENCE IS NOT GOOD
        #          → Tavily Web Search
        # -------------------------------------------------------

        elif use_web_fallback:

            print("  Local evidence insufficient → searching web...")

            web_results = search_web(
                sq.text,
                max_results=5,
            )

            if not web_results:

                sub_answers.append(
                    {
                        "sub_query": sq.text,
                        "answered": False,
                        "reason": (
                            "Local SEC retrieval was not confident "
                            "and Tavily returned no usable results."
                        ),
                        "evidence_type": "none",
                        "sources": [],
                    }
                )

                continue

            context_chunks = []

            for result in web_results:

                context_chunks.append(
                    {
                        "text": result.get("text", ""),
                        "company": result.get(
                            "title",
                            "Web source",
                        ),
                        "filing_type": "Web",
                        "fiscal_year": "N/A",
                        "filed_date": "N/A",
                        "section": result.get(
                            "title",
                            "Web result",
                        ),
                        "source_url": result.get(
                            "url",
                            "",
                        ),
                    }
                )

            evidence_type = "Web search"

        else:

            sub_answers.append(
                {
                    "sub_query": sq.text,
                    "answered": False,
                    "reason": gate_result.reason,
                    "evidence_type": "none",
                    "sources": [],
                }
            )

            continue

        # -------------------------------------------------------
        # STEP 4 — GENERATION
        # -------------------------------------------------------

        if mode == "live":

            answer_text = generate_llm(
                sq.text,
                context_chunks,
                api_key=api_key,
            )

        else:

            answer_text = mock_generate(
                sq.text,
                context_chunks,
            )

        # -------------------------------------------------------
        # STEP 5 — SOURCE INFORMATION
        # -------------------------------------------------------

        sources = []

        for chunk in context_chunks:

            if chunk.get("filing_type") == "Web":

                sources.append(
                    format_web_source(
                        {
                            "title": chunk.get(
                                "section",
                                "Web source",
                            ),
                            "url": chunk.get(
                                "source_url",
                                "",
                            ),
                        }
                    )
                )

            else:

                sources.append(
                    format_citation(chunk)
                )

        sub_answers.append(
            {
                "sub_query": sq.text,
                "answered": True,
                "answer": answer_text,
                "evidence_type": evidence_type,
                "sources": sources,
            }
        )

    return sub_answers


def print_final_answer(
    question: str,
    sub_answers: list,
):

    print(
        f"\n{'=' * 70}"
        f"\nFINAL ANSWER for: {question}"
        f"\n{'=' * 70}"
    )

    for sa in sub_answers:

        print(
            f"\n[{sa['sub_query']}]"
        )

        if sa["answered"]:

            print(sa["answer"])

            print(
                f"\nEvidence type: "
                f"{sa['evidence_type']}"
            )

            for source in sa["sources"]:

                print(
                    f"  - {source}"
                )

        else:

            print(
                f"DECLINED TO ANSWER: "
                f"{sa['reason']}"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--no-web",
        action="store_true",
    )

    args = parser.parse_args()

    retriever = HybridRetriever(
        "data/processed/chunks.json"
    )

    demo_questions = [

        "What was Apple's total net sales in fiscal year 2024?",

        "How did Apple's revenue compare to Microsoft's revenue in fiscal year 2024?",

        "What is Microsoft's relationship with OpenAI?",

        "What was Apple's revenue from selling rocket ships in 2024?",
    ]

    questions = (
        [args.question]
        if args.question
        else demo_questions
    )

    for q in questions:

        sub_answers = answer_question(
            q,
            retriever,
            mode=args.mode,
            use_web_fallback=not args.no_web,
        )

        print_final_answer(
            q,
            sub_answers,
        )