
"""
retrieval.py — Hybrid retrieval for SEC 10-K filing chunks.

Features:
  1. TF-IDF cosine similarity.
  2. BM25 keyword relevance.
  3. Reciprocal Rank Fusion (RRF).
  4. Financial-query expansion for revenue and financial statements.
  5. Removal of zero-relevance results.
  6. Small, explainable boosts for relevant financial evidence.
  7. Diagnostic metadata to help inspect retrieved chunks.

Important:
A retrieval score is NOT proof that a financial claim is correct.
The answer-generation and validation stages must verify figures against
the actual retrieved source text.
"""

import json
import re
import math
from collections import Counter
from typing import List, Dict, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# Text and metadata utilities
# ---------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Tokenize text while preserving numbers and ticker-like terms."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def normalize_ticker(value) -> str:
    """Normalize ticker metadata for exact company filtering."""
    if value is None:
        return ""
    return str(value).strip().upper()


def safe_text(value) -> str:
    """Convert missing or non-string metadata to safe text."""
    return str(value) if value is not None else ""


def normalize_text(text: str) -> str:
    """Normalize text for simple keyword comparisons."""
    return re.sub(r"\s+", " ", safe_text(text).lower()).strip()


# ---------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------

class BM25:
    """Classic Okapi BM25 ranking implementation."""

    def __init__(
        self,
        corpus_tokens: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens

        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.n_docs = len(corpus_tokens)

        total_tokens = sum(self.doc_lens)
        self.avgdl = total_tokens / self.n_docs if self.n_docs else 0.0

        self.doc_freqs = []
        df = Counter()

        for doc in corpus_tokens:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)

            for term in freqs:
                df[term] += 1

        self.idf = {
            term: math.log(
                1 + (self.n_docs - freq + 0.5) / (freq + 0.5)
            )
            for term, freq in df.items()
        }

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        if not self.n_docs or self.avgdl <= 0:
            return 0.0

        score = 0.0
        freqs = self.doc_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]

        for term in query_tokens:
            if term not in freqs:
                continue

            idf = self.idf.get(term, 0.0)
            frequency = freqs[term]

            denominator = frequency + self.k1 * (
                1 - self.b + self.b * dl / self.avgdl
            )

            if denominator:
                score += (
                    idf
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                )

        return score

    def rank(self, query: str) -> List[Tuple[int, float]]:
        """Return only documents with positive BM25 relevance."""
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = [
            (i, self.score(query_tokens, i))
            for i in range(self.n_docs)
        ]

        # Critical: zero-score documents must not be treated as matches.
        scores = [(i, score) for i, score in scores if score > 0]

        return sorted(scores, key=lambda item: item[1], reverse=True)


# ---------------------------------------------------------------------
# Query analysis and expansion
# ---------------------------------------------------------------------

FINANCIAL_QUERY_TERMS = {
    "revenue",
    "revenues",
    "sales",
    "net sales",
    "total revenue",
    "annual revenue",
    "yearly revenue",
    "revenue growth",
    "financial performance",
    "gross profit",
    "operating income",
    "net income",
    "cash flow",
    "financial statement",
    "financial statements",
}


def is_financial_query(query: str) -> bool:
    """Identify questions likely to require financial-statement evidence."""
    q = normalize_text(query)

    return any(term in q for term in FINANCIAL_QUERY_TERMS)


def is_revenue_query(query: str) -> bool:
    """Identify revenue-focused questions for targeted retrieval."""
    q = normalize_text(query)

    return any(
        term in q
        for term in (
            "revenue",
            "revenues",
            "net sales",
            "total sales",
            "annual sales",
            "sales growth",
        )
    )


def extract_years(query: str) -> List[str]:
    """Extract distinct four-digit years explicitly mentioned in a query."""
    years = re.findall(r"\b(?:19|20)\d{2}\b", safe_text(query))
    return list(dict.fromkeys(years))


def expand_financial_query(query: str) -> str:
    """
    Add related financial-statement terminology.

    Expansion improves recall; it does not establish that retrieved
    documents contain verified revenue figures.
    """
    q = normalize_text(query)

    if is_revenue_query(q):
        additions = (
            " total revenue annual revenue revenues net sales "
            "consolidated statements of operations "
            "consolidated statements of income "
            "consolidated income statement "
            "financial statements reported revenue "
            "fiscal year "
        )
        return f"{query} {additions}"

    if is_financial_query(q):
        additions = (
            " consolidated financial statements "
            "annual report financial results "
            "reported amounts fiscal year "
        )
        return f"{query} {additions}"

    return query


# ---------------------------------------------------------------------
# Evidence diagnostics
# ---------------------------------------------------------------------

MONEY_PATTERN = re.compile(
    r"""
    (?:
        \$\s*
        |
        USD\s*
    )?
    \b
    \d{1,3}(?:,\d{3})+(?:\.\d+)?
    |
    (?:
        \$\s*
        |
        USD\s*
    )
    \d+(?:\.\d+)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

REVENUE_TERMS = (
    "revenue",
    "revenues",
    "net sales",
    "total sales",
    "total revenue",
)

FINANCIAL_TABLE_TERMS = (
    "consolidated statements of operations",
    "consolidated statements of income",
    "consolidated income statement",
    "net sales",
    "total revenue",
    "total revenues",
    "revenue by segment",
    "results of operations",
    "financial statements",
)


def inspect_evidence(
    chunk: Dict,
    query: str,
) -> Dict:
    """
    Add transparent evidence indicators.

    These flags help diagnose relevance. They are not a financial audit
    and do not prove that a number is the correct reported revenue.
    """
    text = safe_text(chunk.get("text"))
    section = safe_text(chunk.get("section"))
    combined = normalize_text(f"{section} {text}")
    query_normalized = normalize_text(query)

    contains_revenue_term = any(
        term in combined for term in REVENUE_TERMS
    )

    contains_financial_statement_signal = any(
        term in combined for term in FINANCIAL_TABLE_TERMS
    )

    contains_money_amount = bool(MONEY_PATTERN.search(text))

    requested_years = extract_years(query)
    years_present = [
        year for year in requested_years
        if re.search(rf"\b{re.escape(year)}\b", text)
    ]

    # Small, explainable ranking boost. It is deliberately not large
    # enough to replace the actual retrieval signals.
    evidence_boost = 0.0

    if is_revenue_query(query):
        if contains_revenue_term:
            evidence_boost += 0.002

        if contains_financial_statement_signal:
            evidence_boost += 0.002

        if contains_money_amount:
            evidence_boost += 0.001

        if years_present:
            evidence_boost += min(0.001 * len(years_present), 0.002)

    return {
        "contains_revenue_term": contains_revenue_term,
        "contains_financial_statement_signal": (
            contains_financial_statement_signal
        ),
        "contains_money_amount": contains_money_amount,
        "requested_years_found": years_present,
        "evidence_boost": round(evidence_boost, 5),
    }


# ---------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------

class HybridRetriever:
    def __init__(self, chunks_path: str):
        with open(chunks_path, "r", encoding="utf-8") as file:
            loaded_chunks = json.load(file)

        if not isinstance(loaded_chunks, list):
            raise ValueError(
                "The chunks JSON must contain a list of chunk objects."
            )

        self.chunks = [
            chunk for chunk in loaded_chunks
            if isinstance(chunk, dict)
            and safe_text(chunk.get("text")).strip()
        ]

        if not self.chunks:
            raise ValueError(
                f"No usable text chunks found in: {chunks_path}"
            )

        # Preserve all original metadata while ensuring safe text access.
        self.texts = [
            safe_text(chunk.get("text"))
            for chunk in self.chunks
        ]

        # TF-IDF retrieval.
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

        # BM25 retrieval.
        tokenized = [tokenize(text) for text in self.texts]
        self.bm25 = BM25(tokenized)

    def _tfidf_rank(
        self,
        query: str,
    ) -> List[Tuple[int, float]]:
        """Return only documents with positive cosine similarity."""
        q_vec = self.vectorizer.transform([query])

        if q_vec.nnz == 0:
            return []

        similarities = cosine_similarity(
            q_vec,
            self.tfidf_matrix,
        )[0]

        # Critical: don't assign RRF ranks to zero-similarity documents.
        ranked = [
            (idx, float(score))
            for idx, score in enumerate(similarities)
            if score > 0
        ]

        return sorted(
            ranked,
            key=lambda item: item[1],
            reverse=True,
        )

    def _eligible_indices(
        self,
        company_filter: Optional[str],
    ) -> List[int]:
        """Return corpus indices allowed by the company filter."""
        if not company_filter:
            return list(range(len(self.chunks)))

        wanted_ticker = normalize_ticker(company_filter)

        return [
            idx
            for idx, chunk in enumerate(self.chunks)
            if normalize_ticker(chunk.get("ticker")) == wanted_ticker
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        company_filter: str = None,
    ) -> List[Dict]:
        """
        Search using TF-IDF + BM25 + reciprocal rank fusion.

        Existing call compatibility:
            retriever.search(
                query,
                top_k=5,
                company_filter="AAPL"
            )

        Returns fewer than top_k results when fewer genuine matches exist.
        """
        if not safe_text(query).strip():
            return []

        if top_k <= 0:
            return []

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        eligible = self._eligible_indices(company_filter)

        if not eligible:
            return []

        eligible_set = set(eligible)

        # For financial questions, search both the original query and
        # a terminology-expanded version to improve recall.
        queries = [query]
        expanded_query = expand_financial_query(query)

        if expanded_query != query:
            queries.append(expanded_query)

        ranking_sets = []

        for search_query in queries:
            tfidf_ranked = [
                (idx, score)
                for idx, score in self._tfidf_rank(search_query)
                if idx in eligible_set
            ]

            bm25_ranked = [
                (idx, score)
                for idx, score in self.bm25.rank(search_query)
                if idx in eligible_set
            ]

            ranking_sets.append(("tfidf", tfidf_ranked))
            ranking_sets.append(("bm25", bm25_ranked))

        fused_scores = {idx: 0.0 for idx in eligible}
        tfidf_scores = {idx: 0.0 for idx in eligible}
        bm25_scores = {idx: 0.0 for idx in eligible}
        matched_signals = {idx: [] for idx in eligible}

        # Fuse each positive-relevance ranking. Rank positions are
        # calculated only among genuine positive-score matches.
        for signal_name, ranking in ranking_sets:
            for rank_position, (idx, raw_score) in enumerate(ranking):
                fused_scores[idx] += (
                    1.0 / (rrf_k + rank_position + 1)
                )

                if signal_name == "tfidf":
                    tfidf_scores[idx] = max(
                        tfidf_scores[idx],
                        float(raw_score),
                    )
                else:
                    bm25_scores[idx] = max(
                        bm25_scores[idx],
                        float(raw_score),
                    )

                if signal_name not in matched_signals[idx]:
                    matched_signals[idx].append(signal_name)

        # Don't return chunks that matched none of the ranking signals.
        matched_indices = [
            idx for idx in eligible
            if fused_scores[idx] > 0
        ]

        if not matched_indices:
            return []

        results = []

        for idx in matched_indices:
            chunk = dict(self.chunks[idx])

            diagnostics = inspect_evidence(chunk, query)

            raw_rrf_score = fused_scores[idx]
            final_score = (
                raw_rrf_score + diagnostics["evidence_boost"]
            )

            chunk["tfidf_score"] = round(
                float(tfidf_scores[idx]), 4
            )
            chunk["bm25_score"] = round(
                float(bm25_scores[idx]), 4
            )
            chunk["rrf_score"] = round(raw_rrf_score, 6)
            chunk["fused_score"] = round(final_score, 6)

            chunk["matched_signals"] = matched_signals[idx]
            chunk["company_filter_applied"] = (
                normalize_ticker(company_filter)
                if company_filter else None
            )

            chunk.update(diagnostics)

            # This is intentionally not called "verified".
            chunk["evidence_status"] = (
                "potential_financial_evidence"
                if (
                    diagnostics["contains_revenue_term"]
                    or diagnostics["contains_financial_statement_signal"]
                )
                else "general_retrieval_match"
            )

            results.append(chunk)

        results.sort(
            key=lambda item: item["fused_score"],
            reverse=True,
        )

        return results[:top_k]


# ---------------------------------------------------------------------
# Manual retrieval diagnostics
# ---------------------------------------------------------------------

if __name__ == "__main__":
    retriever = HybridRetriever(
        "data/processed/chunks.json"
    )

    test_queries = [
        (
            "What was Apple's total revenue in fiscal year 2024?",
            "AAPL",
        ),
        (
            "What was Microsoft's total revenue in fiscal year 2024?",
            "MSFT",
        ),
        (
            "What was Nvidia's annual revenue in 2023 and 2024?",
            "NVDA",
        ),
        (
            "Microsoft OpenAI partnership Azure",
            "MSFT",
        ),
        (
            "Apple Greater China net sales decline",
            "AAPL",
        ),
    ]

    for query, ticker in test_queries:
        print("\n" + "=" * 90)
        print(f"QUERY: {query}")
        print(f"COMPANY FILTER: {ticker}")
        print("=" * 90)

        results = retriever.search(
            query,
            top_k=5,
            company_filter=ticker,
        )

        if not results:
            print("No positive-relevance chunks found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(
                f"\n#{rank} "
                f"[{result.get('ticker', 'UNKNOWN')} "
                f"{result.get('fiscal_year', 'UNKNOWN')} | "
                f"{result.get('section', 'UNKNOWN')}]"
            )

            print(
                f"  TF-IDF: {result['tfidf_score']:.4f} | "
                f"BM25: {result['bm25_score']:.4f} | "
                f"RRF: {result['rrf_score']:.6f} | "
                f"Final: {result['fused_score']:.6f}"
            )

            print(
                f"  Revenue term: "
                f"{result['contains_revenue_term']} | "
                f"Financial-statement signal: "
                f"{result['contains_financial_statement_signal']} | "
                f"Money amount: "
                f"{result['contains_money_amount']}"
            )

            print(
                f"  Requested years found: "
                f"{result['requested_years_found']}"
            )

            print(
                f"  Evidence status: "
                f"{result['evidence_status']}"
            )

            print(
                f"  Text: "
                f"{result['text'][:500].replace(chr(10), ' ')}"
            )