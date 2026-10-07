
from __future__ import annotations

import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_DIR = "data/index"

DEFAULT_K = 16
DEFAULT_TOP_N = 10


# ============================================================
# TOKENIZATION
# ============================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def tokenize(text: str) -> List[str]:
    """Convert text into normalized tokens."""

    tokens = _TOKEN_RE.findall(text.lower())

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


# ============================================================
# CHUNK
# ============================================================

@dataclass
class Chunk:
    """Represents one documentation chunk."""

    tool: str
    source_file: str
    section_path: str
    text: str


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:
    """
    Hybrid documentation retriever.

    Retrieval pipeline:

        Query
          ↓
        Dense retrieval
          +
        BM25 retrieval
          ↓
        Reciprocal Rank Fusion
          ↓
        Lexical reranking
          ↓
        Final relevant chunks
    """

    def __init__(self, index_dir: str = INDEX_DIR):

        self.index_dir = index_dir

        dense_path = os.path.join(
            index_dir,
            "dense.pkl",
        )

        bm25_path = os.path.join(
            index_dir,
            "bm25.pkl",
        )

        embedder_path = os.path.join(
            index_dir,
            "embedder.pkl",
        )

        # ----------------------------------------------------
        # Validate index files
        # ----------------------------------------------------

        if not os.path.exists(dense_path):
            raise FileNotFoundError(
                f"Dense index not found: {dense_path}"
            )

        if not os.path.exists(bm25_path):
            raise FileNotFoundError(
                f"BM25 index not found: {bm25_path}"
            )

        # ----------------------------------------------------
        # Load dense index
        # ----------------------------------------------------

        with open(
            dense_path,
            "rb",
        ) as f:
            dense_data = pickle.load(f)

        self.chunks = dense_data["chunks"]

        # Support both old and new index formats
        if "vectors" in dense_data:
            vectors = dense_data["vectors"]
        elif "embeddings" in dense_data:
            vectors = dense_data["embeddings"]
        else:
            raise KeyError(
                "Dense index does not contain 'vectors' "
                "or 'embeddings'. Rebuild the index."
            )

        self.dense_vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Normalize vectors
        # ----------------------------------------------------

        norms = np.linalg.norm(
            self.dense_vectors,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        self.dense_vectors = (
            self.dense_vectors / norms
        )

        # ----------------------------------------------------
        # Load BM25
        # ----------------------------------------------------

        with open(
            bm25_path,
            "rb",
        ) as f:
            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["bm25"]

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        self.embedder = None

        # First try the saved local embedder.
        if os.path.exists(embedder_path):

            try:

                with open(
                    embedder_path,
                    "rb",
                ) as f:
                    self.embedder = pickle.load(f)

                print(
                    "[retriever] Loaded saved embedding model."
                )

            except Exception as exc:

                print(
                    "[retriever] Could not load saved "
                    f"embedder: {exc}"
                )

                self.embedder = None

        # ----------------------------------------------------
        # If no saved embedder exists, use get_embedder()
        # ----------------------------------------------------

        if self.embedder is None:

            try:

                from embedder import get_embedder

                print(
                    "[retriever] Creating embedding model..."
                )

                self.embedder = get_embedder(
                    prefer_transformer=True
                )

                print(
                    "[retriever] Embedding model created:"
                    f" {self.embedder.name}"
                )

            except Exception as exc:

                raise RuntimeError(
                    "Could not load or create the embedding model."
                ) from exc

        # ----------------------------------------------------
        # Validate embedding dimensionality
        # ----------------------------------------------------

        try:

            test_vector = self.embedder.embed(
                ["test"]
            )

            test_vector = np.asarray(
                test_vector,
                dtype=np.float32,
            )

            if test_vector.ndim == 2:
                test_dim = test_vector.shape[1]
            else:
                test_dim = test_vector.shape[0]

            index_dim = self.dense_vectors.shape[1]

            if test_dim != index_dim:

                raise RuntimeError(
                    "Embedding dimension mismatch.\n"
                    f"Index dimension: {index_dim}\n"
                    f"Model dimension: {test_dim}\n\n"
                    "Please rebuild the index using the "
                    "same embedding model."
                )

        except RuntimeError:
            raise

        except Exception as exc:

            raise RuntimeError(
                "The embedding model could not generate "
                "a test embedding."
            ) from exc

    # ========================================================
    # DENSE SEARCH
    # ========================================================

    def dense_search(
        self,
        query: str,
        k: int = DEFAULT_K,
    ) -> List[Tuple[int, float]]:

        query_vector = self.embedder.embed(
            [query]
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.ndim == 2:
            query_vector = query_vector[0]

        norm = np.linalg.norm(
            query_vector
        )

        if norm != 0:
            query_vector = (
                query_vector / norm
            )

        scores = self.dense_vectors @ query_vector

        k = min(
            k,
            len(scores),
        )

        indices = np.argsort(
            scores
        )[::-1][:k]

        return [
            (
                int(index),
                float(scores[index]),
            )
            for index in indices
        ]

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def bm25_search(
        self,
        query: str,
        k: int = DEFAULT_K,
    ) -> List[Tuple[int, float]]:

        tokens = tokenize(query)

        if not tokens:
            return []

        scores = self.bm25.get_scores(
            tokens
        )

        k = min(
            k,
            len(scores),
        )

        indices = np.argsort(
            scores
        )[::-1][:k]

        return [
            (
                int(index),
                float(scores[index]),
            )
            for index in indices
        ]

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        rrf_k: int = 60,
    ) -> List[Tuple[Chunk, float]]:

        dense_results = self.dense_search(
            query,
            k=k,
        )

        bm25_results = self.bm25_search(
            query,
            k=k,
        )

        rrf_scores = defaultdict(float)

        # ----------------------------------------------------
        # Dense rankings
        # ----------------------------------------------------

        for rank, (idx, _score) in enumerate(
            dense_results
        ):

            rrf_scores[idx] += (
                1.0
                / (
                    rrf_k
                    + rank
                    + 1
                )
            )

        # ----------------------------------------------------
        # BM25 rankings
        # ----------------------------------------------------

        for rank, (idx, _score) in enumerate(
            bm25_results
        ):

            rrf_scores[idx] += (
                1.0
                / (
                    rrf_k
                    + rank
                    + 1
                )
            )

        # ----------------------------------------------------
        # Sort combined results
        # ----------------------------------------------------

        ranked = sorted(
            rrf_scores.items(),
            key=lambda x: -x[1],
        )[:k]

        return [
            (
                self.chunks[idx],
                score,
            )
            for idx, score in ranked
        ]

    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        query: str,
        candidates: List[
            Tuple[Chunk, float]
        ],
        top_n: int = DEFAULT_TOP_N,
    ) -> List[Tuple[Chunk, float]]:

        q_terms = set(
            tokenize(query)
        )

        rescored = []

        for chunk, base_score in candidates:

            c_terms = tokenize(
                chunk.text
            )

            c_terms_set = set(
                c_terms
            )

            # ------------------------------------------------
            # Query-term overlap
            # ------------------------------------------------

            overlap = (
                len(
                    q_terms
                    & c_terms_set
                )
                / max(
                    len(q_terms),
                    1,
                )
            )

            # ------------------------------------------------
            # Exact phrase bonus
            # ------------------------------------------------

            phrase_bonus = 0.0

            if query.lower() in chunk.text.lower():
                phrase_bonus = 0.3

            # ------------------------------------------------
            # Section bonus
            # ------------------------------------------------

            section_bonus = 0.0

            section_lower = (
                chunk.section_path.lower()
            )

            if any(
                token in section_lower
                for token in q_terms
            ):
                section_bonus = 0.1

            # ------------------------------------------------
            # Final score
            # ------------------------------------------------

            rerank_score = (
                base_score
                + overlap
                + phrase_bonus
                + section_bonus
            )

            rescored.append(
                (
                    chunk,
                    rerank_score,
                )
            )

        rescored.sort(
            key=lambda x: -x[1]
        )

        return rescored[:top_n]

    # ========================================================
    # PUBLIC RETRIEVE METHOD
    # ========================================================

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        top_n: int = DEFAULT_TOP_N,
    ) -> List[Tuple[Chunk, float]]:

        candidates = self.hybrid_search(
            query,
            k=k,
        )

        return self.rerank(
            query,
            candidates,
            top_n=top_n,
        )


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    retriever = HybridRetriever()

    test_queries = [
        "how do I set a memory limit on a container",
        "what does CrashLoopBackOff mean",
        "how do I configure a liveness probe",
        "what happens when a Kubernetes node fails",
        "what happens when a Docker container stops",
        "what is Terraform state",
    ]

    for query in test_queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retriever.retrieve(
            query,
            k=16,
            top_n=10,
        )

        for i, (chunk, score) in enumerate(
            results,
            start=1,
        ):

            print("\n" + "-" * 60)
            print(f"RESULT #{i}")
            print("-" * 60)

            print(
                f"Score: {score:.4f}"
            )

            print(
                f"Tool: {getattr(chunk, 'tool', 'Unknown')}"
            )

            print(
                f"Source: {getattr(chunk, 'source_file', 'Unknown')}"
            )

            print(
                f"Section: {getattr(chunk, 'section_path', 'Unknown')}"
            )

            print(
                getattr(
                    chunk,
                    "text",
                    "",
                )[:500]
            )

