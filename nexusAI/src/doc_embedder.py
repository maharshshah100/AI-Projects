"""
Embedding backend for the document retrieval layer (Phase 4), same
pluggable-interface pattern used in DevDocs and proven there: a real API
backend for production, and a locally-computed fallback that needs no
network access, so this module is fully testable even without a Voyage
AI key.

VoyageEmbedder  -> the real one. Used automatically the moment a
                    VOYAGE_API_KEY is available.
LocalTestEmbedder -> TF-IDF + truncated SVD, computed entirely locally.
                    Used here ONLY to verify the Qdrant integration and
                    retrieval logic work correctly end-to-end without
                    requiring a live key during development.
"""
from __future__ import annotations
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LocalTestEmbedder:
    name = "local-tfidf-svd-128d"
    dimension = 128

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=self.dimension, random_state=42)
        self._fitted = False

    def fit(self, texts: list[str]):
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(self.dimension, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if n_components < 1:
            # Tiny corpus (e.g. a one-paragraph notes file): SVD is impossible
            # and pointless -- use the normalized TF-IDF vectors directly.
            self._use_svd = False
            self.dimension = tfidf.shape[1]
        else:
            self._use_svd = True
            if n_components < self.dimension:
                self.svd = TruncatedSVD(n_components=n_components, random_state=42)
                self.dimension = n_components
            self.svd.fit(tfidf)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("Call .fit() first")
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf) if getattr(self, "_use_svd", True) else tfidf.toarray()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (vecs / norms).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class VoyageEmbedder:
    name = "voyage-3"
    dimension = 1024  # voyage-3's native output dimension

    def __init__(self, api_key: str | None = None):
        import voyageai
        self.client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))

    def fit(self, texts: list[str]):
        return self  # pretrained, nothing to fit

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embed(texts, model=self.name, input_type="document")
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.client.embed([text], model=self.name, input_type="query")
        return result.embeddings[0]

    def rerank(self, query: str, documents: list[str], top_k: int = 4):
        result = self.client.rerank(query, documents, model="rerank-2", top_k=top_k)
        return [(r.index, r.relevance_score) for r in result.results]


def get_embedder(prefer_voyage: bool = True):
    if prefer_voyage and os.environ.get("VOYAGE_API_KEY"):
        try:
            emb = VoyageEmbedder()
            print(f"[embedder] Using {emb.name} (real Voyage AI embeddings)")
            return emb
        except Exception as e:
            print(f"[embedder] Voyage AI unavailable ({e}); falling back to local test embedder.")
    emb = LocalTestEmbedder()
    print(f"[embedder] Using {emb.name} (local fallback, no API key needed)")
    return emb
