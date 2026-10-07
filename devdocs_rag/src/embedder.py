"""
Embedding backend, made swappable on purpose.

WHY THIS FILE EXISTS:
Real semantic embeddings (bge-small-en-v1.5, OpenAI text-embedding-3-small,
etc.) come from Hugging Face or a paid API. In a locked-down sandbox with
no outbound access to huggingface.co, you can't pull those weights. Rather
than fake the results, this module gives you TWO real, working backends
behind the same interface:

  - TransformerEmbedder  -> the "real" one (fastembed / bge-small).
                             Use this when you run the project on your own
                             machine with normal internet access.
  - LocalTfidfEmbedder    -> a genuinely computed, dependency-light fallback
                             (TF-IDF -> truncated SVD, i.e. classic LSA).
                             No downloads required. Weaker than a
                             transformer model at capturing paraphrases/
                             synonyms, but it's a real dense embedding
                             space, not a mock, and hybrid search (this
                             + BM25) still works correctly on top of it.

Swap which one `get_embedder()` returns and nothing else in the pipeline
needs to change -- that's the point of having an interface here at all.
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LocalTfidfEmbedder:
    """A real, locally-computed dense embedding space. No network calls."""

    name = "local-tfidf-svd-256d"

    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, texts: list[str]):
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call .fit() on the corpus before .embed()")
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        # normalize so cosine similarity == dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return [v for v in (vecs / norms)]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "LocalTfidfEmbedder":
        with open(path, "rb") as f:
            return pickle.load(f)


class TransformerEmbedder:
    """
    Real transformer embeddings via fastembed. Requires access to
    huggingface.co to download model weights on first run -- works fine
    on a normal machine, just not inside this network-restricted sandbox.
    """

    name = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=self.name)

    def fit(self, texts: list[str]):
        return self  # nothing to fit, it's a pretrained model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return list(self.model.embed(texts))


def get_embedder(prefer_transformer: bool = True):
    """
    Tries the real transformer embedder first; falls back to the local
    TF-IDF+SVD embedder if model weights can't be downloaded (e.g. in a
    network-restricted environment like this sandbox).

    Set env var DEVDOCS_FORCE_LOCAL_EMBEDDER=1 to skip the transformer
    attempt entirely (useful for fast iteration when you already know
    Hugging Face isn't reachable).
    """
    import os
    if os.environ.get("DEVDOCS_FORCE_LOCAL_EMBEDDER") == "1":
        prefer_transformer = False

    if prefer_transformer:
        try:
            emb = TransformerEmbedder()
            print(f"[embedder] Using {emb.name} (real transformer embeddings)")
            return emb
        except Exception as e:
            print(f"[embedder] Transformer embedder unavailable ({e}); "
                  f"falling back to local TF-IDF+SVD embedder.")
    emb = LocalTfidfEmbedder()
    print(f"[embedder] Using {emb.name} (local fallback, no download needed)")
    return emb
