"""
Document retrieval: indexes narrative text chunks (MD&A notes, commentary)
into Qdrant and retrieves the most relevant ones for a given question.

This is used ONLY for open-ended/narrative questions. Exact financial
figures never come through this path -- they're answered directly from
the structured data. This module is specifically for "why did X happen" /
"what does management say about Y" style questions, where the answer
lives in prose, not a number.

Uses Qdrant's own native capabilities rather than hand-rolled fusion/
reranking logic -- the DevDocs lesson applied directly: less custom code
in the retrieval path means fewer places for a bug to hide.
"""
from __future__ import annotations
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from doc_chunker import DocChunk, chunk_narrative_text
from doc_embedder import get_embedder

COLLECTION_NAME = "finsight_documents"


class DocumentRetriever:
    def __init__(self, embedder=None, qdrant_path: str | None = None):
        """
        qdrant_path=None uses Qdrant's in-memory mode (fine for a single
        session / testing). Pass a real path (e.g. './qdrant_data') for
        persistence across app restarts in the actual deployed product.
        """
        self.embedder = embedder or get_embedder()
        self.client = QdrantClient(location=qdrant_path or ":memory:")
        self._collection_ready = False
        self._chunks_by_id: dict[str, DocChunk] = {}

    def index_document(self, text: str, source_name: str) -> int:
        chunks = chunk_narrative_text(text, source_name)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        # The local fallback embedder must be fitted on the corpus before
        # first use; the Voyage embedder is pretrained and has no _fitted flag.
        if hasattr(self.embedder, "fit") and getattr(self.embedder, "_fitted", True) is False:
            self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)

        if not self._collection_ready:
            dim = len(vectors[0])
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            self._collection_ready = True

        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid4())
            self._chunks_by_id[point_id] = chunk
            points.append(PointStruct(
                id=point_id, vector=vector,
                payload={"text": chunk.text, "section": chunk.section,
                         "source": source_name, "chunk_id": chunk.chunk_id},
            ))
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self._collection_ready:
            return []
        query_vector = (
            self.embedder.embed_query(query) if hasattr(self.embedder, "embed_query")
            else self.embedder.embed([query])[0]
        )
        results = self.client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, limit=top_k,
        )
        return [
            {
                "text": p.payload["text"],
                "section": p.payload["section"],
                "source": p.payload["source"],
                "score": round(p.score, 4),
            }
            for p in results.points
        ]

    def search_with_confidence_gate(self, query: str, top_k: int = 3, threshold: float = 0.75) -> dict:
        """
        Wraps search() with a refusal decision, same pattern proven in
        DevDocs: if the top result's score is below `threshold`, report
        low confidence rather than returning results that look
        plausible but aren't actually relevant.

        IMPORTANT -- READ BEFORE TUNING THIS THRESHOLD:
        The default here was NOT calibrated against real Voyage AI
        embeddings (no API key was available while building this). It
        was observed, with the LOCAL fallback embedder, that lexical
        collisions occur -- e.g. "capital of France" scoring highly
        against "Capital Structure and Debt" purely on shared word
        overlap, despite unrelated meaning. This is a known weakness of
        TF-IDF-style embeddings specifically, not of the retrieval
        architecture. Voyage's real embeddings are trained to separate
        meaning, not just tokens, so this collision is expected to
        largely disappear -- but the threshold MUST be re-measured
        against real Voyage scores (the same way DevDocs' threshold was
        calibrated against a real golden set) before trusting this
        number in production. Do not ship this default without doing so.
        """
        results = self.search(query, top_k=top_k)
        confident = bool(results) and results[0]["score"] >= threshold
        return {"confident": confident, "results": results if confident else []}


if __name__ == "__main__":
    embedder = get_embedder(prefer_voyage=False)  # explicit local mode for this sandbox test

    text = open("data/sample/Nimbus_MDA_Notes.txt").read()
    chunks = chunk_narrative_text(text, "Nimbus_MDA_Notes")
    embedder.fit([c.text for c in chunks])  # LocalTestEmbedder needs fitting before first use

    retriever = DocumentRetriever(embedder=embedder)
    n = retriever.index_document(text, "Nimbus_MDA_Notes")
    print(f"Indexed {n} chunks into Qdrant\n")

    test_queries = [
        "Why did gross margin decline in 2025?",
        "What is management's plan for the dividend policy?",
        "How many independent directors are on the board?",
        "What are the main risks to next year's outlook?",
        "What's the capital of France?",  # should retrieve nothing relevant
    ]
    for q in test_queries:
        print(f"=== Q: {q} ===")
        results = retriever.search(q, top_k=2)
        for r in results:
            print(f"  score={r['score']:.3f} [{r['section']}]")
            print(f"    {r['text'][:150]}...")
        print()
