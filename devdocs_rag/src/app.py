"""
FastAPI backend for DevDocs RAG.

Run with:  uvicorn src.app:app --reload --port 8000
Then:      curl -X POST localhost:8000/query -H "Content-Type: application/json" \
                -d '{"question": "how do I set a memory limit on a container?"}'
"""
from __future__ import annotations
import sys
import time

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, "src")
from retrieve import HybridRetriever  # noqa: E402
from generate import answer_question  # noqa: E402

app = FastAPI(title="DevDocs RAG", version="0.1.0")

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


class QueryRequest(BaseModel):
    question: str


class SourceOut(BaseModel):
    n: int
    tool: str
    file: str
    section: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    refused: bool
    top_score: float
    sources: list[SourceOut]
    latency_ms: int


@app.on_event("startup")
def _startup():
    get_retriever()  # load indexes once, not per-request


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.time()
    result = answer_question(req.question, get_retriever())
    latency_ms = int((time.time() - start) * 1000)
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        refused=result["refused"],
        top_score=result["top_score"],
        sources=[SourceOut(**s) for s in result["sources"]],
        latency_ms=latency_ms,
    )
