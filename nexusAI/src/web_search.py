"""
Phase 5: web search fallback (Tavily).

Used ONLY when a question can't be answered from the uploaded documents
(e.g. "what's a healthy debt-to-equity ratio for retail?"). Results are
always labeled as web-sourced in the UI so a user never mistakes them for
something taken from their own documents.

Speed: search_depth="basic" keeps this fast (typically 1-2 seconds).
Never raises -- a failed/missing search returns an empty result with a
reason, so chat degrades gracefully instead of crashing.
"""
from __future__ import annotations
import os

from dotenv import load_dotenv
load_dotenv()


def web_search(query: str, max_results: int = 4, api_key: str | None = None) -> dict:
    """Returns {"ok": bool, "results": [{"title","url","content"}], "error": str|None}."""
    api_key = api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"ok": False, "results": [], "error": "TAVILY_API_KEY not set"}
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, search_depth="basic", max_results=max_results, timeout=8)
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": (r.get("content") or "")[:700]}
            for r in resp.get("results", [])
        ]
        return {"ok": True, "results": results, "error": None}
    except Exception as e:  # network error, bad key, rate limit, etc.
        return {"ok": False, "results": [], "error": f"{type(e).__name__}: {str(e)[:150]}"}


def format_web_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[W{i}] {r['title']} ({r['url']})\n{r['content']}" for i, r in enumerate(results, 1)
    )


if __name__ == "__main__":
    print("No key:", web_search("typical debt to equity ratio retail industry"))
    print("Fake key:", web_search("test", api_key="fake")["error"])
