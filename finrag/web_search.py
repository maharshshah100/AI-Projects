"""
web_search.py

Tavily web-search fallback for FinRAG.

Purpose:
- Search the web when local SEC retrieval is not confident enough.
- Return clean evidence that can be passed to the Groq model.
- Keep API-key handling outside the Streamlit UI.
"""

import os

from dotenv import load_dotenv
from tavily import TavilyClient


# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def get_tavily_api_key():
    """
    Load Tavily API key from:
    1. Local .env
    2. Streamlit Secrets

    Returns:
        str | None
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if api_key:
        return api_key

    # Streamlit Cloud support
    try:
        import streamlit as st

        api_key = st.secrets["TAVILY_API_KEY"]

        if api_key:
            return api_key

    except (KeyError, FileNotFoundError):
        pass

    return None


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def search_web(
    query: str,
    max_results: int = 5,
):
    """
    Search the web using Tavily.

    Args:
        query: User's financial research question.
        max_results: Maximum number of results to return.

    Returns:
        List of dictionaries containing:
        title, url, text, score
    """

    api_key = get_tavily_api_key()

    if not api_key:
        return []

    try:

        client = TavilyClient(
            api_key=api_key
        )

        response = client.search(
            query=query,

            search_depth="advanced",

            max_results=max_results,

            include_answer=False,

            include_raw_content=False,
        )

        results = []

        for item in response.get(
            "results",
            [],
        ):

            results.append(
                {
                    "title": item.get(
                        "title",
                        "Web source",
                    ),

                    "url": item.get(
                        "url",
                        "",
                    ),

                    "text": item.get(
                        "content",
                        "",
                    ),

                    "score": item.get(
                        "score",
                        0.0,
                    ),
                }
            )

        return results

    except Exception as e:

        print(
            f"Tavily search failed: {e}"
        )

        return []


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_web_source(
    source: dict,
):
    """
    Format a web source for display in Streamlit.
    """

    title = source.get(
        "title",
        "Web source",
    )

    url = source.get(
        "url",
        "",
    )

    if url:

        return (
            f"**{title}** — {url}"
        )

    return f"**{title}**"


# ---------------------------------------------------------------------------
# Simple standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    results = search_web(
        "Apple revenue 2024 10-K",
        max_results=3,
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n[{i}] "
            f"{result['title']}"
        )

        print(
            result["url"]
        )

        print(
            result["text"][:500]
        )