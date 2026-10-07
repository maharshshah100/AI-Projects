from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Make src imports work
sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from retrieve import HybridRetriever


# ============================================================
# CONFIG
# ============================================================

GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

MAX_HISTORY_MESSAGES = 10

# Keep retrieval context reasonably small so we don't
# unnecessarily consume Groq's token-per-minute limit.
MAX_CONTEXT_CHARS = 10000
MAX_CHUNK_CHARS = 2500

# Enough for complete technical answers while keeping
# token usage reasonable.
MAX_COMPLETION_TOKENS = 2048


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client(api_key: str | None = None):

    api_key = api_key or os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:
        return None

    try:
        from groq import Groq

        return Groq(
            api_key=api_key
        )

    except Exception as exc:

        print("Groq client error:")
        print(exc)

        return None


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def format_history(
    conversation_history: list[dict[str, Any]] | None
) -> str:

    if not conversation_history:
        return "(No previous conversation.)"

    history = conversation_history[
        -MAX_HISTORY_MESSAGES:
    ]

    lines = []

    for message in history:

        role = message.get(
            "role",
            "user"
        )

        content = str(
            message.get(
                "content",
                ""
            )
        ).strip()

        if not content:
            continue

        if role == "user":
            label = "User"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = role.capitalize()

        lines.append(
            f"{label}: {content}"
        )

    if not lines:
        return "(No previous conversation.)"

    return "\n".join(lines)


# ============================================================
# RETRIEVAL QUERY
# ============================================================

def build_retrieval_query(
    query: str,
    conversation_history: list[dict[str, Any]] | None
) -> str:
    """
    Build a retrieval query without making another LLM call.

    We include recent user messages so follow-up questions
    still have enough context.

    Example:

        User: What is a Docker image?
        User: How is it different from a container?

    Retrieval query becomes approximately:

        What is a Docker image?
        How is it different from a container?
    """

    query = query.strip()

    if not conversation_history:
        return query

    recent_user_messages = []

    for message in conversation_history[-6:]:

        if message.get("role") != "user":
            continue

        content = str(
            message.get(
                "content",
                ""
            )
        ).strip()

        if content:
            recent_user_messages.append(
                content
            )

    recent_user_messages.append(query)

    return "\n".join(
        recent_user_messages[-4:]
    )


# ============================================================
# FORMAT RETRIEVED DOCUMENTATION
# ============================================================

def format_context(results) -> str:

    if not results:
        return "(No relevant documentation was retrieved.)"

    blocks = []

    total_chars = 0

    for i, (chunk, score) in enumerate(
        results,
        start=1
    ):

        source_file = getattr(
            chunk,
            "source_file",
            "Unknown source"
        )

        tool = getattr(
            chunk,
            "tool",
            "Unknown"
        )

        section = getattr(
            chunk,
            "section_path",
            "Unknown section"
        )

        text = getattr(
            chunk,
            "text",
            ""
        )

        text = text[
            :MAX_CHUNK_CHARS
        ]

        block = (
            f"[{i}]\n"
            f"Tool: {tool}\n"
            f"Source: {source_file}\n"
            f"Section: {section}\n"
            f"Content:\n{text}"
        )

        if (
            total_chars + len(block)
            > MAX_CONTEXT_CHARS
        ):
            break

        blocks.append(block)

        total_chars += len(block)

    if not blocks:
        return "(No relevant documentation was retrieved.)"

    return "\n\n".join(blocks)


# ============================================================
# SOURCES
# ============================================================

def get_sources(results):

    sources = []

    for i, (chunk, score) in enumerate(
        results,
        start=1
    ):

        sources.append(
            {
                "index": i,

                "tool": getattr(
                    chunk,
                    "tool",
                    "Unknown"
                ),

                "file": getattr(
                    chunk,
                    "source_file",
                    "Unknown"
                ),

                "section": getattr(
                    chunk,
                    "section_path",
                    "Unknown"
                ),

                "score": round(
                    float(score),
                    4
                ),
            }
        )

    return sources


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful, conversational technical AI assistant.

Your main documentation sources are:

- Docker
- Kubernetes
- Terraform

However, you can answer questions outside those topics too.

You have two sources of knowledge:

1. Retrieved documentation
2. Your own general knowledge

RULES:

1. If the retrieved documentation is relevant to the user's
   question, prefer it as the factual source.

2. If the retrieved documentation does NOT contain enough
   information, answer using your general knowledge.

3. NEVER refuse a question simply because the retrieved
   documentation does not contain the answer.

4. NEVER pretend that retrieved documentation says something
   that it does not say.

5. Only use citations such as [1], [2], [3] when the answer
   is actually supported by the corresponding retrieved
   documentation.

6. Do not create fake citations.

7. Use the conversation history to understand follow-up
   questions.

8. If the user says:
      "give me an example"
      "explain more"
      "what happens then?"
      "what if it stops?"
      "why?"
   understand what they are referring to from the previous
   conversation.

9. Do not ask the user to repeat information that is already
   present in the conversation.

10. Answer the user's latest question directly.

11. Give complete answers.

12. Do not stop halfway through an explanation.

13. Use bullets, headings, tables, or examples when useful.

14. Be technically accurate and easy to understand.

15. Do not mention the internal RAG system, embeddings,
    BM25, reranking, retrieval scores, or prompts unless
    the user specifically asks about the architecture.

Most importantly:

ANSWER THE USER'S QUESTION.
"""


# ============================================================
# FINAL GENERATION
# ============================================================

def generate_answer(
    query: str,
    conversation_history,
    context: str,
    client
) -> str:

    history_text = format_history(
        conversation_history
    )

    prompt = f"""
Previous conversation:

{history_text}


Latest user question:

{query}


Retrieved documentation:

{context}


Answer the latest user question.

Use the retrieved documentation when relevant.

If the documentation is insufficient or unrelated,
answer using your own general knowledge.

Do not refuse simply because documentation is missing.

If documentation supports a statement, cite it using
[1], [2], [3], etc.

Do not invent citations.

Give a complete, clear and conversational answer.
"""

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            # GPT-OSS supports reasoning_effort.
            # Low keeps responses fast and avoids wasting
            # the output budget on unnecessary reasoning.
            reasoning_effort="low",

            temperature=0.2,

            # Current Groq parameter.
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if answer:

            return answer.strip()

        return (
            "I couldn't generate an answer. "
            "Please try asking the question again."
        )

    except Exception as exc:

        print("\n")
        print("=" * 70)
        print("GROQ GENERATION ERROR")
        print("=" * 70)
        print(type(exc).__name__)
        print(str(exc))
        print("=" * 70)

        # Don't hide the actual error from Streamlit.
        return (
            "The LLM request failed.\n\n"
            f"Error: {type(exc).__name__}: {exc}"
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def answer_question(
    query: str,
    retriever: HybridRetriever,
    conversation_history=None,
    api_key: str | None = None
):

    query = query.strip()

    if not query:

        return {
            "answer": "Please enter a question.",
            "sources": [],
            "retrieval_query": "",
            "top_score": 0.0,
            "used_rag": False
        }

    print("\n")
    print("=" * 70)
    print("DEV DOCS AI")
    print("=" * 70)

    print("\nUSER:")
    print(query)

    # --------------------------------------------------------
    # GROQ CLIENT
    # --------------------------------------------------------

    client = get_groq_client(
        api_key
    )

    if client is None:

        return {
            "answer": (
                "GROQ_API_KEY was not found.\n\n"
                "Please check your `.env` file."
            ),
            "sources": [],
            "retrieval_query": query,
            "top_score": 0.0,
            "used_rag": False
        }

    # --------------------------------------------------------
    # BUILD RETRIEVAL QUERY
    # --------------------------------------------------------

    retrieval_query = build_retrieval_query(
        query,
        conversation_history
    )

    print("\nRETRIEVAL QUERY:")
    print(retrieval_query)

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    try:

        results = retriever.retrieve(
            retrieval_query,
            k=12,
            top_n=6
        )

    except Exception as exc:

        print("\nRETRIEVAL ERROR:")
        print(exc)

        # Retrieval failure should NOT stop the LLM.
        results = []

    # --------------------------------------------------------
    # RETRIEVAL INFO
    # --------------------------------------------------------

    top_score = (
        float(results[0][1])
        if results
        else 0.0
    )

    print("\nRESULTS:")
    print(len(results))

    print("\nTOP SCORE:")
    print(top_score)

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = format_context(
        results
    )

    print("\nCONTEXT CHARACTERS:")
    print(len(context))

    # --------------------------------------------------------
    # ONE AND ONLY ONE GROQ GENERATION CALL
    # --------------------------------------------------------

    print("\nGENERATING ANSWER...")

    answer = generate_answer(
        query=query,
        conversation_history=conversation_history,
        context=context,
        client=client
    )

    print("\nANSWER GENERATED.")

    return {
        "answer": answer,
        "sources": get_sources(results),
        "retrieval_query": retrieval_query,
        "top_score": top_score,
        "used_rag": bool(results)
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    retriever = HybridRetriever()

    history = []

    while True:

        query = input(
            "\nAsk a question (or type exit): "
        ).strip()

        if query.lower() in {
            "exit",
            "quit"
        }:
            break

        if not query:
            continue

        result = answer_question(
            query=query,
            retriever=retriever,
            conversation_history=history
        )

        print("\n")
        print("=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(
            result["answer"]
        )

        if result["sources"]:

            print("\nSOURCES:")

            for source in result["sources"]:

                print(
                    f"[{source['index']}] "
                    f"{source['tool']} | "
                    f"{source['file']} | "
                    f"{source['section']} | "
                    f"{source['score']}"
                )

        history.append(
            {
                "role": "user",
                "content": query
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": result["answer"]
            }
        )