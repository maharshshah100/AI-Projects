"""
Streamlit frontend for DevDocs Conversational RAG.

Run:

    python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# Allow imports from src/
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "src",
    ),
)

import streamlit as st

from retrieve import HybridRetriever
from generate import answer_question


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DevDocs AI",
    page_icon="📚",
    layout="centered",
)


# ============================================================
# HEADER
# ============================================================

st.title("📚 DevDocs AI")

st.caption(
    "A conversational AI assistant for Docker, Kubernetes, "
    "Terraform, and general technical questions."
)


# ============================================================
# API KEY CHECK
# ============================================================

if not os.environ.get("GROQ_API_KEY"):

    st.warning(
        "GROQ_API_KEY is not configured. "
        "Add it to your `.env` file before asking questions.",
        icon="⚠️",
    )


# ============================================================
# LOAD RETRIEVER
# ============================================================

@st.cache_resource
def load_retriever():

    return HybridRetriever()


try:

    retriever = load_retriever()

except Exception as exc:

    st.error(
        "The document index could not be loaded."
    )

    st.code(str(exc))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("About")

    st.markdown(
        """
        **Knowledge sources**

        - Docker
        - Kubernetes
        - Terraform
        - General LLM knowledge

        **Architecture**

        - Conversation understanding
        - Hybrid retrieval
        - Dense search
        - BM25 search
        - Reranking
        - LLM generation
        - Conversational memory

        **Behavior**

        If relevant documentation exists, the assistant
        uses it.

        If the documentation does not contain the answer,
        the LLM can still answer using its general knowledge.
        """
    )

    st.subheader("Try these")

    examples = [
        "What is a Docker image?",
        "What is the difference between a Docker image and a container?",
        "What is Kubernetes?",
        "What is the difference between Docker and Kubernetes?",
        "Explain Terraform providers.",
        "What is the TCP three-way handshake?",
        "What is machine learning?",
    ]

    for example in examples:

        if st.button(
            example,
            use_container_width=True,
        ):

            st.session_state[
                "pending_query"
            ] = example


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        sources = message.get(
            "sources",
            [],
        )

        if sources:

            with st.expander(
                f"Sources ({len(sources)})"
            ):

                for source in sources:

                    st.markdown(
                        f"- **[{source['tool']}]** "
                        f"`{source['file']}` → "
                        f"*{source['section']}* "
                        f"(score {source['score']})"
                    )


# ============================================================
# USER INPUT
# ============================================================

query = st.chat_input(
    "Ask me anything..."
)


# ============================================================
# SIDEBAR BUTTON INPUT
# ============================================================

if "pending_query" in st.session_state:

    query = st.session_state.pop(
        "pending_query"
    )


# ============================================================
# PROCESS QUESTION
# ============================================================

if query:

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Take a snapshot BEFORE adding the current question.
    #
    # This means generate.py receives the previous conversation
    # and can understand follow-up questions naturally.
    # --------------------------------------------------------

    previous_history = list(
        st.session_state.messages
    )

    # --------------------------------------------------------
    # Show user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            result = answer_question(
                query=query,
                retriever=retriever,
                conversation_history=previous_history,
            )

        answer = result.get(
            "answer",
            "I couldn't generate an answer.",
        )

        st.markdown(answer)

        # ----------------------------------------------------
        # Show retrieval information quietly
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )

        if sources:

            with st.expander(
                f"Sources ({len(sources)})"
            ):

                for source in sources:

                    st.markdown(
                        f"- **[{source['tool']}]** "
                        f"`{source['file']}` → "
                        f"*{source['section']}* "
                        f"(score {source['score']})"
                    )

        # ----------------------------------------------------
        # Show whether documentation was retrieved.
        # This is intentionally subtle.
        # ----------------------------------------------------

        if result.get("used_rag"):

            st.caption(
                "📚 Relevant documentation was used."
            )

        else:

            st.caption(
                "🤖 Answer generated from general knowledge."
            )

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )