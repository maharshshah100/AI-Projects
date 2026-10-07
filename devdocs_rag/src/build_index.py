
from __future__ import annotations

import glob
import json
import os
import pickle
import sys

from rank_bm25 import BM25Okapi


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/raw"
INDEX_DIR = "data/index"

sys.path.insert(0, "src")

from chunker import chunk_document, Chunk
from embedder import get_embedder


# Import the single tokenization implementation
from retrieve import tokenize


# ============================================================
# BUILD INDEX
# ============================================================

def build():

    os.makedirs(
        INDEX_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load and chunk documentation
    # --------------------------------------------------------

    all_chunks: list[Chunk] = []

    tool_dirs = {
        "kubernetes": f"{DATA_DIR}/kubernetes/*.md",
        "docker": f"{DATA_DIR}/docker/*.md",
        "terraform": f"{DATA_DIR}/terraform/*.mdx",
    }

    doc_count = 0

    for tool, pattern in tool_dirs.items():

        for path in sorted(
            glob.glob(pattern)
        ):

            with open(
                path,
                encoding="utf-8",
            ) as f:

                text = f.read()

            fname = os.path.basename(
                path
            )

            chunks = chunk_document(
                text,
                fname,
                tool=tool,
            )

            all_chunks.extend(
                chunks
            )

            doc_count += 1

    print(
        f"Chunked {doc_count} docs across "
        f"{len(tool_dirs)} tools into "
        f"{len(all_chunks)} chunks"
    )

    for tool in tool_dirs:

        n = sum(
            1
            for c in all_chunks
            if c.tool == tool
        )

        print(
            f"  {tool}: {n} chunks"
        )

    if not all_chunks:
        raise RuntimeError(
            "No documentation chunks were created. "
            "Check data/raw."
        )

    # --------------------------------------------------------
    # Dense embeddings
    # --------------------------------------------------------

    texts = [
        c.text
        for c in all_chunks
    ]

    print(
        "Creating embedding model..."
    )

    embedder = get_embedder(
        prefer_transformer=True
    )

    print(
        f"Using embedding model: "
        f"{embedder.name}"
    )

    # Local TF-IDF/SVD requires fitting.
    # Transformer embeddings do not.
    embedder.fit(
        texts
    )

    print(
        "Embedding chunks..."
    )

    embeddings = embedder.embed(
        texts
    )

    embeddings = [
        list(vector)
        for vector in embeddings
    ]

    # --------------------------------------------------------
    # Save dense index
    # --------------------------------------------------------

    dense_data = {
        "vectors": embeddings,
        "embeddings": embeddings,
        "chunks": all_chunks,
        "embedder_name": embedder.name,
    }

    dense_path = os.path.join(
        INDEX_DIR,
        "dense.pkl",
    )

    with open(
        dense_path,
        "wb",
    ) as f:

        pickle.dump(
            dense_data,
            f,
        )

    # --------------------------------------------------------
    # Save embedder when supported
    # --------------------------------------------------------

    if hasattr(
        embedder,
        "save",
    ):

        embedder_path = os.path.join(
            INDEX_DIR,
            "embedder.pkl",
        )

        embedder.save(
            embedder_path
        )

        print(
            f"Saved embedding model: "
            f"{embedder_path}"
        )

    else:

        print(
            "Embedding model does not require "
            "a local pickle file."
        )

    print(
        f"Saved dense index: "
        f"{len(embeddings)} vectors, "
        f"dim={len(embeddings[0])}, "
        f"model={embedder.name}"
    )

    # --------------------------------------------------------
    # BM25 keyword index
    # --------------------------------------------------------

    tokenized = [
        tokenize(c.text)
        for c in all_chunks
    ]

    bm25 = BM25Okapi(
        tokenized
    )

    bm25_path = os.path.join(
        INDEX_DIR,
        "bm25.pkl",
    )

    with open(
        bm25_path,
        "wb",
    ) as f:

        pickle.dump(
            {
                "bm25": bm25,
                "chunks": all_chunks,
            },
            f,
        )

    print(
        "Saved BM25 index"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    meta = [
        {
            "chunk_id": c.chunk_id,
            "source_file": c.source_file,
            "tool": c.tool,
            "section_path": c.section_path,
            "contains_code": c.contains_code,
            "char_len": len(c.text),
        }
        for c in all_chunks
    ]

    meta_path = os.path.join(
        INDEX_DIR,
        "chunks_meta.json",
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            meta,
            f,
            indent=2,
        )

    print(
        f"Wrote metadata for "
        f"{len(meta)} chunks to "
        f"{meta_path}"
    )

    print(
        "\nIndex build completed successfully."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build()
