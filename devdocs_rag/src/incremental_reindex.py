"""
Incremental reindexing.

The naive approach to "a new doc page arrived" is: re-run build_index.py
on everything. That works at 35 docs. It does NOT work once you have
5,000 docs and a new page shows up every few minutes (which is the real
situation any production docs-RAG system faces) -- re-embedding
everything on every update is wasteful and slow.

This script instead:
  1. Chunks ONLY the new document
  2. Embeds ONLY the new chunks (using the already-fitted embedder --
     for the TF-IDF+SVD fallback this means transforming into the
     EXISTING vector space, not re-fitting it)
  3. Appends to the existing dense matrix and chunk list
  4. Rebuilds the BM25 index (BM25 has no incremental-update API in
     rank_bm25, so this one part IS a full rebuild -- documented
     honestly below rather than hidden)

Run: python3 src/incremental_reindex.py path/to/new_doc.md kubernetes

A ready-to-use example is included at
data/incremental_demo/cli_workspaces_index.mdx (Terraform's CLI workspaces
doc, deliberately kept OUT of data/raw/terraform so this script has a
genuinely new document to add). Try:

    python3 src/incremental_reindex.py data/incremental_demo/cli_workspaces_index.mdx terraform

Then query "how do Terraform workspaces work" before and after running
it -- before, retrieval finds nothing relevant; after, it does.
"""
from __future__ import annotations
import pickle
import sys

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, "src")
from chunker import chunk_document  # noqa: E402
from retrieve import tokenize  # noqa: E402

INDEX_DIR = "data/index"


def incremental_add(doc_path: str, tool: str):
    fname = doc_path.split("/")[-1]
    text = open(doc_path, encoding="utf-8").read()
    new_chunks = chunk_document(text, fname, tool=tool)
    print(f"New doc '{fname}' -> {len(new_chunks)} new chunks")

    # --- Dense: embed only the new chunks, in the EXISTING vector space ---
    with open(f"{INDEX_DIR}/dense.pkl", "rb") as f:
        dense_data = pickle.load(f)

    try:
        embedder = pickle.load(open(f"{INDEX_DIR}/embedder.pkl", "rb"))
        new_vecs = embedder.embed([c.text for c in new_chunks])
    except FileNotFoundError:
        raise RuntimeError(
            "No saved embedder found (this happens if the index was built "
            "with the real transformer embedder, which needs no fitting -- "
            "in that case just call TransformerEmbedder().embed() directly "
            "on the new chunks instead)."
        )

    dense_data["embeddings"] = dense_data["embeddings"] + list(new_vecs)
    dense_data["chunks"] = dense_data["chunks"] + new_chunks
    with open(f"{INDEX_DIR}/dense.pkl", "wb") as f:
        pickle.dump(dense_data, f)
    print(f"Dense index: {len(dense_data['chunks'])} total chunks (was "
          f"{len(dense_data['chunks']) - len(new_chunks)})")

    # --- BM25: honestly, this part has to fully rebuild ---
    # rank_bm25's BM25Okapi has no incremental API -- its IDF statistics
    # are computed once over the whole corpus. Rebuilding it from the
    # (now-larger) chunk list is cheap at this scale (milliseconds for
    # hundreds of chunks) but WOULD become the bottleneck at real scale
    # (millions of docs), at which point you'd reach for a proper search
    # engine (Elasticsearch/Typesense) that supports incremental indexing
    # natively instead of rank_bm25.
    all_chunks = dense_data["chunks"]
    tokenized = [tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    with open(f"{INDEX_DIR}/bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)
    print("BM25 index rebuilt (see comment above for why this part isn't incremental)")

    return len(new_chunks)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 src/incremental_reindex.py <doc_path> <tool_name>")
        sys.exit(1)
    incremental_add(sys.argv[1], sys.argv[2])
