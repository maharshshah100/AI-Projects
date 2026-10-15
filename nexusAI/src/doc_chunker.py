"""
Chunker for narrative text (MD&A notes, commentary) -- deliberately much
simpler than DevDocs' code-aware chunker, because this content has no code
blocks to protect. The one thing that matters here: never split a
paragraph in half, since a paragraph is usually one complete thought/topic
in a document like this (each section discusses one theme: margins, debt,
dividends, etc.).
"""
from __future__ import annotations
import re
from dataclasses import dataclass

SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9\s&,]{6,}$")  # ALL-CAPS section headers in our MD&A


@dataclass
class DocChunk:
    text: str
    section: str
    chunk_id: str


def chunk_narrative_text(text: str, source_name: str = "document") -> list[DocChunk]:
    """
    Splits on blank-line-separated paragraphs, tracking the most recent
    ALL-CAPS section header so each chunk knows which section it came
    from (used for citations later).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_section = "Introduction"
    idx = 0
    for para in paragraphs:
        first_line = para.split("\n")[0].strip()
        if SECTION_HEADER_RE.match(first_line):
            current_section = first_line.title()
            # If the header line has more text after it (rare), keep the rest
            rest = "\n".join(para.split("\n")[1:]).strip()
            if not rest:
                continue
            para = rest

        chunks.append(DocChunk(text=para, section=current_section,
                                chunk_id=f"{source_name}::{idx}"))
        idx += 1

    return chunks


if __name__ == "__main__":
    text = open("data/sample/Nimbus_MDA_Notes.txt").read()
    chunks = chunk_narrative_text(text, "Nimbus_MDA_Notes")
    print(f"Produced {len(chunks)} chunks\n")
    for c in chunks:
        print(f"[{c.section}] ({len(c.text)} chars)")
        print(f"  {c.text[:100]}...")
        print()
