"""
Code-aware chunker for markdown documentation.

The naive approach (split every N tokens) will happily cut a YAML/HCL code
block in half. That produces chunks where a retrieved "example" is
syntactically broken -- and the LLM will confidently "complete" it with a
hallucinated ending. This module fixes that by:

  1. Splitting on markdown headers first (so a chunk never mixes two
     unrelated sections).
  2. Within a section, walking line by line and NEVER breaking inside a
     ``` fenced code block, even if that makes the chunk larger than the
     target size.
  3. Keeping a code block attached to the prose immediately above it
     (the explanation and the example are one unit -- splitting them
     apart loses the context of what the code is for).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*```")

TARGET_CHARS = 1800   # ~ 450-500 tokens, a reasonable chunk size
MIN_CHARS = 200        # don't emit tiny orphan chunks


@dataclass
class Chunk:
    text: str
    source_file: str
    section_path: str          # e.g. "Deployments > Updating a Deployment"
    contains_code: bool
    tool: str = "unknown"       # "kubernetes" | "docker" | "terraform"
    chunk_id: str = ""


def split_into_sections(md_text: str) -> list[tuple[str, str]]:
    """Split a markdown doc into (header_path, section_body) pairs."""
    lines = md_text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    header_stack: list[str] = []
    current_lines: list[str] = []
    in_code_fence = False

    def flush():
        if current_lines:
            path = " > ".join(header_stack) if header_stack else "(intro)"
            sections.append((path, current_lines.copy()))

    for line in lines:
        if FENCE_RE.match(line):
            in_code_fence = not in_code_fence
            current_lines.append(line)
            continue

        m = HEADER_RE.match(line)
        if m and not in_code_fence:
            flush()
            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # trim stack to this header's depth, then push
            header_stack = header_stack[: level - 1]
            header_stack.append(title)
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush()
    return [(path, "\n".join(body_lines)) for path, body_lines in sections]


def chunk_section_code_aware(text: str) -> list[str]:
    """
    Break one section's text into ~TARGET_CHARS chunks WITHOUT ever
    splitting inside a fenced code block.
    """
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    in_fence = False

    def buf_flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("".join(buf))
        buf, buf_len = [], 0

    i = 0
    while i < len(lines):
        line = lines[i]

        if FENCE_RE.match(line) and not in_fence:
            # Starting a code block: if we're already near the target size,
            # flush what we have BEFORE the code block starts, so the
            # block begins a fresh chunk and has room to live in one piece.
            if buf_len >= TARGET_CHARS * 0.6:
                buf_flush()
            in_fence = True
            buf.append(line)
            buf_len += len(line)
            i += 1
            continue

        if FENCE_RE.match(line) and in_fence:
            in_fence = False
            buf.append(line)
            buf_len += len(line)
            i += 1
            continue

        if in_fence:
            # Never break here, no matter how long the block is.
            buf.append(line)
            buf_len += len(line)
            i += 1
            continue

        # Normal prose line
        buf.append(line)
        buf_len += len(line)
        if buf_len >= TARGET_CHARS:
            buf_flush()
        i += 1

    buf_flush()
    # merge tiny trailing chunks into the previous one
    merged: list[str] = []
    for c in chunks:
        if merged and len(c.strip()) < MIN_CHARS:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    return [c for c in merged if c.strip()]


def chunk_document(md_text: str, source_file: str, tool: str = "unknown") -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_path, body in split_into_sections(md_text):
        if not body.strip():
            continue
        for piece in chunk_section_code_aware(body):
            contains_code = "```" in piece
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    source_file=source_file,
                    section_path=section_path,
                    contains_code=contains_code,
                    tool=tool,
                )
            )
    for idx, c in enumerate(chunks):
        c.chunk_id = f"{tool}::{source_file}::{idx}"
    return chunks


def naive_chunk_document(md_text: str, source_file: str, size: int = TARGET_CHARS) -> list[Chunk]:
    """The 'wrong' baseline, kept here on purpose so we can measure the
    difference: fixed-size chunking with NO awareness of code fences."""
    chunks = []
    for i in range(0, len(md_text), size):
        piece = md_text[i:i + size]
        if piece.strip():
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    source_file=source_file,
                    section_path="(naive, no section tracking)",
                    contains_code="```" in piece,
                    chunk_id=f"{source_file}::naive::{i}",
                )
            )
    return chunks


def code_block_integrity_rate(chunks: list[Chunk]) -> float:
    """
    Metric: of all chunks that contain a code fence marker, what fraction
    have a BALANCED number of ``` markers (i.e. the block wasn't cut off
    at the chunk boundary)? This is the number that proves the chunker
    is actually solving the problem it claims to solve.
    """
    code_chunks = [c for c in chunks if c.contains_code]
    if not code_chunks:
        return 1.0
    intact = sum(1 for c in code_chunks if c.text.count("```") % 2 == 0)
    return intact / len(code_chunks)
