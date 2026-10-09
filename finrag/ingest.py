"""
ingest.py — Structure-aware chunking for SEC filings.

Unlike naive fixed-size chunking (which can slice a table or a sentence in
half), this splits along the document's own structure first (====== SECTION
======= markers, which correspond to real 10-K Item headers), then further
splits long sections into paragraph-level chunks — never mid-sentence, never
mid-table-row. Every chunk carries full source metadata so a retrieved chunk
can always be traced back to an exact company, filing, and section.
"""
import re
import json
import os
from dataclasses import dataclass, asdict
from typing import List

SOURCE_HEADER_RE = re.compile(r"^(SOURCE|FILED|COMPANY|CIK|URL):\s*(.*)$")
SECTION_RE = re.compile(r"^===\s*(.+?)\s*===$")

MAX_CHUNK_CHARS = 900   # keeps chunks small enough to be precise, large enough to keep context
MIN_CHUNK_CHARS = 80    # avoid tiny orphan chunks (e.g. a lone header line)


@dataclass
class Chunk:
    chunk_id: str
    company: str
    ticker: str
    filing_type: str
    fiscal_year: str
    filed_date: str
    source_url: str
    section: str
    text: str


def parse_document(path: str) -> List[Chunk]:
    with open(path, "r") as f:
        raw = f.read()

    lines = raw.split("\n")
    meta = {"SOURCE": "", "FILED": "", "COMPANY": "", "CIK": "", "URL": ""}
    body_start = 0
    for i, line in enumerate(lines):
        m = SOURCE_HEADER_RE.match(line.strip())
        if m:
            meta[m.group(1)] = m.group(2).strip()
            body_start = i + 1
        elif line.strip() == "" and i > 0 and meta["SOURCE"]:
            continue

    # derive structured metadata
    company_full = meta["COMPANY"]
    ticker_match = re.search(r"\(([A-Z]+)\)", company_full)
    ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"
    company = company_full.split("(")[0].strip()

    fy_match = re.search(r"(20\d\d)", meta["SOURCE"])
    fiscal_year = fy_match.group(1) if fy_match else "UNKNOWN"
    filing_type = "10-K" if "10-K" in meta["SOURCE"] else "10-Q" if "10-Q" in meta["SOURCE"] else "UNKNOWN"

    body = "\n".join(lines[body_start:])

    # split on section markers
    sections = []
    current_section = "General"
    current_text = []
    for line in body.split("\n"):
        sm = SECTION_RE.match(line.strip())
        if sm:
            if current_text:
                sections.append((current_section, "\n".join(current_text).strip()))
            current_section = sm.group(1)
            current_text = []
        else:
            current_text.append(line)
    if current_text:
        sections.append((current_section, "\n".join(current_text).strip()))

    chunks = []
    chunk_counter = 0
    for section_name, section_text in sections:
        if not section_text.strip():
            continue
        # split section into paragraph blocks (blank-line separated)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]

        buffer = ""
        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if len(candidate) > MAX_CHUNK_CHARS and buffer:
                # flush buffer as its own chunk, start new buffer with this paragraph
                if len(buffer) >= MIN_CHUNK_CHARS:
                    chunk_counter += 1
                    chunks.append(Chunk(
                        chunk_id=f"{ticker}_{fiscal_year}_{chunk_counter:03d}",
                        company=company, ticker=ticker, filing_type=filing_type,
                        fiscal_year=fiscal_year, filed_date=meta["FILED"],
                        source_url=meta["URL"], section=section_name, text=buffer,
                    ))
                buffer = para
            else:
                buffer = candidate
        if buffer and len(buffer) >= MIN_CHUNK_CHARS:
            chunk_counter += 1
            chunks.append(Chunk(
                chunk_id=f"{ticker}_{fiscal_year}_{chunk_counter:03d}",
                company=company, ticker=ticker, filing_type=filing_type,
                fiscal_year=fiscal_year, filed_date=meta["FILED"],
                source_url=meta["URL"], section=section_name, text=buffer,
            ))

    return chunks


def build_corpus(raw_dir: str, out_path: str) -> List[Chunk]:
    all_chunks = []
    for fname in sorted(os.listdir(raw_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(raw_dir, fname)
            chunks = parse_document(path)
            all_chunks.extend(chunks)
            print(f"  {fname}: {len(chunks)} chunks")

    with open(out_path, "w") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2)

    return all_chunks


if __name__ == "__main__":
    print("Chunking filings...")
    chunks = build_corpus("data/raw", "data/processed/chunks.json")
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Companies: {sorted(set(c.ticker for c in chunks))}")
    print(f"Sections: {sorted(set(c.section for c in chunks))}")
    print(f"\nSample chunk:")
    print(json.dumps(asdict(chunks[5]), indent=2)[:600])
