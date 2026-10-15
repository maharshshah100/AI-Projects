"""
Pipeline: takes whatever the user uploads (Excel workbooks, CSV statements,
narrative notes as .txt/.md/.pdf) and builds one FinSightSession holding
everything the UI and chat need: normalized data, ratios, anomalies,
charts, per-chart captions, final recommendations, and a document
retriever for narrative questions.

Never raises on bad input -- problems are collected in session.warnings /
session.error so the UI can show a friendly message instead of a traceback.
"""
from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass, field

from ingest import ingest_workbook, ingest_csv, detect_statement_type
from normalize import normalize_statement, NormalizedStatement
from analysis import analyze, AnalysisResult
from chart_engine import generate_all_charts
from chart_insights import generate_all_chart_captions, polish_captions_with_llm, CHART_ORDER
from doc_embedder import get_embedder, LocalTestEmbedder
from doc_retrieve import DocumentRetriever

STATEMENT_TYPES = ("Income Statement", "Balance Sheet", "Cash Flow Statement")


@dataclass
class FinSightSession:
    company_name: str = "Your Company"
    normalized: dict[str, NormalizedStatement] = field(default_factory=dict)
    result: AnalysisResult | None = None
    charts: dict = field(default_factory=dict)
    captions: dict[str, str] = field(default_factory=dict)
    chart_order: list[str] = field(default_factory=list)
    recommendations: str = ""
    retriever: DocumentRetriever | None = None
    doc_names: list[str] = field(default_factory=list)
    statement_files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    embedder_name: str = ""


def classify_by_fields(norm: NormalizedStatement) -> str | None:
    """Fallback when a file/sheet name doesn't say what it is: look at WHICH
    line items it contains."""
    f = norm.fields
    if "total_assets" in f or "total_current_assets" in f:
        return "Balance Sheet"
    if "cfo" in f or "ending_cash" in f or "capex" in f:
        return "Cash Flow Statement"
    if "revenue" in f or "net_income" in f:
        return "Income Statement"
    return None


def _company_from_title(path: str, ext: str, fallback: str) -> str:
    try:
        if ext == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True)
            v = wb.worksheets[0]["A1"].value
            wb.close()
        else:
            with open(path, encoding="utf-8", errors="ignore") as f:
                v = f.readline().split(",")[0]
        v = (v or "").strip()
        # Titles like "Nimbus Retail Inc." are fine; skip header-ish junk.
        if v and len(v) < 60 and not any(c.isdigit() for c in v[:4]):
            return v
    except Exception:
        pass
    return fallback


def _read_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception:
        return ""


def build_session(files: list[tuple[str, bytes]], use_llm_polish: bool = True) -> FinSightSession:
    """files: list of (filename, raw_bytes)."""
    s = FinSightSession()
    raw_statements: dict[str, object] = {}
    narrative_docs: list[tuple[str, str]] = []
    company_guess = None

    with tempfile.TemporaryDirectory() as tmp:
        for fname, data in files:
            ext = os.path.splitext(fname)[1].lower()
            path = os.path.join(tmp, os.path.basename(fname))
            with open(path, "wb") as f:
                f.write(data)
            try:
                if ext in (".xlsx", ".xlsm"):
                    company_guess = company_guess or _company_from_title(path, ".xlsx", os.path.splitext(fname)[0])
                    found = ingest_workbook(path)
                    if not found:
                        s.warnings.append(f"'{fname}': no financial statement tables detected (needs a header row with periods like FY2024 or Q1 2025).")
                    for sheet, stmt in found.items():
                        kind = detect_statement_type(sheet) or classify_by_fields(normalize_statement(stmt))
                        if kind is None:
                            s.warnings.append(f"'{fname}' / sheet '{sheet}': couldn't tell if this is an Income Statement, Balance Sheet or Cash Flow -- skipped.")
                            continue
                        if kind in raw_statements:
                            s.warnings.append(f"More than one {kind} supplied; using the latest one ('{fname}').")
                        stmt.sheet_name = kind
                        raw_statements[kind] = stmt
                        s.statement_files[kind] = fname
                elif ext == ".csv":
                    company_guess = company_guess or _company_from_title(path, ".csv", os.path.splitext(fname)[0])
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        head = "".join(f.readline() for _ in range(4))
                    kind = detect_statement_type(head, fname)
                    tmp_name = kind or "Unknown"
                    found = ingest_csv(path, sheet_name=tmp_name)
                    if not found:
                        s.warnings.append(f"'{fname}': no financial statement table detected.")
                        continue
                    stmt = found[tmp_name]
                    if kind is None:
                        kind = classify_by_fields(normalize_statement(stmt))
                        if kind is None:
                            s.warnings.append(f"'{fname}': couldn't tell which statement this is -- skipped.")
                            continue
                    stmt.sheet_name = kind
                    raw_statements[kind] = stmt
                    s.statement_files[kind] = fname
                elif ext in (".txt", ".md"):
                    narrative_docs.append((fname, data.decode("utf-8", errors="ignore")))
                elif ext == ".pdf":
                    text = _read_pdf_text(path)
                    if text.strip():
                        narrative_docs.append((fname, text))
                    else:
                        s.warnings.append(f"'{fname}': couldn't extract text from this PDF (scanned image PDFs aren't supported yet).")
                else:
                    s.warnings.append(f"'{fname}': unsupported file type ({ext}).")
            except Exception as e:
                s.warnings.append(f"'{fname}': could not be read ({type(e).__name__}: {str(e)[:120]}).")

    s.company_name = company_guess or "Your Company"

    # ---- structured analysis ----
    s.normalized = {k: normalize_statement(v) for k, v in raw_statements.items()}
    if "Income Statement" not in s.normalized:
        if not narrative_docs:
            s.error = ("No Income Statement was found. Upload at least an Income Statement "
                       "(Excel or CSV with Revenue and Net Income by period). A Balance Sheet and "
                       "Cash Flow Statement are optional but unlock liquidity, leverage and cash analysis.")
            return s
    else:
        try:
            s.result = analyze(s.normalized)
            s.charts = generate_all_charts(s.result, s.normalized)
            rule_caps = generate_all_chart_captions(s.result, s.normalized)
            s.chart_order = [k for k in CHART_ORDER if k in s.charts]
            try:
                if use_llm_polish:
                    s.captions, s.recommendations = polish_captions_with_llm(
                        rule_caps, s.result, s.normalized, s.company_name)
                else:
                    raise RuntimeError("polish disabled")
            except Exception:
                # LLM unavailable (no key / network): rule-based captions + actions
                s.captions, s.recommendations = polish_captions_with_llm(
                    rule_caps, s.result, s.normalized, s.company_name, api_key="")
            for miss in [t for t in STATEMENT_TYPES if t not in s.normalized]:
                s.warnings.append(f"No {miss} supplied -- analysis that depends on it was skipped.")
        except Exception as e:
            s.error = f"Analysis failed: {type(e).__name__}: {str(e)[:200]}"
            return s

    # ---- narrative documents -> retriever ----
    if narrative_docs:
        embedder = get_embedder()
        retriever = DocumentRetriever(embedder=embedder)
        try:
            for name, text in narrative_docs:
                retriever.index_document(text, name)
        except Exception as e:  # e.g. Voyage unreachable -> local fallback
            s.warnings.append(f"Voyage embeddings unavailable ({type(e).__name__}); used the local fallback embedder instead.")
            embedder = LocalTestEmbedder()
            retriever = DocumentRetriever(embedder=embedder)
            for name, text in narrative_docs:
                retriever.index_document(text, name)
        s.retriever = retriever
        s.doc_names = [n for n, _ in narrative_docs]
        s.embedder_name = embedder.name

    return s


def load_sample(name: str, base_dir: str = "data/sample") -> FinSightSession:
    """Loads one of the bundled demo companies."""
    sets = {
        "Nimbus Retail (declining margins)": ["Nimbus_Retail_Financial_Statements.xlsx", "Nimbus_MDA_Notes.txt"],
        "Solstice Technologies (healthy)": ["Solstice_Technologies_Financial_Statements.xlsx"],
        "Anchor Manufacturing (distressed)": ["Anchor_Manufacturing_Financial_Statements.xlsx"],
    }
    files = []
    for fn in sets[name]:
        with open(os.path.join(base_dir, fn), "rb") as f:
            files.append((fn, f.read()))
    return build_session(files)


SAMPLE_NAMES = [
    "Nimbus Retail (declining margins)",
    "Solstice Technologies (healthy)",
    "Anchor Manufacturing (distressed)",
]


if __name__ == "__main__":
    for n in SAMPLE_NAMES:
        s = load_sample(n)
        print(f"{n}: company={s.company_name!r} charts={len(s.charts)} anomalies="
              f"{len(s.result.anomalies) if s.result else None} docs={s.doc_names} err={s.error}")
