import sys, os, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from pipeline import build_session, load_sample, SAMPLE_NAMES

print("=== bundled samples ===")
for n in SAMPLE_NAMES:
    s = load_sample(n)
    print(f"{n:38s} company={s.company_name!r:28s} charts={len(s.charts)} anomalies={len(s.result.anomalies)} docs={s.doc_names} err={s.error}")

def files(*names):
    out = []
    for n in names:
        with open(f"data/sample/{n}", "rb") as f:
            out.append((n, f.read()))
    return out

print("\n=== 3 CSVs uploaded together (types auto-detected from content) ===")
s = build_session(files("Bristlecone_Income_Statement.csv", "Bristlecone_Balance_Sheet.csv", "Bristlecone_Cash_Flow.csv"))
print(s.company_name, s.statement_files, "anomalies:", len(s.result.anomalies), "charts:", len(s.charts), "warnings:", s.warnings)

print("\n=== CSV with a meaningless filename (must classify by content) ===")
data = open("data/sample/Bristlecone_Balance_Sheet.csv","rb").read().replace(b"Balance Sheet", b"Statement X")
s2 = build_session([("upload1.csv", open("data/sample/Bristlecone_Income_Statement.csv","rb").read().replace(b"Income Statement", b"Statement Y")),
                    ("upload2.csv", data)])
print(s2.statement_files, "err:", s2.error, "warnings:", s2.warnings)

print("\n=== Income-statement-only quarterly ===")
s = build_session(files("Vantage_Robotics_Quarterly_Income_Statement.xlsx"))
print(s.company_name, "charts:", list(s.charts), "warnings:", s.warnings)

print("\n=== Garbage / edge inputs (must never raise) ===")
for label, fl in [("empty list", []),
                  ("only a txt narrative", [("notes.txt", b"Revenue grew because of new stores. Margins fell due to costs.")]),
                  ("unsupported file", [("pic.png", b"\x89PNG")]),
                  ("corrupt xlsx", [("bad.xlsx", b"not really excel")]),
                  ("csv w/o periods", [("x.csv", b"a,b\n1,2\n")])]:
    s = build_session(fl)
    print(f"{label:22s} error={bool(s.error)} warnings={len(s.warnings)} retriever={s.retriever is not None}")
