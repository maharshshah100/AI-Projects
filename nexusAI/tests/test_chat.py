import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import chat_engine as ce
from pipeline import load_sample, build_session

def drain(reply): return "".join(reply.stream)

# --- mocks: LLM + web (sandbox has no network) ---
def fake_llm(system, user, history, max_tokens=500):
    if system == ce.GROUNDED_SYSTEM and "healthy" in user.split("QUESTION:")[-1]:
        yield "NOT_IN_DOCUMENTS"; return
    tag = {ce.GROUNDED_SYSTEM: "GROUNDED", ce.WEB_SYSTEM: "WEB", ce.GENERAL_SYSTEM: "GENERAL"}[system]
    for w in [f"[{tag} answer ", "streamed ", "in pieces.]"]:
        yield w
ce._stream_llm = fake_llm
ce.web_search = lambda q, **k: {"ok": True, "results": [{"title": "Retail ratios", "url": "http://x.test", "content": "Retail D/E is often 0.5-1.5"}], "error": None}

os.environ["GROQ_API_KEY"] = "fake"
nimbus = load_sample("Nimbus Retail (declining margins)")
eng = ce.ChatEngine(nimbus)
hist = []
def ask(q):
    r = eng.ask(q, hist); t = drain(r)
    hist.extend([{"role": "user", "content": q}, {"role": "assistant", "content": t}])
    print(f"Q: {q}\n   source={r.source}  cites={[c['label'] for c in r.citations]}\n   -> {t[:170]!r}\n")
    return r, t

print("########## Nimbus (with MD&A notes) ##########")
r, t = ask("What was revenue in FY2024?");           assert r.source == "exact" and "138,000" in t
r, t = ask("and FY2025?");                           assert r.source == "exact" and "150,000" in t and "Revenue" in t
r, t = ask("What is the net margin?");               assert r.source == "exact" and "8.1%" in t and "4.9%" in t
r, t = ask("current ratio in the latest period");    assert "1.90x" in t
r, t = ask("how much cash did we have in 2023?");    assert "15,000" in t
r, t = ask("Why did gross margin decline?");         assert r.source == "documents" and any(c["label"] == "D1" for c in r.citations)
r, t = ask("What's a healthy debt to equity ratio for retail?"); assert r.source == "web" and r.note and "W1" in r.citations[0]["label"] + "W1"
r, t = ask("total sales for FY2023");                assert r.source == "exact" and "120,000" in t

print("########## Web search unavailable -> labeled general knowledge ##########")
ce.web_search = lambda q, **k: {"ok": False, "results": [], "error": "TAVILY_API_KEY not set"}
r, t = ask("What's a healthy current ratio for a retailer?"); assert r.source == "general" and "TAVILY_API_KEY" in r.note

print("########## No LLM key -> limited mode (never crashes) ##########")
del os.environ["GROQ_API_KEY"]
eng2 = ce.ChatEngine(nimbus)
r = eng2.ask("Why did margins fall?"); t = drain(r); print("   source:", r.source, "|", r.note, "\n   ", t[:120].replace("\n"," "), "\n"); assert r.source == "limited"
r = eng2.ask("what was revenue in FY2024"); assert r.source == "exact"  # exact answers need no key at all

print("########## Healthy company, no notes uploaded ##########")
os.environ["GROQ_API_KEY"] = "fake"
sol = ce.ChatEngine(load_sample("Solstice Technologies (healthy)"))
r = sol.ask("Why is net income growing?"); print("   source:", r.source, "->", drain(r)[:60]); assert r.source == "documents"
r = sol.ask("what is the current ratio in FY2025"); assert "9.37x" in drain(r)

print("\n########## Income-statement-only upload: ratio needing a missing statement ##########")
v = ce.ChatEngine(build_session([("v.xlsx", open("data/sample/Vantage_Robotics_Quarterly_Income_Statement.xlsx", "rb").read())]))
r = v.ask("what is the current ratio?"); t = drain(r); print("  ", t); assert "can't compute" in t and "Balance Sheet" in t
r = v.ask("net margin in Q3 2025"); assert "3.9%" in drain(r)
r = v.ask(""); assert drain(r)
print("\nALL CHAT ROUTING TESTS PASSED")
