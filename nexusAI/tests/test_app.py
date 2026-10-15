import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
import chat_engine as ce

def fake_llm(system, user, history, max_tokens=500):
    if system == ce.GROUNDED_SYSTEM and "healthy" in user.split("QUESTION:")[-1]:
        yield "NOT_IN_DOCUMENTS"; return
    for w in ["Mocked ", "streamed ", "answer."]:
        yield w
ce._stream_llm = fake_llm
ce.web_search = lambda q, **k: {"ok": True, "results": [{"title": "T", "url": "http://x", "content": "c"}], "error": None}

def check(at, label):
    assert not at.exception, f"{label}: EXCEPTION {[e.value for e in at.exception]}"
    print(f"  ok: {label}")

at = AppTest.from_file(os.path.join(ROOT, "streamlit_app.py"), default_timeout=90)
at.run(); check(at, "initial load (nothing uploaded)")
assert at.chat_input[0].disabled, "chat should be disabled before documents are loaded"

at.button(key="load_sample").click().run(); check(at, "load Nimbus sample")
charts = at.get("plotly_chart"); print("  plotly charts rendered:", len(charts)); assert len(charts) == 7
print("  KPI metrics:", [(m.label, m.value) for m in at.metric])
assert [m.label for m in at.metric][:3] == ["Net margin", "Current ratio", "Debt-to-equity"]
assert not at.chat_input[0].disabled

def chat(q):
    at.chat_input[0].set_value(q).run(); check(at, f"chat: {q}")
    msgs = at.chat_message
    return " ".join(m.value for m in msgs[-1].markdown)

os.environ["GROQ_API_KEY"] = "fake"
t = chat("What was revenue in the latest period?");         print("   ->", t[:70]); assert "150,000" in t
t = chat("Why did gross margin decline?");                  print("   ->", t[:70]); assert "Mocked" in t
t = chat("What's a healthy current ratio for retail?");     print("   ->", t[:70]); assert "Mocked" in t  # went to web path
assert len(at.session_state["messages"]) == 6
print("  sources used:", [m.get("source") for m in at.session_state["messages"] if m["role"] == "assistant"])

at.selectbox(key="sample_choice").select("Solstice Technologies (healthy)")
at.button(key="load_sample").click().run(); check(at, "switch to healthy sample")
assert at.session_state["messages"] == [], "chat should reset on new documents"
print("  Solstice findings metric:", [m.value for m in at.metric if m.label == "Findings"])

at.selectbox(key="sample_choice").select("Anchor Manufacturing (distressed)")
at.button(key="load_sample").click().run(); check(at, "switch to distressed sample")
print("  Anchor findings metric:", [m.value for m in at.metric if m.label == "Findings"])
print("\nAPP TEST PASSED")
