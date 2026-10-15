"""
FinSight -- AI Financial Analyst (Streamlit UI)

Left:  upload documents -> instant analysis: KPIs, each chart with its own
       insight underneath, then recommended actions at the end.
Right: chat about the documents (exact figures, document-grounded answers,
       and clearly-labeled web suggestions).

Run:   streamlit run streamlit_app.py
"""
import os
import sys
import time
import re

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "src"))
os.chdir(APP_DIR)  # so relative paths work no matter where you launch from

from dotenv import load_dotenv
load_dotenv(os.path.join(APP_DIR, ".env"))

import streamlit as st

from pipeline import build_session, load_sample, SAMPLE_NAMES
from chat_engine import ChatEngine
from chart_insights import CHART_TITLES
from analysis import fmt_pct, fmt_ratio

st.set_page_config(page_title="FinSight — AI Financial Analyst", page_icon="📊", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1500px;}
  .badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-bottom: 6px;}
  .b-exact {background:#e6f4ea; color:#137333;}
  .b-documents {background:#e8f0fe; color:#1a56c4;}
  .b-web {background:#fef3e0; color:#b45f06;}
  .b-general {background:#f3e8fd; color:#7b2cbf;}
  .b-limited {background:#f1f3f4; color:#5f6368;}
  .hero {font-size: 2rem; font-weight: 700; margin-bottom: 0;}
  .sub {color:#5f6368; margin-top: -4px; margin-bottom: 8px;}
</style>
""", unsafe_allow_html=True)

BADGES = {
    "exact": ("b-exact", "🧮 Exact figure — computed, no AI guesswork"),
    "documents": ("b-documents", "📄 From your documents"),
    "web": ("b-web", "🌐 From the web — not in your documents"),
    "general": ("b-general", "💭 General knowledge — not from your documents"),
    "limited": ("b-limited", "⚠️ Limited mode"),
}

# ----------------------------------------------------------------------------
# state
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("session", None)
ss.setdefault("engine", None)
ss.setdefault("messages", [])
ss.setdefault("pending", None)


def set_session(sess):
    ss.session = sess
    ss.engine = ChatEngine(sess) if (sess.result is not None or sess.retriever is not None) else None
    ss.messages = []
    ss.pending = None


# ----------------------------------------------------------------------------
# sidebar: key status + help
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Setup status")
    for label, env, hint in [
        ("Groq (AI answers)", "GROQ_API_KEY", "free at console.groq.com"),
        ("Voyage (document search)", "VOYAGE_API_KEY", "free at dash.voyageai.com"),
        ("Tavily (web search)", "TAVILY_API_KEY", "free at tavily.com"),
    ]:
        ok = bool(os.environ.get(env))
        st.markdown(f"{'✅' if ok else '⚪'} **{label}**" + ("" if ok else f"  \n<small>add `{env}` to .env — {hint}</small>"),
                    unsafe_allow_html=True)
    st.caption("The app works without keys: exact figures, charts and rule-based insights need none. "
               "Keys unlock AI answers, better document search and web suggestions.")
    st.divider()
    st.markdown("**Upload tips**\n- Excel with Income Statement / Balance Sheet / Cash Flow sheets, or one CSV per statement\n"
                "- Add notes (.txt / .md / .pdf) to ask 'why' questions\n- Header row with periods like FY2024 or Q1 2025")
    if st.button("🔄 Start over", use_container_width=True):
        ss.session = ss.engine = ss.pending = None
        ss.messages = []
        st.rerun()

# ----------------------------------------------------------------------------
# header
# ----------------------------------------------------------------------------
st.markdown('<p class="hero">📊 FinSight</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload financial statements → instant analysis, charts and recommendations → ask anything.</p>',
            unsafe_allow_html=True)

left, right = st.columns([1.3, 1], gap="large")

# ----------------------------------------------------------------------------
# LEFT: documents + analysis
# ----------------------------------------------------------------------------
with left:
    with st.container(border=True):
        st.markdown("#### 1 · Your documents")
        files = st.file_uploader(
            "Drop financial statements and/or notes", type=["xlsx", "xlsm", "csv", "txt", "md", "pdf"],
            accept_multiple_files=True, key="uploader", label_visibility="collapsed")
        c1, c2 = st.columns([1, 1])
        if c1.button("Analyze documents", type="primary", disabled=not files, use_container_width=True, key="analyze"):
            with st.spinner("Reading, normalizing and analyzing…"):
                set_session(build_session([(f.name, f.getvalue()) for f in files]))
        with c2:
            sample = st.selectbox("…or try a sample company", SAMPLE_NAMES, label_visibility="collapsed", key="sample_choice")
        if st.button("Load sample company", use_container_width=True, key="load_sample"):
            with st.spinner("Analyzing sample…"):
                set_session(load_sample(sample, base_dir=os.path.join(APP_DIR, "data", "sample")))

    sess = ss.session
    if sess is None:
        st.info("👆 Upload your statements (or load a sample company) to see the analysis. "
                "No documents yet? The sample companies show a declining retailer, a healthy tech firm and a distressed manufacturer.")
    else:
        if sess.error:
            st.error(sess.error)
        for w in sess.warnings:
            st.warning(w, icon="⚠️")

        if sess.result is not None:
            r = sess.result
            p = r.periods[-1]
            ratios = r.ratios[p]
            high = [a for a in r.anomalies if a.severity == "high"]
            st.markdown(f"#### 2 · {sess.company_name} — analysis ({r.periods[0]} → {p})")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Net margin", fmt_pct(ratios.net_margin))
            k2.metric("Current ratio", fmt_ratio(ratios.current_ratio))
            k3.metric("Debt-to-equity", fmt_ratio(ratios.debt_to_equity))
            k4.metric("Findings", f"{len(r.anomalies)}", delta=f"{len(high)} high severity" if high else "none high",
                      delta_color="inverse" if high else "off")

            for key in sess.chart_order:
                with st.container(border=True):
                    st.plotly_chart(sess.charts[key], key=f"chart_{key}")
                    st.info(sess.captions.get(key, ""), icon="💡")

            st.markdown("#### 3 · Recommended actions")
            with st.container(border=True):
                st.markdown(sess.recommendations or "No specific actions flagged.")
            if sess.doc_names:
                st.caption(f"Notes indexed for Q&A: {', '.join(sess.doc_names)}"
                           + (f" · embeddings: {sess.embedder_name}" if sess.embedder_name else ""))

# ----------------------------------------------------------------------------
# RIGHT: chat
# ----------------------------------------------------------------------------
with right:
    st.markdown("#### 💬 Ask about your documents")
    ready = ss.engine is not None
    box = st.container(height=640, border=True)

    prompt = st.chat_input("Ask a question…" if ready else "Load documents first", disabled=not ready, key="chat_input")
    if ss.pending and ready:
        prompt, ss.pending = ss.pending, None

    def render_message(m):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                cls, label = BADGES.get(m.get("source", "limited"), BADGES["limited"])
                st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)
                if m.get("note"):
                    st.caption(m["note"])
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("citations"):
                    with st.expander("Sources"):
                        for c in m["citations"]:
                            st.markdown(f"- **{c['label']}** — {c['detail']}")
                if m.get("latency") is not None:
                    st.caption(f"⏱ first words in {m['latency']:.1f}s")

    with box:
        if not ss.messages and not prompt:
            if ready:
                st.markdown("**Try asking:**")
                for q in ["What was revenue in the latest period?", "Which margin is falling fastest and why?",
                          "What should we do about our debt?", "What's a healthy current ratio for this industry?"]:
                    if st.button(q, key=f"starter_{q}"):
                        ss.pending = q
                        st.rerun()
            else:
                st.caption("Load a sample company or upload your documents, then ask questions here — "
                           "exact figures, answers grounded in your documents, and web suggestions when the documents don't cover it.")
        for m in ss.messages:
            render_message(m)

        if prompt and ready:
            history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
            ss.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            t0 = time.time()
            reply = ss.engine.ask(prompt, history)
            first = {"t": None}

            def timed():
                for chunk in reply.stream:
                    if first["t"] is None:
                        first["t"] = time.time() - t0
                    yield chunk

            with st.chat_message("assistant"):
                cls, label = BADGES.get(reply.source, BADGES["limited"])
                st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)
                if reply.note:
                    st.caption(reply.note)
                # Materialize the stream so empty/model-error responses are visible and
                # malformed Markdown from generated answers is normalized before rendering.
                try:
                    text = "".join(str(chunk) for chunk in timed()).strip()
                except Exception as exc:
                    text = f"AI response failed ({type(exc).__name__}): {str(exc)[:400]}"
                text = text.replace("∗", "*").replace("\u2217", "*")
                text = re.sub(r"\*\*`?([-−$\d,%.]+)`?\*\*\s*to\s*\*\*`?([-−$\d,%.]+)`?\*\*", r"**\1 to \2**", text, flags=re.IGNORECASE)
                if not text:
                    text = ("FinSight received an empty response from the AI provider. "
                            "Check the terminal output for the underlying API error and verify GROQ_MODEL in .env.")
                st.markdown(text)
                if reply.citations:
                    with st.expander("Sources"):
                        for c in reply.citations:
                            st.markdown(f"- **{c['label']}** — {c['detail']}")
                if first["t"] is not None:
                    st.caption(f"⏱ first words in {first['t']:.1f}s")
            ss.messages.append({"role": "assistant", "content": text, "source": reply.source, "note": reply.note,
                                "citations": reply.citations, "latency": first["t"]})
