import math
import re
import pandas as pd
import requests
import streamlit as st

API = "http://127.0.0.1:8000"
st.set_page_config(page_title="Enterprise RAG", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background:
  radial-gradient(900px 500px at 10% -10%, rgba(99,102,241,.35), transparent 60%),
  radial-gradient(800px 500px at 100% 0%, rgba(236,72,153,.22), transparent 55%),
  #070a14; }
.block-container { padding-top: 1.5rem; max-width: 1150px; }
header[data-testid="stHeader"] { background: transparent; }
.hero { padding: 1.2rem 0 .4rem 0; }
.hero h1 { font-size: 3.2rem; font-weight: 800; letter-spacing: -1px; margin: 0;
  background: linear-gradient(90deg,#60a5fa,#a78bfa 50%,#f472b6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color:#94a3b8; font-size:1.1rem; margin:.4rem 0 0 0; }
.pipe { display:flex; gap:10px; margin: 1rem 0 .4rem 0; flex-wrap: wrap; }
.step { flex:1; min-width:150px; padding:12px 14px; border-radius:14px;
  background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09); }
.step b { color:#e2e8f0; font-size:.92rem; } .step span { display:block; color:#94a3b8; font-size:.78rem; margin-top:2px; }
.step .n { display:inline-block; width:22px; height:22px; line-height:22px; text-align:center; border-radius:50%;
  background: linear-gradient(135deg,#3b82f6,#8b5cf6); color:white; font-size:.75rem; margin-right:6px; }
.answer { padding: 1.3rem 1.5rem; border-radius: 18px; font-size: 1.08rem; line-height: 1.7; color:#f1f5f9;
  background: linear-gradient(135deg, rgba(59,130,246,.14), rgba(139,92,246,.14));
  border: 1px solid rgba(139,92,246,.35); box-shadow: 0 10px 40px rgba(99,102,241,.15); }
.cite { display:inline-block; padding:1px 10px; margin:0 2px; border-radius:999px; font-size:.78rem;
  background: rgba(52,211,153,.15); color:#6ee7b7; border:1px solid rgba(52,211,153,.4); }
.badge { display:inline-block; padding: 2px 11px; border-radius:999px; font-size:.75rem;
  background: rgba(96,165,250,.15); color:#93c5fd; border:1px solid rgba(96,165,250,.35); margin-left:6px; }
.q { color:#cbd5e1; font-size:1.05rem; margin: 1rem 0 .5rem 0; }
.bar { height:6px; border-radius:6px; background: rgba(255,255,255,.1); overflow:hidden; margin:8px 0 2px 0; }
.bar > div { height:100%; background: linear-gradient(90deg,#34d399,#60a5fa,#a78bfa); }
.small { color:#94a3b8; font-size:.8rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.035);
  border:1px solid rgba(255,255,255,.09) !important; border-radius:16px; }
[data-testid="stMetric"] { background: rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.08);
  padding:12px 16px; border-radius:14px; }
[data-testid="stMetricValue"] { font-weight:800; }
[data-testid="stSidebar"] { background:#080c1a; border-right:1px solid rgba(255,255,255,.06); }
.stButton > button { border-radius:12px; border:1px solid rgba(255,255,255,.14); transition: all .15s; }
.stButton > button:hover { border-color:#818cf8; transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: linear-gradient(90deg,#3b82f6,#8b5cf6,#ec4899);
  border:none; color:white; font-weight:700; padding:.55rem 1.6rem; }
.stTextInput input { border-radius:14px; padding:14px 16px; font-size:1.05rem; }
button[data-baseweb="tab"] { font-weight:600; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

EXAMPLES = [
    "What is the maximum input voltage of the LM317?",
    "Which GPIO pin is connected to the user LED on the Pico?",
    "What does HTTP status code 404 mean?",
]
CITE = re.compile(r"\[([^\[\]]+?\.(?:pdf|txt)),\s*p?(\d+)\]", re.I)


def set_q(text):
    st.session_state.q = text


def fmt_answer(text: str) -> str:
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = CITE.sub(r"<span class='cite'>📄 \1 · p\2</span>", text)
    return text.replace("\n", "<br>")


def get_metrics():
    try:
        return requests.get(f"{API}/metrics", timeout=5).json()
    except Exception:
        return None


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 📊 Live observability")
    st.button("Refresh", use_container_width=True)
    m = get_metrics()
    if m is None:
        st.error("Backend not reachable")
    elif not m.get("requests"):
        st.info("No queries yet.")
    else:
        a, b = st.columns(2)
        a.metric("Requests", m["requests"])
        b.metric("Cost", f"${m['total_est_cost_usd']:.4f}")
        a.metric("p50", f"{m['p50_ms']} ms")
        b.metric("p95", f"{m['p95_ms']} ms")
        st.caption(f"Tokens in/out: {m['total_input_tokens']} / {m['total_output_tokens']}")

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
  <h1>Enterprise RAG</h1>
  <p>Ask your documents anything. Hybrid retrieval, cross-encoder reranking, cited answers.</p>
</div>
<div class="pipe">
  <div class="step"><b><span class="n">1</span>Ingest</b><span>Parse and chunk PDFs</span></div>
  <div class="step"><b><span class="n">2</span>Hybrid search</b><span>BM25 + vectors, RRF</span></div>
  <div class="step"><b><span class="n">3</span>Rerank</b><span>Cross-encoder top 5</span></div>
  <div class="step"><b><span class="n">4</span>Generate</b><span>Gemini with citations</span></div>
</div>
""", unsafe_allow_html=True)

tab_ask, tab_dash = st.tabs(["💬 Ask", "📈 Dashboard"])

# ---------------- Ask tab ----------------
with tab_ask:
    question = st.text_input("Question", key="q", label_visibility="collapsed",
                             placeholder="Ask something about your documents...")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        col.button(ex, on_click=set_q, args=(ex,), use_container_width=True)
    ask = st.button("✨ Ask", type="primary")

    if ask and question.strip():
        with st.spinner("Searching, reranking and generating..."):
            try:
                r = requests.post(f"{API}/query", json={"question": question}, timeout=120)
            except Exception as e:
                st.error(f"Cannot reach backend: {e}")
                st.stop()
        if r.status_code != 200:
            st.error(f"Backend error: {r.text}")
        else:
            st.session_state.history.insert(0, {"q": question, "d": r.json()})

    if st.session_state.history:
        item = st.session_state.history[0]
        d = item["d"]
        st.markdown(f"<div class='q'>🙋 {item['q']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer'>{fmt_answer(d['answer'])}</div>", unsafe_allow_html=True)
        t = d["timings_ms"]
        st.write("")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", f"{t['total_ms']} ms")
        c2.metric("Retrieval", f"{t['retrieval_ms']} ms")
        c3.metric("Rerank", f"{t['rerank_ms']} ms")
        c4.metric("Generation", f"{t['generation_ms']} ms")
        c5.metric("Tokens", d["tokens"]["input"] + d["tokens"]["output"])
        st.caption(f"Model: {d['model']}")

        st.markdown("#### 📚 Sources")
        for s in d["sources"]:
            rel = 1 / (1 + math.exp(-s["score"]))
            with st.container(border=True):
                st.markdown(f"**📄 {s['filename']}** <span class='badge'>page {s['page']}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='bar'><div style='width:{rel*100:.0f}%'></div></div>"
                            f"<span class='small'>relevance {rel*100:.0f}%</span>", unsafe_allow_html=True)
                with st.expander("Show passage"):
                    st.text(s["text"])
    else:
        st.info("Type a question or click an example to get started.")

# ---------------- Dashboard tab ----------------
with tab_dash:
    m = get_metrics()
    if not m or not m.get("requests"):
        st.info("Ask a few questions first, then come back here.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total requests", m["requests"])
        c2.metric("p50 latency", f"{m['p50_ms']} ms")
        c3.metric("p95 latency", f"{m['p95_ms']} ms")
        c4.metric("Est. cost", f"${m['total_est_cost_usd']:.4f}")

        st.markdown("#### Average time per stage (ms)")
        st.bar_chart(pd.DataFrame({
            "stage": ["Retrieval", "Rerank", "Generation"],
            "ms": [m["avg_retrieval_ms"], m["avg_rerank_ms"], m["avg_generation_ms"]],
        }).set_index("stage"))

        if st.session_state.history:
            st.markdown("#### Latency per question (this session)")
            rows = [{"question": h["q"][:40], **{k: v for k, v in h["d"]["timings_ms"].items() if k != "total_ms"}}
                    for h in reversed(st.session_state.history)]
            df = pd.DataFrame(rows).set_index("question")
            st.bar_chart(df)
            st.markdown("#### Earlier answers")
            for h in st.session_state.history:
                with st.expander(h["q"]):
                    st.markdown(h["d"]["answer"])