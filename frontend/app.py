import math
import requests
import streamlit as st

API = "http://127.0.0.1:8000"
st.set_page_config(page_title="Enterprise RAG", page_icon="📚", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #1e2a5a 0%, #0b1020 45%, #070a14 100%); }
.block-container { padding-top: 2rem; max-width: 1100px; }
.hero h1 { font-size: 2.8rem; font-weight: 800; margin-bottom: 0;
  background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #94a3b8; font-size: 1.05rem; margin-top: .3rem; }
.badge { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: .78rem;
  background: rgba(96,165,250,.15); color:#93c5fd; border:1px solid rgba(96,165,250,.35); margin-right:6px; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.10) !important; border-radius: 16px; }
[data-testid="stMetric"] { background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08);
  padding: 12px 14px; border-radius: 12px; }
[data-testid="stSidebar"] { background: #0a0f1e; border-right: 1px solid rgba(255,255,255,.06); }
.stButton > button { border-radius: 10px; border: 1px solid rgba(255,255,255,.15); }
.stButton > button[kind="primary"] { background: linear-gradient(90deg,#3b82f6,#8b5cf6); border: none; color: white; font-weight: 600; }
.src-title { font-weight: 600; color: #e2e8f0; }
.bar { height: 6px; border-radius: 6px; background: rgba(255,255,255,.1); overflow: hidden; margin: 6px 0 2px 0; }
.bar > div { height: 100%; background: linear-gradient(90deg,#34d399,#60a5fa); }
.small { color:#94a3b8; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

EXAMPLES = [
    "What is the maximum input voltage of the LM317?",
    "Which GPIO pin is connected to the user LED on the Pico?",
    "What does HTTP status code 404 mean?",
]


def set_question(text):
    st.session_state.q = text


# ---------- Sidebar: observability ----------
with st.sidebar:
    st.markdown("### 📊 Observability")
    st.button("Refresh metrics")
    try:
        m = requests.get(f"{API}/metrics", timeout=5).json()
        if m.get("requests"):
            c1, c2 = st.columns(2)
            c1.metric("Requests", m["requests"])
            c2.metric("Cost (est.)", f"${m['total_est_cost_usd']}")
            c1.metric("p50 ms", m["p50_ms"])
            c2.metric("p95 ms", m["p95_ms"])
            st.caption(f"Tokens in/out: {m['total_input_tokens']} / {m['total_output_tokens']}")
            st.caption(f"Avg ms - retrieval {m['avg_retrieval_ms']}, rerank {m['avg_rerank_ms']}, "
                       f"generation {m['avg_generation_ms']}")
        else:
            st.info("No queries yet.")
    except Exception:
        st.error("Backend not reachable")

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <h1>Enterprise RAG</h1>
  <p>Ask your documents anything. Hybrid search, reranking and cited answers.</p>
  <span class="badge">BM25 + Dense</span><span class="badge">RRF Fusion</span>
  <span class="badge">Cross-Encoder</span><span class="badge">Cited Sources</span>
</div>
""", unsafe_allow_html=True)
st.write("")

# ---------- Question box ----------
question = st.text_input("Your question", key="q", placeholder="e.g. What is the maximum input voltage of the LM317?",
                         label_visibility="collapsed")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, on_click=set_question, args=(ex,), use_container_width=True)

ask = st.button("🔍 Ask", type="primary")

# ---------- Run query ----------
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

# ---------- Show latest result ----------
if st.session_state.history:
    item = st.session_state.history[0]
    d = item["d"]
    st.markdown(f"#### 💬 {item['q']}")
    with st.container(border=True):
        st.markdown(d["answer"])
    t = d["timings_ms"]
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
            st.markdown(f"<span class='src-title'>📄 {s['filename']}</span> "
                        f"<span class='badge'>page {s['page']}</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='bar'><div style='width:{rel*100:.0f}%'></div></div>"
                        f"<span class='small'>relevance {rel*100:.0f}%</span>", unsafe_allow_html=True)
            with st.expander("Show passage"):
                st.text(s["text"])

    if len(st.session_state.history) > 1:
        st.markdown("#### 🕘 Earlier questions")
        for old in st.session_state.history[1:]:
            with st.expander(old["q"]):
                st.markdown(old["d"]["answer"])
else:
    st.info("Type a question or click an example above to get started.")
    