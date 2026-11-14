"""StratgLens — Streamlit UI (3-pane: capabilities | report | decision loop).

Calls POST /stratglens on the FastAPI service. The service runs the multi-agent
pipeline; this page only sends the request and renders the response. No secrets here.

Run:
  #   PowerShell:  $env:API_BASE_URL="http://127.0.0.1:8000"   (or the Render URL)
  streamlit run stratglens_app.py
"""

import os

import httpx
import streamlit as st

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "https://signal-roadmap.onrender.com")

CAPABILITIES = [
    ("🔎 query_competitor_db", "Structured competitor + pricing database"),
    ("🌐 web_research", "Live web search for recent competitor moves"),
    ("📚 retrieve_documents", "RAG retrieval over analyst reports (with citations)"),
    ("💬 analyze_customer_feedback", "Customer-feedback analysis for unmet needs"),
]

EXAMPLES = [
    "team collaboration software",
    "project management tools for agencies",
    "AI note-taking apps for sales teams",
]

st.set_page_config(page_title="StratgLens", layout="wide")

# --- Sidebar ----------------------------------------------------------------
st.sidebar.title("StratgLens")
st.sidebar.caption("Competitive landscape + product whitespace, agent-generated.")

st.sidebar.subheader("Agent settings")
st.sidebar.text_input("Model", "gemini-3.5-flash", disabled=True)
base_url = st.sidebar.text_input("API base URL", DEFAULT_BASE_URL).rstrip("/")
if st.sidebar.button("New analysis"):
    st.session_state.pop("result", None)
    st.rerun()
if st.sidebar.button("Check API health"):
    try:
        r = httpx.get(f"{base_url}/health", timeout=10.0)
        st.sidebar.success(f"Healthy: {r.json()}") if r.status_code == 200 \
            else st.sidebar.error(f"HTTP {r.status_code}")
    except httpx.HTTPError as exc:
        st.sidebar.error(f"Unreachable: {exc}")

st.sidebar.subheader("Capabilities (tools)")
for name, desc in CAPABILITIES:
    st.sidebar.markdown(f"**{name}**  \n<span style='color:gray'>{desc}</span>",
                        unsafe_allow_html=True)

st.sidebar.subheader("Specialists")
st.sidebar.caption(
    "Supervisor fans out to competitor · evidence · whitespace agents, "
    "then a synthesis step fuses the report."
)

# --- Header -----------------------------------------------------------------
st.title("🧭 StratgLens Agent")
st.markdown(
    "*Evidence-backed competitive landscape and ranked product whitespace, "
    "for Product Managers and Product Strategy leaders.*"
)
st.caption(
    "Google ADK · 3 specialists + synthesis · tools: query_competitor_db, "
    "web_research, retrieve_documents, analyze_customer_feedback. "
    "Each specialist runs a Think → Act → Observe loop."
)

# --- Input ------------------------------------------------------------------
st.markdown("**Example markets:**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["market_input"] = ex

market = st.text_input(
    "Target market or strategy question",
    st.session_state.get("market_input", "team collaboration software"),
)
run = st.button("Run analysis", type="primary")

if run:
    if not market.strip():
        st.error("Enter a target market or question.")
    else:
        with st.spinner("Running StratgLens pipeline (3 specialists + synthesis)…"):
            try:
                resp = httpx.post(
                    f"{base_url}/stratglens", json={"message": market}, timeout=240.0
                )
                st.session_state["result"] = (resp.status_code, resp.json())
            except httpx.HTTPError as exc:
                st.session_state["result"] = (0, {"error": str(exc)})
            except ValueError:
                st.session_state["result"] = (resp.status_code, {"error": "non-JSON response"})

# --- Output: report (left) + decision loop (right) --------------------------
result = st.session_state.get("result")
if result:
    status, data = result
    if status != 200 or not isinstance(data, dict) or "answer" not in data:
        st.error(f"HTTP {status}")
        st.json(data)
    else:
        report_col, loop_col = st.columns([3, 2])

        with report_col:
            st.subheader("📈 Strategy report")
            st.markdown(data.get("answer", ""))

        with loop_col:
            st.subheader("🔁 Decision loop")
            turns = data.get("turns", [])
            for i, turn in enumerate(turns, start=1):
                agent = turn.get("agent", f"agent {i}")
                with st.container(border=True):
                    st.markdown(f"**Turn {i} · `{agent}`**")
                    if turn.get("think"):
                        st.markdown("🧠 **THINK**")
                        st.caption(turn["think"][:400])
                    for a in turn.get("acts", []):
                        st.markdown(f"🛠️ **ACT** → `{a.get('tool')}`")
                        if a.get("args"):
                            st.code(str(a["args"]), language="json")
                        if a.get("observation"):
                            with st.expander("👁️ OBSERVE"):
                                st.code(a["observation"], language="json")
                    if not turn.get("acts") and turn.get("output"):
                        st.caption("Synthesis step (no tools) — produced the final report.")

        with st.expander("Raw JSON"):
            st.json(data)
