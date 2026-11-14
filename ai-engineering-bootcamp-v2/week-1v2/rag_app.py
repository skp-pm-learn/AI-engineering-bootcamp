"""Minimal Streamlit UI for the Week 1 v2 RAG service (/ingest + /ask).

The FastAPI service is the source of truth: this UI only sends HTTP requests
and renders the JSON. No embedding, retrieval, or generation happens here.

Run:
  # point at your live service (preferred: env var, no hardcoded secrets)
  #   PowerShell:  $env:API_BASE_URL="https://signal-roadmap.onrender.com"
  streamlit run rag_app.py

The API base URL can also be set/overridden in the sidebar at runtime.
"""

import os

import httpx
import streamlit as st

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "https://signal-roadmap.onrender.com")


def call_json(method: str, url: str, payload: dict | None = None):
    """Thin HTTP wrapper. Returns (status_code, data) or (0, {"error": ...})."""
    try:
        if method == "POST":
            response = httpx.post(url, json=payload, timeout=180.0)
        else:
            response = httpx.get(url, timeout=10.0)
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, response.text
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


st.set_page_config(page_title="Northwind RAG", layout="centered")
st.title("Northwind RAG — /ingest + /ask")
st.caption("The FastAPI service does all RAG work. This page is just a client.")

# --- Sidebar: where the API lives (not a secret; keys stay on the server) ---
base_url = st.sidebar.text_input("API base URL", DEFAULT_BASE_URL).rstrip("/")
st.sidebar.caption("Set API_BASE_URL env var or edit here. No API keys needed in the UI.")
if st.sidebar.button("Check API health"):
    status, data = call_json("GET", f"{base_url}/health")
    if status == 200:
        st.sidebar.success(f"Healthy: {data}")
    else:
        st.sidebar.error(f"Unreachable (HTTP {status}): {data}")

ingest_tab, ask_tab, agent_tab = st.tabs(["Ingest", "Ask", "Agent"])

# --- Ingest -----------------------------------------------------------------
with ingest_tab:
    st.subheader("Ingest a document")
    with st.form("ingest_form"):
        document_id = st.text_input("document_id", "my-doc-1")
        source = st.text_input("source (optional)", "")
        text = st.text_area("text", height=200, placeholder="Paste document text here...")
        ingest_submitted = st.form_submit_button("Ingest", type="primary")

    if ingest_submitted:
        if not text.strip() or not document_id.strip():
            st.error("Both text and document_id are required.")
        else:
            payload = {"text": text, "document_id": document_id}
            if source.strip():
                payload["source"] = source.strip()
            with st.spinner("Calling /ingest..."):
                status, data = call_json("POST", f"{base_url}/ingest", payload)
            if status == 200:
                st.success(
                    f"Indexed **{data.get('chunks_indexed')}** chunks "
                    f"for document_id **{data.get('document_id')}** "
                    f"(status: {data.get('status')})."
                )
            else:
                st.error(f"HTTP {status}")
            st.json(data)

# --- Ask --------------------------------------------------------------------
with ask_tab:
    st.subheader("Ask a question")
    with st.form("ask_form"):
        question = st.text_area(
            "Question",
            "How many days per week can I work remotely?",
            height=90,
        )
        top_k = st.slider("top_k", min_value=1, max_value=10, value=5)
        ask_submitted = st.form_submit_button("Ask", type="primary")

    if ask_submitted:
        if not question.strip():
            st.error("Question is required.")
        else:
            with st.spinner("Calling /ask..."):
                status, data = call_json(
                    "POST", f"{base_url}/ask", {"question": question, "top_k": top_k}
                )

            if status != 200 or not isinstance(data, dict) or "answer" not in data:
                st.error(f"HTTP {status}")
                st.json(data)
            else:
                answer = data.get("answer", {})
                text_answer = answer.get("answer", "")
                confidence = answer.get("confidence")
                sources_needed = answer.get("sources_needed")

                # Clear refusal vs grounded-answer signal.
                if sources_needed:
                    st.warning(
                        "⚠️ **Refusal / insufficient context** — the service could "
                        "not answer from the ingested documents."
                    )
                else:
                    st.success("✅ **Answered from the documents**")

                st.markdown("### Answer")
                st.write(text_answer)
                st.caption(
                    f"confidence: {confidence} | sources_needed: {sources_needed}"
                )

                # Citations: unique document_ids used + the retrieved chunks.
                retrieved = data.get("retrieved", [])
                cited_docs = sorted(
                    {r.get("document_id") for r in retrieved if r.get("document_id")}
                )
                st.markdown("### Citations")
                if cited_docs:
                    st.markdown(
                        "**Documents retrieved:** "
                        + " ".join(f"`{d}`" for d in cited_docs)
                    )
                if data.get("retrieved_chunk_ids"):
                    st.markdown("**Retrieved chunk IDs:**")
                    st.code("\n".join(data["retrieved_chunk_ids"]))
                if retrieved:
                    st.markdown("**Scores:**")
                    st.table(
                        [
                            {
                                "chunk_id": r.get("id"),
                                "document_id": r.get("document_id"),
                                "source": r.get("source"),
                                "score": round(r.get("score", 0), 3),
                            }
                            for r in retrieved
                        ]
                    )

                # Runtime metadata (carried through from Session 1).
                cols = st.columns(4)
                cols[0].metric("Model", str(data.get("model", "-")))
                cols[1].metric("Tokens", str(data.get("tokens_used", "-")))
                cols[2].metric("Latency", f"{data.get('latency_ms', '-')} ms")
                cols[3].metric("Cost", f"${data.get('cost_usd', '-')}")

                with st.expander("Raw JSON"):
                    st.json(data)

# --- Agent ------------------------------------------------------------------
with agent_tab:
    st.subheader("Ask the ADK agent")
    st.caption(
        "Calls POST /agent. The Gemini agent decides when to use its "
        "search_policies tool, then answers with citations. Needs GOOGLE_API_KEY "
        "set on the API server."
    )
    with st.form("agent_form"):
        goal = st.text_area(
            "Goal / message",
            "How much does Northwind reimburse for home office equipment, and how often?",
            height=90,
        )
        agent_submitted = st.form_submit_button("Run agent", type="primary")

    if agent_submitted:
        if not goal.strip():
            st.error("A goal / message is required.")
        else:
            with st.spinner("Calling /agent..."):
                status, data = call_json("POST", f"{base_url}/agent", {"message": goal})

            if status != 200 or not isinstance(data, dict) or "answer" not in data:
                st.error(f"HTTP {status}")
                st.json(data)
            else:
                if data.get("stopped"):
                    st.warning("⚠️ Stopped — the agent hit its step limit.")
                else:
                    st.success("✅ Agent completed")

                st.markdown("### Answer")
                st.write(data.get("answer", ""))

                steps = data.get("steps", [])
                st.markdown(f"### Steps ({len(steps)})")
                if not steps:
                    st.caption("The agent answered without calling any tool.")
                for i, step in enumerate(steps, start=1):
                    st.markdown(f"**{i}. Act → `{step.get('tool')}`**")
                    obs = step.get("observation")
                    if obs:
                        st.markdown("Observe:")
                        st.code(obs, language="json")

                with st.expander("Raw JSON"):
                    st.json(data)
