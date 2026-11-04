# Session 2 — RAG Assignment Submission

**Live service:** https://signal-roadmap.onrender.com
**Repo folder:** `ai-engineering-bootcamp-v2/week-1v2`

I extended my Session 1 typed FastAPI `/ask` service into a retrieval-augmented
generation (RAG) service over a sample corporate policy corpus, added a Streamlit
UI, evaluated it with a golden set, and implemented one add-on (chunk-size
comparison).

---

## 1. What I built

**Architecture — the API is the source of truth; nothing re-implements RAG in the UI.**

- **Vector store:** Pinecone serverless (cosine, 1536-dim), configured entirely
  from environment variables — no secrets in code.
- **Embeddings:** OpenAI `text-embedding-3-small`, used identically at ingest and
  query time so stored and query vectors share one space.
- **Chunking:** LangChain `RecursiveCharacterTextSplitter` (default 800 / 100,
  configurable per request).

**Endpoints (`main.py`):**

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ingest` | Chunk → embed → upsert a document (metadata: `document_id`, `chunk_index`, `source`). Returns `{document_id, chunks_indexed, status}`. Empty input → 400. |
| POST | `/ask` | Embed question → retrieve top-k → ground the prompt → generate. Returns the answer plus `retrieved_chunk_ids`, `tokens_used`, `cost_usd`. |
| GET | `/debug/retrieve?q=` | Retrieval-only view (top-k chunks + scores), no LLM call. |
| GET | `/debug/pinecone` | Confirms Pinecone is configured and reachable. |
| GET | `/health` | Liveness check. |

**Grounding prompt** (answer only from context, cite `document_id`, refuse if
insufficient) — full template in `main.py` (`GROUNDING_INSTRUCTIONS`).

**Streamlit UI (`rag_app.py`):** an Ingest tab and an Ask tab. The Ask tab shows
a clear grounded-vs-refusal banner plus citations (document IDs, retrieved chunk
IDs, scores). API base URL comes from the `API_BASE_URL` env var or the sidebar —
no secrets in the UI.

**Corpus:** the Northwind Robotics sample policy pack (6 docs → 65 chunks),
ingested with stable policy IDs as `document_id` (POL-101, POL-114, POL-207,
SPEC-WB9, POL-220, POL-118).

---

## 2. Basic assignment — proof

**Grounded answer (cites the source):**

> Q: *"How many days per week can I work remotely?"*
> A: *"Employees may work remotely up to three days per week (POL-101)."*
> `confidence: 1.0`, `sources_needed: false`, retrieved chunks all from POL-101.

**Refusal (fact not in the corpus):**

> Q: *"What is the parental leave policy?"*
> A: *"I cannot answer from the available Northwind policies."*
> `confidence: 0.1`, `sources_needed: true`.

Full curl commands and raw JSON responses are in
[`RAG_SUBMISSION.md`](./RAG_SUBMISSION.md).

### Golden-set eval

Run against the live service, `top_k=5`. Retrieval hit = expected doc in top-5;
faithfulness = answer supported by retrieved text; correctness = matches expected.

| # | Question | Expected answer | Expected doc | Hit? | Faithful? | Correct? |
|---|----------|-----------------|--------------|:---:|:---:|:---:|
| 1 | How many remote days are allowed? | Up to 3 days/week | POL-101 | ✅ | ✅ | ✅ |
| 2 | What is the mileage rate over 50 miles? | 45p per mile | POL-114 | ✅ | ✅ | ✅ |
| 3 | How fast must a lost laptop be reported? | Within 1 hour | POL-207 | ✅ | ✅ | ✅ |
| 4 | What is the WB-9 payload limit? | 25 kg | SPEC-WB9 | ✅ | ✅ | ✅ |
| 5 | What is the minimum password length? | ≥ 14 characters | POL-207 | ✅ | ✅ | ✅ |
| 6 | What is the parental leave policy? *(refusal)* | *Not in corpus → refuse* | — | n/a | ✅ | ✅ |

**Result:** 5/5 grounded questions pass retrieval, faithfulness, and correctness;
the out-of-corpus question correctly refuses.

---

## 3. Add-on — chunk-size comparison

**Why this add-on:** the corpus is short, fact-dense policy text and the queries
are pinpoint lookups, so chunk size is the highest-leverage retrieval knob — and
800/100 was never tuned. It needs no new dependencies or secrets and produces a
clean quantitative before/after. (Script: `chunk_size_eval.py`.)

**Method:** ingest the corpus at three configs into **isolated Pinecone
namespaces** (production index untouched), then score the golden set by the rank
and top-1 similarity of the first chunk from the expected document.

**Before / after — rank (#) and top-1 score of the correct chunk:**

| Question | Expected | 800/100 *(baseline)* | 400/80 | 250/50 |
|----------|----------|:---:|:---:|:---:|
| Remote days | POL-101 | #1 (0.597) | #1 (0.685) | #1 (0.701) |
| Mileage rate | POL-114 | #1 (0.483) | #1 (0.559) | #1 (0.642) |
| Lost laptop | POL-207 | #1 (0.543) | #1 (0.522) | #1 (0.677) |
| WB-9 payload | SPEC-WB9 | #1 (0.648) | #1 (0.735) | #1 (0.735) |
| Password length | POL-207 | #1 (0.441) | #1 (0.574) | #1 (0.614) |

**Findings:**

- Every config already retrieves the correct document at **rank #1** (the corpus
  is small), so nothing *misses*.
- **Smaller chunks raise the top-1 similarity score on every question**
  (mileage +0.16, password +0.17). A higher score means a larger margin between
  the relevant chunk and the noise, which tightens the top-5 and makes the
  refusal threshold more reliable.
- **Trade-off:** 250/50 produces ~4× more chunks (257 vs 65); 400/80 captures
  most of the gain at ~2.3× chunks.

**Recommendation:** adopt **250/50** for best precision, or **400/80** as the
frugal middle ground — both beat the current 800/100 for this fact-lookup
workload.

---

## 4. How to run

```bash
# API (from week-1v2, with OPENAI_API_KEY and PINECONE_API_KEY in .env)
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Streamlit UI (points at live service by default)
streamlit run rag_app.py
```

On Render, the same env vars are set in the service's Environment tab; Pinecone
is managed, so the index persists across redeploys.
