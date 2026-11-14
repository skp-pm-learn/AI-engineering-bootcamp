# Session 3 — Agent Assignment Submission

**Repo folder:** `ai-engineering-bootcamp-v2/week-1v2`
**Stack (one word):** **ADK** — Google Agent Development Kit, Gemini `gemini-3.5-flash`

I built a single-agent ADK system on top of my Session 1/2 FastAPI + Pinecone RAG
capstone. The agent answers Northwind Robotics policy questions, grounded only in
documents it retrieves at runtime through **one real tool** that reuses my Session 2
retrieval.

---

## Agent vs. workflow

> **This is an agent because** the model itself decides *whether* and *how* to call
> the `search_policies` tool at runtime — it rewrites the question into a retrieval
> query, reads the tool's observation, and could loop or refine — rather than running
> a fixed, pre-wired retrieve-then-generate sequence.
>
> (It would be a *workflow* if the code always called retrieval then generation in a
> hardcoded order. Here the LLM drives control flow — that is the agent distinction.)

---

## 1. What I built

**One root agent, one real tool, no router** (the job is a single multi-step task, so
multi-agent routing is not needed yet).

| Piece | Detail |
|-------|--------|
| **Agent** | `stratglens_policy_agent` (Gemini `gemini-3.5-flash`). Instruction states the goal, constraints (answer only from retrieved passages, cite `document_id`, refuse when unsupported), and what "done" looks like. |
| **Tool** | `search_policies(query, top_k)` — a thin wrapper over `vectorstore.query()`, my Session 2 Pinecone retrieval (cosine, 1536-dim, `text-embedding-3-small`). Returns passages with `document_id`, `source`, and `score` so the agent can cite sources. |
| **Loop control** | Think → Act → Observe logging over the ADK event stream, plus a hard `MAX_TOOL_CALLS = 5` cap so the loop cannot run forever. |
| **Resilience** | Retries transient `503 UNAVAILABLE` (model overload) with exponential backoff; tool errors are returned as `{status: ...}` observations the model can see, never an uncaught crash. |

**Files:**

- `agent.py` — the ADK agent (patterns copied from the sample `adk-multi-agent-systems/demo1_routing.py`: `Agent`, `Runner`, `InMemorySessionService`, the async run loop).
- `vectorstore.py` — reused unchanged from Session 2 (the tool body).
- `SESSION3_RUN_LOG.txt` — saved run log (below).

---

## 2. Proof — one task that *must* call the tool

The question asks for a private, company-specific figure that exists only in my
ingested corpus (£350 every 36 months). No LLM can answer it from memory, so a
correct, cited answer is proof the tool actually executed.

```
USER: How much does Northwind reimburse for home office equipment, and how often?
----------------------------------------------------------------------
  [ACT]     call search_policies({'query': 'home office equipment reimbursement amount frequency'})
  [OBSERVE] search_policies -> status=ok
  [ANSWER]  Northwind reimburses up to £350 (or local equivalent) once every 36 months
            for approved home office equipment (POL-101). Claims are processed through
            the expenses policy (POL-101, POL-114).
```

**Think → Act → Observe mapping** (ADK streams events; the labels map to the three words):

| Phase | ADK event | Evidence in the log |
|-------|-----------|---------------------|
| **THINK** | model plans the call | reformulated the question into the query `"home office equipment reimbursement amount frequency"` (reasoned, not echoed) |
| **ACT** | `function_call` | `[ACT] call search_policies(...)` |
| **OBSERVE** | `function_response` | `[OBSERVE] search_policies -> status=ok` (real passages from Pinecone, 65-vector index) |
| **ANSWER** | final `text` | grounded answer citing POL-101 and POL-114 |

Full log: [`SESSION3_RUN_LOG.txt`](./SESSION3_RUN_LOG.txt).

---

## 3. How to run

```bash
cd ai-engineering-bootcamp-v2/week-1v2
# .env needs GOOGLE_API_KEY (Gemini), plus OPENAI_API_KEY + PINECONE_API_KEY
#   (the agent reasons with Gemini; retrieval embeds with OpenAI and searches Pinecone)
pip install google-adk

python agent.py "How much does Northwind reimburse for home office equipment, and how often?"
python agent.py "What is the remote work policy?"   # any Northwind policy question
```

The agent prints each `[ACT]` / `[OBSERVE]` / `[ANSWER]` step as it runs.

---

## 4. Notes / decisions

- **Dual-provider by design:** Gemini drives the agent; OpenAI embeddings + Pinecone
  power retrieval. This reuses the exact RAG stack from Session 2 rather than
  rebuilding it.
- **Model choice:** the sample demo's `gemini-2.5-flash` is retired for new API keys
  (404), and `gemini-3.8-flash` was frequently overloaded (503), so I settled on the
  stable `gemini-3.5-flash` and added 503 retry/backoff.
- **Why no router yet:** the assignment is one multi-step job. A 3-agent Option-A
  structure (competitor / evidence / whitespace specialists + a synthesis step) is
  sketched for the StratgLens capstone direction, but single-agent is the smallest
  thing that satisfies "one ADK agent + one real tool."
