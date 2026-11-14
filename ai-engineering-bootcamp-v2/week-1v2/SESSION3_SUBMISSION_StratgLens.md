# Session 3 — StratgLens Multi-Agent Submission

**Repo folder:** `ai-engineering-bootcamp-v2/week-1v2`
**Stack (one word):** **ADK** — Google Agent Development Kit, Gemini `gemini-3.5-flash-lite`
**Live service:** `https://signal-roadmap.onrender.com`  ·  endpoint `POST /stratglens`

StratgLens is my capstone agent: for a target market, it produces an **evidence-backed
competitive landscape** and a **ranked list of product whitespace opportunities** for
Product Managers and Product Strategy leaders. It extends the single-agent Session 3
work (`agent.py`) into a **multi-agent system**.

---

## Agent vs. workflow

> **This is an agent because** each specialist decides at runtime which of its tools to
> call and how to phrase the query, and the supervisor fans work out to the specialists
> the task needs — control flow is driven by the models, not a fixed script. The one
> fixed part is the final synthesis step, by design (see "Honest notes").

---

## 1. Architecture (Option A: 3 specialists + synthesis)

```
Target market
     │
     ▼  supervisor fans out
 ┌──────────────────┬──────────────────┬────────────────────┐
 │ competitor_agent │  evidence_agent  │  whitespace_agent  │
 │ query_competitor │ retrieve_        │ analyze_customer_  │
 │ _db + web_       │ documents (RAG)  │ feedback           │
 │ research         │                  │                    │
 └────────┬─────────┴────────┬─────────┴─────────┬──────────┘
          └──────────────────┴───────────────────┘
                             ▼
                      synthesis_agent  (no tools)
                             ▼
      COMPETITIVE LANDSCAPE  +  RANKED WHITESPACE OPPORTUNITIES
```

| Specialist | Tools | Job |
|------------|-------|-----|
| `competitor_agent` | `query_competitor_db`, `web_research` | Where competitors are differentiated (DB facts + recent web moves) |
| `evidence_agent` | `retrieve_documents` | Supporting passages **with citations** (analyst reports/filings) |
| `whitespace_agent` | `analyze_customer_feedback` | Unmet customer needs, ranked |
| `synthesis_agent` | *(none)* | Fuses the three into the two deliverables |

The four tools return **stubbed sample data** (no external keys) so the decision loop and
synthesis can be demoed instantly; each tool body is a clean swap point for a real source.

**Files:**
- `stratglens.py` — the deployable multi-agent module + structured `run_stratglens()`.
- `main.py` — `POST /stratglens` endpoint (Session 1 `/ask` and Session 3 `/agent` untouched).
- `stratglens_app.py` — Streamlit UI (3-pane: capabilities · report · decision loop).
- `SESSION3_STRATGLENS_RUN_LOG.txt` — the run log below.

---

## 2. Proof — one run, Think → Act → Observe

Target market: **team collaboration software**. Full log:
[`SESSION3_STRATGLENS_RUN_LOG.txt`](./SESSION3_STRATGLENS_RUN_LOG.txt).

Decision loop (abbreviated — every ACT is a real tool call, every OBS its result):

```
--- competitor_agent ---
ACT: query_competitor_db({'market': 'team collaboration software'})
OBS: {... Acme Collab / Flowly / Tasklytic, pricing, differentiators ...}
ACT: web_research({'query': 'team collaboration software competitors news ...'})
OBS: {... Flowly $40M Series B; Acme on-prem tier ...}

--- evidence_agent ---
ACT: retrieve_documents({'query': 'differentiation team collaboration software'})
OBS: {... Gartner Collab MQ 2026 p.12; Forrester Wave Q3-2026 p.7 ...}

--- whitespace_agent ---
ACT: analyze_customer_feedback({'segment': 'team collaboration software'})
OBS: {... offline mode (128), usage-based pricing (94), guest access (61) ...}

--- synthesis_agent ---  (no tools → final report)
```

**Final report (excerpt):**

> ### COMPETITIVE LANDSCAPE
> 1. **Acme Collab** — Enterprise ($18/user/mo). Native SSO, SOC2, granular admin controls,
>    new on-prem tier; admin control & compliance are the primary enterprise differentiator
>    **[Gartner Collab MQ 2026, p. 12]**.
> 2. **Flowly** — SMB ($8/user/mo). Rapid onboarding, AI meeting notes, only vendor scoring
>    >4/5 on mobile usability **[Forrester Wave Q3-2026, p. 7]**.
> 3. **Tasklytic** — Developer-first ($12/user/mo). Deep GitHub/Jira integration, automation API.
>
> ### RANKED WHITESPACE OPPORTUNITIES
> 1. **Offline mode for field teams** (128 mentions) — no competitor offers offline-first.
> 2. **Usage-based pricing** (94 mentions, churn-risk) — all players exposed on per-seat cost.
> 3. **Built-in client/guest access** (61 mentions) — external collaboration overcharged across vendors.

Every landscape claim carries a citation from the EVIDENCE block, and each whitespace item
is ranked with a one-line rationale — exactly the two deliverables the job calls for.

---

## 3. Endpoint + UI

**`POST /stratglens`** returns `{answer, turns[]}`, where each turn carries the agent's
`think`, its `acts[]` (tool + args + truncated observation), and its `output`. No API keys
or environment data appear in the payload; failures surface as the exception type only.

```bash
# Local
curl -s -X POST http://127.0.0.1:8000/stratglens \
  -H "Content-Type: application/json" \
  -d '{"message": "team collaboration software"}'

# Render
curl -s -X POST https://signal-roadmap.onrender.com/stratglens \
  -H "Content-Type: application/json" \
  -d '{"message": "project management tools for agencies"}'
```

**Streamlit UI** (3-pane, like the reference design):

```bash
cd ai-engineering-bootcamp-v2/week-1v2
streamlit run stratglens_app.py        # set sidebar API URL to localhost or Render
```

- **Left:** agent settings + the 4 capabilities (tools) + specialists note.
- **Center:** example markets, input, and the 📈 strategy report.
- **Right:** 🔁 decision loop — per-turn THINK / ACT / OBSERVE cards.

---

## 4. Honest notes / decisions

- **Resilience:** each specialist retries transient `503` (model overload) and dropped
  connections (`httpx` read/connect errors) with backoff, so one blip doesn't fail the run.
  A real 503 mid-test was absorbed this way.
- **Model & quota:** `gemini-2.5-flash` is retired for new keys and `3.8-flash` is often
  overloaded, so StratgLens uses `gemini-3.5-flash-lite` (higher free-tier requests/day).
  Each run makes 4 model calls; on the Google **free tier (20 req/day per model)** a burst
  of runs can hit `429 RESOURCE_EXHAUSTED` — that is an account limit, not a bug, and the
  endpoint returns a clean 502.
- **Why synthesis is fixed (not routed):** the deliverable needs competitor + evidence +
  whitespace *fused*, which pure routing (one query → one specialist) can't do. Option A keeps
  routing honest for the specialists and adds a deterministic synthesis step on top.
- **Tools are stubbed:** the report proves the wiring and synthesis, not real market data;
  each tool body is the swap point for real RAG / DB / web APIs.
