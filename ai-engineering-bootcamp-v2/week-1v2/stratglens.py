"""StratgLens — competitive-landscape + product-whitespace agent (deployable).

Architecture (Option A): a supervisor-style fan-out to three specialists, then a
synthesis step that fuses their outputs into two deliverables:
  1. an evidence-backed competitive landscape, and
  2. a ranked list of product whitespace opportunities.

Specialists and their stubbed tools (sample data — no external keys needed):
  - competitor_agent : query_competitor_db + web_research
  - evidence_agent   : retrieve_documents (RAG-style passages with citations)
  - whitespace_agent : analyze_customer_feedback

This module exposes a structured `run_stratglens()` that returns the final report
PLUS a per-agent Think -> Act -> Observe "decision loop", so a FastAPI endpoint and
a Streamlit UI can render the reasoning. Patterns reuse agent.py / demo1_routing.py.

Requires GOOGLE_API_KEY (Gemini). The tools are stubbed, so no other keys are used.
"""

import asyncio
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.errors import ServerError

# Transient failures worth retrying: Gemini overload (503) and dropped/timed-out
# connections mid-stream (httpx read/connect errors).
TRANSIENT_ERRORS = (httpx.TransportError,)

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")
load_dotenv(THIS_DIR.parent / ".env")

MODEL = "gemini-3.5-flash-lite"  # lite = higher free-tier RPD; per-model quota

# The four capabilities, surfaced so a UI can list them (name -> description).
CAPABILITIES = {
    "query_competitor_db": "Structured competitor + pricing database",
    "web_research": "Live web search for recent competitor moves",
    "retrieve_documents": "RAG retrieval over analyst reports / filings (with citations)",
    "analyze_customer_feedback": "Customer-feedback analysis for unmet needs",
}


# =====================================================================
# TOOLS (stubbed sample data — swap each body for a real source later)
# =====================================================================

def query_competitor_db(market: str) -> dict:
    """Query the structured competitor + pricing database for a market."""
    return {
        "market": market,
        "competitors": [
            {"name": "Acme Collab", "positioning": "enterprise", "price": "$18/user/mo",
             "differentiated_on": ["SOC2 + granular admin controls", "native SSO"]},
            {"name": "Flowly", "positioning": "SMB / prosumer", "price": "$8/user/mo",
             "differentiated_on": ["fastest onboarding", "polished mobile app"]},
            {"name": "Tasklytic", "positioning": "developer-first", "price": "$12/user/mo",
             "differentiated_on": ["deep GitHub/Jira integration", "automation API"]},
        ],
    }


def web_research(query: str) -> dict:
    """Run a live web search for recent competitor moves, launches, and news."""
    return {
        "query": query,
        "findings": [
            {"source": "techcrunch.com", "date": "2026-10-28",
             "note": "Flowly raised a $40M Series B; announced AI meeting-notes feature."},
            {"source": "acmecollab.com/blog", "date": "2026-09-15",
             "note": "Acme launched an on-prem tier aimed at regulated industries."},
        ],
    }


def retrieve_documents(query: str) -> dict:
    """RAG retrieval over the document corpus; returns passages WITH citations."""
    return {
        "query": query,
        "passages": [
            {"doc": "Gartner Collab MQ 2026", "page": 12,
             "text": "Buyers increasingly cite administrative control and compliance as "
                     "the primary differentiator at the enterprise tier."},
            {"doc": "Forrester Wave Q3-2026", "page": 7,
             "text": "Mobile experience remains uneven across vendors; only Flowly scores "
                     "above 4/5 on mobile usability."},
        ],
    }


def analyze_customer_feedback(segment: str) -> dict:
    """Analyze customer feedback for a segment; surface unmet needs with frequency."""
    return {
        "segment": segment,
        "unmet_needs": [
            {"need": "offline mode for field teams", "mentions": 128, "sentiment": "frustrated"},
            {"need": "usage-based pricing instead of per-seat", "mentions": 94, "sentiment": "churn-risk"},
            {"need": "built-in client/guest access without extra seats", "mentions": 61, "sentiment": "requested"},
        ],
    }


# =====================================================================
# AGENTS (3 specialists + synthesis)
# =====================================================================

competitor_agent = Agent(
    name="competitor_agent", model=MODEL,
    description="Finds where competitors are differentiated (DB + recent web moves).",
    instruction="You are a competitive-intelligence specialist. Use query_competitor_db for "
                "canonical competitor/pricing facts, then web_research for recent news. "
                "Summarize where each competitor is differentiated.",
    tools=[query_competitor_db, web_research],
)

evidence_agent = Agent(
    name="evidence_agent", model=MODEL,
    description="Retrieves evidence and citations from the document corpus.",
    instruction="You are a research-evidence specialist. Use retrieve_documents to find "
                "supporting passages. Always return the claim AND its citation (doc + page).",
    tools=[retrieve_documents],
)

whitespace_agent = Agent(
    name="whitespace_agent", model=MODEL,
    description="Surfaces underserved customer needs and ranks them as whitespace.",
    instruction="You are a product-whitespace specialist. Use analyze_customer_feedback to find "
                "unmet needs, then rank them by frequency and churn risk.",
    tools=[analyze_customer_feedback],
)

synthesis_agent = Agent(
    name="synthesis_agent", model=MODEL,
    instruction="You are a product-strategy synthesizer. You are given three evidence blocks: "
                "COMPETITOR, EVIDENCE, and WHITESPACE. Produce two sections:\n"
                "1) COMPETITIVE LANDSCAPE — where each competitor is differentiated, every claim "
                "backed by a citation from the EVIDENCE block.\n"
                "2) RANKED WHITESPACE OPPORTUNITIES — underserved needs ranked by opportunity size "
                "and how exposed competitors are on them; one line of rationale each.\n"
                "Do not invent facts beyond the three blocks.",
    tools=[],
)


# =====================================================================
# STRUCTURED RUN (Think -> Act -> Observe per agent)
# =====================================================================

def _truncate(resp, limit: int = 300) -> str:
    try:
        text = json.dumps(resp, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        text = str(resp)
    return text[:limit] + ("…" if len(text) > limit else "")


async def _collect_once(agent, message: str) -> dict:
    """Run one agent once, capturing {think, acts[], output}."""
    service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="stratglens", session_service=service)
    session = await service.create_session(app_name="stratglens", user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=message)])

    think_parts: list[str] = []
    acts: list[dict] = []
    output = ""

    async for event in runner.run_async(
        user_id="user1", session_id=session.id, new_message=content
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                acts.append({"tool": fc.name, "args": dict(fc.args), "observation": None})
            elif getattr(part, "function_response", None):
                fr = part.function_response
                obs = _truncate(fr.response)
                for a in reversed(acts):
                    if a["tool"] == fr.name and a["observation"] is None:
                        a["observation"] = obs
                        break
                else:
                    acts.append({"tool": fr.name, "args": {}, "observation": obs})
            elif getattr(part, "text", None) and part.text.strip():
                if event.is_final_response():
                    output = part.text.strip()
                else:
                    think_parts.append(part.text.strip())

    return {"think": " ".join(think_parts).strip(), "acts": acts, "output": output}


async def _collect(agent, message: str, max_retries: int = 3) -> dict:
    """Run one agent with retry/backoff on transient overloads and network drops."""
    for attempt in range(1, max_retries + 1):
        try:
            return await _collect_once(agent, message)
        except ServerError as exc:
            if exc.code == 503 and attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
        except TRANSIENT_ERRORS:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            raise


async def _run_pipeline(message: str) -> dict:
    """Fan out to the 3 specialists, then synthesize. Returns {final, turns[]}."""
    competitor = await _collect(competitor_agent, f"Analyze competitors in: {message}")
    evidence = await _collect(evidence_agent, f"Find evidence on differentiation in: {message}")
    whitespace = await _collect(whitespace_agent, f"Find unmet customer needs in: {message}")

    blocks = (
        f"TARGET: {message}\n\n"
        f"COMPETITOR:\n{competitor['output']}\n\n"
        f"EVIDENCE:\n{evidence['output']}\n\n"
        f"WHITESPACE:\n{whitespace['output']}"
    )
    synthesis = await _collect(synthesis_agent, blocks)

    turns = [
        {"agent": "competitor_agent", **competitor},
        {"agent": "evidence_agent", **evidence},
        {"agent": "whitespace_agent", **whitespace},
        {"agent": "synthesis_agent", **synthesis},
    ]
    return {"final": synthesis["output"] or "(no synthesis produced)", "turns": turns}


async def run_stratglens(message: str) -> dict:
    """Public entry point: {final, turns[]}.

    Each specialist (_collect) retries transient 503s and dropped connections on
    its own, so a single network blip no longer fails the whole pipeline.
    """
    return await _run_pipeline(message)


# --- CLI for quick local checks ---------------------------------------------

async def _main():
    import sys
    market = " ".join(sys.argv[1:]) or "team collaboration software"
    print(f"TARGET MARKET: {market}\n" + "=" * 60)
    result = await run_stratglens(market)
    for turn in result["turns"]:
        print(f"\n--- {turn['agent']} ---")
        if turn["think"]:
            print(f"THINK: {turn['think'][:160]}")
        for a in turn["acts"]:
            print(f"ACT:   {a['tool']}({a['args']})")
            if a["observation"]:
                print(f"OBS:   {a['observation'][:120]}")
    print("\n" + "=" * 60 + "\nFINAL REPORT:\n")
    print(result["final"])


if __name__ == "__main__":
    asyncio.run(_main())
