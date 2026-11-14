"""
StratgLens — Session 3 capstone (Option A: 3-way routing + final synthesis)

Built on the structure of demo1_routing.py:
  - 3 specialist agents (routed to, one per query)  -> proves routing
  - 4 stubbed tools grouped across the 3 specialists -> proves tool attachment
  - a synthesis_agent (no tools) that fuses the three outputs into the
    two real deliverables: a competitive landscape + a ranked whitespace list

Everything returns hardcoded data for now, so routing + synthesis can be
verified before wiring in real RAG / DB / web APIs.

Run: python stratglens_routing.py
"""

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

MODEL = "gemini-2.5-flash"

# =====================================================================
# TOOLS (4 stubs — swap each body for the real source later)
# =====================================================================

# --- competitor_agent's tools ---

def query_competitor_db(market: str) -> dict:
    """Query the structured competitor + pricing database for a market.

    Returns each known competitor with their positioning, headline price,
    and the one or two features they are differentiated on.
    """
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
    """Run a live web search for recent competitor moves, launches, and news.

    Use to enrich the structured DB with things that happened too recently
    to be in it (funding, launches, pricing changes, acquisitions).
    """
    return {
        "query": query,
        "findings": [
            {"source": "techcrunch.com", "date": "2026-10-28",
             "note": "Flowly raised a $40M Series B; announced AI meeting-notes feature."},
            {"source": "acmecollab.com/blog", "date": "2026-09-15",
             "note": "Acme launched an on-prem deployment tier aimed at regulated industries."},
        ],
    }

# --- evidence_agent's tool (RAG) ---

def retrieve_documents(query: str) -> dict:
    """Semantic retrieval over the document corpus (analyst reports, filings,
    product docs, G2/Gartner excerpts). Returns passages WITH citations so
    every landscape claim can be evidence-backed.
    """
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

# --- whitespace_agent's tool ---

def analyze_customer_feedback(segment: str) -> dict:
    """Analyze customer feedback (reviews, support tickets, interviews) for a
    segment and surface recurring UNMET needs, each with a frequency signal.
    """
    return {
        "segment": segment,
        "unmet_needs": [
            {"need": "offline mode for field teams", "mentions": 128, "sentiment": "frustrated"},
            {"need": "usage-based pricing instead of per-seat", "mentions": 94, "sentiment": "churn-risk"},
            {"need": "built-in client/guest access without extra seats", "mentions": 61, "sentiment": "requested"},
        ],
    }

# =====================================================================
# SPECIALIST AGENTS (3) — name + description drive the router's choice
# =====================================================================

competitor_agent = Agent(
    name="competitor_agent", model=MODEL,
    description="Finds where competitors are differentiated — features and pricing from "
                "the structured database, plus recent moves from the web.",
    instruction="You are a competitive-intelligence specialist. Use query_competitor_db for "
                "canonical competitor/pricing facts, then web_research for recent news. "
                "Summarize where each competitor is differentiated.",
    tools=[query_competitor_db, web_research],
)

evidence_agent = Agent(
    name="evidence_agent", model=MODEL,
    description="Retrieves evidence and citations from the document corpus (analyst reports, "
                "filings, product docs) to back up landscape claims.",
    instruction="You are a research-evidence specialist. Use retrieve_documents to find "
                "supporting passages. Always return the claim AND its citation (doc + page).",
    tools=[retrieve_documents],
)

whitespace_agent = Agent(
    name="whitespace_agent", model=MODEL,
    description="Surfaces underserved and unmet customer needs from feedback, and ranks them "
                "as product whitespace opportunities.",
    instruction="You are a product-whitespace specialist. Use analyze_customer_feedback to find "
                "unmet needs, then rank them by frequency and churn risk.",
    tools=[analyze_customer_feedback],
)

# =====================================================================
# ROUTER — same pattern as demo1's root_agent (one query -> one specialist)
# =====================================================================

stratglens_router = Agent(
    name="stratglens_router", model=MODEL,
    instruction="Route the user's question to exactly one specialist: competitor_agent "
                "(competitor features/pricing/news), evidence_agent (citations from documents), "
                "or whitespace_agent (unmet customer needs). Never answer directly.",
    sub_agents=[competitor_agent, evidence_agent, whitespace_agent],
)

# =====================================================================
# SYNTHESIS — no tools; fuses the 3 specialists' outputs (Option A)
# =====================================================================

synthesis_agent = Agent(
    name="synthesis_agent", model=MODEL,
    instruction="You are a product-strategy synthesizer. You are given three evidence blocks: "
                "COMPETITOR, EVIDENCE, and WHITESPACE. Produce two sections:\n"
                "1) COMPETITIVE LANDSCAPE — where each competitor is differentiated, every claim "
                "backed by a citation from the EVIDENCE block.\n"
                "2) RANKED WHITESPACE OPPORTUNITIES — underserved needs ranked by opportunity size "
                "and how exposed competitors are on them. Explain each ranking in one line.\n"
                "Do not invent facts beyond the three blocks.",
    tools=[],
)

# =====================================================================
# RUNNER (unchanged from demo1)
# =====================================================================

async def ask(agent, message):
    service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="stratglens", session_service=service)
    session = await service.create_session(app_name="stratglens", user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=message)])
    async for event in runner.run_async(user_id="user1", session_id=session.id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            return event.content.parts[0].text
    return "(no response)"


async def full_report(market: str) -> str:
    """Option A orchestration: gather all three specialists, then synthesize."""
    competitor = await ask(competitor_agent, f"Analyze competitors in the {market} market.")
    evidence = await ask(evidence_agent, f"Find evidence on differentiation in the {market} market.")
    whitespace = await ask(whitespace_agent, f"Find unmet needs for customers in the {market} market.")
    blocks = (
        f"COMPETITOR:\n{competitor}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"WHITESPACE:\n{whitespace}"
    )
    return await ask(synthesis_agent, blocks)


async def main():
    # Part 1 — prove routing: each query should reach exactly one specialist.
    print("=" * 60)
    print("PART 1 — ROUTING (one query -> one specialist)")
    print("=" * 60)
    routing_tests = [
        ("COMPETITOR", "What is Flowly's pricing and what are they differentiated on?"),
        ("EVIDENCE", "What do analyst reports say differentiates the enterprise tier?"),
        ("WHITESPACE", "What are customers in the SMB segment asking for that they don't get?"),
    ]
    for label, query in routing_tests:
        print(f"\n--- {label} ---")
        print(f"User: {query}\n")
        print(f"Agent: {await ask(stratglens_router, query)}\n")

    # Part 2 — prove the real deliverables: landscape + ranked whitespace.
    print("=" * 60)
    print("PART 2 — FULL STRATGLENS REPORT (synthesis)")
    print("=" * 60)
    market = "team collaboration software"
    print(f"\nTarget market: {market}\n")
    print(await full_report(market))


if __name__ == "__main__":
    asyncio.run(main())
