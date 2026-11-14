"""StratgLens — Session 3 ADK agent (single grounded retrieval agent).

ONE root agent, ONE real tool (Pinecone retrieval reused from vectorstore.py),
a hard step limit so the loop can't run forever, and Think/Act/Observe logging
of ADK's event stream.

Patterns copied from the ADK sample:
  adk-multi-agent-systems/demo1_routing.py
    - Agent(name, model, instruction, tools=[...])
    - Runner + InMemorySessionService
    - the async run loop over runner.run_async(...)

Stack note (dual-provider, by design):
  - Gemini reasons / calls tools   -> needs GOOGLE_API_KEY
  - vectorstore.query embeds + searches Pinecone -> needs OPENAI_API_KEY + PINECONE_API_KEY
  (This is the same OpenAI+Pinecone retrieval your FastAPI /ask already uses.)

Run:
  cd ai-engineering-bootcamp-v2/week-1v2
  #   .env must contain GOOGLE_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY
  python agent.py                              # runs the built-in demo questions
  python agent.py "What is the remote work policy?"   # ask your own question
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.errors import ServerError

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")
load_dotenv(THIS_DIR.parent / ".env")

MODEL = "gemini-3.5-flash"  # 2.5-flash is retired for new keys; 3.8 is frequently overloaded (503)

# Hard ceiling on tool calls per question, so a confused loop can't run forever.
MAX_TOOL_CALLS = 5


# =====================================================================
# TOOL (one real tool — thin wrapper over the retrieval you already built)
# =====================================================================

def search_policies(query: str, top_k: int = 5) -> dict:
    """Search the Northwind policy knowledge base for passages relevant to a query.

    Returns the top matching chunks, each with its document_id and source, so the
    answer can cite where every fact came from. If retrieval is not configured
    (missing keys / empty index), returns a clearly-labelled placeholder instead
    of crashing, so the agent still runs end-to-end.
    """
    try:
        import vectorstore  # reused from the Session 2 RAG capstone

        matches = vectorstore.query(query, top_k=top_k)
        if not matches:
            return {"status": "empty", "query": query, "passages": []}
        return {
            "status": "ok",
            "query": query,
            "passages": [
                {
                    "document_id": m.get("document_id"),
                    "source": m.get("source"),
                    "score": m.get("score"),
                    "text": m.get("text"),
                }
                for m in matches
            ],
        }
    except Exception as exc:  # noqa: BLE001 - tool must degrade, not crash the agent
        return {
            "status": "placeholder",
            "query": query,
            "note": f"Retrieval unavailable ({type(exc).__name__}: {exc}). "
                    "Wire real keys in .env to enable Pinecone retrieval.",
            "passages": [],
        }


# =====================================================================
# ROOT AGENT (goal, constraints, and what "done" looks like)
# =====================================================================

INSTRUCTION = """\
You are StratgLens, a policy assistant for Northwind Robotics.

GOAL: Answer the user's question about Northwind policies, grounded ONLY in
passages returned by the search_policies tool.

HOW TO WORK:
1. Call search_policies with the user's question to retrieve context.
2. If the first results are thin or off-topic, you may refine the query and call
   search_policies again — but do NOT call it more than a few times.

CONSTRAINTS:
- Use ONLY facts found in retrieved passages. Never use outside knowledge.
- Cite the document_id next to each fact you use, e.g. (POL-101).
- If the passages do not contain enough information, DO NOT guess: say you cannot
  answer from the available Northwind policies and suggest what document is needed.

DONE LOOKS LIKE: a short grounded answer with an inline (document_id) citation for
every claim, OR an explicit "I can't answer this from the available policies."
"""

root_agent = Agent(
    name="stratglens_policy_agent",
    model=MODEL,
    description="Answers Northwind policy questions grounded in retrieved documents, with citations.",
    instruction=INSTRUCTION,
    tools=[search_policies],
)


# =====================================================================
# RUNNER + Think / Act / Observe logging
# =====================================================================

def _log_event(event) -> int:
    """Print an ADK event as THINK / ACT / OBSERVE. Returns # of tool calls seen."""
    tool_calls = 0
    if not (event.content and event.content.parts):
        return tool_calls

    for part in event.content.parts:
        # ACT: the model decided to call a tool.
        if getattr(part, "function_call", None):
            fc = part.function_call
            tool_calls += 1
            print(f"  [ACT]     call {fc.name}({dict(fc.args)})")
        # OBSERVE: the tool returned a result.
        elif getattr(part, "function_response", None):
            fr = part.function_response
            resp = fr.response
            status = resp.get("status") if isinstance(resp, dict) else resp
            print(f"  [OBSERVE] {fr.name} -> status={status}")
        # THINK / ANSWER: model text (reasoning or the final response).
        elif getattr(part, "text", None) and part.text.strip():
            tag = "ANSWER" if event.is_final_response() else "THINK"
            print(f"  [{tag}]  {part.text.strip()}")
    return tool_calls


async def _run_once(agent, message: str) -> str:
    """One full agent run: logs each step and enforces the tool-call cap."""
    service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="stratglens", session_service=service)
    session = await service.create_session(app_name="stratglens", user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=message)])

    total_tool_calls = 0
    final = "(no response)"
    async for event in runner.run_async(
        user_id="user1", session_id=session.id, new_message=content
    ):
        total_tool_calls += _log_event(event)

        # Hard stop: the step limit the loop cannot exceed.
        if total_tool_calls > MAX_TOOL_CALLS:
            print(f"  [STOP]    tool-call limit ({MAX_TOOL_CALLS}) reached — aborting.")
            return "Stopped: exceeded the maximum number of retrieval steps."

        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text or final
    return final


async def ask(agent, message: str, max_retries: int = 3) -> str:
    """Run the agent, retrying on transient 503/UNAVAILABLE model overloads.

    The Gemini API returns 503 when a model is briefly overloaded. That is a
    server-side spike, not an agent error, so we back off and retry the whole run.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await _run_once(agent, message)
        except ServerError as exc:
            if exc.code == 503 and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [RETRY]   model overloaded (503); retry {attempt}/{max_retries - 1} in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise


async def main():
    questions = sys.argv[1:] or [
        "What is Northwind's remote work policy?",
        "How many vacation days do new employees get?",
    ]
    for q in questions:
        print("=" * 70)
        print(f"USER: {q}")
        print("-" * 70)
        answer = await ask(root_agent, q)
        print("-" * 70)
        print(f"FINAL: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
