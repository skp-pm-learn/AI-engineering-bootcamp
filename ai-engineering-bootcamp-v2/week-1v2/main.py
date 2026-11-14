"""Week 1 v2 demo API: one compact `/ask` endpoint for the intro class.

Run:
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import AliasChoices, BaseModel, Field, ValidationError

import vectorstore

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")
load_dotenv(THIS_DIR.parent / ".env")

app = FastAPI(title="Week 1 v2 /ask Demo")
_client: OpenAI | None = None

ModelName = Literal["gpt-4o-mini", "gpt-4o", "o3-mini"]
DEFAULT_MODEL: ModelName = "gpt-4o-mini"
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


class Answer(BaseModel):
    """The model output shape we want every caller to receive."""

    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


DEFAULT_TOP_K = 5


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    model: ModelName | None = None
    force_bad: bool = False
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0, le=20)


class IngestRequest(BaseModel):
    # text / document_id are validated in the handler so empty input returns a
    # clean 400 (Pydantic min_length would surface as a 422 instead).
    text: str
    document_id: str
    source: str | None = None
    metadata: dict | None = None
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


class AttemptResult(BaseModel):
    attempt: int
    step: str
    ok: bool
    message: str
    raw_output: str | None = None
    validation_error: str | None = None


class RetrievedChunk(BaseModel):
    id: str
    document_id: str | None = None
    source: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    retrieved_chunk_ids: list[str]
    retrieved: list[RetrievedChunk]
    attempts: list[AttemptResult]


class AgentStep(BaseModel):
    tool: str
    observation: str | None = None


class AgentRequest(BaseModel):
    # The user's goal/question. Either "message" or "goal" is accepted in the body.
    message: str = Field(
        min_length=1, validation_alias=AliasChoices("message", "goal")
    )


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    stopped: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent")
async def run_agent_endpoint(body: AgentRequest) -> AgentResponse:
    """Invoke the Google ADK agent on a user goal.

    Returns the final grounded answer plus a short steps[] trace (tool name +
    truncated observation per call). Never returns API keys or environment data.

    curl example (local):
      curl -s -X POST http://127.0.0.1:8000/agent \
        -H "Content-Type: application/json" \
        -d '{"message": "How much does Northwind reimburse for home office equipment, and how often?"}'

    curl example (Render):
      curl -s -X POST https://signal-roadmap.onrender.com/agent \
        -H "Content-Type: application/json" \
        -d '{"message": "What is the remote work policy?"}'
    """
    # Imported lazily so the Session 1 /ask path never depends on google-adk.
    import agent as adk_agent

    try:
        result = await adk_agent.run_agent(body.message)
    except Exception as exc:  # noqa: BLE001
        # Surface a clean error without leaking internals/keys.
        raise HTTPException(
            status_code=502,
            detail=f"Agent run failed: {type(exc).__name__}",
        ) from exc

    return AgentResponse(
        answer=result["final"],
        steps=[AgentStep(**step) for step in result["steps"]],
        stopped=result.get("stopped", False),
    )


@app.get("/debug/pinecone")
def debug_pinecone() -> dict:
    """Confirm Pinecone is configured and reachable (no OpenAI tokens spent)."""
    return vectorstore.check_pinecone()


@app.get("/debug/retrieve")
def debug_retrieve(q: str, top_k: int = 5) -> dict:
    """Retrieval-only debug view: embed `q` and return top_k chunks + scores.

    Does NOT call the LLM — use it to verify retrieval quality before wiring
    generation into /ask. Example:
      curl -s "http://127.0.0.1:8000/debug/retrieve?q=remote%20work%20policy"
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' must not be empty.")
    matches = vectorstore.query(q, top_k=top_k)
    return {"query": q, "top_k": top_k, "matches": matches}


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk, embed, and upsert a document into the Pinecone vector store.

    curl example:
      curl -s -X POST http://127.0.0.1:8000/ingest \
        -H "Content-Type: application/json" \
        -d '{
              "document_id": "rag-intro",
              "source": "rag_notes.md",
              "text": "Retrieval-Augmented Generation combines a retriever with a generator. ..."
            }'

    Chunk sizing is configurable per request via chunk_size / chunk_overlap
    (defaults 800 / 100). Empty text or document_id returns a 400.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    if not body.document_id.strip():
        raise HTTPException(
            status_code=400, detail="Field 'document_id' must not be empty."
        )
    if body.chunk_overlap >= body.chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    chunks = splitter.split_text(body.text)
    if not chunks:
        raise HTTPException(
            status_code=400, detail="Input produced no chunks to index."
        )

    base_metadata = dict(body.metadata or {})
    if body.source:
        base_metadata["source"] = body.source

    chunks_indexed = vectorstore.upsert_chunks(
        chunks, base_metadata, body.document_id
    )
    return IngestResponse(
        document_id=body.document_id,
        chunks_indexed=chunks_indexed,
        status="ok",
    )


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_per_1k, output_per_1k = MODEL_PRICES_PER_1K.get(
        model, MODEL_PRICES_PER_1K[DEFAULT_MODEL]
    )
    return (prompt_tokens / 1000 * input_per_1k) + (
        completion_tokens / 1000 * output_per_1k
    )


def usage_counts(completion) -> tuple[int, int, int]:
    usage = completion.usage
    if usage is None:
        return 0, 0, 0
    return usage.total_tokens, usage.prompt_tokens, usage.completion_tokens


def call_structured_model(question: str, model: ModelName) -> tuple[Answer, int, int, int]:
    completion = get_client().chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    total_tokens, prompt_tokens, completion_tokens = usage_counts(completion)
    return parsed, total_tokens, prompt_tokens, completion_tokens


def call_malformed_json_once(question: str, model: ModelName) -> tuple[str, int, int, int]:
    """Demo-only path: force one malformed response so students can see retry."""

    completion = get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY JSON using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' instead of a number."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    total_tokens, prompt_tokens, completion_tokens = usage_counts(completion)
    return raw, total_tokens, prompt_tokens, completion_tokens


GROUNDING_INSTRUCTIONS = (
    "You are a policy assistant for Northwind Robotics. Answer the user's "
    "question using ONLY the context passages provided below. Each passage is "
    "labelled with its document_id.\n"
    "Rules:\n"
    "1. Use only facts found in the context. Do not rely on outside knowledge.\n"
    "2. Cite the document_id next to the facts you use, e.g. (POL-101). If you "
    "use several passages, cite each one.\n"
    "3. If the context does not contain enough information to answer, do not "
    "guess. Say you cannot answer from the available Northwind policies, set "
    "sources_needed to true, and give a low confidence.\n"
    "4. Set confidence to reflect how fully the context supports your answer.\n"
)


def build_grounded_prompt(question: str, chunks: list[dict]) -> str:
    """Build the retrieval-augmented prompt: instructions + context + question."""
    if chunks:
        context = "\n\n".join(
            f"[{i + 1}] (document_id: {chunk.get('document_id')})\n{chunk.get('text')}"
            for i, chunk in enumerate(chunks)
        )
    else:
        context = "(no relevant context was retrieved)"

    return (
        f"{GROUNDING_INSTRUCTIONS}\n"
        "=== CONTEXT ===\n"
        f"{context}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {question}"
    )


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    model = body.model or DEFAULT_MODEL

    # Retrieval-augmented generation: embed the question, pull top-k chunks, and
    # ground the prompt on them. Retrieval runs once; the generation loop below
    # (including the force_bad validation demo) reuses the grounded prompt.
    retrieved = vectorstore.query(body.question, top_k=body.top_k)
    retrieved_chunks = [
        RetrievedChunk(
            id=chunk["id"],
            document_id=chunk.get("document_id"),
            source=chunk.get("source"),
            score=chunk.get("score"),
        )
        for chunk in retrieved
    ]
    retrieved_chunk_ids = [chunk.id for chunk in retrieved_chunks]
    grounded_prompt = build_grounded_prompt(body.question, retrieved)

    last_error: str | None = None
    attempts: list[AttemptResult] = []
    total_tokens_used = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    start = time.perf_counter()

    for attempt in range(2):
        try:
            if body.force_bad and attempt == 0:
                raw, tokens_used, prompt_tokens, completion_tokens = call_malformed_json_once(
                    grounded_prompt, model
                )
                total_tokens_used += tokens_used
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                try:
                    answer = Answer.model_validate_json(raw)
                except ValidationError as exc:
                    last_error = str(exc)
                    attempts.append(
                        AttemptResult(
                            attempt=attempt + 1,
                            step="forced_bad_json",
                            ok=False,
                            message="Validation failed, so the endpoint retries with structured output.",
                            raw_output=raw,
                            validation_error=str(exc),
                        )
                    )
                    continue

                attempts.append(
                    AttemptResult(
                        attempt=attempt + 1,
                        step="forced_bad_json",
                        ok=True,
                        message="Unexpectedly passed validation.",
                        raw_output=raw,
                    )
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_structured_model(
                    grounded_prompt, model
                )
                total_tokens_used += tokens_used
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                attempts.append(
                    AttemptResult(
                        attempt=attempt + 1,
                        step="structured_output",
                        ok=True,
                        message="Structured output matched the Answer schema.",
                    )
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(
                model, total_prompt_tokens, total_completion_tokens
            )
            return AskResponse(
                answer=answer,
                tokens_used=total_tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved=retrieved_chunks,
                attempts=attempts,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            attempts.append(
                AttemptResult(
                    attempt=attempt + 1,
                    step="structured_output",
                    ok=False,
                    message="Structured output failed validation.",
                    validation_error=str(exc),
                )
            )

    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
