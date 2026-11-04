"""Pinecone vector store support for the Week 1 v2 `/ask` demo (Session 2 RAG).

All configuration comes from environment variables so no secrets live in code.
The same embedding model is used at ingest time and query time, which is what
keeps stored vectors and query vectors in the same space.

Required env var:
  PINECONE_API_KEY     your Pinecone API key

Optional env vars (sensible defaults shown):
  PINECONE_INDEX       index name                     (default: "ask-rag")
  PINECONE_CLOUD       serverless cloud               (default: "aws")
  PINECONE_REGION      serverless region              (default: "us-east-1")
  EMBEDDING_MODEL      OpenAI embedding model         (default: "text-embedding-3-small")

OPENAI_API_KEY is already required by the main app and is reused here for embeddings.
"""

import os
import time

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# text-embedding-3-small produces 1536-dimensional vectors. The Pinecone index
# must be created with exactly this dimension, so we pin it to the model.
EMBED_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_INDEX = "ask-rag"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"

_pc: Pinecone | None = None
_embed_client: OpenAI | None = None


def embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embedding_dimension() -> int:
    model = embedding_model()
    if model not in EMBED_DIMENSIONS:
        raise ValueError(
            f"Unknown embedding model {model!r}; add its dimension to EMBED_DIMENSIONS."
        )
    return EMBED_DIMENSIONS[model]


def index_name() -> str:
    return os.getenv("PINECONE_INDEX", DEFAULT_INDEX)


def get_pinecone() -> Pinecone:
    """Lazy singleton Pinecone client, built from PINECONE_API_KEY."""
    global _pc
    if _pc is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")
        _pc = Pinecone(api_key=api_key)
    return _pc


def get_embed_client() -> OpenAI:
    """Lazy singleton OpenAI client used only for embeddings."""
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI()
    return _embed_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed one or more texts with the configured model (same model everywhere)."""
    response = get_embed_client().embeddings.create(
        model=embedding_model(),
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_text(text: str) -> list[float]:
    """Embed a single text — convenience wrapper over embed_texts."""
    return embed_texts([text])[0]


def ensure_index() -> str:
    """Return the index name, creating the serverless index if it does not exist."""
    pc = get_pinecone()
    name = index_name()
    existing = pc.list_indexes().names()
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=embedding_dimension(),
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", DEFAULT_CLOUD),
                region=os.getenv("PINECONE_REGION", DEFAULT_REGION),
            ),
        )
        # A newly created serverless index is not immediately ready for upserts.
        for _ in range(30):
            if pc.describe_index(name).status.get("ready"):
                break
            time.sleep(1)
    return name


def get_index(create: bool = False):
    """Return a Pinecone index handle. With create=True, create it if missing."""
    pc = get_pinecone()
    name = ensure_index() if create else index_name()
    return pc.Index(name)


def upsert_chunks(chunks: list[str], base_metadata: dict, document_id: str) -> int:
    """Embed chunks and upsert them into the index; return the count upserted.

    Each vector id is "{document_id}::{chunk_index}" so re-ingesting the same
    document overwrites its chunks instead of duplicating them. The chunk text is
    stored in metadata under "text" so retrieval can return it later. Pinecone
    metadata values must be scalars (str/number/bool) or lists of strings.
    """
    if not chunks:
        return 0

    index = get_index(create=True)
    embeddings = embed_texts(chunks)

    vectors = []
    for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        metadata = {
            **base_metadata,
            "document_id": document_id,
            "chunk_index": chunk_index,
            "text": chunk,
        }
        vectors.append(
            {
                "id": f"{document_id}::{chunk_index}",
                "values": embedding,
                "metadata": metadata,
            }
        )

    # Upsert in batches — Pinecone recommends <= 100 vectors per request.
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[start : start + batch_size])

    return len(vectors)


def query(text: str, top_k: int = 5) -> list[dict]:
    """Embed `text` and return the top_k most similar chunks from the index.

    Uses the same embedding model as ingest, so query and stored vectors share
    one space. Returns a flat list of matches with score and metadata; does not
    call any LLM.
    """
    index = get_index(create=False)
    vector = embed_text(text)
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)

    matches = []
    for match in result.get("matches", []):
        metadata = match.get("metadata") or {}
        matches.append(
            {
                "id": match.get("id"),
                "score": match.get("score"),
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
                "source": metadata.get("source"),
                "text": metadata.get("text"),
            }
        )
    return matches


def check_pinecone() -> dict:
    """Health/debug check: confirm Pinecone is configured and reachable.

    Returns a plain dict (never raises) describing configuration and connectivity,
    so it is safe to call from a route or a script.
    """
    status: dict = {
        "configured": bool(os.getenv("PINECONE_API_KEY")),
        "index_name": index_name(),
        "embedding_model": embedding_model(),
        "reachable": False,
        "index_exists": False,
    }

    try:
        status["expected_dimension"] = embedding_dimension()
    except ValueError as exc:
        status["error"] = str(exc)
        return status

    if not status["configured"]:
        status["error"] = "PINECONE_API_KEY is not set."
        return status

    try:
        pc = get_pinecone()
        names = pc.list_indexes().names()
        status["reachable"] = True
        status["available_indexes"] = names
        status["index_exists"] = status["index_name"] in names
        if status["index_exists"]:
            stats = pc.Index(status["index_name"]).describe_index_stats()
            status["vector_count"] = stats.get("total_vector_count", 0)
            status["index_dimension"] = stats.get("dimension")
    except Exception as exc:  # noqa: BLE001 - debug helper reports any failure
        status["error"] = f"{type(exc).__name__}: {exc}"

    return status
