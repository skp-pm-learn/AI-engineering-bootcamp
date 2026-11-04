"""Chunk-size comparison for the Northwind RAG corpus.

Ingests the corpus at several (chunk_size, chunk_overlap) configs into ISOLATED
Pinecone namespaces (production default namespace is untouched), runs the golden
set against each, and reports, per question:
  - rank of the first retrieved chunk from the EXPECTED document (1 = top hit)
  - cosine score of that best matching chunk

Finer chunks usually improve precision for pinpoint fact lookups. Run with the
corpus files present and PINECONE/OPENAI env vars set.
"""

import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import NotFoundError

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")
load_dotenv(THIS_DIR.parent / ".env")

import vectorstore  # noqa: E402  (after load_dotenv)

CORPUS = THIS_DIR / "northwind_handbook"
DOCS = {
    "doc1_handbook.txt": "POL-101",
    "doc2_expenses.txt": "POL-114",
    "doc3_security.txt": "POL-207",
    "doc4_product.txt": "SPEC-WB9",
    "doc5_it_acceptable_use.txt": "POL-220",
    "doc6_facilities.txt": "POL-118",
}

# (chunk_size, chunk_overlap) configs to compare. First is the current baseline.
CONFIGS = [(800, 100), (400, 80), (250, 50)]

# golden set: (question, expected document_id)
GOLDEN = [
    ("How many days per week can I work remotely?", "POL-101"),
    ("What is the mileage rate for driving over 50 miles?", "POL-114"),
    ("How quickly must I report a lost laptop?", "POL-207"),
    ("What is the WB-9 robot payload limit?", "SPEC-WB9"),
    ("What is the minimum password length?", "POL-207"),
]
TOP_K = 5


def ingest_config(index, namespace: str, size: int, overlap: int) -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    total = 0
    for filename, doc_id in DOCS.items():
        text = (CORPUS / filename).read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        embeddings = vectorstore.embed_texts(chunks)
        vectors = [
            {
                "id": f"{doc_id}::{i}",
                "values": emb,
                "metadata": {"document_id": doc_id, "chunk_index": i, "text": chunk},
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        for start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[start : start + 100], namespace=namespace)
        total += len(vectors)
    return total


def wait_until_indexed(index, namespace: str, expected: int, timeout: int = 60) -> None:
    for _ in range(timeout):
        stats = index.describe_index_stats()
        ns = stats.get("namespaces", {}).get(namespace, {})
        if ns.get("vector_count", 0) >= expected:
            return
        time.sleep(1)


def score_config(index, namespace: str):
    """Return per-question (rank, score) for the expected document."""
    results = []
    for question, expected in GOLDEN:
        vector = vectorstore.embed_text(question)
        res = index.query(
            vector=vector, top_k=TOP_K, include_metadata=True, namespace=namespace
        )
        rank, score = None, None
        for position, match in enumerate(res.get("matches", []), start=1):
            if (match.get("metadata") or {}).get("document_id") == expected:
                rank, score = position, match.get("score")
                break
        results.append((question, expected, rank, score))
    return results


def clear_namespace(index, namespace: str) -> None:
    try:
        index.delete(delete_all=True, namespace=namespace)
    except NotFoundError:
        pass  # namespace doesn't exist yet — nothing to clear


def main() -> None:
    index = vectorstore.get_index(create=False)
    per_config = {}
    for size, overlap in CONFIGS:
        namespace = f"eval-{size}-{overlap}"
        clear_namespace(index, namespace)  # fresh start
        count = ingest_config(index, namespace, size, overlap)
        wait_until_indexed(index, namespace, count)
        per_config[(size, overlap)] = score_config(index, namespace)
        print(f"config {size}/{overlap}: {count} chunks ingested into '{namespace}'")

    # Comparison table
    print("\n=== RANK of first expected-doc chunk (1=top, 'miss'=not in top-5) ===")
    header = "question".ljust(50) + "expected".ljust(10) + "".join(
        f"{s}/{o}".ljust(14) for s, o in CONFIGS
    )
    print(header)
    for i, (question, expected) in enumerate(GOLDEN):
        row = question[:48].ljust(50) + expected.ljust(10)
        for cfg in CONFIGS:
            _, _, rank, score = per_config[cfg][i]
            cell = f"{'miss' if rank is None else '#'+str(rank)} ({'' if score is None else round(score,3)})"
            row += cell.ljust(14)
        print(row)

    # cleanup
    for size, overlap in CONFIGS:
        clear_namespace(index, f"eval-{size}-{overlap}")
    print("\ncleaned up eval namespaces.")


if __name__ == "__main__":
    main()
