# Basic RAG Proof — Northwind Policy Assistant

**Live service:** `https://signal-roadmap.onrender.com`

My `/ask` endpoint does retrieval-augmented generation: it embeds the question
(`text-embedding-3-small`), retrieves the top-5 matching chunks from Pinecone,
and prompts the model to answer **only** from that context, cite the
`document_id`, and refuse if the context doesn't cover the question. Every
response lists the `retrieved_chunk_ids` it grounded on.

## 1. Question answered FROM the docs (with citation)

```bash
curl -s -X POST https://signal-roadmap.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days per week can I work remotely?"}'
```

```json
{
  "answer": {
    "answer": "Employees may work remotely up to three days per week (POL-101).",
    "confidence": 1.0,
    "sources_needed": false
  },
  "tokens_used": 980,
  "model": "gpt-4o-mini",
  "latency_ms": 1920,
  "cost_usd": 0.00016,
  "retrieved_chunk_ids": ["POL-101::5", "POL-101::4", "POL-101::6", "POL-101::7", "POL-101::3"],
  "retrieved": [
    {"id": "POL-101::5", "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.597},
    {"id": "POL-101::4", "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.551},
    {"id": "POL-101::6", "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.544},
    {"id": "POL-101::7", "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.489},
    {"id": "POL-101::3", "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.427}
  ],
  "attempts": [
    {"attempt": 1, "step": "structured_output", "ok": true,
     "message": "Structured output matched the Answer schema."}
  ]
}
```

Correct answer ("up to three days per week"), **cites POL-101**, confidence 1.0,
and the grounding chunks are all from POL-101 (`doc1_handbook.txt`).

## 2. Question NOT in the docs → refusal

```bash
curl -s -X POST https://signal-roadmap.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the parental leave policy?"}'
```

```json
{
  "answer": {
    "answer": "I cannot answer from the available Northwind policies.",
    "confidence": 0.1,
    "sources_needed": true
  },
  "tokens_used": 918,
  "model": "gpt-4o-mini",
  "latency_ms": 993,
  "cost_usd": 0.000148,
  "retrieved_chunk_ids": ["POL-101::7", "POL-101::9", "POL-101::8", "POL-101::17", "POL-101::1"],
  "retrieved": [
    {"id": "POL-101::7",  "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.512},
    {"id": "POL-101::9",  "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.503},
    {"id": "POL-101::8",  "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.478},
    {"id": "POL-101::17", "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.364},
    {"id": "POL-101::1",  "document_id": "POL-101", "source": "doc1_handbook.txt", "score": 0.328}
  ],
  "attempts": [
    {"attempt": 1, "step": "structured_output", "ok": true,
     "message": "Structured output matched the Answer schema."}
  ]
}
```

Parental leave is **intentionally absent** from the corpus, so the model
**refuses** instead of guessing: `sources_needed: true`, confidence 0.1. (It
still retrieves nearby chunks, but none contain the fact, so it declines.)

## Why this proves basic RAG works

- **Grounded answers** pull real facts from the ingested docs and cite the
  source `document_id`.
- **Out-of-scope questions** trigger a refusal instead of a hallucination.
- Every response is transparent about *what* it retrieved
  (`retrieved_chunk_ids`), so grounding is auditable.
