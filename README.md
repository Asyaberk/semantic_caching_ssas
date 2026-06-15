# SSAS Semantic Cache Seeder

A pipeline that scans an SSAS cube, generates natural-language question / MDX pairs using an LLM, and uploads them to Qdrant for semantic caching.

When a user asks a business question, the cache layer finds the closest stored question by vector similarity and returns its pre-generated MDX — avoiding a live LLM call. On a cache miss, the system generates a fresh MDX and writes it back to Qdrant automatically (write-through caching).

---

## Architecture

```
SSAS Cube Schema (mock or real)
        │
        ▼
Question Generator Agent  ──► gpt-5-nano
        │
        ▼
MDX Generator Agent       ──► gpt-5-nano
        │
        ▼
Qdrant Uploader Agent     ──► text-embedding-3-small → Qdrant
```

**Monitoring:** Every LLM call is traced in Langfuse (model, tokens, cost, latency).  
**Deduplication:** New records with cosine similarity ≥ 0.95 to an existing record are skipped.  
**Resume:** On restart the pipeline reads the existing Qdrant count and only generates missing records.

---

## Project Structure

```
ssas_project/
├── backend/
│   ├── agents/
│   │   ├── question_agent.py    # generates business questions from cube schema
│   │   ├── mdx_agent.py         # generates MDX queries for each question
│   │   └── uploader_agent.py    # embeds questions and upserts to Qdrant
│   ├── mock/
│   │   ├── cube_schema.py       # mock Sales cube (dimensions, measures, members)
│   │   └── cube_formatter.py    # formats schema as LLM-readable text
│   ├── models/
│   │   └── schemas.py           # QAPair, PipelineState Pydantic models
│   ├── services/
│   │   └── schema_provider.py   # MockSchemaProvider / SSASSchemaProvider
│   ├── config.py                # settings loaded from .env
│   ├── demo_router.py           # /demo/query endpoint (cache hit / miss demo)
│   ├── main.py                  # FastAPI app, pipeline endpoints, static files
│   └── orchestrator.py          # coordinates all agents, background thread
├── frontend/
│   ├── index.html               # Admin UI  →  localhost:8002/
│   └── demo.html                # Demo UI   →  localhost:8002/demo
├── .env.example                 # environment variable template
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── test_agent.py                # quick end-to-end smoke test
└── verify.py                   # verifies Qdrant contents + semantic search
```

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Fill in the values below
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | Model name (default: `gpt-5-nano`) |
| `OPENAI_THINKING_EFFORT` | `minimal` / `medium` / `high` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model (default: `text-embedding-3-small`) |
| `QDRANT_URL` | Qdrant cluster URL |
| `QDRANT_PORT` | Qdrant port (6333 for cloud, 443 for custom HTTPS) |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION_NAME` | Collection name (default: `ssas_qa_cache`) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host URL |
| `TARGET_QUESTION_COUNT` | Records to generate per cube (default: `200`) |
| `QUESTIONS_PER_BATCH` | Questions per LLM batch (default: `20`) |
| `USE_MOCK_CUBE` | `true` = mock data, `false` = real SSAS |
| `SSAS_URL` | SSAS XMLA endpoint (only when `USE_MOCK_CUBE=false`) |

### 2. Start with Docker

```bash
docker compose up --build -d
```

### 3. Open the UIs

| URL | Description |
|---|---|
| `http://localhost:8002` | Admin UI — start/stop pipeline, live progress |
| `http://localhost:8002/demo` | Demo UI — semantic cache hit/miss demo |
| `http://localhost:8002/docs` | FastAPI auto-generated API docs |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/pipeline/start` | Start the seeding pipeline |
| `POST` | `/pipeline/stop` | Request a clean stop |
| `GET` | `/pipeline/status` | Live pipeline state (JSON) |
| `POST` | `/demo/query` | Semantic cache lookup — hit or miss |

### Demo query example

```bash
curl -X POST http://localhost:8002/demo/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Net Revenue for Turkey in 2025?", "cube_name": "Sales"}'
```

Response:
```json
{
  "status": "hit",
  "source": "cache",
  "similarity": 1.0,
  "mdx": "SELECT {[Measures].[Net Revenue]} ON COLUMNS ...",
  "response_time_ms": 312
}
```

---

## Switching to Real SSAS

When the real SSAS endpoint is available, update `.env`:

```env
USE_MOCK_CUBE=false
SSAS_URL=https://your-ssas-endpoint/xmla
```

Then implement `SSASSchemaProvider` in `backend/services/schema_provider.py`. All agents, the orchestrator, and the UI remain unchanged.

---

## Running Tests

```bash
# End-to-end pipeline smoke test (inside the container)
docker exec ssas_seeder_backend python /app/test_agent.py

# Verify Qdrant contents and semantic search
docker exec ssas_seeder_backend python /app/verify.py
```
