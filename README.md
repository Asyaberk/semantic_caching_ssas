# SSAS Semantic Cache

A semantic caching layer for SSAS (SQL Server Analysis Services) cubes.

The system generates natural-language question / MDX pairs from cube schemas using an LLM, stores them in Qdrant for vector search, and serves cached MDX responses when users ask business questions — without hitting the LLM on every request.

When a user asks a question:
- **Cache Hit** → returns the stored MDX in ~50 ms, no LLM cost
- **Cache Miss** → LLM generates a fresh MDX, stores it, and auto-generates 5 similar phrasings for future hits
- **Negative Feedback** → the user's original question is re-cached with a fresh MDX; admin is notified

---

## Architecture

```
SSAS Bridge API  ──►  Schema Provider
                             │
                    Question Generator (LLM)
                             │
                    MDX Generator (LLM)
                             │
                    Qdrant Uploader
                      │          │
                 Qdrant        PostgreSQL
               (vector search)  (source of truth)
```

**Cubes:** 9 SSAS cubes (Accruement, CreditDebit, VesselOrder, Waiting, GangPoint, …)  
**Pipeline:** Per-cube quotas (`QUESTIONS_PER_CUBE`), dedup at 0.95 cosine similarity  
**Monitoring:** Every LLM call traced in Langfuse (model, tokens, cost, latency)  
**Feedback loop:** Negative feedback triggers background MDX regeneration for the user's phrasing

---

## Project Structure

```
ssas_project/
├── backend/
│   ├── agents/
│   │   ├── question_agent.py     # generates business questions from cube schema
│   │   ├── mdx_agent.py          # generates MDX queries for each question
│   │   └── uploader_agent.py     # embeds questions, upserts to Qdrant
│   ├── db/
│   │   └── database.py           # PostgreSQL helpers (save, fetch, feedback)
│   ├── mock/
│   │   ├── cube_schema.py        # mock cube schema for offline dev
│   │   └── cube_formatter.py     # formats schema as LLM-readable text
│   ├── models/
│   │   └── schemas.py            # QAPair, PipelineState Pydantic models
│   ├── services/
│   │   └── schema_provider.py    # MockSchemaProvider / SSASSchemaProvider
│   ├── config.py                 # settings from .env
│   ├── admin_router.py           # /admin/* CRUD endpoints
│   ├── demo_router.py            # /demo/query, /demo/feedback, /demo/execute
│   ├── main.py                   # FastAPI app, pipeline endpoints, static files
│   └── orchestrator.py           # coordinates agents, background thread
├── frontend/
│   ├── index.html                # Admin UI  →  localhost:8002/
│   └── demo.html                 # Demo UI   →  localhost:8002/demo
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | Model name (default: `gpt-4o-mini`) |
| `OPENAI_EMBEDDING_MODEL` | Embedding model (default: `text-embedding-3-small`) |
| `QDRANT_URL` | Qdrant cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION_NAME` | Collection name (default: `ssas_qa_cache`) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host URL |
| `POSTGRES_*` | PostgreSQL connection settings |
| `SSAS_URL` | SSAS Bridge base URL |
| `SSAS_API_KEY` | SSAS Bridge API key |
| `USE_MOCK_CUBE` | `true` = offline mock, `false` = real SSAS |
| `QUESTIONS_PER_CUBE` | Questions to generate per cube (default: `50`) |
| `QUESTIONS_PER_BATCH` | LLM batch size (default: `20`) |
| `SIMILARITY_THRESHOLD` | Cache hit threshold, 0–1 (default: `0.75`) |

### 2. Start with Docker

```bash
docker compose up --build -d
```

### 3. Open the UIs

| URL | Description |
|---|---|
| `http://localhost:8002` | Admin UI — pipeline control, cache management |
| `http://localhost:8002/demo` | Demo UI — semantic cache with live SSAS execution |
| `http://localhost:8002/docs` | FastAPI interactive API docs |

---

## How It Works

### Seeding Pipeline

```
Admin clicks Start
    → Orchestrator discovers all cube names via SSAS Bridge
    → For each cube:
        1. Load existing pairs from PostgreSQL → upload to Qdrant
        2. While uploaded < QUESTIONS_PER_CUBE:
             generate questions (LLM)  →  generate MDX (LLM)
             →  save to PostgreSQL  →  upload to Qdrant
```

### Query Flow

```
User asks question
    → embed with text-embedding-3-small
    → search Qdrant (all cubes, no filter)
    → similarity ≥ 0.75  →  Cache HIT  →  return stored MDX
    → similarity < 0.75  →  Cache MISS →  LLM generates MDX
                                        →  save to PostgreSQL + Qdrant
                                        →  generate 5 similar phrasings
                                        →  cache all 5
```

### Feedback Flow

```
User clicks 👎
    → pair flagged in PostgreSQL + Qdrant (feedback='negative')
    → admin sees user's original question in red under the cached question
    → background thread:
        - uses cached pair's MDX as reference
        - LLM adapts MDX for the user's exact phrasing
        - new pair saved to PostgreSQL + Qdrant
```

### SSAS Execution

```
User clicks ▶ Run on SSAS
    → sends MDX to SSAS Bridge /api/v1/mdx/query
    → auto-fixes: date year keys (e.g. &[2025] → date range)
    → fallback: if MDX fails, runs simplified aggregate query
    → real cube data shown as table in UI
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/pipeline/start` | Start the seeding pipeline |
| `POST` | `/pipeline/stop` | Request a clean stop |
| `GET` | `/pipeline/status` | Live pipeline state |
| `GET` | `/admin/cache` | List cached pairs (filterable) |
| `PUT` | `/admin/cache/{id}` | Edit a cached pair |
| `DELETE` | `/admin/cache/{id}` | Delete a cached pair |
| `POST` | `/demo/query` | Semantic cache lookup |
| `POST` | `/demo/feedback` | Submit feedback for a result |
| `POST` | `/demo/execute` | Run MDX against SSAS and return data |

### Example: query

```bash
curl -X POST http://localhost:8002/demo/query \
  -H "Content-Type: application/json" \
  -d '{"question": "VAT total for Turkey in 2025?"}'
```

```json
{
  "status": "hit",
  "source": "cache",
  "similarity": 0.97,
  "matched_question": "VAT TOTAL for Turkey in 2025?",
  "cube_name": "cubeAccruement",
  "mdx": "SELECT {[Measures].[VAT TOTAL]} ON COLUMNS FROM [cubeAccruement] ...",
  "pair_id": "b7621e10-...",
  "response_time_ms": 48
}
```

### Example: execute MDX

```bash
curl -X POST http://localhost:8002/demo/execute \
  -H "Content-Type: application/json" \
  -d '{"mdx": "SELECT {[Measures].[Accruement Count]} ON COLUMNS FROM [cubeAccruement]"}'
```

```json
{
  "columns": [{"name": "[Measures].[Accruement Count]", "type": "Object"}],
  "rows": [{"[Measures].[Accruement Count]": 4861269}],
  "row_count": 1,
  "elapsed_ms": 8,
  "error": null
}
```

---

## SSAS Cubes

| Cube | Domain |
|---|---|
| `cubeAccruement` | Port service accruals — invoices, VAT, GRT by country/goods/date |
| `cubeCreditDebit` | Credit-debit invoice movements by customer type and date |
| `cubeDwellTimeRoro` | Ro-Ro vehicle dwell time at port |
| `cubeGangPoint` | Dock worker gang performance and point system |
| `cubeGeneralJobOrders` | General operation job orders by goods and area |
| `cubeOtherJobOrders` | Non-vessel equipment and service job orders |
| `cubeVesselJobOrder` | Vessel-specific job orders (crane, transport, etc.) |
| `cubeVesselOrder` | Vessel berthing orders — mooring time, operation time |
| `cubeWaiting` | Vessel and equipment waiting time analysis |

---

## Development

```bash
# Run locally without Docker
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8002

# Check PostgreSQL state
docker exec ssas_postgres psql -U ssas -d ssas_cache \
  -c "SELECT cube_name, COUNT(*), COUNT(feedback) FROM qa_pairs GROUP BY cube_name;"
```
