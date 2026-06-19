# SSAS Semantic Cache

A semantic caching layer for SSAS (SQL Server Analysis Services) cubes.

The system generates natural-language question / MDX pairs from cube schemas using an LLM, stores them in Qdrant for vector search, and serves cached MDX responses when users ask business questions — without hitting the LLM on every request.

---

## How It Works

Every incoming question goes through the `QueryResolverAgent`:

```
User question
    → embed (text-embedding-3-small)
    → Qdrant semantic search
    → extract named entities (year, country, company, goods, container ID)
    → compare entities with matched cache entry
         ├── same entities          → Cache Hit       (~50 ms, no LLM)
         ├── year / entity differs  → Template Hit     (fill {{YEAR}}/{{COUNTRY}} etc.)
         │                            or Patch         (regex / LLM MDX edit)
         │                            → write-through to Qdrant
         └── topic mismatch        → Cache Miss        (LLM generates fresh MDX)
                                      → save + build template + cache 5 paraphrases
```

### Parameterized MDX Templates

When a Q&A pair is stored, entity values in its MDX are replaced with placeholders:

```
Cache MDX:  WHERE ([Year].&[2025] * [Country].&[Turkey])
Template:   WHERE ([Year].&[{{YEAR}}] * [Country].&[{{COUNTRY}}])
```

When a similar question arrives with different entities, the template is filled directly — no LLM call, no regex. One cached pair can serve any year × country × company × container ID combination.

### Write-Through

Patched and template-filled results are immediately written back to Qdrant with `force=True` (bypassing the 0.95 dedup threshold). The next identical question is a direct **Cache Hit**.

---

## Architecture

```
SSAS Bridge API  ──►  Schema Provider
                            │
                    Question Generator (LLM)
                            │
                    MDX Generator (LLM + Langfuse tracing)
                            │
                    Qdrant Uploader
                      │          │
                 Qdrant        PostgreSQL
               (vector search)  (admin + logs)
                      │
              QueryResolverAgent
              ┌───────┴──────────┐
          entity_checker    mdx_template
          (mismatch type)   ({{PLACEHOLDERS}})
```

**Cubes:** 9 SSAS cubes (Accruement, CreditDebit, VesselOrder, Waiting, …)  
**Similarity threshold:** 0.75 cosine similarity (configurable)  
**Monitoring:** Every LLM call traced in Langfuse (model, tokens, cost, latency)  
**Feedback:** Negative feedback flags the pair; admin edits MDX in the Cache tab

---

## Project Structure

```
ssas_project/
├── backend/
│   ├── agents/
│   │   ├── entity_agent.py       # extracts year, country, company, goods, container ID
│   │   ├── mdx_agent.py          # generates MDX queries via LLM + Langfuse tracing
│   │   ├── query_resolver.py     # central decision agent (Hit/Template/Patch/Miss)
│   │   ├── question_agent.py     # generates business questions from cube schema
│   │   └── uploader_agent.py     # embeds questions, upserts to Qdrant
│   ├── db/
│   │   └── database.py           # PostgreSQL helpers (save, fetch, log, feedback)
│   ├── models/
│   │   └── schemas.py            # QAPair (+ mdx_template, entity_map), PipelineState
│   ├── services/
│   │   ├── entity_checker.py     # NONE / YEAR / ENTITY / MAJOR mismatch classification
│   │   ├── mdx_patcher.py        # regex year patch + LLM entity patch
│   │   ├── mdx_template.py       # make_template / fill_template with {{PLACEHOLDERS}}
│   │   └── schema_provider.py    # MockSchemaProvider / SSASSchemaProvider
│   ├── config.py                 # settings from .env
│   ├── admin_router.py           # /admin/* CRUD + query log endpoints
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
| `LANGFUSE_HOST` | Langfuse host (default: `https://cloud.langfuse.com`) |
| `POSTGRES_URL` | PostgreSQL connection string |
| `SSAS_URL` | SSAS Bridge base URL |
| `SSAS_API_KEY` | SSAS Bridge API key |
| `USE_MOCK_CUBE` | `true` = offline mock, `false` = real SSAS |
| `QUESTIONS_PER_CUBE` | Questions to generate per cube (default: `50`) |
| `SIMILARITY_THRESHOLD` | Cache hit threshold, 0–1 (default: `0.75`) |

### 2. Start with Docker

```bash
docker compose up --build -d
```

### 3. Open the UIs

| URL | Description |
|---|---|
| `http://localhost:8002` | Admin UI — pipeline, cache management, query history |
| `http://localhost:8002/demo` | Demo UI — semantic cache with live SSAS execution |
| `http://localhost:8002/docs` | FastAPI interactive API docs |

---

## Query Resolution Detail

### Status types

| Status | Meaning |
|---|---|
| **Hit** | Exact semantic match — MDX returned from Qdrant directly |
| **Template Hit** | Matched pair has a template; user entities filled in — no LLM |
| **Patched** | Year or entity differed — MDX edited and written back to cache |
| **Miss** | No match above threshold — LLM generates fresh MDX |
| **Failed** | SSAS execution error — logged for admin review |

### Entity types extracted

`year` · `country` · `company / customer` · `goods type` · `container / vessel ID`

---

## Admin Panel

### Cache tab
Lists all cached Q&A pairs. Filter by:
- **Question** keyword (searches question text)
- **MDX keyword** (searches MDX content — e.g. `2024`, `Turkey`)
- **Cube** name
- **Feedback** status (Flagged / Verified)

### Cube Explorer tab
Reads the live SSAS catalog and lets admins:
- inspect cubes, dimensions, measures, hierarchies, levels, and cardinalities
- preview bounded hierarchy-member data with a selected measure
- run read-only MDX exactly as written, with a 500-row display limit

### Query History tab
Logs every incoming query with outcome, similarity score, matched cache entry, and mismatch type. Failed queries have a **Save to cache** button for manual MDX entry.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/pipeline/start` | Start the seeding pipeline |
| `POST` | `/pipeline/stop` | Request a clean stop |
| `GET` | `/pipeline/status` | Live pipeline state |
| `GET` | `/admin/cache` | List cached pairs (filterable) |
| `PUT` | `/admin/cache/{id}` | Edit a cached pair's MDX |
| `DELETE` | `/admin/cache/{id}` | Delete a cached pair |
| `GET` | `/admin/query-log` | Query history with outcomes |
| `GET` | `/admin/cubes` | List live SSAS cubes |
| `GET` | `/admin/cubes/{cube}/schema` | List dimensions and measures |
| `GET` | `/admin/cubes/{cube}/hierarchies` | List hierarchy and level metadata |
| `GET` | `/admin/cubes/{cube}/members` | Preview bounded member data |
| `POST` | `/admin/cubes/{cube}/execute` | Run exact read-only MDX with bounded output |
| `POST` | `/demo/query` | Semantic cache lookup |
| `POST` | `/demo/feedback` | Submit feedback for a result |
| `POST` | `/demo/execute` | Run MDX against SSAS and return data |

### Example: query

```bash
curl -X POST http://localhost:8002/demo/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Total accruement for Turkey in 2025?"}'
```

```json
{
  "status": "template",
  "source": "template",
  "mismatch": "year",
  "similarity": 0.96,
  "matched_question": "Total accruement for Turkey in 2025?",
  "cube_name": "cubeAccruement",
  "mdx": "SELECT {[Measures].[Accruement Count]} ON COLUMNS FROM [cubeAccruement] WHERE ...",
  "pair_id": "b7621e10-...",
  "response_time_ms": 62
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
  -c "SELECT cube_name, COUNT(*), COUNT(mdx_template) AS templated FROM qa_pairs GROUP BY cube_name;"

# Tail live logs
docker logs -f ssas_seeder_backend
```
