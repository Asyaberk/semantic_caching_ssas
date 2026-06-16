"""
Demo Query Router

Exposes a single POST /demo/query endpoint that demonstrates
semantic caching in action:

  1. Embed the incoming question (OpenAI)
  2. Search Qdrant for the closest cached Q&A pair
  3a. Cache HIT  (similarity >= threshold) → return cached MDX instantly
  3b. Cache MISS (similarity <  threshold) → call LLM to generate new MDX

The response includes timing, similarity score, and source ("cache" | "llm")
so the demo UI can clearly show the difference.
"""

import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient

from backend.config import settings
from backend.agents.mdx_agent import MDXGeneratorAgent
from backend.agents.uploader_agent import QdrantUploaderAgent
from backend.services.schema_provider import get_schema_provider

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/demo", tags=["demo"])

# ── Shared clients (initialised once) ────────────────────────────────────────
_openai    = OpenAI(api_key=settings.openai_api_key)
_qdrant    = QdrantClient(
    url=settings.qdrant_url,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key,
    https=True,
)
_mdx_agent  = MDXGeneratorAgent(provider=get_schema_provider())
_uploader   = QdrantUploaderAgent()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:  str
    cube_name: str   = "Sales"
    threshold: float = 0.75   # similarity score above which we consider it a cache hit


class QueryResponse(BaseModel):
    status:           str            # "hit" | "miss"
    source:           str            # "cache" | "llm"
    question:         str            # original question
    matched_question: str | None     # cached question (hit only)
    similarity:       float | None   # cosine similarity (hit only)
    mdx:              str            # MDX query (from cache or freshly generated)
    response_time_ms: int            # wall-clock time in milliseconds
    cube_name:        str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Semantic cache lookup demo endpoint.

    Embeds the question, searches Qdrant, and returns either a cached MDX
    (cache hit) or a freshly generated MDX (cache miss), along with timing
    and similarity metadata for the UI to display.
    """
    t0 = time.perf_counter()

    # When semantic cache is disabled, skip Qdrant and always call LLM
    if not settings.enable_semantic_cache:
        logger.info("Semantic cache disabled — calling LLM directly for '%s'.", req.question)
        vector = _embed(req.question)
        pair   = _mdx_agent.generate_for_question(
            question=req.question, cube_name=req.cube_name
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return QueryResponse(
            status="miss", source="llm",
            question=req.question, matched_question=None, similarity=None,
            mdx=pair.mdx, response_time_ms=elapsed_ms, cube_name=req.cube_name,
        )

    # 1. Embed the question
    vector = _embed(req.question)

    # 2. Search Qdrant
    results = _qdrant.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=vector,
        limit=1,
        with_payload=True,
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # 3a. Cache HIT
    if results and results[0].score >= req.threshold:
        best    = results[0]
        payload = best.payload

        logger.info(
            "Cache HIT for '%s' — matched '%s' (score %.4f)",
            req.question, payload.get("question"), best.score,
        )

        return QueryResponse(
            status           = "hit",
            source           = "cache",
            question         = req.question,
            matched_question = payload.get("question"),
            similarity       = round(best.score, 4),
            mdx              = payload.get("mdx", ""),
            response_time_ms = elapsed_ms,
            cube_name        = req.cube_name,
        )

    # 3b. Cache MISS — generate MDX with LLM, then save to Qdrant
    best_score = round(results[0].score, 4) if results else None
    logger.info(
        "Cache MISS for '%s' (best score: %s) — calling LLM.",
        req.question, best_score,
    )

    pair = _mdx_agent.generate_for_question(
        question  = req.question,
        cube_name = req.cube_name,
    )

    # Write-through: save the newly generated pair so future queries hit cache
    try:
        _uploader.upload([pair])
        logger.info("Write-through: saved new pair to Qdrant for '%s'.", req.question)
    except Exception as exc:
        logger.warning("Write-through save failed (non-fatal): %s", exc)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return QueryResponse(
        status           = "miss",
        source           = "llm",
        question         = req.question,
        matched_question = results[0].payload.get("question") if results else None,
        similarity       = best_score,
        mdx              = pair.mdx,
        response_time_ms = elapsed_ms,
        cube_name        = req.cube_name,
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    response = _openai.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding
