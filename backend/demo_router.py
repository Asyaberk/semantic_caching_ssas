"""
Demo Query Router

POST /demo/query    — semantic cache lookup via QueryResolverAgent
POST /demo/feedback — record user feedback
POST /demo/execute  — run MDX against SSAS Bridge, return tabular data
"""

import logging
import threading

import httpx
import re as _re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

from backend.config import settings
from backend.agents.query_resolver import QueryResolverAgent
from backend.db.database import (
    save_pairs, save_feedback, save_feedback_by_id, get_pair_by_id,
)
from backend.models.schemas import QAPair
from backend.agents.mdx_agent import MDXGeneratorAgent
from backend.agents.uploader_agent import QdrantUploaderAgent
from backend.services.schema_provider import get_schema_provider

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/demo", tags=["demo"])

# ── Shared clients ────────────────────────────────────────────────────────────
_resolver  = QueryResolverAgent()
_openai    = OpenAI(api_key=settings.openai_api_key)
_qdrant    = QdrantClient(
    url=settings.qdrant_url,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key,
    https=True,
)
_mdx_agent = MDXGeneratorAgent(provider=get_schema_provider())
_uploader  = QdrantUploaderAgent()


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:  str
    cube_name: str | None = None
    threshold: float = 0.75


class QueryResponse(BaseModel):
    status:           str          # "hit" | "template" | "patched" | "miss"
    source:           str          # "cache" | "template" | "patched" | "llm"
    question:         str
    matched_question: str | None
    similarity:       float | None
    mdx:              str
    response_time_ms: int
    cube_name:        str
    pair_id:          str | None
    mismatch:         str | None = None


class FeedbackRequest(BaseModel):
    pair_id:       str | None = None
    question:      str        = ""
    cube_name:     str        = ""
    feedback:      str        = ""    # "positive" | "negative"
    user_question: str | None = None


class FeedbackResponse(BaseModel):
    saved:    bool
    feedback: str


class ExecuteRequest(BaseModel):
    mdx:         str
    data_source: str = "main"


class ExecuteResponse(BaseModel):
    columns:    list[dict]
    rows:       list[dict]
    row_count:  int
    elapsed_ms: int | None = None
    error:      str | None = None
    simplified: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Resolve question via QueryResolverAgent (hit / patched / miss)."""
    result = _resolver.resolve(
        question=req.question,
        cube_name=req.cube_name,
        threshold=req.threshold,
    )
    return QueryResponse(
        status           = result.status,
        source           = result.source,
        question         = result.question,
        matched_question = result.matched_question,
        similarity       = result.similarity,
        mdx              = result.mdx,
        response_time_ms = result.response_time_ms,
        cube_name        = result.cube_name,
        pair_id          = result.pair_id,
        mismatch         = result.mismatch,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    """
    Record user feedback for a cached answer.

    positive → marks pair as verified good quality.
    negative → flags pair for admin review. Pair stays in cache (Option A).
               A fresh MDX for the user's phrasing is generated in background.
    """
    if req.feedback not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="feedback must be 'positive' or 'negative'")

    if req.pair_id:
        saved = save_feedback_by_id(
            pair_id       = req.pair_id,
            feedback      = req.feedback,
            user_question = req.user_question or req.question or None,
        )
    else:
        saved = save_feedback(
            question  = req.question,
            cube_name = req.cube_name,
            feedback  = req.feedback,
        )

    # Update Qdrant payload flag
    flagged = (req.feedback == "negative")
    try:
        if req.pair_id:
            _qdrant.set_payload(
                collection_name=settings.qdrant_collection_name,
                payload={"flagged": flagged},
                points=PointIdsList(points=[req.pair_id]),
            )
        elif req.question and req.cube_name:
            _flag_in_qdrant(req.question, req.cube_name, flagged=flagged)
    except Exception as exc:
        logger.warning("Could not update Qdrant flag: %s", exc)

    # Negative feedback → generate fresh MDX for user's phrasing in background
    if req.feedback == "negative" and req.user_question:
        cached_pair = get_pair_by_id(req.pair_id) if req.pair_id else None
        cached_q    = cached_pair["question"] if cached_pair else req.question
        cube        = (cached_pair or {}).get("cube_name") or req.cube_name or "cubeAccruement"

        if req.user_question.strip() and req.user_question.strip() != cached_q.strip():
            reference_mdx = (cached_pair or {}).get("mdx", "")
            threading.Thread(
                target=_generate_and_cache_miss,
                args=(req.user_question.strip(), cube, reference_mdx),
                daemon=True,
            ).start()

    return FeedbackResponse(saved=saved, feedback=req.feedback)


@router.post("/execute", response_model=ExecuteResponse)
def execute_mdx(req: ExecuteRequest):
    """
    Forward MDX to the SSAS Bridge and return tabular data.

    Retry strategy:
      1. Fix common year-key format (&[2025] → date range) and run
      2. If 0 rows or error → strip WHERE + ROW dimensions, run aggregate only
      3. If aggregate also fails → return error
    """
    def _run(mdx: str) -> dict:
        url     = f"{settings.ssas_url}/api/v1/mdx/query"
        headers = {"X-API-Key": settings.ssas_api_key, "Content-Type": "application/json"}
        resp    = httpx.post(url, headers=headers, json={"mdx": mdx, "dataSource": req.data_source}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _fix_year_keys(mdx: str) -> str:
        """
        Convert &[2025] year keys to full date-range sets that SSAS accepts.
        Handles: [Dim].[YearHierarchy].&[2025]
        """
        def _to_range(m):
            prefix = m.group(1)   # everything up to the year key
            year   = m.group(2)
            return (
                "{" + prefix + ".&[" + year + "-01-01T00:00:00]"
                ":" + prefix + ".&[" + year + "-12-31T00:00:00]}"
            )
        # Pattern: [Anything].[Anything].&[2025]
        return _re.sub(
            r'(\[[^\]]+\]\.\[[^\]]+\])\.&\[(\d{4})\]',
            _to_range,
            mdx,
        )

    def _aggregate_only(mdx: str) -> tuple[str, bool]:
        """
        Strip ROW axis and WHERE clause, keep only COLUMNS (measures) and FROM.
        Returns (simplified_mdx, did_simplify).
        """
        cube_m   = _re.search(r'FROM\s+\[([^\]]+)\]', mdx, _re.IGNORECASE)
        col_m    = _re.search(r'SELECT\s*\{([^}]+)\}\s*ON\s+COLUMNS', mdx, _re.IGNORECASE | _re.DOTALL)
        if cube_m and col_m:
            cube     = cube_m.group(1)
            measures = "{" + col_m.group(1).strip() + "}"
            return f"SELECT {measures} ON COLUMNS FROM [{cube}]", True
        return "", False

    orig_error = ""

    # ── Attempt 1: date-fixed MDX ─────────────────────────────────────────────
    fixed_mdx = _fix_year_keys(req.mdx)
    try:
        data = _run(fixed_mdx)
        rows = data.get("rows", [])
        if rows:
            return ExecuteResponse(
                columns    = data.get("columns", []),
                rows       = rows,
                row_count  = data.get("rowCount", len(rows)),
                elapsed_ms = data.get("elapsedMs"),
                simplified = False,
            )
        # 0 rows — fall through to aggregate
        logger.info("MDX returned 0 rows — trying aggregate fallback.")
    except httpx.HTTPStatusError as exc:
        orig_error = exc.response.text
        logger.info("MDX failed (%s) — trying aggregate fallback.", exc.response.status_code)
    except Exception as exc:
        orig_error = str(exc)
        logger.info("MDX error: %s — trying aggregate fallback.", exc)

    # ── Attempt 2: aggregate only (no rows, no WHERE) ─────────────────────────
    simple_mdx, did_simplify = _aggregate_only(req.mdx)
    if did_simplify:
        try:
            logger.info("Aggregate fallback: %s", simple_mdx[:120])
            data = _run(simple_mdx)
            rows = data.get("rows", [])
            return ExecuteResponse(
                columns    = data.get("columns", []),
                rows       = rows,
                row_count  = data.get("rowCount", len(rows)),
                elapsed_ms = data.get("elapsedMs"),
                simplified = True,
                error      = "Showing totals only — dimension filters could not be applied.",
            )
        except Exception as exc2:
            logger.warning("Aggregate fallback also failed: %s", exc2)

    # ── All attempts failed ────────────────────────────────────────────────────
    return ExecuteResponse(
        columns=[], rows=[], row_count=0, simplified=False,
        error=f"SSAS query failed: {orig_error[:300]}" if orig_error else "No results returned from SSAS.",
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _flag_in_qdrant(question: str, cube_name: str, flagged: bool) -> None:
    import uuid as _uuid
    point_id = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"{cube_name}::{question}"))
    _qdrant.set_payload(
        collection_name=settings.qdrant_collection_name,
        payload={"flagged": flagged},
        points=PointIdsList(points=[point_id]),
    )


def _generate_and_cache_miss(question: str, cube_name: str, reference_mdx: str = "") -> None:
    """Generate a fresh MDX for a negatively-flagged user question and cache it."""
    try:
        if reference_mdx:
            prompt = (
                f"You are an MDX expert.\n"
                f"User asked: {question}\n\n"
                f"A similar MDX query is:\n{reference_mdx}\n\n"
                f"Write an MDX query for the user's question. "
                f"Use the same cube (FROM clause) and adjust as needed. "
                f"Return only the raw MDX query."
            )
            resp = _openai.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
            )
            mdx = resp.choices[0].message.content.strip()
        else:
            pair = _mdx_agent.generate_for_question(question=question, cube_name=cube_name)
            mdx  = pair.mdx

        new_pair = QAPair(question=question, mdx=mdx, cube_name=cube_name)
        save_pairs([new_pair])
        _uploader.upload([new_pair])
        logger.info("Cached fresh MDX for flagged question '%s'.", question[:60])
    except Exception as exc:
        logger.warning("Failed to generate MDX for flagged question (non-fatal): %s", exc)
