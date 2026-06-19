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
    get_pair_by_id,
    log_execution,
    save_feedback,
    save_feedback_by_id,
    save_pairs,
)
from backend.models.schemas import QAPair
from backend.agents.mdx_agent import MDXGeneratorAgent
from backend.agents.uploader_agent import QdrantUploaderAgent
from backend.services.schema_provider import get_schema_provider
from backend.services.mdx_execution import execute_with_repair, extract_cube_name

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
    status:           str          # "hit" | "template" | "patched" | "miss" | "needs_clarification" | "not_answerable"
    source:           str          # "cache" | "template" | "patched" | "llm" | "validation"
    question:         str
    matched_question: str | None
    similarity:       float | None
    mdx:              str
    response_time_ms: int
    cube_name:        str
    pair_id:          str | None
    mismatch:         str | None = None
    feedback_message: str | None = None
    suggestions:      list[str] | None = None


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
    cube_name:   str | None = None
    pair_id:     str | None = None
    question:    str = ""
    data_source: str = settings.ssas_data_source


class ExecuteResponse(BaseModel):
    status:     str
    columns:    list[dict]
    rows:       list[dict]
    row_count:  int
    elapsed_ms: int | None = None
    error:      str | None = None
    executed_mdx: str
    attempt:       str
    validated:     bool
    cache_updated: bool = False


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
        feedback_message = result.feedback_message,
        suggestions      = result.suggestions,
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
    """Execute MDX and report the exact query accepted by SSAS."""
    def _run(mdx: str) -> dict:
        url     = f"{settings.ssas_url}/api/v1/mdx/query"
        headers = {"X-API-Key": settings.ssas_api_key, "Content-Type": "application/json"}
        resp    = httpx.post(url, headers=headers, json={"mdx": mdx, "dataSource": req.data_source}, timeout=30)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(resp.text[:1000]) from exc
        return resp.json()

    def _llm_repair(mdx: str, error_msg: str, cube: str, question: str) -> str | None:
        """Ask the LLM to fix an MDX query given the SSAS error."""
        try:
            from backend.mock.cube_formatter import format_cube_for_llm
            schema = format_cube_for_llm(cube, get_schema_provider()) if cube else ""

            prompt = (
                "You repair MDX without changing the user's business question.\n"
                f"User question: {question or '(not supplied)'}\n"
                f"Target cube: {cube}\n"
                f"Error: {error_msg[:500]}\n\n"
                f"Broken MDX:\n{mdx}\n\n"
                f"Cube schema (use ONLY these names):\n{schema[:8000]}\n\n"
                "Fix only the error while preserving every filter, measure, "
                "grouping and date constraint. Never simplify to a total. "
                "Return ONLY the complete MDX query.\n\n"
                "Rules:\n"
                "1. Keep the FROM cube unchanged.\n"
                "2. Do not invent schema names or members.\n"
                "3. Do not remove WHERE, ROWS, COLUMNS, subselects or NON EMPTY.\n"
                "4. Do not use .Members on single-level attribute hierarchies.\n"
            )

            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                timeout=45,
            )
            raw = resp.choices[0].message.content.strip()
            raw = _re.sub(r'```[a-z]*\n?', '', raw, flags=_re.IGNORECASE).strip('`').strip()
            logger.info("LLM MDX repair produced:\n%s", raw[:300])
            return raw
        except Exception as exc:
            logger.error("LLM MDX repair failed: %s", exc)
            return None

    mdx_cube = extract_cube_name(req.mdx)
    cube = req.cube_name or mdx_cube
    if not mdx_cube:
        raise HTTPException(status_code=400, detail="The MDX query must include a FROM [cube] clause.")
    if req.cube_name and req.cube_name != mdx_cube:
        raise HTTPException(
            status_code=400,
            detail=f"The selected cube ({req.cube_name}) does not match the MDX cube ({mdx_cube}).",
        )

    result = execute_with_repair(
        mdx=req.mdx,
        cube_name=cube,
        question=req.question,
        run_mdx=_run,
        repair_mdx=_llm_repair,
    )

    cache_updated = False
    if req.pair_id and result.status == "success" and result.attempt in {"year_fix", "llm_repair"}:
        try:
            from backend.db.database import update_pair_mdx

            cache_updated = update_pair_mdx(req.pair_id, result.executed_mdx)
            if cache_updated:
                _qdrant.set_payload(
                    collection_name=settings.qdrant_collection_name,
                    payload={"mdx": result.executed_mdx, "mdx_template": None, "entity_map": None},
                    points=PointIdsList(points=[req.pair_id]),
                )
        except Exception as exc:
            cache_updated = False
            logger.warning("Validated MDX cache update failed: %s", exc)

    log_execution(
        question=req.question,
        pair_id=req.pair_id,
        cube_name=cube,
        status=result.status,
        attempt=result.attempt,
        row_count=result.row_count,
        error=result.error,
    )

    return ExecuteResponse(
        status=result.status,
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
        elapsed_ms=result.elapsed_ms,
        error=result.error,
        executed_mdx=result.executed_mdx,
        attempt=result.attempt,
        validated=result.validated,
        cache_updated=cache_updated,
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
