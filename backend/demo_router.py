"""
Demo Query Router

POST /demo/query   — semantic cache lookup (hit → cache, miss → LLM + write-through)
POST /demo/feedback — record user feedback (+/-) for a cached answer

On cache MISS the endpoint also:
  1. Saves the new pair to PostgreSQL and Qdrant (write-through)
  2. Generates 5 paraphrases of the question in the background and caches
     them with the same MDX so future similar queries become cache hits.
"""

import logging
import time
import threading

from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList, Filter, FieldCondition, MatchValue

from backend.config import settings
from backend.agents.mdx_agent import MDXGeneratorAgent
from backend.agents.uploader_agent import QdrantUploaderAgent
from backend.db.database import save_pairs, save_feedback, save_feedback_by_id, get_pair_by_id
from backend.models.schemas import QAPair
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

PARAPHRASE_COUNT = 5   # similar questions to generate and cache on a miss


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:  str
    cube_name: str | None = None   # optional — if omitted, search all cubes
    threshold: float = 0.75


class QueryResponse(BaseModel):
    status:           str          # "hit" | "miss"
    source:           str          # "cache" | "llm"
    question:         str
    matched_question: str | None
    similarity:       float | None
    mdx:              str
    response_time_ms: int
    cube_name:        str          # cube that answered the question
    pair_id:          str | None


class FeedbackRequest(BaseModel):
    pair_id:       str | None = None  # preferred — direct UUID of the cached pair
    question:      str        = ""    # fallback if pair_id not provided
    cube_name:     str        = ""
    feedback:      str        = ""    # "positive" | "negative"
    user_question: str | None = None  # the actual text the user typed (may differ from cached q)


class FeedbackResponse(BaseModel):
    saved:    bool
    feedback: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Semantic cache lookup demo endpoint.
    """
    import uuid as _uuid
    _NS = _uuid.NAMESPACE_DNS

    t0 = time.perf_counter()

    # When semantic cache is disabled, skip Qdrant and always call LLM
    if not settings.enable_semantic_cache:
        cube = req.cube_name or "cubeAccruement"
        logger.info("Semantic cache disabled — calling LLM directly for '%s'.", req.question)
        pair       = _mdx_agent.generate_for_question(question=req.question, cube_name=cube)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        pair_id    = str(_uuid.uuid5(_NS, f"{cube}::{req.question}"))
        return QueryResponse(
            status="miss", source="llm",
            question=req.question, matched_question=None, similarity=None,
            mdx=pair.mdx, response_time_ms=elapsed_ms,
            cube_name=cube, pair_id=pair_id,
        )

    # 1. Embed the question
    vector = _embed(req.question)

    # 2. Search Qdrant
    qdrant_filter = None
    if req.cube_name:
        qdrant_filter = Filter(
            must=[FieldCondition(key="cube_name", match=MatchValue(value=req.cube_name))]
        )

    results = _qdrant.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=vector,
        query_filter=qdrant_filter,
        limit=1,
        with_payload=True,
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # 3a. Cache HIT — cube is taken from the matched payload
    if results and results[0].score >= req.threshold:
        best     = results[0]
        payload  = best.payload
        hit_cube = payload.get("cube_name") or req.cube_name or "unknown"
        pair_id  = str(_uuid.uuid5(_NS, f"{hit_cube}::{payload.get('question', '')}"))

        logger.info(
            "Cache HIT for '%s' — matched '%s' in cube '%s' (score %.4f)",
            req.question, payload.get("question"), hit_cube, best.score,
        )

        return QueryResponse(
            status           = "hit",
            source           = "cache",
            question         = req.question,
            matched_question = payload.get("question"),
            similarity       = round(best.score, 4),
            mdx              = payload.get("mdx", ""),
            response_time_ms = elapsed_ms,
            cube_name        = hit_cube,
            pair_id          = pair_id,
        )

    # 3b. Cache MISS — need cube_name to generate MDX; fall back to first cube
    cube      = req.cube_name or "cubeAccruement"
    best_score = round(results[0].score, 4) if results else None
    logger.info(
        "Cache MISS for '%s' (best score: %s) — calling LLM for cube '%s'.",
        req.question, best_score, cube,
    )

    pair    = _mdx_agent.generate_for_question(question=req.question, cube_name=cube)
    pair_id = str(_uuid.uuid5(_NS, f"{cube}::{req.question}"))

    # Write-through: save original pair
    try:
        save_pairs([pair])
        _uploader.upload([pair])
        logger.info("Write-through: saved '%s' to PostgreSQL + Qdrant.", req.question)
    except Exception as exc:
        logger.warning("Write-through save failed (non-fatal): %s", exc)

    # Background: generate and cache paraphrases so future similar queries hit cache
    threading.Thread(
        target=_cache_paraphrases,
        args=(req.question, req.cube_name, pair.mdx),
        daemon=True,
    ).start()

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return QueryResponse(
        status           = "miss",
        source           = "llm",
        question         = req.question,
        matched_question = results[0].payload.get("question") if results else None,
        similarity       = best_score,
        mdx              = pair.mdx,
        response_time_ms = elapsed_ms,
        cube_name        = cube,
        pair_id          = pair_id,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    """
    Record user feedback for a cached answer.

    'positive' → marks the pair as verified good quality.
    'negative' → flags the pair for admin review. The pair remains in the
                 cache and still serves future queries (option A behaviour);
                 admins can edit or delete it via the admin panel.
    """
    if req.feedback not in ("positive", "negative"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="feedback must be 'positive' or 'negative'")

    # Use pair_id directly when available (works for both HIT and MISS)
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

    # For negative feedback also update the Qdrant payload so admin UIs
    # that query Qdrant directly can see the flag.
    flagged = (req.feedback == "negative")
    try:
        if req.pair_id:
            from qdrant_client.models import PointIdsList
            _qdrant.set_payload(
                collection_name=settings.qdrant_collection_name,
                payload={"flagged": flagged},
                points=PointIdsList(points=[req.pair_id]),
            )
        elif req.question and req.cube_name:
            _flag_in_qdrant(req.question, req.cube_name, flagged=flagged)
    except Exception as exc:
        logger.warning("Could not update Qdrant flag: %s", exc)

    # On negative feedback: if the user asked something different from the cached question,
    # treat their phrasing as a cache miss and generate a fresh MDX for it in the background.
    if req.feedback == "negative" and req.user_question:
        cached_pair = get_pair_by_id(req.pair_id) if req.pair_id else None
        cached_q    = cached_pair["question"] if cached_pair else req.question
        cube        = (cached_pair or {}).get("cube_name") or req.cube_name or "cubeAccruement"

        if req.user_question.strip() and req.user_question.strip() != cached_q.strip():
            logger.info(
                "Negative feedback: generating fresh MDX for user question '%s' in cube '%s'.",
                req.user_question[:60], cube,
            )
            reference_mdx = (cached_pair or {}).get("mdx", "")
            threading.Thread(
                target=_generate_and_cache_miss,
                args=(req.user_question.strip(), cube, reference_mdx),
                daemon=True,
            ).start()

    return FeedbackResponse(saved=saved, feedback=req.feedback)


# ── Private helpers ───────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    response = _openai.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def _cache_paraphrases(question: str, cube_name: str, mdx: str) -> None:
    """
    Generate PARAPHRASE_COUNT alternative phrasings of `question`, then
    cache each one (with the same MDX) if it is not already in Qdrant.

    Runs in a background thread so the API response is not delayed.
    """
    logger.info("Generating %d paraphrases for: '%s'", PARAPHRASE_COUNT, question[:60])
    try:
        prompt = (
            f"Generate exactly {PARAPHRASE_COUNT} alternative ways to ask the following "
            f"business question. Return only a JSON object with key \"questions\" containing "
            f"a list of strings. Do not include the original question.\n\n"
            f"Original: {question}"
        )
        response = _openai.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        import json
        data = json.loads(response.choices[0].message.content)
        paraphrases: list[str] = data.get("questions", [])

        if not paraphrases:
            logger.warning("No paraphrases returned for '%s'.", question[:60])
            return

        # Build QAPairs reusing the already-generated MDX
        pairs = [
            QAPair(question=q, mdx=mdx, cube_name=cube_name)
            for q in paraphrases
            if q.strip()
        ]

        if pairs:
            save_pairs(pairs)
            uploaded = _uploader.upload(pairs)   # dedup filter inside uploader
            logger.info(
                "Cached %d/%d paraphrases for '%s'.", uploaded, len(pairs), question[:60]
            )

    except Exception as exc:
        logger.warning("Paraphrase caching failed (non-fatal): %s", exc)


def _generate_and_cache_miss(question: str, cube_name: str, reference_mdx: str = "") -> None:
    """
    Generate a fresh MDX for a question that received negative feedback,
    then save it to PostgreSQL + Qdrant as a new cache entry.

    Uses the reference_mdx (from the flagged pair) as context so we do NOT
    need to re-fetch the SSAS schema -- making this robust to API downtime.
    """
    try:
        logger.info("Generating MDX for flagged user question: '%s'", question[:60])

        if reference_mdx:
            # Adapt the existing MDX for the new question phrasing (no schema needed)
            prompt = (
                f"You are an MDX expert for Microsoft SQL Server Analysis Services.\n"
                f"The user asked: {question}\n\n"
                f"A similar existing MDX query is:\n{reference_mdx}\n\n"
                f"Write an MDX query specifically for the user's question above. "
                f"Use the same cube (FROM clause) and adjust dimensions/measures as needed. "
                f"Return only the raw MDX query, no explanation."
            )
            response = _openai.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
            )
            mdx = response.choices[0].message.content.strip()
        else:
            pair = _mdx_agent.generate_for_question(question=question, cube_name=cube_name)
            mdx  = pair.mdx

        new_pair = QAPair(question=question, mdx=mdx, cube_name=cube_name)
        save_pairs([new_pair])
        uploaded = _uploader.upload([new_pair])
        logger.info(
            "Cached fresh MDX for flagged question '%s' (uploaded: %d).", question[:60], uploaded
        )
    except Exception as exc:
        logger.warning("Failed to generate MDX for flagged question (non-fatal): %s", exc)



def _flag_in_qdrant(question: str, cube_name: str, flagged: bool) -> None:
    """Update the flagged field in the Qdrant payload for a specific point."""
    import uuid as _uuid
    _NS = _uuid.NAMESPACE_DNS
    # Use double-colon separator to match QdrantUploaderAgent._make_id()
    point_id = str(_uuid.uuid5(_NS, f"{cube_name}::{question}"))

    _qdrant.set_payload(
        collection_name=settings.qdrant_collection_name,
        payload={"flagged": flagged},
        points=PointIdsList(points=[point_id]),
    )


class ExecuteRequest(BaseModel):
    mdx:         str
    data_source: str = "main"


class ExecuteResponse(BaseModel):
    columns:   list[dict]
    rows:      list[dict]
    row_count: int
    elapsed_ms: int | None = None
    error:     str | None = None


@router.post("/execute", response_model=ExecuteResponse)
def execute_mdx(req: ExecuteRequest):
    """
    Forward a raw MDX query to the SSAS Bridge and return the tabular result.

    Auto-fixes common LLM MDX mistakes:
      - Date years written as &[2025] instead of &[Calendar 2025]
    If the first attempt fails with a date error, applies fixes and retries once.
    """
    import re as _re, httpx

    def _run(mdx: str) -> dict:
        url     = f"{settings.ssas_url}/api/v1/mdx/query"
        headers = {"X-API-Key": settings.ssas_api_key, "Content-Type": "application/json"}
        resp    = httpx.post(url, headers=headers, json={"mdx": mdx, "dataSource": req.data_source}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _fix_date_years(mdx: str) -> str:
        """
        Replace LLM-generated year keys like [Dim].[Year].&[2025]
        with a full date range filter SSAS accepts.
        """
        def _year_to_range(m):
            dim  = m.group(1)
            year = m.group(2)
            return (
                "{" + dim + ".[Date].[Date].&[" + year + "-01-01T00:00:00]"
                ":" + dim + ".[Date].[Date].&[" + year + "-12-31T00:00:00]}"
            )
        return _re.sub(r'(\[[^\]]+\])\.\[Year\]\.&\[(\d{4})\]', _year_to_range, mdx)

    def _simplify_mdx(mdx: str) -> tuple[str, str]:
        """
        Fallback: extract the cube name and measures from the original MDX
        and run a simple aggregate (no rows, no WHERE).
        Returns (simplified_mdx, note_for_user).
        """
        # Extract FROM [cubeName]
        cube_m = _re.search(r'FROM\s+\[([^\]]+)\]', mdx, _re.IGNORECASE)
        cube   = cube_m.group(1) if cube_m else None
        # Extract measures block between first { and first } on COLUMNS line
        col_m  = _re.search(r'SELECT\s*\{([^}]+)\}\s*ON\s+COLUMNS', mdx, _re.IGNORECASE | _re.DOTALL)
        if cube and col_m:
            measures = "{" + col_m.group(1).strip() + "}"
            simple   = f"SELECT {measures} ON COLUMNS FROM [{cube}]"
            note     = "⚠ Simplified query (row/filter dimensions removed due to MDX generation issue — showing totals)"
            return simple, note
        return "", ""

    orig_error = ""
    body       = ""

    # ── Attempt 1: apply date fix pre-emptively, then run ───────────────────
    fixed_mdx = _fix_date_years(req.mdx)
    try:
        data = _run(fixed_mdx)
        rows = data.get("rows", [])
        # Got data — return immediately
        if rows:
            return ExecuteResponse(
                columns    = data.get("columns", []),
                rows       = rows,
                row_count  = data.get("rowCount", len(rows)),
                elapsed_ms = data.get("elapsedMs"),
            )
        # Ran OK but 0 rows — fall through to simplified attempt
        logger.info("MDX ran OK but returned 0 rows — trying simplified version.")
        orig_error = ""
    except httpx.HTTPStatusError as exc:
        body      = exc.response.text.lower()
        orig_error = exc.response.text
        logger.info("MDX failed (%s) — trying simplified version.", exc.response.status_code)
    except Exception as exc:
        orig_error = str(exc)
        logger.info("MDX error: %s — trying simplified version.", exc)

    # ── Attempt 2: simplified aggregate (just measures, no row/WHERE filters) ─
    try:
        simple_mdx, note = _simplify_mdx(req.mdx)
        if simple_mdx:
            logger.info("Trying simplified aggregate: %s", simple_mdx[:120])
            data = _run(simple_mdx)
            rows = data.get("rows", [])
            return ExecuteResponse(
                columns    = data.get("columns", []),
                rows       = rows,
                row_count  = data.get("rowCount", len(rows)),
                elapsed_ms = data.get("elapsedMs"),
                error      = note if note else None,
            )
    except Exception as exc2:
        logger.warning("Simplified MDX also failed: %s", exc2)

    # ── All attempts failed ──────────────────────────────────────────────────
    return ExecuteResponse(
        columns=[], rows=[], row_count=0,
        error=f"SSAS query failed: {orig_error[:300]}" if orig_error else "No results returned from SSAS."
    )

