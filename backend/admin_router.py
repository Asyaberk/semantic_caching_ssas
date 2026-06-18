"""
Admin Cache Management Router

Exposes CRUD endpoints for the qa_pairs table so admins can inspect,
edit, and delete cached Q&A pairs from the admin UI.

GET    /admin/cache              — paginated list of all pairs
PUT    /admin/cache/{pair_id}    — update MDX for a pair (PostgreSQL + Qdrant)
DELETE /admin/cache/{pair_id}    — delete a pair from PostgreSQL + Qdrant
"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList, Filter, FieldCondition, MatchValue

from backend.config  import settings
from backend.db.database import (
    get_all_pairs, update_pair_mdx, delete_pair,
    get_query_log, save_pairs,
)
from backend.models.schemas import QAPair

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

_qdrant = QdrantClient(
    url=settings.qdrant_url,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key,
    https=True,
)


# ── Models ────────────────────────────────────────────────────────────────────

class CacheListResponse(BaseModel):
    total:     int
    page:      int
    page_size: int
    items:     list[dict]


class UpdateMdxRequest(BaseModel):
    mdx: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/cache", response_model=CacheListResponse)
async def list_cache(
    cube_name:  str | None = Query(None, description="Filter by cube name"),
    feedback:   str | None = Query(None, description="Filter by feedback: positive | negative"),
    page:       int        = Query(1,    ge=1),
    page_size:  int        = Query(50,   ge=1, le=200),
):
    """
    Return a paginated list of cached Q&A pairs for the admin UI.
    Optionally filter by cube name or feedback status.
    """
    rows, total = get_all_pairs(
        cube_name=cube_name,
        feedback=feedback,
        page=page,
        page_size=page_size,
    )
    return CacheListResponse(
        total=total, page=page, page_size=page_size, items=rows
    )


@router.put("/cache/{pair_id}")
async def update_cache_entry(pair_id: str, body: UpdateMdxRequest):
    """
    Update the MDX for a cached pair in both PostgreSQL and Qdrant.

    Qdrant payload is patched in-place so the vector (embedding) is preserved;
    only the mdx field in the payload is changed.
    """
    updated_pg = update_pair_mdx(pair_id, body.mdx)
    if not updated_pg:
        raise HTTPException(status_code=404, detail="Pair not found in PostgreSQL.")

    # Update the mdx field in the Qdrant payload without re-embedding
    try:
        _qdrant.set_payload(
            collection_name=settings.qdrant_collection_name,
            payload={"mdx": body.mdx},
            points=PointIdsList(points=[pair_id]),
        )
        logger.info("Updated MDX in Qdrant for pair %s.", pair_id)
    except Exception as exc:
        logger.warning("Qdrant payload update failed (non-fatal): %s", exc)

    return {"updated": True, "pair_id": pair_id}


@router.delete("/cache/{pair_id}")
async def delete_cache_entry(pair_id: str):
    """
    Delete a cached pair from both PostgreSQL and Qdrant.
    """
    deleted_pg = delete_pair(pair_id)
    if not deleted_pg:
        raise HTTPException(status_code=404, detail="Pair not found in PostgreSQL.")

    try:
        _qdrant.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=PointIdsList(points=[pair_id]),
        )
        logger.info("Deleted pair %s from Qdrant.", pair_id)
    except Exception as exc:
        logger.warning("Qdrant delete failed (non-fatal): %s", exc)

    return {"deleted": True, "pair_id": pair_id}


# ── Query Log ─────────────────────────────────────────────────────────────────

class QueryLogResponse(BaseModel):
    total:     int
    page:      int
    page_size: int
    items:     list[dict]


@router.get("/query-log", response_model=QueryLogResponse)
async def list_query_log(
    action:    str | None = Query(None, description="hit | patched | miss | failed"),
    mismatch:  str | None = Query(None, description="none | year | entity | major"),
    page:      int        = Query(1,    ge=1),
    page_size: int        = Query(50,   ge=1, le=200),
):
    """
    Return paginated query log rows, newest first.
    Filter by action and/or mismatch type.
    """
    rows, total = get_query_log(
        action=action, mismatch=mismatch,
        page=page, page_size=page_size,
    )
    return QueryLogResponse(total=total, page=page, page_size=page_size, items=rows)


class AddToCacheRequest(BaseModel):
    question:  str
    mdx:       str
    cube_name: str


@router.post("/query-log/add-to-cache")
async def add_to_cache(req: AddToCacheRequest):
    """
    Save a failed / patched query as a new cached pair in PostgreSQL + Qdrant.
    Called from the admin Query Log UI.
    """
    import uuid as _uuid
    from backend.agents.uploader_agent import QdrantUploaderAgent

    pair = QAPair(
        id        = str(_uuid.uuid4()),
        cube_name = req.cube_name,
        question  = req.question,
        mdx       = req.mdx,
    )
    try:
        saved = save_pairs([pair])
        uploader = QdrantUploaderAgent()
        uploader.upload([pair])
        logger.info("Admin: added '%s' to cache (cube=%s).", req.question, req.cube_name)
        return {"saved": saved > 0, "pair_id": pair.id}
    except Exception as exc:
        logger.error("add-to-cache failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
