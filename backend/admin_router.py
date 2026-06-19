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

import httpx
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
from backend.services.cube_explorer import (
    build_member_preview_mdx,
    shape_result,
    validate_readonly_mdx,
)
from backend.services.schema_provider import get_schema_provider

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

_qdrant = QdrantClient(
    url=settings.qdrant_url,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key,
    https=True,
)
_schema_provider = get_schema_provider()


# ── Models ────────────────────────────────────────────────────────────────────

class CacheListResponse(BaseModel):
    total:     int
    page:      int
    page_size: int
    items:     list[dict]


class UpdateMdxRequest(BaseModel):
    mdx: str


class ExplorerExecuteRequest(BaseModel):
    mdx: str
    limit: int = 200


def _cube_or_404(cube_name: str) -> dict:
    try:
        cube = next(
            (item for item in _schema_provider.get_cubes() if item.get("name") == cube_name),
            None,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not read SSAS metadata: {exc}") from exc
    if not cube:
        raise HTTPException(status_code=404, detail="Cube not found.")
    return cube


def _execute_bridge(mdx: str) -> dict:
    try:
        response = httpx.post(
            f"{settings.ssas_url}/api/v1/mdx/query",
            headers={"X-API-Key": settings.ssas_api_key, "Content-Type": "application/json"},
            json={"mdx": mdx, "dataSource": settings.ssas_data_source},
            timeout=45,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=422, detail=exc.response.text[:1000]) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"SSAS connection error: {exc}") from exc


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/cubes")
def list_cubes():
    """List the cubes currently exposed by the configured SSAS data source."""
    try:
        cubes = _schema_provider.get_cubes()
        return {"items": cubes, "total": len(cubes), "data_source": settings.ssas_data_source}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not read the SSAS cube list: {exc}") from exc


@router.get("/cubes/{cube_name}/schema")
def cube_schema(cube_name: str):
    """Return measures and dimensions without expensive member expansion."""
    cube = _cube_or_404(cube_name)
    try:
        dimensions = _schema_provider.get_dimensions(cube_name)
        measures = _schema_provider.get_measures(cube_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not read the cube schema: {exc}") from exc
    return {"cube": cube, "dimensions": dimensions, "measures": measures}


@router.get("/cubes/{cube_name}/hierarchies")
def cube_hierarchies(cube_name: str, dimension_name: str = Query(...)):
    """Load hierarchy and level metadata for a selected dimension."""
    _cube_or_404(cube_name)
    dimensions = _schema_provider.get_dimensions(cube_name)
    dimension = next(
        (
            item for item in dimensions
            if dimension_name in {item.get("name"), item.get("unique_name")}
        ),
        None,
    )
    if not dimension:
        raise HTTPException(status_code=404, detail="Dimension not found.")
    try:
        items = _schema_provider.get_dimension_hierarchies(cube_name, dimension["name"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not read the hierarchy list: {exc}") from exc
    return {"dimension": dimension, "items": items, "total": len(items)}


@router.get("/cubes/{cube_name}/members")
def preview_members(
    cube_name: str,
    dimension_name: str = Query(...),
    hierarchy_unique_name: str = Query(...),
    measure_unique_name: str = Query(...),
    limit: int = Query(100, ge=1, le=500),
):
    """List bounded hierarchy members with a selected measure."""
    _cube_or_404(cube_name)
    hierarchies = _schema_provider.get_dimension_hierarchies(cube_name, dimension_name)
    if hierarchy_unique_name not in {item.get("uniqueName") or item.get("unique_name") for item in hierarchies}:
        raise HTTPException(status_code=400, detail="Hierarchy not found in this dimension.")
    measures = _schema_provider.get_measures(cube_name)
    if measure_unique_name not in {item.get("unique_name") for item in measures}:
        raise HTTPException(status_code=400, detail="Measure not found in this cube.")

    mdx = build_member_preview_mdx(
        cube_name=cube_name,
        hierarchy_unique_name=hierarchy_unique_name,
        measure_unique_name=measure_unique_name,
        limit=limit,
    )
    return shape_result(_execute_bridge(mdx), mdx, limit)


@router.post("/cubes/{cube_name}/execute")
def explorer_execute(cube_name: str, body: ExplorerExecuteRequest):
    """Execute a read-only MDX query exactly as written, with bounded output."""
    _cube_or_404(cube_name)
    limit = min(max(body.limit, 1), 500)
    try:
        mdx = validate_readonly_mdx(body.mdx, cube_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return shape_result(_execute_bridge(mdx), mdx, limit)

@router.get("/cache", response_model=CacheListResponse)
async def list_cache(
    cube_name:    str | None  = Query(None),
    feedback:     str | None  = Query(None, description="positive | negative"),
    search:       str | None  = Query(None, description="keyword in question text"),
    mdx_search:   str | None  = Query(None, description="keyword in MDX content (e.g. 2024, Turkey)"),
    has_template: bool | None = Query(None, description="true = only pairs with template"),
    page:         int         = Query(1,  ge=1),
    page_size:    int         = Query(50, ge=1, le=200),
):
    """
    Return a paginated list of cached Q&A pairs for the admin UI.
    Filters: cube, feedback, question keyword, MDX content keyword, template presence.
    """
    rows, total = get_all_pairs(
        cube_name=cube_name, feedback=feedback,
        search=search, mdx_search=mdx_search,
        has_template=has_template,
        page=page, page_size=page_size,
    )
    return CacheListResponse(total=total, page=page, page_size=page_size, items=rows)


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
            payload={"mdx": body.mdx, "mdx_template": None, "entity_map": None},
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
    from backend.agents.uploader_agent import QdrantUploaderAgent

    pair_id = QdrantUploaderAgent._make_id(req.cube_name, req.question)
    pair = QAPair(
        id        = pair_id,
        cube_name = req.cube_name,
        question  = req.question,
        mdx       = req.mdx,
    )
    try:
        saved = save_pairs([pair])
        uploader = QdrantUploaderAgent()
        uploader.upload([pair], force=True)
        logger.info("Admin: added '%s' to cache (cube=%s).", req.question, req.cube_name)
        return {"saved": saved > 0, "pair_id": pair_id}
    except Exception as exc:
        logger.error("add-to-cache failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
