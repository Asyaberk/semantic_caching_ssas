"""Safe query builders and response shaping for the admin Cube Explorer."""

from __future__ import annotations

import re

from backend.services.mdx_execution import extract_cube_name


def build_member_preview_mdx(
    *, cube_name: str, hierarchy_unique_name: str, measure_unique_name: str, limit: int
) -> str:
    """Build a bounded member listing query from already-validated schema names."""
    return (
        f"SELECT {{{measure_unique_name}}} ON COLUMNS, "
        f"NON EMPTY HEAD({hierarchy_unique_name}.Members, {limit}) ON ROWS "
        f"FROM [{cube_name}]"
    )


def validate_readonly_mdx(mdx: str, cube_name: str) -> str:
    """Validate that an admin console query is read-only and targets one cube."""
    cleaned = mdx.strip()
    if not cleaned:
        raise ValueError("MDX cannot be empty.")
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        raise ValueError("Cube Explorer only allows SELECT or WITH queries.")
    query_cube = extract_cube_name(cleaned)
    if not query_cube:
        raise ValueError("The MDX query must include a FROM [cube] clause.")
    if query_cube != cube_name:
        raise ValueError(
            f"The selected cube ({cube_name}) does not match the MDX cube ({query_cube})."
        )
    return cleaned


def shape_result(data: dict, mdx: str, limit: int) -> dict:
    """Return a bounded, UI-friendly representation of a Bridge result."""
    all_rows = data.get("rows") or []
    rows = all_rows[:limit]
    source_count = data.get("rowCount", len(all_rows))
    return {
        "columns": data.get("columns") or [],
        "rows": rows,
        "row_count": len(rows),
        "source_row_count": source_count,
        "elapsed_ms": data.get("elapsedMs"),
        "truncated": bool(data.get("truncated")) or len(all_rows) > limit or source_count > len(rows),
        "mdx": mdx,
    }
