"""Helpers for classifying cached MDX validation results."""

from __future__ import annotations


def classify_bridge_result(data: dict) -> tuple[str, int]:
    """Classify an SSAS Bridge response without treating empty data as success."""
    rows = data.get("rows") or []
    row_count = int(data.get("rowCount", len(rows)) or 0)
    return ("success" if rows else "no_data", row_count)
