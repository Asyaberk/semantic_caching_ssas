"""Static checks that reject ambiguous dimension-only MDX references."""

from __future__ import annotations

import re


def validate_hierarchy_references(mdx: str, dimensions: list[dict]) -> None:
    """Require an explicit hierarchy when a dimension exposes multiple choices."""
    ambiguous = []
    for dimension in dimensions:
        if int(dimension.get("hierarchy_count") or 0) <= 1:
            continue
        unique_name = str(dimension.get("unique_name") or "")
        name = str(dimension.get("name") or "").strip("[]")
        candidates = {unique_name, f"[{name}]"}
        for candidate in candidates:
            if not candidate:
                continue
            pattern = rf"{re.escape(candidate)}\s*\.\s*(?:Children|Members)\b"
            if re.search(pattern, mdx, re.IGNORECASE):
                ambiguous.append(candidate)
                break
    if ambiguous:
        names = ", ".join(sorted(set(ambiguous)))
        raise ValueError(
            f"MDX uses an ambiguous dimension reference ({names}). "
            "Use an exact hierarchy unique name from the cube schema."
        )
