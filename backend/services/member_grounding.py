"""Question-aware SSAS member grounding for MDX generation prompts."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.services.schema_provider import SchemaProvider


_QUOTED_RE = re.compile(r"""["']([^"']{2,80})["']""")
_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9&./_-]{2,}(?:\s+[A-Z][A-Za-z0-9&./_-]{2,}){0,2}\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_IGNORE = {
    "How", "What", "Which", "Show", "List", "Give", "Find", "Total",
    "Count", "Average", "Minimum", "Maximum", "Compare", "Top", "Bottom",
}


class MemberGroundingError(ValueError):
    def __init__(self, unmatched: list[str]):
        self.unmatched = unmatched
        values = ", ".join(unmatched)
        super().__init__(
            "I could not confirm these values as exact SSAS members: "
            f"{values}. Use Admin → Cube Explorer to find the exact member caption, "
            "then ask again using that value in quotes."
        )


def extract_member_candidates(question: str) -> list[str]:
    """Extract likely member values without treating ordinary metrics as members."""
    quoted = [match.strip() for match in _QUOTED_RE.findall(question)]
    candidates: list[str] = list(quoted)
    candidates.extend(
        match.strip() for match in _CAPITALIZED_RE.findall(question)
        if not any(match.strip() in quoted_item for quoted_item in quoted)
    )

    cleaned: list[str] = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip(" ?.,:;")
        if not candidate or candidate in _IGNORE or _YEAR_RE.fullmatch(candidate):
            continue
        key = candidate.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(candidate)
    return cleaned[:3]


def find_grounded_members(
    question: str,
    cube_name: str,
    provider: SchemaProvider,
    *,
    per_search_limit: int = 5,
    max_results: int = 20,
) -> dict:
    """
    Search candidate values across cube dimensions.

    Calls are parallel and bounded because grounding only runs on cache misses.
    """
    dimensions = [
        item for item in provider.get_dimensions(cube_name)
        if item.get("name") != "Measures"
    ]
    measures = provider.get_measures(cube_name)

    schema_words = set()
    for item in dimensions + measures:
        text = " ".join([
            str(item.get("name") or ""),
            str(item.get("caption") or ""),
            str(item.get("unique_name") or ""),
        ])
        schema_words.update(re.findall(r"[A-Za-z0-9]+", text.casefold()))

    candidates = [
        candidate for candidate in extract_member_candidates(question)
        if not set(re.findall(r"[A-Za-z0-9]+", candidate.casefold())).issubset(schema_words)
    ]
    if not candidates:
        return {"candidates": [], "matches": [], "unmatched": []}
    tasks = {}
    matches: list[dict] = []

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(dimensions)))) as executor:
        for candidate in candidates:
            for dimension in dimensions:
                future = executor.submit(
                    provider.search_members,
                    cube_name,
                    candidate,
                    dimension.get("name"),
                    per_search_limit,
                )
                tasks[future] = (candidate, dimension)

        for future in as_completed(tasks):
            candidate, dimension = tasks[future]
            try:
                items = future.result()
            except Exception:
                continue
            for item in items:
                unique_name = item.get("unique_name") or ""
                if not unique_name:
                    continue
                matches.append({
                    "requested_text": candidate,
                    "caption": item.get("caption") or "",
                    "unique_name": unique_name,
                    "dimension_name": item.get("dimension_name") or dimension.get("name") or "",
                    "hierarchy_name": item.get("hierarchy_name") or "",
                    "level_name": item.get("level_name") or "",
                })
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

    deduped = []
    seen = set()
    for item in matches:
        key = (item["requested_text"].casefold(), item["unique_name"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    matched_candidates = {item["requested_text"].casefold() for item in deduped}
    unmatched = [item for item in candidates if item.casefold() not in matched_candidates]
    return {"candidates": candidates, "matches": deduped, "unmatched": unmatched}


def format_grounding_for_prompt(grounding: dict) -> str:
    matches = grounding.get("matches") or []
    unmatched = grounding.get("unmatched") or []
    lines = [
        "Question-specific member grounding:",
        "- Use a member filter only when an exact unique name is listed below.",
        "- Never construct a member key from the user's wording.",
    ]
    if matches:
        for item in matches:
            lines.append(
                f"  - Requested '{item['requested_text']}' -> "
                f"{item['caption']} ({item['unique_name']})"
            )
    else:
        lines.append("  - No exact member matches were found.")
    if unmatched:
        lines.append(f"  - Unmatched candidate values: {', '.join(unmatched)}")
        lines.append("  - Do not add filters for unmatched candidate values.")
    return "\n".join(lines)
