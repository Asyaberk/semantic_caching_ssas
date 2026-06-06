"""
Converts cube schema data into plain-text descriptions suitable for LLM prompts.

The formatter accepts a SchemaProvider instance so it works transparently
with both mock and real SSAS data. The rest of the application calls
format_cube_for_llm() without knowing which data source is active.
"""

from backend.services.schema_provider import SchemaProvider, get_schema_provider


def format_cube_for_llm(cube_name: str, provider: SchemaProvider | None = None) -> str:
    """
    Returns a plain-text description of the given cube's schema.

    Args:
        cube_name: The name of the cube to describe (e.g. "Sales").
        provider:  A SchemaProvider instance. If omitted, the provider is
                   resolved automatically from the environment config.

    Returns:
        A multi-line string ready to be embedded in an LLM prompt.
    """
    if provider is None:
        provider = get_schema_provider()

    dimensions = provider.get_dimensions(cube_name)
    measures   = provider.get_measures(cube_name)
    members    = provider.get_members(cube_name)

    lines: list[str] = []

    # ── Cube header ──────────────────────────────────────────
    lines.append(f"Cube: {cube_name}")
    lines.append("")

    # ── Measures ─────────────────────────────────────────────
    lines.append("Measures (values that can be aggregated or queried):")
    for measure in measures:
        lines.append(f"  - {measure['caption']} ({measure['unique_name']})")
        lines.append(f"    Description: {measure['description']}")
        lines.append(f"    Aggregation: {measure['aggregation']}")
    lines.append("")

    # ── Dimensions & known members ────────────────────────────
    lines.append("Dimensions (axes along which measures can be sliced or filtered):")
    for dim in dimensions:
        lines.append(f"  - {dim['caption']} ({dim['unique_name']})")
        lines.append(f"    Description: {dim['description']}")

        dim_members = members.get(dim["unique_name"], [])
        if dim_members:
            captions = [m["caption"] for m in dim_members]
            lines.append(f"    Known members: {', '.join(captions)}")

    lines.append("")

    # ── Date hierarchy detail ─────────────────────────────────
    year_members    = members.get("[Date].[Calendar].[Year]", [])
    quarter_members = members.get("[Date].[Calendar].[Quarter]", [])
    month_members   = members.get("[Date].[Calendar].[Month]", [])

    if any([year_members, quarter_members, month_members]):
        lines.append("Date hierarchy detail:")

    if year_members:
        captions = [m["caption"] for m in year_members]
        lines.append(f"  Years available: {', '.join(captions)}")

    if quarter_members:
        captions = [m["caption"] for m in quarter_members]
        lines.append(f"  Quarters available: {', '.join(captions)}")

    if month_members:
        # Show only first 6 months to keep the prompt concise
        captions = [m["caption"] for m in month_members[:6]]
        lines.append(f"  Months available (sample): {', '.join(captions)} (... and more)")

    return "\n".join(lines)


def get_all_cube_names(provider: SchemaProvider | None = None) -> list[str]:
    """
    Returns the names of all cubes exposed by the schema provider.
    Useful when iterating over multiple cubes during question generation.
    """
    if provider is None:
        provider = get_schema_provider()
    return [cube["name"] for cube in provider.get_cubes()]


def get_measure_names(cube_name: str, provider: SchemaProvider | None = None) -> list[str]:
    """Returns a list of measure captions for the given cube."""
    if provider is None:
        provider = get_schema_provider()
    return [m["caption"] for m in provider.get_measures(cube_name)]


def get_dimension_names(cube_name: str, provider: SchemaProvider | None = None) -> list[str]:
    """Returns a list of dimension captions for the given cube."""
    if provider is None:
        provider = get_schema_provider()
    return [d["caption"] for d in provider.get_dimensions(cube_name)]
