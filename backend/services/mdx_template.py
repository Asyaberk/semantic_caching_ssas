"""
MDX Template Service

Converts a concrete MDX query + its extracted entities into a parameterized
template where entity values are replaced by placeholders:

  {{YEAR}}     — calendar year  (e.g. 2025 → {{YEAR}})
  {{COUNTRY}}  — country name   (e.g. Turkey → {{COUNTRY}})
  {{COMPANY}}  — company name
  {{GOODS}}    — goods / cargo type

Usage
─────
  template, entity_map = make_template(mdx, entities)
  filled = fill_template(template, user_entities)

Why this matters
────────────────
Without templates each new year/country combination requires a YEAR or
ENTITY patch (regex or LLM).  With templates ONE cached pair can serve ALL
entity combinations for the same question structure — fill placeholders and
return a Template Hit in milliseconds, with zero LLM calls.
"""

from __future__ import annotations
import re
import logging
from backend.agents.entity_agent import QuestionEntities

logger = logging.getLogger(__name__)

PLACEHOLDER_YEAR      = "{{YEAR}}"
PLACEHOLDER_COUNTRY   = "{{COUNTRY}}"
PLACEHOLDER_COMPANY   = "{{COMPANY}}"
PLACEHOLDER_GOODS     = "{{GOODS}}"
PLACEHOLDER_CONTAINER = "{{CONTAINER}}"


# ── Template generation ───────────────────────────────────────────────────────

def make_template(mdx: str, entities: QuestionEntities) -> tuple[str, dict]:
    """
    Replace known entity values in MDX with typed placeholders.

    Returns:
        template   — MDX string with {{PLACEHOLDERS}}
        entity_map — dict mapping placeholder → original value
                     e.g. {"YEAR": "2025", "COUNTRY": "Turkey"}

    Safe if entities is empty — returns the original MDX unchanged and an
    empty map (no placeholders).
    """
    template   = mdx
    entity_map: dict[str, str] = {}

    # ── Year replacement ──────────────────────────────────────────────────────
    # Matches: .&[2025]  or  [Year].&[2025]  or  [Calendar 2025] (in member keys)
    # Strategy: replace first year found to keep template consistent.
    for year in sorted(entities.years, key=lambda y: -y):   # largest first to avoid partial overlap
        yr_str = str(year)
        # Pattern 1: .&[2025]
        if f".&[{yr_str}]" in template:
            template = template.replace(f".&[{yr_str}]", f".&[{PLACEHOLDER_YEAR}]")
            entity_map["YEAR"] = yr_str
            break
        # Pattern 2: [Calendar 2025]
        if f"[Calendar {yr_str}]" in template:
            template = template.replace(f"[Calendar {yr_str}]", f"[Calendar {PLACEHOLDER_YEAR}]")
            entity_map["YEAR"] = yr_str
            break
        # Pattern 3: bare year at end of member key  &[2025-01-01T...]
        if re.search(rf'&\[{yr_str}-\d{{2}}-\d{{2}}', template):
            template = re.sub(
                rf'&\[{yr_str}(-\d{{2}}-\d{{2}}T[^\]]*)\]',
                rf'&[{PLACEHOLDER_YEAR}\1]',
                template,
            )
            entity_map["YEAR"] = yr_str
            break

    # ── Country replacement ───────────────────────────────────────────────────
    for country in entities.countries:
        if not country:
            continue
        patterns = [
            (f".&[{country}]",  f".&[{PLACEHOLDER_COUNTRY}]"),
            (f'"{country}"',    f'"{PLACEHOLDER_COUNTRY}"'),
        ]
        for src, dst in patterns:
            if src in template:
                template = template.replace(src, dst)
                entity_map["COUNTRY"] = country
                break
        if "COUNTRY" in entity_map:
            break

    # ── Company replacement ───────────────────────────────────────────────────
    for company in entities.companies:
        if not company:
            continue
        src = f".&[{company}]"
        if src in template:
            template = template.replace(src, f".&[{PLACEHOLDER_COMPANY}]")
            entity_map["COMPANY"] = company
            break

    # ── Goods replacement ─────────────────────────────────────────────────────
    for goods in entities.goods:
        if not goods:
            continue
        src = f".&[{goods}]"
        if src in template:
            template = template.replace(src, f".&[{PLACEHOLDER_GOODS}]")
            entity_map["GOODS"] = goods
            break

    # ── Container / vessel ID replacement ───────────────────────────────────
    for container in entities.containers:
        if not container:
            continue
        src = f".&[{container}]"
        if src in template:
            template = template.replace(src, f".&[{PLACEHOLDER_CONTAINER}]")
            entity_map["CONTAINER"] = container
            break

    if entity_map:
        logger.debug("Template created — placeholders: %s", list(entity_map.keys()))
    else:
        logger.debug("No entity values found in MDX — template equals original MDX.")

    return template, entity_map


# ── Template filling ──────────────────────────────────────────────────────────

def fill_template(
    template:     str,
    user_entities: QuestionEntities,
    original_map:  dict[str, str],
) -> str | None:
    """
    Fill all placeholders in template with user entity values.

    Returns the filled MDX string, or None if a required placeholder
    cannot be satisfied by the user's entities.

    original_map is used to determine which placeholder types are required
    (it was created alongside the template during cache write).
    """
    result = template

    if "YEAR" in original_map:
        if not user_entities.years:
            logger.debug("Template fill failed — YEAR required but not found in query.")
            return None
        result = result.replace(
            f".&[{PLACEHOLDER_YEAR}]",
            f".&[{user_entities.years[0]}]",
        )
        # Also handle Calendar prefix variant
        result = result.replace(
            f"[Calendar {PLACEHOLDER_YEAR}]",
            f"[Calendar {user_entities.years[0]}]",
        )
        # Handle date-range variant (year prefix in ISO dates)
        result = re.sub(
            rf'&\[{re.escape(PLACEHOLDER_YEAR)}(-\d{{2}}-\d{{2}}T[^\]]*)\]',
            rf'&[{user_entities.years[0]}\1]',
            result,
        )

    if "COUNTRY" in original_map:
        if not user_entities.countries:
            logger.debug("Template fill failed — COUNTRY required but not found in query.")
            return None
        result = result.replace(
            f".&[{PLACEHOLDER_COUNTRY}]",
            f".&[{user_entities.countries[0]}]",
        )
        result = result.replace(
            f'"{PLACEHOLDER_COUNTRY}"',
            f'"{user_entities.countries[0]}"',
        )

    if "COMPANY" in original_map:
        if not user_entities.companies:
            logger.debug("Template fill failed — COMPANY required but not found in query.")
            return None
        result = result.replace(
            f".&[{PLACEHOLDER_COMPANY}]",
            f".&[{user_entities.companies[0]}]",
        )

    if "GOODS" in original_map:
        if not user_entities.goods:
            logger.debug("Template fill failed — GOODS required but not found in query.")
            return None
        result = result.replace(
            f".&[{PLACEHOLDER_GOODS}]",
            f".&[{user_entities.goods[0]}]",
        )

    if "CONTAINER" in original_map:
        if not user_entities.containers:
            logger.debug("Template fill failed — CONTAINER required but not found in query.")
            return None
        result = result.replace(
            f".&[{PLACEHOLDER_CONTAINER}]",
            f".&[{user_entities.containers[0]}]",
        )

    # Sanity check: no unfilled placeholders remain
    unfilled = re.findall(r'\{\{[A-Z]+\}\}', result)
    if unfilled:
        logger.warning("Template fill left unfilled placeholders: %s", unfilled)
        return None

    return result


def has_placeholders(template: str) -> bool:
    """Return True if the template string contains any placeholders."""
    return bool(re.search(r'\{\{[A-Z]+\}\}', template))


def extract_entity_map(template_json: str | dict | None) -> dict:
    """Safely parse entity_map from JSON string or dict."""
    if not template_json:
        return {}
    if isinstance(template_json, dict):
        return template_json
    import json
    try:
        return json.loads(template_json)
    except Exception:
        return {}
