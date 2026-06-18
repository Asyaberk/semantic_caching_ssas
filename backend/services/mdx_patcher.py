"""
MDX Patcher

Takes a cached MDX query and patches it to match the user's actual entities.

Two patch strategies:
  YEAR   — regex substitution of year literals (fast, no LLM)
  ENTITY — LLM rewrites the WHERE / FILTER clause with new entity values
"""

from __future__ import annotations

import re
import logging

from openai import OpenAI

from backend.agents.entity_agent import QuestionEntities
from backend.config import settings

logger = logging.getLogger(__name__)
_openai = OpenAI(api_key=settings.openai_api_key)


# ── Year patching ─────────────────────────────────────────────────────────────

def _replace_year_in_member_key(mdx: str, old_year: int, new_year: int) -> str:
    """
    Replace every occurrence of a 4-digit year used as an MDX member key.

    Handles:
      .&[2025]                   → .&[2024]
      .&[2025-01-01T00:00:00]   → .&[2024-01-01T00:00:00]
      Calendar 2025              → Calendar 2024
    """
    year_str     = str(old_year)
    new_year_str = str(new_year)

    # bare key: .&[2025]
    mdx = re.sub(
        rf'\.&\[{year_str}\]',
        f'.&[{new_year_str}]',
        mdx,
    )
    # date-prefixed key: .&[2025-...
    mdx = re.sub(
        rf'\.&\[{year_str}(-\d{{2}}-\d{{2}}T\d{{2}}:\d{{2}}:\d{{2}}\])',
        f'.&[{new_year_str}\\1',
        mdx,
    )
    # Calendar label: Calendar 2025
    mdx = re.sub(
        rf'\bCalendar {year_str}\b',
        f'Calendar {new_year_str}',
        mdx,
    )
    # Plain year as string literal in STRTOMEMBER: [Calendar 2025]
    mdx = re.sub(
        rf'\[Calendar {year_str}\]',
        f'[Calendar {new_year_str}]',
        mdx,
    )
    return mdx


def patch_years(
    mdx: str,
    cached_entities: QuestionEntities,
    user_entities:   QuestionEntities,
) -> str:
    """
    Replace year references in MDX from cached years → user years.

    If counts match (1 cached → 1 user), does a direct substitution.
    If multiple years exist, replaces in sorted order (oldest→oldest, etc.).
    """
    cached_years = sorted(cached_entities.years)
    user_years   = sorted(user_entities.years)

    if not cached_years or not user_years:
        logger.warning("patch_years called but year lists are empty — skipping.")
        return mdx

    # Pair up years positionally; if counts differ, use the first user year for all
    pairs = list(zip(cached_years, user_years))
    if len(cached_years) > len(user_years):
        # More cached years than user years — fill with last user year
        pairs += [(y, user_years[-1]) for y in cached_years[len(user_years):]]

    for old_y, new_y in pairs:
        if old_y != new_y:
            mdx = _replace_year_in_member_key(mdx, old_y, new_y)
            logger.info("MDX year patched: %d → %d", old_y, new_y)

    return mdx


# ── Entity patching (LLM) ─────────────────────────────────────────────────────

def patch_entities_llm(
    original_question: str,
    user_question:     str,
    cached_mdx:        str,
    cube_name:         str,
) -> str:
    """
    Ask the LLM to adapt the cached MDX for the user's question.

    The LLM receives:
      - The original question the MDX was written for
      - The user's new question
      - The cached MDX

    It returns a minimally modified MDX that answers the user's question.
    """
    prompt = (
        f"You are an MDX expert for Microsoft SQL Server Analysis Services.\n\n"
        f"The following MDX was written for this question:\n"
        f"  Original: {original_question}\n\n"
        f"MDX:\n{cached_mdx}\n\n"
        f"The user is now asking:\n"
        f"  User: {user_question}\n\n"
        f"Adapt the MDX minimally to answer the user's question. "
        f"Keep the same cube (FROM [{cube_name}]) and overall structure. "
        f"Only change the member references or filters that differ. "
        f"Return ONLY the raw MDX query, no explanation."
    )

    try:
        resp = _openai.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        patched = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        patched = re.sub(r"^```(?:mdx|sql)?|```$", "", patched, flags=re.MULTILINE).strip()
        logger.info("LLM entity patch completed for '%s'.", user_question[:60])
        return patched
    except Exception as exc:
        logger.warning("LLM entity patch failed (non-fatal): %s", exc)
        return cached_mdx   # fallback: return original unmodified
