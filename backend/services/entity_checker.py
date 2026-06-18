"""
Entity Mismatch Checker

Compares the entities extracted from the user's question against those
from the cached (matched) question and classifies the difference:

  NONE   — all critical entities match → safe to return cached MDX
  YEAR   — only the year(s) differ     → MDX can be patched cheaply
  ENTITY — country/company/goods differ → LLM patch needed
  MAJOR  — structure is too different  → treat as cache miss
"""

from __future__ import annotations

import logging
from enum import Enum

from backend.agents.entity_agent import QuestionEntities

logger = logging.getLogger(__name__)


class MismatchType(str, Enum):
    NONE   = "none"    # all entities match — safe cache hit
    YEAR   = "year"    # only year(s) differ — cheap MDX patch
    ENTITY = "entity"  # country/company/goods differ — LLM patch
    MAJOR  = "major"   # too different — treat as full miss


def _normalise(values: list[str]) -> set[str]:
    return {v.strip().lower() for v in values}


def check_mismatch(
    user: QuestionEntities,
    cached: QuestionEntities,
) -> MismatchType:
    """
    Classify the entity difference between user question and cached question.

    Rules (evaluated in order):
    1. If both have no critical entities → NONE (nothing to compare)
    2. If years differ and nothing else does → YEAR
    3. If named entities (country/company/goods) differ → ENTITY
    4. If both years AND named entities differ → MAJOR
    5. Otherwise → NONE
    """

    year_mismatch    = set(user.years)    != set(cached.years)
    country_mismatch = _normalise(user.countries) != _normalise(cached.countries)
    company_mismatch = _normalise(user.companies) != _normalise(cached.companies)
    goods_mismatch   = _normalise(user.goods)     != _normalise(cached.goods)

    named_mismatch = country_mismatch or company_mismatch or goods_mismatch

    if not year_mismatch and not named_mismatch:
        return MismatchType.NONE

    if year_mismatch and not named_mismatch:
        logger.info(
            "Year mismatch: user=%s cached=%s → YEAR patch",
            user.years, cached.years,
        )
        return MismatchType.YEAR

    if named_mismatch and not year_mismatch:
        logger.info(
            "Named entity mismatch (country=%s company=%s goods=%s) → ENTITY patch",
            country_mismatch, company_mismatch, goods_mismatch,
        )
        return MismatchType.ENTITY

    # Both year and named entities differ
    logger.info("Both year and named entities differ → MAJOR mismatch (treat as miss)")
    return MismatchType.MAJOR
