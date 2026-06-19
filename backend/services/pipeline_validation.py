"""SSAS quality gate for MDX candidates produced by the seeding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import httpx

from backend.models.schemas import QAPair
from backend.services.cache_validation import classify_bridge_result


RunMdx = Callable[[str], dict]


@dataclass
class PairValidation:
    pair: QAPair
    status: str
    row_count: int = 0
    error: str | None = None


def run_ssas_mdx(mdx: str) -> dict:
    """Execute MDX through the configured SSAS Bridge."""
    from backend.config import settings

    response = httpx.post(
        f"{settings.ssas_url}/api/v1/mdx/query",
        headers={"X-API-Key": settings.ssas_api_key, "Content-Type": "application/json"},
        json={"mdx": mdx, "dataSource": settings.ssas_data_source},
        timeout=45,
    )
    response.raise_for_status()
    return response.json()


def validate_pair_candidates(
    pairs: list[QAPair],
    run_mdx: RunMdx = run_ssas_mdx,
) -> list[PairValidation]:
    """Validate candidates and preserve truthful no-data and failure outcomes."""
    outcomes = []
    for pair in pairs:
        try:
            status, row_count = classify_bridge_result(run_mdx(pair.mdx))
            outcomes.append(PairValidation(pair, status, row_count))
        except Exception as exc:
            outcomes.append(PairValidation(pair, "failed", error=str(exc)[:1000]))
    return outcomes
