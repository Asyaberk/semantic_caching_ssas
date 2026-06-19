"""Truthful, testable MDX execution orchestration.

The executor never removes filters or axes to manufacture a result.  It may
retry a failed query with a narrowly-scoped year-key correction and an
LLM-provided repair, but always reports the exact MDX that SSAS executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


RunMdx = Callable[[str], dict]
RepairMdx = Callable[[str, str, str, str], str | None]


DATE_DIMENSIONS: dict[str, str] = {
    "cubeAccruement": "AccruementDate",
    "cubeCreditDebit": "CreditDebitDate",
    "cubeDwellTimeRoro": "DwellDate",
    "cubeGangPoint": "GangDate",
    "cubeGeneralJobOrders": "SuccessDate",
    "cubeOtherJobOrders": "SuccessDate",
    "cubeVesselJobOrder": "SuccessDate",
    "cubeVesselOrder": "MoorageDate",
    "cubeWaiting": "WaitingStartDate",
}


@dataclass
class ExecutionResult:
    status: str
    columns: list[dict] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    elapsed_ms: int | None = None
    error: str | None = None
    executed_mdx: str = ""
    attempt: str = "original"

    @property
    def validated(self) -> bool:
        return self.status in {"success", "no_data"}


def extract_cube_name(mdx: str) -> str:
    """Return the cube in the FROM clause, or an empty string."""
    match = re.search(r"\bFROM\s+\[([^\]]+)\]", mdx, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def fix_bare_year_keys(mdx: str, cube_name: str) -> str:
    """Replace bare four-digit member keys with the cube's date range.

    This retry is deliberately narrow and is only used after SSAS rejects the
    original query. Unknown cubes are left untouched instead of guessing a
    generic date dimension.
    """
    date_dimension = DATE_DIMENSIONS.get(cube_name)
    if not date_dimension:
        return mdx

    def to_range(match: re.Match) -> str:
        year = match.group(2)
        return (
            f"{{[{date_dimension}].[Date].&[{year}-01-01T00:00:00]"
            f":[{date_dimension}].[Date].&[{year}-12-31T00:00:00]}}"
        )

    return re.sub(
        r"(\[[^\]]+\]\.\[[^\]]+\])\.&\[(\d{4})\]",
        to_range,
        mdx,
    )


def _successful_result(data: dict, mdx: str, attempt: str) -> ExecutionResult:
    rows = data.get("rows") or []
    return ExecutionResult(
        status="success" if rows else "no_data",
        columns=data.get("columns") or [],
        rows=rows,
        row_count=data.get("rowCount", len(rows)),
        elapsed_ms=data.get("elapsedMs"),
        executed_mdx=mdx,
        attempt=attempt,
    )


def execute_with_repair(
    *,
    mdx: str,
    cube_name: str,
    question: str,
    run_mdx: RunMdx,
    repair_mdx: RepairMdx,
) -> ExecutionResult:
    """Execute MDX without silently changing the requested business meaning."""
    last_error = ""

    try:
        return _successful_result(run_mdx(mdx), mdx, "original")
    except Exception as exc:  # bridge exceptions are normalised by the caller
        last_error = str(exc)

    fixed_mdx = fix_bare_year_keys(mdx, cube_name)
    if fixed_mdx != mdx:
        try:
            return _successful_result(run_mdx(fixed_mdx), fixed_mdx, "year_fix")
        except Exception as exc:
            last_error = str(exc)

    repaired_mdx = repair_mdx(mdx, last_error, cube_name, question)
    if repaired_mdx and repaired_mdx.strip() and repaired_mdx.strip() != mdx.strip():
        repaired_mdx = repaired_mdx.strip()
        if extract_cube_name(repaired_mdx) != cube_name:
            return ExecutionResult(
                status="failed",
                error="MDX onarımı hedef cube'u değiştirdiği için reddedildi.",
                executed_mdx=mdx,
                attempt="failed",
            )
        try:
            return _successful_result(run_mdx(repaired_mdx), repaired_mdx, "llm_repair")
        except Exception as exc:
            last_error = str(exc)

    return ExecutionResult(
        status="failed",
        error=last_error[:500] or "SSAS sorgusu çalıştırılamadı.",
        executed_mdx=mdx,
        attempt="failed",
    )
