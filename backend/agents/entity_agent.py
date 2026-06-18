"""
Entity Extractor Agent

Extracts structured named entities from a natural-language question:
  - years       : [2024, 2025]
  - countries   : ["Turkey", "Germany"]
  - companies   : ["Premium", "Standard", "Bikes Ltd"]
  - goods       : ["Bikes", "Accessories"]
  - containers  : ["CONT-123"]
  - date_range  : "Q1", "H2", "2025-06"

Years are extracted with a regex (free, instant).
Named entities use a lightweight LLM call — result is cached per question.
"""

import re
import logging
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

from backend.config import settings

logger = logging.getLogger(__name__)

_openai = OpenAI(api_key=settings.openai_api_key)


class QuestionEntities(BaseModel):
    years:      list[int]   = []
    countries:  list[str]   = []
    companies:  list[str]   = []
    goods:      list[str]   = []
    containers: list[str]   = []
    date_range: str | None  = None   # "Q1", "H1", "2025-06"

    def is_empty(self) -> bool:
        return not any([self.years, self.countries, self.companies,
                        self.goods, self.containers, self.date_range])

    def critical_keys(self) -> dict:
        """Return the subset of entities used for mismatch detection."""
        return {
            "years":     self.years,
            "countries": self.countries,
            "companies": self.companies,
            "goods":     self.goods,
        }


_YEAR_RE       = re.compile(r'\b(20\d{2})\b')
_QUARTER_RE    = re.compile(r'\b(Q[1-4])\b', re.IGNORECASE)
_HALFYEAR_RE   = re.compile(r'\b(H[12])\b', re.IGNORECASE)
_YEARMONTH_RE  = re.compile(r'\b(20\d{2}-(?:0[1-9]|1[0-2]))\b')


def _extract_years(text: str) -> list[int]:
    return sorted({int(y) for y in _YEAR_RE.findall(text)})


def _extract_date_range(text: str) -> str | None:
    m = _YEARMONTH_RE.search(text)
    if m:
        return m.group(1)
    m = _QUARTER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _HALFYEAR_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


@lru_cache(maxsize=512)
def _llm_extract(question: str) -> dict:
    """
    LLM-based extraction for named entities (countries, companies, goods, containers).
    Result is cached in-process to avoid repeated calls for the same question.
    """
    prompt = (
        "Extract named entities from this business question. "
        "Reply with a JSON object with these keys (use empty lists if not found):\n"
        '  "countries": list of country names\n'
        '  "companies": list of company or customer names\n'
        '  "goods": list of goods/product types\n'
        '  "containers": list of container or vessel IDs\n\n'
        f"Question: {question}\n\n"
        "Reply ONLY with the JSON object, no explanation."
    )
    try:
        resp = _openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        import json as _json
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        return _json.loads(raw)
    except Exception as exc:
        logger.warning("LLM entity extraction failed (non-fatal): %s", exc)
        return {}


def extract_entities(question: str, use_llm: bool = True) -> QuestionEntities:
    """
    Extract all named entities from a question.

    Args:
        question: The user's natural-language question.
        use_llm:  If True, use LLM for named entities (countries/companies/goods).
                  Set False for offline/test mode.

    Returns:
        QuestionEntities with all extracted fields.
    """
    years      = _extract_years(question)
    date_range = _extract_date_range(question)

    llm_data: dict = {}
    if use_llm:
        llm_data = _llm_extract(question)

    return QuestionEntities(
        years      = years,
        countries  = llm_data.get("countries", []),
        companies  = llm_data.get("companies", []),
        goods      = llm_data.get("goods", []),
        containers = llm_data.get("containers", []),
        date_range = date_range,
    )
