"""
Question quality and cube-routing guard for user-facing demo queries.

The guard deliberately stays deterministic. It prevents obviously bad or
out-of-scope prompts from falling through to the LLM with the wrong default cube,
and it suggests the most likely cube when the question contains schema terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.services.schema_provider import SchemaProvider


_WORD_RE = re.compile(r"[\wğüşöçıİĞÜŞÖÇ]+", re.UNICODE)
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "how", "in",
    "is", "me", "of", "on", "show", "the", "there", "to", "total", "what",
    "with",
    "adet", "bir", "bu", "da", "de", "en", "göre", "hangi", "kaç", "kac",
    "kadar", "listele", "mi", "mı", "mu", "mü", "ne", "nedir", "olan",
    "olarak", "toplam", "var", "ve", "ya", "yıl", "yili", "yılı",
}

_CHATTER = {
    "hi", "hello", "hey", "merhaba", "selam", "test", "deneme", "asdf",
    "lorem", "naber", "nasılsın", "nasilsin",
}

_CUBE_HINTS = {
    "cubeAccruement": {"accruement", "accrual", "tahakkuk", "invoice", "vat", "grt"},
    "cubeCreditDebit": {"credit", "debit", "alacak", "borç", "borc", "movement"},
    "cubeDwellTimeRoro": {"dwell", "roro", "ro", "vehicle", "bekleme"},
    "cubeGangPoint": {"gang", "point", "worker", "işçi", "isci", "puan"},
    "cubeGeneralJobOrders": {"general", "job", "order", "operation", "work"},
    "cubeOtherJobOrders": {"other", "equipment", "service", "job", "order"},
    "cubeVesselJobOrder": {"vessel", "ship", "job", "order", "crane", "transport"},
    "cubeVesselOrder": {"vessel", "ship", "berth", "berthing", "moorage", "order"},
    "cubeWaiting": {"waiting", "wait", "vessel", "ship", "equipment", "bekleme"},
}


@dataclass(frozen=True)
class QuestionGuardResult:
    status: str
    valid: bool
    message: str
    suggested_cube: str | None = None
    confidence: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def tokenize(text: str) -> set[str]:
    """Return normalized searchable tokens."""
    normalized = text.replace("İ", "i").casefold()
    words = set(_WORD_RE.findall(normalized))
    expanded: set[str] = set()
    for word in words:
        expanded.add(word)
        for part in _CAMEL_RE.split(word):
            if part:
                expanded.add(part.casefold())
        if word.startswith("cube") and len(word) > 4:
            expanded.add(word[4:])
    return {w for w in expanded if len(w) > 1}


def quick_validate_question(question: str) -> QuestionGuardResult | None:
    """Catch empty/chatter prompts before any expensive embedding or LLM call."""
    cleaned = " ".join(question.split())
    tokens = tokenize(cleaned)
    meaningful = [t for t in tokens if t not in _STOP_WORDS and not _YEAR_RE.fullmatch(t)]

    if not cleaned:
        return _needs_clarification("Lütfen SSAS cube verileriyle ilgili bir iş sorusu yazın.")

    if len(cleaned) < 8 or tokens <= _CHATTER:
        return _needs_clarification(
            "Soru çok kısa veya iş bağlamı taşımıyor. Hangi metriği, hangi kırılımı ve mümkünse yılı yazın."
        )

    if not meaningful:
        return _needs_clarification(
            "Soru yeterince açık değil. Örn: “2025 Türkiye toplam tahakkuk nedir?” gibi metrik + filtre belirtin."
        )

    return None


def route_question_to_cube(
    question: str,
    provider: SchemaProvider,
    requested_cube: str | None = None,
) -> QuestionGuardResult:
    """
    Determine whether the question is answerable from known cube schema terms.

    If a cube is explicitly requested by the caller, validate against that cube
    and return it unless the prompt is clearly out of scope.
    """
    quick = quick_validate_question(question)
    if quick:
        return quick

    try:
        cubes = provider.get_cubes()
    except Exception:
        cubes = []

    if not cubes:
        if requested_cube:
            return QuestionGuardResult(
                status="ok",
                valid=True,
                message="Cube şeması okunamadı; seçilen cube ile devam ediliyor.",
                suggested_cube=requested_cube,
                confidence=0.4,
            )
        return _needs_clarification("Cube şeması şu an okunamadı; lütfen daha sonra tekrar deneyin.")

    known_names = {c.get("name") for c in cubes if c.get("name")}
    if requested_cube and requested_cube not in known_names:
        return QuestionGuardResult(
            status="not_answerable",
            valid=False,
            message=f"“{requested_cube}” adında bilinen bir cube yok.",
            suggestions=_generic_suggestions(),
        )

    scores = _score_cubes(question, provider, cubes)
    if requested_cube:
        score = scores.get(requested_cube, (0, []))
        return QuestionGuardResult(
            status="ok",
            valid=True,
            message="Seçilen cube ile devam ediliyor.",
            suggested_cube=requested_cube,
            confidence=min(1.0, score[0] / 8) if score else 0.5,
            matched_terms=score[1] if score else [],
        )

    ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
    top_cube, (top_score, top_terms) = ranked[0] if ranked else (None, (0, []))
    second_score = ranked[1][1][0] if len(ranked) > 1 else 0

    if not top_cube or top_score <= 0:
        return QuestionGuardResult(
            status="not_answerable",
            valid=False,
            message=(
                "Bu sorunun mevcut SSAS cube şemasında karşılığını bulamadım. "
                "Cube verileriyle ilişkili metrik, tarih, ülke, gemi, sipariş veya bekleme gibi alanları belirtin."
            ),
            suggestions=_generic_suggestions(),
        )

    if top_score < 2 or (second_score and top_score - second_score < 1):
        return QuestionGuardResult(
            status="needs_clarification",
            valid=False,
            message=(
                "Soru hangi cube/veri alanına ait yeterince net değil. "
                f"En yakın adaylar: {', '.join(name for name, _ in ranked[:3])}."
            ),
            suggestions=_generic_suggestions(),
        )

    return QuestionGuardResult(
        status="ok",
        valid=True,
        message=f"Soru {top_cube} cube'una yönlendirildi.",
        suggested_cube=top_cube,
        confidence=min(1.0, top_score / 8),
        matched_terms=top_terms,
    )


def _score_cubes(
    question: str,
    provider: SchemaProvider,
    cubes: list[dict],
) -> dict[str, tuple[int, list[str]]]:
    q_tokens = tokenize(question)
    scores: dict[str, tuple[int, list[str]]] = {}

    for cube in cubes:
        cube_name = cube.get("name")
        if not cube_name:
            continue

        keywords = set()
        keywords |= tokenize(cube_name)
        keywords |= tokenize(cube.get("caption") or "")
        for alias in cube.get("aliases") or []:
            keywords |= tokenize(str(alias))
        keywords |= _CUBE_HINTS.get(cube_name, set())

        try:
            for dim in provider.get_dimensions(cube_name):
                keywords |= tokenize(dim.get("name") or "")
                keywords |= tokenize(dim.get("caption") or "")
                keywords |= tokenize(dim.get("unique_name") or "")
                for alias in dim.get("aliases") or []:
                    keywords |= tokenize(str(alias))
        except Exception:
            pass

        try:
            for measure in provider.get_measures(cube_name):
                keywords |= tokenize(measure.get("name") or "")
                keywords |= tokenize(measure.get("caption") or "")
                keywords |= tokenize(measure.get("unique_name") or "")
                for alias in measure.get("aliases") or []:
                    keywords |= tokenize(str(alias))
        except Exception:
            pass

        keywords = {k for k in keywords if k not in _STOP_WORDS}
        matched = sorted(q_tokens & keywords)
        score = len(matched)

        phrase = question.casefold()
        for hint in _CUBE_HINTS.get(cube_name, set()):
            if len(hint) > 3 and hint in phrase and hint not in matched:
                matched.append(hint)
                score += 1

        scores[cube_name] = (score, sorted(set(matched)))

    return scores


def _needs_clarification(message: str) -> QuestionGuardResult:
    return QuestionGuardResult(
        status="needs_clarification",
        valid=False,
        message=message,
        suggestions=_generic_suggestions(),
    )


def _generic_suggestions() -> list[str]:
    return [
        "Metrik yazın: count, amount, waiting time, accruement gibi.",
        "Kırılım/filtre ekleyin: ülke, gemi, müşteri, ekipman veya yıl.",
        "Örnek: “2025 Türkiye toplam tahakkuk nedir?”",
    ]
