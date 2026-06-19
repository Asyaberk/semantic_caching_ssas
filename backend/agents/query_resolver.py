"""
Query Resolver Agent

Single entry point for all user query resolution:

  resolve(question, cube_name, threshold) → ResolverResult

Decision flow
─────────────
  1. Embed question  +  extract named entities
  2. Qdrant semantic search
  3a. Score ≥ threshold → entity mismatch check
        NONE   → Cache HIT (entities match, return as-is)
        YEAR / ENTITY:
          i.  Cached pair has mdx_template  → fill placeholders → TEMPLATE HIT
                                              (write-through, no LLM needed)
          ii. No template                   → regex year-patch or LLM entity-patch
                                              → PATCHED (write-through)
        MAJOR  → fall through to step 4
  3b. Score < threshold → step 4
  4.  LLM generates fresh MDX → build template → write-through → MISS

Every resolution is logged to query_log for admin visibility.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid

from dataclasses import dataclass
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from backend.config import settings
from backend.agents.entity_agent import extract_entities, QuestionEntities
from backend.agents.mdx_agent import MDXGeneratorAgent
from backend.agents.uploader_agent import QdrantUploaderAgent
from backend.db.database import save_pairs, log_query
from backend.models.schemas import QAPair
from backend.services.entity_checker import check_mismatch, MismatchType
from backend.services.mdx_patcher import patch_years, patch_entities_llm
from backend.services.mdx_template import (
    make_template, fill_template, has_placeholders, extract_entity_map
)
from backend.services.schema_provider import get_schema_provider
from backend.services.question_guard import quick_validate_question, route_question_to_cube

logger = logging.getLogger(__name__)

_NS = uuid.NAMESPACE_DNS


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ResolverResult:
    status:           str            # "hit" | "template" | "patched" | "miss" | "needs_clarification" | "not_answerable"
    source:           str            # "cache" | "template" | "patched" | "llm" | "validation"
    question:         str
    matched_question: str | None
    similarity:       float | None
    mdx:              str
    response_time_ms: int
    cube_name:        str
    pair_id:          str | None
    mismatch:         str | None = None
    feedback_message: str | None = None
    suggestions:      list[str] | None = None


# ── Agent ─────────────────────────────────────────────────────────────────────

class QueryResolverAgent:
    """
    Resolves a natural-language question to an MDX query via the semantic cache.
    Instantiate once at startup and reuse across requests.
    """

    DEFAULT_THRESHOLD     = 0.75
    PARAPHRASE_COUNT      = 5
    DEFAULT_FALLBACK_CUBE = "cubeAccruement"

    def __init__(self) -> None:
        self._openai    = OpenAI(api_key=settings.openai_api_key)
        self._qdrant    = QdrantClient(
            url=settings.qdrant_url,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=True,
        )
        self._provider  = get_schema_provider()
        self._mdx_agent = MDXGeneratorAgent(provider=self._provider)
        self._uploader  = QdrantUploaderAgent()

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(
        self,
        question:   str,
        cube_name:  str | None = None,
        threshold:  float      = DEFAULT_THRESHOLD,
    ) -> ResolverResult:
        t0 = time.perf_counter()

        quick_validation = quick_validate_question(question)
        if quick_validation:
            return self._validation_result(question, cube_name, t0, quick_validation)

        if not settings.enable_semantic_cache:
            return self._full_miss(question, cube_name, t0, None, None)

        vector        = self._embed(question)
        user_entities = extract_entities(question, use_llm=True)
        logger.info("Entities for '%s': %s", question[:60], user_entities.critical_keys())

        results = self._search(vector, cube_name)

        if results and results[0].score >= threshold:
            result = self._handle_candidate(question, results[0], user_entities, t0)
            if result is not None:
                return result

        best_score = round(results[0].score, 4) if results else None
        matched_q  = results[0].payload.get("question") if results else None

        routing = route_question_to_cube(
            question=question,
            provider=self._provider,
            requested_cube=cube_name,
        )
        if not routing.valid:
            return self._validation_result(
                question=question,
                cube_name=routing.suggested_cube or cube_name,
                t0=t0,
                validation=routing,
                similarity=best_score,
                matched_q=matched_q,
            )

        return self._full_miss(question, routing.suggested_cube or cube_name, t0, best_score, matched_q)

    # ── Private: candidate handling ───────────────────────────────────────────

    def _handle_candidate(
        self,
        question:      str,
        best,
        user_entities: QuestionEntities,
        t0:            float,
    ) -> ResolverResult | None:
        payload    = best.payload
        hit_cube   = payload.get("cube_name") or self.DEFAULT_FALLBACK_CUBE
        cached_q   = payload.get("question", "")
        cached_mdx = payload.get("mdx", "")
        score      = round(best.score, 4)
        pair_id    = str(uuid.uuid5(_NS, f"{hit_cube}::{cached_q}"))

        # Template fields stored in Qdrant payload
        mdx_template = payload.get("mdx_template")
        entity_map   = extract_entity_map(payload.get("entity_map"))

        cached_entities = extract_entities(cached_q, use_llm=True)
        mismatch        = check_mismatch(user_entities, cached_entities)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # ── NONE: exact match ─────────────────────────────────────────────────
        if mismatch == MismatchType.NONE:
            logger.info("Cache HIT '%s' (%.4f)", question[:50], score)
            log_query(
                question=question, user_entities=user_entities.model_dump(),
                matched_id=pair_id, matched_q=cached_q,
                similarity=score, mismatch="none", action="hit", cube_name=hit_cube,
            )
            return ResolverResult(
                status="hit", source="cache", mismatch="none",
                question=question, matched_question=cached_q,
                similarity=score, mdx=cached_mdx,
                response_time_ms=elapsed_ms, cube_name=hit_cube, pair_id=pair_id,
            )

        # ── YEAR or ENTITY: try template first ────────────────────────────────
        if mismatch in (MismatchType.YEAR, MismatchType.ENTITY):
            # ── Option A: Template fill ───────────────────────────────────────
            if mdx_template and entity_map and has_placeholders(mdx_template):
                filled = fill_template(mdx_template, user_entities, entity_map)
                if filled:
                    logger.info(
                        "Template HIT — filled %s for '%s'",
                        list(entity_map.keys()), question[:50]
                    )
                    new_id = self._write_through_patch(
                        question, filled, hit_cube,
                        mdx_template=mdx_template, entity_map=entity_map,
                    )
                    log_query(
                        question=question, user_entities=user_entities.model_dump(),
                        matched_id=pair_id, matched_q=cached_q,
                        similarity=score,
                        mismatch=mismatch.value.lower(),
                        action="template",
                        cube_name=hit_cube, patched_mdx=filled,
                    )
                    return ResolverResult(
                        status="template", source="template",
                        mismatch=mismatch.value.lower(),
                        question=question, matched_question=cached_q,
                        similarity=score, mdx=filled,
                        response_time_ms=int((time.perf_counter() - t0) * 1000),
                        cube_name=hit_cube, pair_id=new_id or pair_id,
                    )

            # ── Option B: Regex/LLM patch (no template available yet) ─────────
            if mismatch == MismatchType.YEAR:
                logger.info("YEAR patch (no template) for '%s'", question[:50])
                patched_mdx = patch_years(cached_mdx, cached_entities, user_entities)
                # Also build template from the original so future hits use it
                tmpl, emap = make_template(cached_mdx, cached_entities)
                self._backfill_template(pair_id, hit_cube, cached_q, cached_mdx, tmpl, emap)
                new_id = self._write_through_patch(question, patched_mdx, hit_cube)
                log_query(
                    question=question, user_entities=user_entities.model_dump(),
                    matched_id=pair_id, matched_q=cached_q,
                    similarity=score, mismatch="year", action="patched",
                    cube_name=hit_cube, patched_mdx=patched_mdx,
                )
                return ResolverResult(
                    status="patched", source="patched", mismatch="year",
                    question=question, matched_question=cached_q,
                    similarity=score, mdx=patched_mdx,
                    response_time_ms=int((time.perf_counter() - t0) * 1000),
                    cube_name=hit_cube, pair_id=new_id or pair_id,
                )

            if mismatch == MismatchType.ENTITY:
                logger.info("ENTITY patch (LLM) for '%s'", question[:50])
                patched_mdx = patch_entities_llm(
                    original_question=cached_q,
                    user_question=question,
                    cached_mdx=cached_mdx,
                    cube_name=hit_cube,
                )
                tmpl, emap = make_template(cached_mdx, cached_entities)
                self._backfill_template(pair_id, hit_cube, cached_q, cached_mdx, tmpl, emap)
                new_id = self._write_through_patch(question, patched_mdx, hit_cube)
                log_query(
                    question=question, user_entities=user_entities.model_dump(),
                    matched_id=pair_id, matched_q=cached_q,
                    similarity=score, mismatch="entity", action="patched",
                    cube_name=hit_cube, patched_mdx=patched_mdx,
                )
                return ResolverResult(
                    status="patched", source="patched", mismatch="entity",
                    question=question, matched_question=cached_q,
                    similarity=score, mdx=patched_mdx,
                    response_time_ms=int((time.perf_counter() - t0) * 1000),
                    cube_name=hit_cube, pair_id=new_id or pair_id,
                )

        # ── MAJOR: fall through to miss ───────────────────────────────────────
        logger.info("MAJOR mismatch — treating as miss.")
        return None

    # ── Private: write-through ────────────────────────────────────────────────

    def _write_through_patch(
        self,
        question:     str,
        mdx:          str,
        cube_name:    str,
        mdx_template: str | None = None,
        entity_map:   dict | None = None,
    ) -> str | None:
        """
        Save a patched/template-filled pair to PostgreSQL + Qdrant in background.
        force=True bypasses the 0.95 dedup threshold so similar questions are stored.
        """
        pair_id = str(uuid.uuid4())
        pair    = QAPair(
            id=pair_id, cube_name=cube_name, question=question, mdx=mdx,
            mdx_template=mdx_template, entity_map=entity_map,
        )

        def _save() -> None:
            try:
                save_pairs([pair])
                self._uploader.upload([pair], force=True)
                logger.info("Write-through: cached '%s'.", question[:60])
            except Exception as exc:
                logger.warning("Write-through failed (non-fatal): %s", exc)

        threading.Thread(target=_save, daemon=True).start()
        return pair_id

    def _backfill_template(
        self,
        pair_id:   str,
        cube_name: str,
        question:  str,
        mdx:       str,
        template:  str,
        emap:      dict,
    ) -> None:
        """
        Update an existing cached pair's Qdrant payload with a newly-generated
        template so future queries against it can use Template Hit instead of
        going through the patch path.
        """
        if not has_placeholders(template):
            return   # nothing to backfill

        def _update() -> None:
            try:
                self._qdrant.set_payload(
                    collection_name=settings.qdrant_collection_name,
                    payload={"mdx_template": template, "entity_map": emap},
                    points=[pair_id],
                )
                logger.info("Backfilled template onto pair %s.", pair_id[:12])
            except Exception as exc:
                logger.warning("Template backfill failed (non-fatal): %s", exc)

        threading.Thread(target=_update, daemon=True).start()

    def _full_miss(
        self,
        question:   str,
        cube_name:  str | None,
        t0:         float,
        similarity: float | None,
        matched_q:  str | None,
    ) -> ResolverResult:
        """Generate fresh MDX, build template, write-through, spawn paraphrases."""
        cube    = cube_name or self.DEFAULT_FALLBACK_CUBE
        pair_id = str(uuid.uuid5(_NS, f"{cube}::{question}"))

        logger.info("Cache MISS '%s' (best: %s) — calling LLM.", question[:60], similarity)
        pair = self._mdx_agent.generate_for_question(question=question, cube_name=cube)

        # Build template from the freshly generated MDX
        user_entities = extract_entities(question, use_llm=False)  # fast, no LLM
        template, emap = make_template(pair.mdx, user_entities)
        pair.mdx_template = template if has_placeholders(template) else None
        pair.entity_map   = emap if emap else None

        try:
            save_pairs([pair])
            self._uploader.upload([pair])
        except Exception as exc:
            logger.warning("Write-through (miss) failed: %s", exc)

        threading.Thread(
            target=self._cache_paraphrases,
            args=(question, cube_name, pair.mdx, template, emap),
            daemon=True,
        ).start()

        log_query(
            question=question, matched_q=matched_q,
            similarity=similarity, action="miss", cube_name=cube,
        )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return ResolverResult(
            status="miss", source="llm", mismatch=None,
            question=question, matched_question=matched_q,
            similarity=similarity, mdx=pair.mdx,
            response_time_ms=elapsed_ms, cube_name=cube, pair_id=pair_id,
        )

    def _validation_result(
        self,
        question:   str,
        cube_name:  str | None,
        t0:         float,
        validation,
        similarity: float | None = None,
        matched_q:  str | None = None,
    ) -> ResolverResult:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        cube = cube_name or validation.suggested_cube or ""

        log_query(
            question=question,
            matched_q=matched_q,
            similarity=similarity,
            mismatch=validation.status,
            action=validation.status,
            cube_name=cube or None,
            error=validation.message,
        )

        return ResolverResult(
            status=validation.status,
            source="validation",
            mismatch=validation.status,
            question=question,
            matched_question=matched_q,
            similarity=similarity,
            mdx="",
            response_time_ms=elapsed_ms,
            cube_name=cube,
            pair_id=None,
            feedback_message=validation.message,
            suggestions=validation.suggestions,
        )

    # ── Private: helpers ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        resp = self._openai.embeddings.create(
            model=settings.openai_embedding_model, input=text,
        )
        return resp.data[0].embedding

    def _search(self, vector: list[float], cube_name: str | None):
        qdrant_filter = None
        if cube_name:
            qdrant_filter = Filter(
                must=[FieldCondition(key="cube_name", match=MatchValue(value=cube_name))]
            )
        return self._qdrant.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=vector,
            query_filter=qdrant_filter,
            limit=1,
            with_payload=True,
        )

    def _cache_paraphrases(
        self,
        question:  str,
        cube_name: str | None,
        mdx:       str,
        template:  str,
        emap:      dict,
    ) -> None:
        cube = cube_name or self.DEFAULT_FALLBACK_CUBE
        try:
            prompt = (
                f"Generate {self.PARAPHRASE_COUNT} distinct paraphrases of this "
                f"business question. Keep the same meaning and entities. "
                f"Return one per line, no numbering:\n{question}"
            )
            resp = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            lines = [
                ln.strip("–-•· ")
                for ln in resp.choices[0].message.content.strip().splitlines()
                if ln.strip()
            ][: self.PARAPHRASE_COUNT]

            pairs = [
                QAPair(
                    id=str(uuid.uuid4()), cube_name=cube, question=ln, mdx=mdx,
                    mdx_template=template if has_placeholders(template) else None,
                    entity_map=emap if emap else None,
                )
                for ln in lines if ln
            ]
            if pairs:
                save_pairs(pairs)
                self._uploader.upload(pairs)
                logger.info("Cached %d paraphrases for '%s'.", len(pairs), question[:50])
        except Exception as exc:
            logger.warning("Paraphrase caching failed (non-fatal): %s", exc)
