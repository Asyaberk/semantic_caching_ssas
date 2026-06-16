"""
Orchestrator

Coordinates the full pipeline:
  1. Discover cube names from the schema provider
  2. For each cube, generate Q&A pairs (LLM) and upload to Qdrant + PostgreSQL

Each cube has its own independent quota (settings.questions_per_cube).
Grand total accumulates across all cubes and is exposed via /pipeline/status.

Old records are never deleted -- dedup threshold (0.95 cosine) in the uploader
prevents adding semantically identical questions twice.
"""

import logging
import threading
from datetime import datetime, timezone

from backend.agents.mdx_agent import MDXGeneratorAgent
from backend.agents.question_agent import QuestionGeneratorAgent
from backend.agents.uploader_agent import QdrantUploaderAgent
from backend.config import settings
from backend.db.database import init_db, load_pairs_for_cube, save_pairs
from backend.models.schemas import PipelineState, QAPair
from backend.services.schema_provider import get_schema_provider

logger = logging.getLogger(__name__)


def get_all_cube_names(provider) -> list[str]:
    """Return the list of cube names from the schema provider.

    The provider exposes get_cubes() which returns a list of dicts,
    each with at minimum a "name" key.
    """
    try:
        cubes = provider.get_cubes()
        return [c["name"] for c in cubes if c.get("name")]
    except Exception as exc:
        logger.warning("Could not fetch cube names: %s -- using empty list.", exc)
        return []


class Orchestrator:
    """
    Thread-safe pipeline manager.

    Usage:
        orchestrator = Orchestrator()
        orchestrator.start()         # launch background thread
        state = orchestrator.get_state()
        orchestrator.stop()          # request clean shutdown
    """

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None
        self._state       = PipelineState()

    # Public API

    def start(self) -> None:
        """
        Launch the pipeline in a background thread.
        Raises RuntimeError if a run is already in progress.
        """
        with self._lock:
            if self._state.status == "running":
                raise RuntimeError("Pipeline is already running.")

            self._stop_event.clear()
            self._state = PipelineState(
                status="running",
                started_at=datetime.now(timezone.utc).isoformat(),
                cube_target=settings.questions_per_cube,
            )

        self._thread = threading.Thread(target=self._run, daemon=True, name="orchestrator")
        self._thread.start()
        logger.info("Pipeline started.")

    def stop(self) -> None:
        """Request a clean shutdown. Current batch finishes first."""
        self._stop_event.set()
        logger.info("Stop requested -- will halt after current batch.")

    def get_state(self) -> PipelineState:
        """Return a snapshot of the current pipeline state."""
        with self._lock:
            return self._state.model_copy()

    # Internal pipeline

    def _run(self) -> None:
        """Main pipeline loop -- runs in a background thread."""
        try:
            init_db()

            provider  = get_schema_provider()
            q_agent   = QuestionGeneratorAgent(provider=provider)
            mdx_agent = MDXGeneratorAgent(provider=provider)
            uploader  = QdrantUploaderAgent()

            cube_names = get_all_cube_names(provider)
            logger.info("Discovered %d cube(s): %s", len(cube_names), cube_names)

            self._update_state(total_cubes=len(cube_names))

            for cube_name in cube_names:
                if self._stop_event.is_set():
                    logger.info("Stop flag detected -- exiting before cube '%s'.", cube_name)
                    break

                self._update_state(current_cube=cube_name)
                logger.info("Processing cube: %s", cube_name)

                self._process_cube(cube_name, q_agent, mdx_agent, uploader)

            final_status = "stopped" if self._stop_event.is_set() else "completed"
            self._update_state(status=final_status)
            logger.info("Pipeline finished with status: %s", final_status)

        except Exception as exc:
            logger.error("Pipeline crashed: %s", exc, exc_info=True)
            self._update_state(status="error", last_error=str(exc))

    def _process_cube(
        self,
        cube_name: str,
        q_agent: QuestionGeneratorAgent,
        mdx_agent: MDXGeneratorAgent,
        uploader: QdrantUploaderAgent,
    ) -> None:
        """
        Generate, embed and upload Q&A pairs for a single cube.

        Each cube has its own independent quota (settings.questions_per_cube).
        Grand total (uploaded_count) accumulates across all cubes.

        Strategy:
          1. Load existing pairs from PostgreSQL (free, no LLM).
          2. Upload any not-yet-in-Qdrant pairs.
          3. If still below per-cube target, generate more via LLM.
        """
        per_cube_target = settings.questions_per_cube
        batch_size      = settings.questions_per_batch
        language        = settings.question_language

        # Reset per-cube counter in state
        self._update_state(cube_uploaded_count=0)

        # Step 0: load from PostgreSQL and sync to Qdrant
        existing_pairs = load_pairs_for_cube(cube_name)
        cube_uploaded  = 0

        if existing_pairs:
            logger.info(
                "Cube '%s': found %d pair(s) in PostgreSQL -- uploading to Qdrant.",
                cube_name, len(existing_pairs),
            )
            pairs_to_upload = [
                p for p in existing_pairs
                if p.mdx or not settings.enable_mdx_generation
            ]
            cube_uploaded = uploader.upload(pairs_to_upload)
            self._update_state(
                questions_generated=self._state.questions_generated + len(existing_pairs),
                mdx_generated=self._state.mdx_generated + len([p for p in existing_pairs if p.mdx]),
                cube_uploaded_count=cube_uploaded,
            )
            self._add_cube_progress(cube_name, cube_uploaded)

        # Step 1-N: generate new pairs until per-cube target is reached
        while not self._stop_event.is_set():
            if cube_uploaded >= per_cube_target:
                logger.info(
                    "Cube '%s': per-cube target of %d reached (%d uploaded) -- moving on.",
                    cube_name, per_cube_target, cube_uploaded,
                )
                break

            remaining = per_cube_target - cube_uploaded
            count     = min(batch_size, remaining)

            logger.info(
                "Cube '%s': %d/%d -- generating %d more questions.",
                cube_name, cube_uploaded, per_cube_target, count,
            )

            # Step A: generate questions
            try:
                questions = q_agent.generate(
                    cube_name=cube_name,
                    count=count,
                    language=language,
                )
                self._update_state(
                    questions_generated=self._state.questions_generated + len(questions)
                )
            except Exception as exc:
                logger.error("Question generation failed: %s -- retrying next batch.", exc)
                self._update_state(last_error=str(exc))
                continue

            # Step B: generate MDX (optional)
            if settings.enable_mdx_generation:
                pairs = mdx_agent.generate_batch(questions=questions, cube_name=cube_name)
                self._update_state(
                    mdx_generated=self._state.mdx_generated + len(pairs)
                )
            else:
                pairs = [
                    QAPair(question=q, mdx="", cube_name=cube_name)
                    for q in questions
                ]
                logger.info("MDX generation disabled -- storing questions only.")

            if not pairs:
                logger.warning("No pairs generated in this batch -- skipping.")
                continue

            # Step C: save to PostgreSQL
            try:
                save_pairs(pairs)
            except Exception as exc:
                logger.warning("PostgreSQL save failed (non-fatal): %s", exc)

            # Step D: upload to Qdrant
            if settings.enable_mdx_generation:
                batch_uploaded = uploader.upload(pairs)
                cube_uploaded += batch_uploaded
                self._update_state(
                    uploaded_count=self._state.uploaded_count + batch_uploaded,
                    cube_uploaded_count=cube_uploaded,
                )
                self._add_cube_progress(cube_name, cube_uploaded)
            else:
                logger.info("Skipping Qdrant upload (MDX generation is disabled).")

    # State helpers

    def _update_state(self, **kwargs) -> None:
        """Thread-safe state update. Always refreshes last_updated."""
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)
            self._state.last_updated = datetime.now(timezone.utc).isoformat()

    def _add_cube_progress(self, cube_name: str, count: int) -> None:
        """Thread-safe update to the per-cube progress dict."""
        with self._lock:
            self._state.cube_progress[cube_name] = count
            self._state.last_updated = datetime.now(timezone.utc).isoformat()
