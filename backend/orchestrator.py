"""
Orchestrator

Coordinates the full seeding pipeline:
  1. Discover all cubes via the SchemaProvider
  2. For each cube, generate questions in batches until the target count is reached
  3. Generate MDX for each question
  4. Upload Q&A pairs to Qdrant

The orchestrator keeps a PipelineState object that is updated in real-time
so that FastAPI endpoints can serve live progress to the UI.

Stop is implemented via a flag so the current batch always completes cleanly
before the process exits.
"""

import logging
import threading
from datetime import datetime, timezone

from backend.agents.mdx_agent       import MDXGeneratorAgent
from backend.agents.question_agent  import QuestionGeneratorAgent
from backend.agents.uploader_agent  import QdrantUploaderAgent
from backend.config                 import settings
from backend.mock.cube_formatter    import get_all_cube_names
from backend.models.schemas         import PipelineState
from backend.services.schema_provider import get_schema_provider

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Top-level coordinator for the SSAS cache-seeding pipeline.

    Thread-safe: start() launches the pipeline in a background thread so
    the FastAPI server remains responsive during a run.

    Usage:
        orch = Orchestrator()
        orch.start()          # non-blocking
        orch.stop()           # requests a clean shutdown
        state = orch.get_state()
    """

    def __init__(self) -> None:
        self._stop_event   = threading.Event()
        self._lock         = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state        = PipelineState()

    # ── Public API ────────────────────────────────────────────────────────

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
                target_count=settings.target_question_count,
            )

        self._thread = threading.Thread(target=self._run, daemon=True, name="orchestrator")
        self._thread.start()
        logger.info("Pipeline started.")

    def stop(self) -> None:
        """
        Request a clean shutdown.
        The current batch will finish before the pipeline stops.
        """
        self._stop_event.set()
        logger.info("Stop requested — will halt after current batch.")

    def get_state(self) -> PipelineState:
        """Return a snapshot of the current pipeline state."""
        with self._lock:
            return self._state.model_copy()

    # ── Internal pipeline ─────────────────────────────────────────────────

    def _run(self) -> None:
        """Main pipeline loop — runs in a background thread."""
        try:
            provider        = get_schema_provider()
            q_agent         = QuestionGeneratorAgent(provider=provider)
            mdx_agent       = MDXGeneratorAgent(provider=provider)
            uploader        = QdrantUploaderAgent()

            cube_names = get_all_cube_names(provider)
            logger.info("Discovered %d cube(s): %s", len(cube_names), cube_names)

            self._update_state(total_cubes=len(cube_names))

            for cube_name in cube_names:
                if self._stop_event.is_set():
                    logger.info("Stop flag detected — exiting before cube '%s'.", cube_name)
                    break

                self._update_state(current_cube=cube_name)
                logger.info("Processing cube: %s", cube_name)

                self._process_cube(cube_name, q_agent, mdx_agent, uploader)

            # Mark as completed only if we were not stopped mid-run
            final_status = "stopped" if self._stop_event.is_set() else "completed"
            self._update_state(status=final_status)
            logger.info("Pipeline finished with status: %s", final_status)

        except Exception as exc:
            logger.error("Pipeline crashed: %s", exc, exc_info=True)
            self._update_state(status="error", last_error=str(exc))

    def _process_cube(
        self,
        cube_name: str,
        q_agent:   QuestionGeneratorAgent,
        mdx_agent: MDXGeneratorAgent,
        uploader:  QdrantUploaderAgent,
    ) -> None:
        """
        Run question → MDX → upload batches for a single cube until the
        target question count is reached or a stop is requested.
        """
        target      = settings.target_question_count
        batch_size  = settings.questions_per_batch
        language    = settings.question_language

        while not self._stop_event.is_set():
            with self._lock:
                already_uploaded = self._state.uploaded_count

            if already_uploaded >= target:
                logger.info(
                    "Target of %d reached for cube '%s' — moving on.", target, cube_name
                )
                break

            remaining   = target - already_uploaded
            count       = min(batch_size, remaining)

            logger.info(
                "Cube '%s': %d/%d uploaded — generating %d more questions.",
                cube_name, already_uploaded, target, count,
            )

            # ── Step A: generate questions ─────────────────────────────
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
                logger.error("Question generation failed: %s — retrying next batch.", exc)
                self._update_state(last_error=str(exc))
                continue

            # ── Step B: generate MDX ───────────────────────────────────
            pairs = mdx_agent.generate_batch(questions=questions, cube_name=cube_name)
            self._update_state(
                mdx_generated=self._state.mdx_generated + len(pairs)
            )

            if not pairs:
                logger.warning("No MDX pairs generated in this batch — skipping upload.")
                continue

            # ── Step C: upload to Qdrant ───────────────────────────────
            uploaded = uploader.upload(pairs)
            self._update_state(
                uploaded_count=self._state.uploaded_count + uploaded
            )

    # ── State helpers ─────────────────────────────────────────────────────

    def _update_state(self, **kwargs) -> None:
        """Thread-safe state update. Always refreshes last_updated."""
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)
            self._state.last_updated = datetime.now(timezone.utc).isoformat()
