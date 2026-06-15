"""
Qdrant Uploader Agent

Takes QAPair objects produced by the MDX Generator Agent, converts each
question into a vector embedding via OpenAI, and upserts the result into
a Qdrant collection.

Key design decisions:
- Embeddings are generated for the question text only (not the MDX).
  At query time, the user's question is embedded and compared against
  stored question embeddings to find the closest cached Q&A pair.
- UUIDs are derived deterministically from (cube_name, question) so that
  re-running the pipeline never creates duplicates — it simply overwrites.
- Upload is done in configurable batches to avoid hitting API rate limits.
"""

import logging
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.config import settings
from backend.models.schemas import QAPair

logger = logging.getLogger(__name__)

EMBEDDING_DIM      = 1536
DUPLICATE_THRESHOLD = 0.95   # cosine similarity above which we skip a new record


class QdrantUploaderAgent:
    """
    Embeds question text and uploads QAPair records to Qdrant.

    Usage:
        agent  = QdrantUploaderAgent()
        count  = agent.upload(pairs)
    """

    def __init__(self) -> None:
        self.openai  = OpenAI(api_key=settings.openai_api_key)
        self.qdrant  = self._connect_qdrant()
        self.collection = settings.qdrant_collection_name
        self._ensure_collection_exists()

    # ── Initialisation ────────────────────────────────────────────────────

    def _connect_qdrant(self) -> QdrantClient:
        """Create and return an authenticated QdrantClient."""
        logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
        return QdrantClient(
            url=settings.qdrant_url,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=True,
        )

    def _ensure_collection_exists(self) -> None:
        """
        Create the Qdrant collection if it does not exist yet.

        Safe to call on every startup — does nothing when the collection
        already exists.
        """
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection in existing:
            logger.info("Collection '%s' already exists — skipping creation.", self.collection)
            return

        logger.info("Creating collection '%s'.", self.collection)
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection '%s' created successfully.", self.collection)

    # ── Public API ────────────────────────────────────────────────────────

    def get_collection_count(self) -> int:
        """
        Return the number of vectors currently stored in the collection.
        Used by the orchestrator to initialise its uploaded_count counter
        so restarts don't re-generate records that already exist.
        """
        try:
            info = self.qdrant.get_collection(self.collection)
            return info.points_count or 0
        except Exception as exc:
            logger.warning("Could not fetch collection count: %s", exc)
            return 0

    def upload(self, pairs: list[QAPair], batch_size: int = 10) -> int:
        """
        Embed questions and upsert all QAPairs into Qdrant.

        Args:
            pairs:      List of QAPair objects to upload.
            batch_size: Number of points sent to Qdrant per API call.

        Returns:
            Number of records successfully uploaded.
        """
        if not pairs:
            logger.warning("upload() called with an empty list — nothing to do.")
            return 0

        uploaded = 0
        skipped  = 0

        # Process in batches
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start: batch_start + batch_size]

            try:
                # Filter out near-duplicates before uploading (Seçenek 3)
                points: list[PointStruct] = []
                for pair in batch:
                    vector = self._embed(pair.question)
                    if self._is_near_duplicate(vector):
                        logger.info(
                            "Skipping near-duplicate: '%s'", pair.question[:60]
                        )
                        skipped += 1
                        continue
                    points.append(self._build_point(pair, vector))

                if not points:
                    continue

                self.qdrant.upsert(
                    collection_name=self.collection,
                    points=points,
                )
                uploaded += len(points)
                logger.info(
                    "Uploaded batch %d–%d (%d uploaded, %d skipped so far).",
                    batch_start + 1,
                    batch_start + len(batch),
                    uploaded,
                    skipped,
                )
            except Exception as exc:
                logger.error(
                    "Batch %d–%d failed — skipping: %s",
                    batch_start + 1,
                    batch_start + len(batch),
                    exc,
                )

        logger.info(
            "Upload complete: %d uploaded, %d skipped as near-duplicates (total input: %d).",
            uploaded, skipped, len(pairs),
        )
        return uploaded

    # ── Private helpers ───────────────────────────────────────────────────

    def _is_near_duplicate(self, vector: list[float]) -> bool:
        """
        Return True if a very similar question (similarity >= DUPLICATE_THRESHOLD)
        already exists in the collection.

        This prevents semantically identical questions from being stored multiple
        times when the pipeline is restarted or the LLM rephrases the same idea.
        """
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=1,
            score_threshold=DUPLICATE_THRESHOLD,
        )
        return len(results) > 0

    def _build_point(self, pair: QAPair, vector: list[float]) -> PointStruct:
        """Convert a QAPair and its pre-computed vector into a Qdrant PointStruct."""
        point_id = self._make_id(pair.cube_name, pair.question)
        payload  = {
            "question":        pair.question,
            "mdx":             pair.mdx,
            "cube_name":       pair.cube_name,
            "dimensions_used": pair.dimensions_used,
            "measures_used":   pair.measures_used,
            "complexity":      pair.complexity.value if pair.complexity else "medium",
        }
        return PointStruct(id=point_id, vector=vector, payload=payload)

    def _embed(self, text: str) -> list[float]:
        """
        Call the OpenAI Embeddings API and return the vector for the given text.

        The model is configurable via OPENAI_EMBEDDING_MODEL in settings
        (default: text-embedding-3-small, dimension: 1536).
        """
        response = self.openai.embeddings.create(
            model=settings.openai_embedding_model,
            input=text,
        )
        return response.data[0].embedding

    @staticmethod
    def _make_id(cube_name: str, question: str) -> str:
        """
        Derive a deterministic UUID from (cube_name, question).

        The same question for the same cube always produces the same UUID,
        so re-running the pipeline overwrites rather than duplicates records.
        """
        key = f"{cube_name}::{question}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
