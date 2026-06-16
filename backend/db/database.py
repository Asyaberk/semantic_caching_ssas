"""
PostgreSQL database layer.

Handles connection setup, schema creation, and CRUD operations for QA pairs.

The database acts as a persistent backup for generated question/MDX pairs so
that the pipeline can be restarted without re-calling the LLM. Qdrant stores
the vector embeddings; PostgreSQL stores the raw text and metadata.

Table: qa_pairs
  id                UUID PRIMARY KEY  (deterministic: uuid5 of cube:question)
  cube_name         VARCHAR
  question          TEXT
  mdx               TEXT (nullable when ENABLE_MDX_GENERATION=false)
  complexity        VARCHAR
  dimensions_used   JSONB
  measures_used     JSONB
  langfuse_trace_id VARCHAR (nullable)
  created_at        TIMESTAMP
"""

import json
import logging
import uuid

import psycopg2
import psycopg2.extras

from backend.config import settings
from backend.models.schemas import QAPair, ComplexityLevel

logger = logging.getLogger(__name__)

# Deterministic namespace — same as uploader_agent so IDs match across both stores
_NS = uuid.NAMESPACE_DNS


def _make_id(cube_name: str, question: str) -> str:
    """Return a deterministic UUID string for a (cube, question) pair."""
    return str(uuid.uuid5(_NS, f"{cube_name}:{question}"))


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection():
    """Return a new psycopg2 connection. Caller is responsible for closing it."""
    return psycopg2.connect(settings.postgres_url)


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the qa_pairs table if it does not exist.
    Safe to call on every startup.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS qa_pairs (
        id                TEXT PRIMARY KEY,
        cube_name         VARCHAR(255) NOT NULL,
        question          TEXT        NOT NULL,
        mdx               TEXT,
        complexity        VARCHAR(50),
        dimensions_used   JSONB,
        measures_used     JSONB,
        langfuse_trace_id VARCHAR(255),
        created_at        TIMESTAMP   DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_qa_pairs_cube ON qa_pairs(cube_name);
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    logger.info("PostgreSQL schema ready (qa_pairs table exists).")


# ── Write ─────────────────────────────────────────────────────────────────────

def save_pairs(pairs: list[QAPair]) -> int:
    """
    Insert QA pairs into PostgreSQL.

    Uses INSERT ... ON CONFLICT DO NOTHING so re-running the pipeline with
    duplicate questions is safe — existing rows are not overwritten.

    Returns the number of rows actually inserted.
    """
    if not pairs:
        return 0

    rows = [
        (
            _make_id(p.cube_name, p.question),
            p.cube_name,
            p.question,
            p.mdx,
            p.complexity.value if p.complexity else None,
            json.dumps(p.dimensions_used or []),
            json.dumps(p.measures_used  or []),
            p.langfuse_trace_id,
        )
        for p in pairs
    ]

    sql = """
        INSERT INTO qa_pairs
            (id, cube_name, question, mdx, complexity,
             dimensions_used, measures_used, langfuse_trace_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
            inserted = cur.rowcount
        conn.commit()

    logger.info("Saved %d new QA pair(s) to PostgreSQL (skipped duplicates).", inserted)
    return inserted


# ── Read ──────────────────────────────────────────────────────────────────────

def load_pairs_for_cube(cube_name: str) -> list[QAPair]:
    """
    Load all stored QA pairs for a cube from PostgreSQL.

    The orchestrator calls this at startup so it can upload existing pairs to
    Qdrant without calling the LLM again.
    """
    sql = """
        SELECT question, mdx, complexity, dimensions_used, measures_used,
               langfuse_trace_id
        FROM   qa_pairs
        WHERE  cube_name = %s
        ORDER  BY created_at ASC
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, (cube_name,))
            rows = cur.fetchall()

    pairs = []
    for row in rows:
        try:
            complexity = ComplexityLevel(row["complexity"]) if row["complexity"] else None
        except ValueError:
            complexity = None

        pairs.append(QAPair(
            question          = row["question"],
            mdx               = row["mdx"] or "",
            cube_name         = cube_name,
            complexity        = complexity,
            dimensions_used   = row["dimensions_used"] or [],
            measures_used     = row["measures_used"]   or [],
            langfuse_trace_id = row["langfuse_trace_id"],
        ))

    logger.info(
        "Loaded %d QA pair(s) from PostgreSQL for cube '%s'.", len(pairs), cube_name
    )
    return pairs


def count_pairs_for_cube(cube_name: str) -> int:
    """Return the number of stored QA pairs for a cube."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE cube_name = %s", (cube_name,)
            )
            return cur.fetchone()[0]
