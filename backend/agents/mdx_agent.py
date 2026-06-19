"""
MDX Generator Agent

Takes a natural language question and the SSAS cube schema, then uses an
LLM to produce a valid MDX query that answers the question.

Questions are processed one at a time so that a failure on a single item
does not abort the entire batch.
"""

import json
import logging
from openai import OpenAI
from backend.config import settings
from backend.mock.cube_formatter import format_cube_for_llm
from backend.models.schemas import QAPair, ComplexityLevel
from backend.services.schema_provider import SchemaProvider, get_schema_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an MDX query expert for SQL Server Analysis Services (SSAS) cubes.
Given a cube schema and a natural language question, write a valid MDX query
that correctly answers the question.

Rules:
- Use exact dimension and hierarchy names from the schema. Never invent names.
- The query must be syntactically valid and executable against the described cube.
- Answer only from the target cube. Keep the FROM clause exactly on that cube.
- If a requested business concept is not present in the schema, do not substitute
  a similar-looking measure or dimension. The application validates this before
  calling you, so prefer precision over guessing.
- Return ONLY a valid JSON object — no markdown, no explanation.

CRITICAL — Year/Date filtering:
- Never use bare year keys like [DateDim].[Year].&[2025] — SSAS rejects these.
- Instead, filter by date range on the Date level:
    WHERE {[AccruementDate].[Date].&[2025-01-01T00:00:00]:[AccruementDate].[Date].&[2025-12-31T00:00:00]}
- Each cube has its own date dimension prefix (AccruementDate, CreditDebitDate, etc.).
  Always use [<CubeName>Date].[Date] for date range filtering.

CRITICAL — Dimension members on ROWS:
- Attribute hierarchies (single-level) do NOT support .Members in ON ROWS.
- Only user hierarchies (multi-level, shown in schema) support .Members.
- For country/company/goods breakdowns, use WHERE clause filtering instead:
    WHERE [AccruementCompany].[CountryName].&[Türkiye]
- To list members of a user hierarchy: [Dim].[HierarchyName].Members ON ROWS

CRITICAL — Member key values:
- Member keys in this SSAS instance use local language (e.g. Türkiye not Turkey).
- Always use the member key exactly as it appears in the cube schema.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MDXGeneratorAgent:
    """
    Generates an MDX query for each natural language question.

    Questions are processed one at a time so that a failure on a single
    item does not abort the entire batch.

    Usage:
        agent = MDXGeneratorAgent()
        pairs = agent.generate_batch(questions=[...], cube_name="Sales")
    """

    def __init__(self, provider: SchemaProvider | None = None):
        self.provider = provider or get_schema_provider()
        self.client   = OpenAI(api_key=settings.openai_api_key)
        # Cache formatted schema text per cube to avoid redundant work
        self._schema_cache: dict[str, str] = {}
        self._langfuse = self._init_langfuse()

    # ── Langfuse tracing ──────────────────────────────────────────────────
    # IMPORTANT: Langfuse integration is intentional and must be kept.
    # It records every LLM call (model, tokens, cost, input/output) to
    # https://langfuse.iotiq.dev for the shared project.
    # Do NOT remove this block — disable by leaving credentials empty in .env.

    def _init_langfuse(self):
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            logger.warning("Langfuse credentials not configured — tracing disabled.")
            return None
        try:
            from langfuse import Langfuse
            lf = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse tracing enabled → %s", settings.langfuse_host)
            return lf
        except Exception as exc:
            logger.warning("Failed to initialise Langfuse (tracing disabled): %s", exc)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_schema_text(self, cube_name: str) -> str:
        """Return formatted schema text, cached per cube name."""
        if cube_name not in self._schema_cache:
            self._schema_cache[cube_name] = format_cube_for_llm(cube_name, self.provider)
        return self._schema_cache[cube_name]

    # ── Public API ────────────────────────────────────────────────────────

    def generate_for_question(self, question: str, cube_name: str) -> QAPair:
        """
        Generate an MDX query for a single natural language question.

        Args:
            question:  Natural language question (e.g. "Net Revenue for Turkey in 2026?").
            cube_name: Name of the target SSAS cube (e.g. "Sales").

        Returns:
            A QAPair with the question, MDX query, and metadata fields populated.

        Raises:
            RuntimeError: If the LLM call fails or the response cannot be parsed.
        """
        schema_text = self._get_schema_text(cube_name)
        user_prompt = self._build_prompt(schema_text, question, cube_name)

        # Start Langfuse trace when available
        trace = None
        if self._langfuse:
            trace = self._langfuse.trace(
                name="mdx_generation",
                input={"question": question, "cube_name": cube_name},
                tags=["mdx-agent"],
            )

        try:
            params: dict = {
                "model": settings.openai_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }

            # reasoning_effort is supported by gpt-5-nano and other reasoning models.
            # Skip the parameter entirely for models that do not support it.
            if settings.openai_thinking_effort:
                params["reasoning_effort"] = settings.openai_thinking_effort

            response = self.client.chat.completions.create(**params)

            raw    = response.choices[0].message.content
            parsed = self._parse_response(raw)

            pair = QAPair(
                question=question,
                mdx=parsed["mdx"],
                cube_name=cube_name,
                dimensions_used=parsed.get("dimensions_used", []),
                measures_used=parsed.get("measures_used", []),
                complexity=ComplexityLevel(parsed.get("complexity", "medium")),
                langfuse_trace_id=trace.id if trace else None,
                upload_status="pending",
            )

            # Log the LLM call as a generation — populates model, tokens and cost in Langfuse.
            if trace:
                trace.generation(
                    name="openai-completion",
                    model=settings.openai_model,
                    input=params["messages"],
                    output=raw,
                    usage={
                        "input":  response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total":  response.usage.total_tokens,
                    },
                )
                trace.update(
                    output={"mdx": pair.mdx, "complexity": str(pair.complexity)},
                )
                self._langfuse.flush()   # force-send before returning

            logger.info("MDX generated for: '%s'", question[:70])
            return pair

        except Exception as exc:
            if trace:
                trace.update(output={"error": str(exc)}, level="ERROR")
            logger.error("MDX generation failed for '%s': %s", question[:70], exc)
            raise RuntimeError(f"MDX generation failed: {exc}") from exc

    def generate_batch(self, questions: list[str], cube_name: str) -> list[QAPair]:
        """
        Generate MDX queries for a list of questions.

        Each question is processed independently. A failure on one item is
        logged and skipped so the rest of the batch can continue.

        Args:
            questions: List of natural language questions from the Question Agent.
            cube_name: Name of the target cube.

        Returns:
            List of successfully generated QAPair objects (failures excluded).
        """
        results: list[QAPair] = []
        failed = 0

        for index, question in enumerate(questions, start=1):
            try:
                pair = self.generate_for_question(question, cube_name)
                results.append(pair)
                logger.info("Batch progress: %d / %d", index, len(questions))
            except Exception as exc:
                failed += 1
                logger.warning(
                    "Skipping question %d / %d — %s", index, len(questions), exc
                )

        logger.info(
            "Batch complete: %d succeeded, %d failed (total %d).",
            len(results), failed, len(questions),
        )
        return results

    # ── Private ───────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> dict:
        """
        Parse and lightly validate the JSON response from the LLM.

        Expected shape: { "mdx": "...", "dimensions_used": [...],
                          "measures_used": [...], "complexity": "..." }
        """
        parsed = json.loads(raw)
        if "mdx" not in parsed:
            raise ValueError(f"LLM response missing 'mdx' key: {raw[:300]}")

        # Normalise complexity to one of the three accepted values
        raw_complexity = str(parsed.get("complexity", "medium")).lower()
        parsed["complexity"] = (
            raw_complexity if raw_complexity in ("simple", "medium", "complex")
            else "medium"
        )
        return parsed

    def _build_prompt(self, schema_text: str, question: str, cube_name: str) -> str:
        """Construct the user prompt sent to the LLM."""
        return f"""\
Below is the schema of the '{cube_name}' SSAS cube:

{schema_text}

Write an MDX query that answers the following question:
"{question}"

Important:
- Use only the schema above for cube '{cube_name}'.
- Keep all user-requested filters and breakdowns. Do not simplify to a grand
  total when the user asked for a country, vessel, customer, date, or ranking.
- If a member value is needed, use the key/caption style shown in the schema.
- Prefer NON EMPTY when putting sets on ROWS.

Return a JSON object with exactly these keys:
{{
  "mdx": "<the complete MDX query string>",
  "dimensions_used": ["<dimension caption>", ...],
  "measures_used":   ["<measure caption>", ...],
  "complexity": "simple" | "medium" | "complex"
}}

Complexity guide:
  simple  — one measure, one filter
  medium  — two or three filters, or a single breakdown
  complex — comparisons, trends, rankings, or more than three filters
"""
