"""
Question Generator Agent

Connects to an LLM and generates diverse, realistic natural language
questions that a business user might ask about a given SSAS cube.

The generated questions are later passed to the MDX Generator Agent,
which produces a corresponding MDX query for each one.

Langfuse tracing is enabled when credentials are configured in the
environment. If they are absent the agent runs without tracing.
"""

import json
import logging
from openai import OpenAI
from backend.config import settings
from backend.mock.cube_formatter import format_cube_for_llm
from backend.services.schema_provider import SchemaProvider, get_schema_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert business analyst specialising in SSAS (SQL Server Analysis Services) cubes.
Your task is to generate realistic, diverse natural language questions that business users
would ask about the provided cube schema.

Guidelines:
- Always use actual member names from the schema (e.g. Turkey, Bikes, 2025, Q1).
- Cover all dimensions and measures across the question set.
- Mix simple single-filter questions with complex multi-dimension ones.
- Every question must be answerable with a single MDX query on this cube.
- Return ONLY a valid JSON object with a "questions" key — no markdown, no explanation.
"""

# Categories guide the LLM to produce varied questions instead of repetitive ones.
QUESTION_CATEGORIES = [
    "Single measure filtered by one dimension member (e.g. Net Revenue for Turkey in 2025)",
    "Year-over-year comparison (e.g. 2024 vs 2025 total revenue)",
    "Customer segment analysis (e.g. Premium vs Standard gross margin)",
    "Product category breakdown (e.g. Bikes vs Accessories order count in 2025)",
    "Multi-dimension filter (e.g. Premium customers in Germany, 2026 Q1 Net Revenue)",
    "Quarterly drill-down (e.g. Q1 through Q4 Gross Margin for 2024)",
    "Monthly trend (e.g. monthly Net Revenue trend across all of 2025)",
    "Top / bottom ranking (e.g. which country had the lowest order count in 2025)",
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QuestionGeneratorAgent:
    """
    Generates natural language business questions for a given SSAS cube.

    Usage:
        agent = QuestionGeneratorAgent()
        questions = agent.generate(cube_name="Sales", count=20, language="en")
    """

    def __init__(self, provider: SchemaProvider | None = None):
        self.provider = provider or get_schema_provider()
        self.client   = OpenAI(api_key=settings.openai_api_key)
        self._langfuse = self._init_langfuse()

    # ── Langfuse ──────────────────────────────────────────────────────────

    def _init_langfuse(self):
        """
        Initialise Langfuse if credentials are present in the environment.
        Returns None and logs a warning when they are not set, so the rest
        of the agent continues to work without tracing.
        """
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            logger.warning("Langfuse credentials not configured — tracing disabled.")
            return None
        try:
            from langfuse import Langfuse
            return Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception as exc:
            logger.warning("Failed to initialise Langfuse: %s", exc)
            return None

    # ── Public API ────────────────────────────────────────────────────────

    def generate(
        self,
        cube_name: str,
        count: int,
        language: str = "en",
    ) -> list[str]:
        """
        Generate `count` natural language questions for the given cube.

        Args:
            cube_name: Name of the SSAS cube (e.g. "Sales").
            count:     Number of questions to generate in this call.
            language:  Output language — "en" for English, "tr" for Turkish.

        Returns:
            A list of question strings.

        Raises:
            RuntimeError: If the LLM call fails or the response cannot be parsed.
        """
        schema_text = format_cube_for_llm(cube_name, self.provider)
        user_prompt = self._build_prompt(schema_text, count, language)

        # Start a Langfuse trace when available
        trace = None
        if self._langfuse:
            trace = self._langfuse.trace(
                name="question_generation",
                input={"cube_name": cube_name, "count": count, "language": language},
                tags=["question-agent"],
            )

        try:
            response = self._call_llm(user_prompt)
            questions = self._parse_response(response.choices[0].message.content)

            # Log successful result to Langfuse
            if trace:
                trace.update(
                    output={"questions_count": len(questions), "questions": questions},
                    usage={
                        "input":  response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                    },
                )

            logger.info("Generated %d questions for cube '%s'.", len(questions), cube_name)
            return questions

        except Exception as exc:
            if trace:
                trace.update(level="ERROR", status_message=str(exc))
            logger.error("Question generation failed for cube '%s': %s", cube_name, exc)
            raise RuntimeError(f"Question generation failed: {exc}") from exc

    # ── Private helpers ───────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str):
        """Send the prompt to the configured OpenAI model and return the raw response."""
        return self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

    def _parse_response(self, raw: str) -> list[str]:
        """
        Parse the JSON response from the LLM.

        The model is instructed to return {"questions": [...]}.
        As a fallback it also handles a bare JSON array.
        """
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("questions", "Questions", "items", "results"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
        raise ValueError(f"Unexpected response format from LLM: {raw[:200]}")

    def _build_prompt(self, schema_text: str, count: int, language: str) -> str:
        """Construct the user-facing prompt sent to the LLM."""
        lang_line = (
            "Write every question in English."
            if language == "en"
            else "Write every question in Turkish."
        )
        categories_text = "\n".join(f"  - {c}" for c in QUESTION_CATEGORIES)

        return f"""\
Below is the schema of an SSAS cube:

{schema_text}

{lang_line}

Generate exactly {count} diverse, realistic business questions a user might ask about this cube.
Cover all of the following question categories:
{categories_text}

Return a JSON object with a single key "questions" containing an array of exactly {count} strings.
Example: {{"questions": ["What is Net Revenue for Turkey in 2025?", ...]}}
"""
