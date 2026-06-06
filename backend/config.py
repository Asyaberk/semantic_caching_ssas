from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded automatically from the .env file.
    Pydantic-settings handles type validation and parsing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-5-nano"
    openai_thinking_effort: str = "minimal"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Qdrant ──────────────────────────────────────────────
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "ssas_qa_cache"

    # ── Langfuse ────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Pipeline ────────────────────────────────────────────
    questions_per_batch: int = 20
    target_question_count: int = 200
    question_language: str = "tr"

    # ── SSAS (real connection, only used when USE_MOCK_CUBE=false) ──
    ssas_url: str = ""
    ssas_api_key: str = ""

    # ── Development ─────────────────────────────────────────
    use_mock_cube: bool = True


# Singleton: all modules import this single instance
settings = Settings()
