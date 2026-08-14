from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``.

    Values are read from environment variables first, then from a ``.env`` file
    located in the project root. Missing values default to ``None`` and are
    validated by callers such as ``OpenRouterEngine``.
    """

    openrouter_api_key: str | None = None
    openrouter_model: str = "z-ai/glm-5.2"

    openrouter_embed_model: str = "google/gemini-embedding-2"
    openrouter_rerank_model: str = "cohere/rerank-4-fast"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_model_cache_dir: str = str(_PROJECT_ROOT / "temp")

    vector_store_path: str = str(_DATA_DIR / "chroma")
    collection_name: str = "personal_notes_gemini_embedding_2_v1"
    support_collection_name: str = "support_gemini_embedding_2_v1"

    documents_dir: str = str(_DATA_DIR / "raw" / "blog")

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __str__(self):
        return self.model_dump_json(indent=2, exclude={"openrouter_api_key"})


def get_settings() -> Settings:
    """Return a fresh ``Settings`` instance.

    This is deliberately uncached so tests can mutate ``os.environ`` between
    instantiations without being affected by a cached value.
    """
    return Settings()
