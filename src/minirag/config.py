from pydantic_settings import BaseSettings, SettingsConfigDict

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``.

    Values are read from environment variables first, then from a ``.env`` file
    located in the project root. Missing values default to ``None`` and are
    validated by callers such as ``OpenRouterEngine``.
    """

    openrouter_api_key: str | None = None
    openrouter_model: str = "z-ai/glm-5.2"

    openrouter_embed_model: str = "openai/text-embedding-3-small"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_model_cache_dir: str = str(_PROJECT_ROOT / "temp")

    vector_store_path: str = str(_PROJECT_ROOT / "store")
    collection_name: str = "defualt_collection"

    documents_dir: str = str(_PROJECT_ROOT / "docs")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Return a fresh ``Settings`` instance.

    This is deliberately uncached so tests can mutate ``os.environ`` between
    instantiations without being affected by a cached value.
    """
    return Settings()
