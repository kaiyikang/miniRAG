from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``.

    Values are read from environment variables first, then from a ``.env`` file
    located in the project root. Missing values default to ``None`` and are
    validated by callers such as ``OpenRouterEngine``.
    """

    openrouter_api_key: str | None = None
    openrouter_model: str = "z-ai/glm-5.2"

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
