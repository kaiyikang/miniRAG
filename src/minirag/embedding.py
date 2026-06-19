from abc import ABC, abstractmethod
import os
import requests

from sentence_transformers import SentenceTransformer


class EmbeddingError(Exception):
    """Raised when the LLM embedding request fails."""


class EmbeddingEngine(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""


class SentenceTransformerEngine(EmbeddingEngine):

    def __init__(self, model: str, cache_dir: str, batch_size: str = 5):

        if not model_name or not cache_dir:
            raise ValueError("Embedding model name or cache dir can not be found!")

        os.makedirs(cache_dir, exist_ok=True)
        self._model = SentenceTransformer(model, cache_folder=cache_dir)
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, batch_size=self._batch_size)
        return embeddings.tolist()


class OpenRouterEmbeddingEngine(EmbeddingEngine):

    BASE_URL = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        settings = get_settings()
        self.model = model or settings.openrouter_embed_model
        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key:
            raise RuntimeError(
                "OpenRouter API key is required. Pass api_key=... or set OPENROUTER_API_KEY."
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = requests.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise EmbeddingError(f"LLM embedding failed: {exc}") from exc

        try:
            return response.json()["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(f"Unexpected response format: {exc}") from exc
