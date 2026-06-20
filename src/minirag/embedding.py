from abc import ABC, abstractmethod
import os
import requests

from sentence_transformers import SentenceTransformer

from minirag.config import get_settings


class EmbeddingError(Exception):
    """Raised when the LLM embedding request fails."""


class EmbeddingEngine(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""


class SentenceTransformerEngine(EmbeddingEngine):

    def __init__(self, model: str, cache_dir: str, batch_size: int = 5):

        if not model or not cache_dir:
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
        model: str,
        api_key: str,
    ):
        self.model = model
        self.api_key = api_key
        if not self.api_key or not self.model:
            raise RuntimeError(
                "OpenRouter model and API key are required. Pass model=... and api_key=..."
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
