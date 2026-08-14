import os

import requests

from minirag.domain.ports import EmbeddingEngine, EmbeddingError


class SentenceTransformerEngine(EmbeddingEngine):

    def __init__(self, model: str, cache_dir: str, batch_size: int = 5):
        if not model or not cache_dir:
            raise ValueError("Embedding model name or cache dir can not be found!")

        from sentence_transformers import SentenceTransformer

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

    def __init__(self, model: str, api_key: str, batch_size: int = 100):
        self._model = model
        self._api_key = api_key
        self._batch_size = batch_size
        if not self._api_key or not self._model:
            raise RuntimeError(
                "OpenRouter model and API key are required. Pass model=... and api_key=..."
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._model, "input": batch},
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise EmbeddingError(f"LLM embedding failed: {exc}") from exc

            payload = response.json()
            if "error" in payload:
                raise EmbeddingError(f"LLM embedding API error: {payload['error']}")

            try:
                data = payload["data"]
                batch_embeddings = [item["embedding"] for item in data]
                if len(batch_embeddings) != len(batch):
                    raise EmbeddingError(
                        f"Embedding count mismatch: expected {len(batch)}, got {len(batch_embeddings)}"
                    )
                all_embeddings.extend(batch_embeddings)
            except (KeyError, IndexError, TypeError) as exc:
                raise EmbeddingError(f"Unexpected response format: {exc}") from exc

        return all_embeddings
