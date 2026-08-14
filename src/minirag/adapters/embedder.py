import math
import os

import requests

from minirag.domain.ports import EmbeddingEngine, EmbeddingError


class SentenceTransformerEngine(EmbeddingEngine):

    def __init__(self, model: str, cache_dir: str, batch_size: int = 5):
        if not model or not cache_dir:
            raise ValueError("Embedding model name or cache dir can not be found!")
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be greater than zero")

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

    def __init__(
        self,
        model: str,
        api_key: str,
        batch_size: int = 100,
        timeout: float = 30.0,
    ):
        self._model = model
        self._api_key = api_key
        if not self._api_key or not self._model:
            raise RuntimeError(
                "OpenRouter model and API key are required. Pass model=... and api_key=..."
            )
        if batch_size <= 0:
            raise ValueError("OpenRouter embedding batch size must be greater than zero")
        if timeout <= 0:
            raise ValueError("OpenRouter embedding timeout must be greater than zero")
        self._batch_size = batch_size
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            payload = self._request_batch(batch)
            all_embeddings.extend(self._parse_batch(payload, expected_count=len(batch)))

        dimensions = {len(embedding) for embedding in all_embeddings}
        if len(dimensions) != 1:
            raise EmbeddingError("Embedding dimensions are inconsistent")
        return all_embeddings

    def _request_batch(self, batch: list[str]) -> object:
        try:
            response = requests.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self._model, "input": batch},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise EmbeddingError(f"LLM embedding failed: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise EmbeddingError("Unexpected embedding response: invalid JSON") from exc

    def _parse_batch(
        self,
        payload: object,
        expected_count: int,
    ) -> list[list[float]]:
        if not isinstance(payload, dict):
            raise EmbeddingError("Unexpected embedding response: expected an object")
        if "error" in payload:
            raise EmbeddingError(f"LLM embedding API error: {payload['error']}")

        data = payload.get("data")
        if not isinstance(data, list):
            raise EmbeddingError("Unexpected embedding response: data must be a list")

        parsed: dict[int, list[float]] = {}
        try:
            for item in data:
                index = item["index"]
                vector = item["embedding"]
                if type(index) is not int or not isinstance(vector, list) or not vector:
                    raise ValueError("invalid index or embedding vector")
                parsed[index] = [float(value) for value in vector]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"Unexpected embedding item: {exc}") from exc

        complete_batch = (
            len(data) == expected_count
            and set(parsed) == set(range(expected_count))
            and all(
                math.isfinite(value)
                for embedding in parsed.values()
                for value in embedding
            )
        )
        if not complete_batch:
            raise EmbeddingError(
                "Embedding results must cover the complete batch with finite vectors"
            )

        return [parsed[index] for index in range(expected_count)]
