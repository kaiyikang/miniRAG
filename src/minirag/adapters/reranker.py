import math
import os
from dataclasses import dataclass

import requests

from minirag.domain.models import SearchedChunk
from minirag.domain.ports import Reranker, RerankerError


class OpenRouterReranker(Reranker):
    BASE_URL = "https://openrouter.ai/api/v1/rerank"

    @dataclass(frozen=True)
    class _Result:
        index: int
        score: float

    def __init__(self, model: str, api_key: str, timeout: float = 30.0):
        if not model or not api_key:
            raise ValueError("OpenRouter rerank model and API key are required")
        if timeout <= 0:
            raise ValueError("OpenRouter rerank timeout must be greater than zero")

        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def rank(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
        chunks: list[SearchedChunk],
    ) -> list[SearchedChunk]:
        query = self._validate_request(query_text, query_embedding)
        if not chunks:
            return []

        payload = self._request_rerank(query, chunks)
        results = self._parse_results(payload, expected_count=len(chunks))
        return self._build_ranked_chunks(chunks, results)

    def _validate_request(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
    ) -> str:
        if query_embedding is not None:
            raise ValueError("OpenRouterReranker doesn't need an embedding")
        if not query_text:
            raise ValueError("query text is required")
        return query_text

    def _request_rerank(
        self,
        query_text: str,
        chunks: list[SearchedChunk],
    ) -> object:
        try:
            response = requests.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "query": query_text,
                    "documents": [chunk.document for chunk in chunks],
                    # rank() returns the complete candidate set. RAGPipeline is
                    # responsible for applying its rerank_k limit.
                    "top_n": len(chunks),
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RerankerError(f"OpenRouter rerank request failed: {exc}") from exc
        return payload

    def _parse_results(
        self,
        payload: object,
        expected_count: int,
    ) -> list[_Result]:
        if not isinstance(payload, dict):
            raise RerankerError("Unexpected OpenRouter rerank response: expected an object")
        if "error" in payload:
            raise RerankerError(f"OpenRouter rerank API error: {payload['error']}")

        results = payload.get("results")
        if not isinstance(results, list):
            raise RerankerError(
                "Unexpected OpenRouter rerank response: results must be a list"
            )
        try:
            parsed_results = [
                self._Result(
                    index=item["index"],
                    score=float(item["relevance_score"]),
                )
                for item in results
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankerError(
                f"Unexpected OpenRouter rerank result format: {exc}"
            ) from exc

        indexes = [result.index for result in parsed_results]
        valid_permutation = (
            len(parsed_results) == expected_count
            and all(type(index) is int for index in indexes)
            and set(indexes) == set(range(expected_count))
            and all(math.isfinite(result.score) for result in parsed_results)
        )
        if not valid_permutation:
            raise RerankerError(
                "Unexpected OpenRouter rerank results: candidates must form a complete "
                "permutation with finite scores"
            )

        return parsed_results

    def _build_ranked_chunks(
        self,
        chunks: list[SearchedChunk],
        results: list[_Result],
    ) -> list[SearchedChunk]:
        ranked_chunks = [
            SearchedChunk(
                chunk_id=chunks[result.index].chunk_id,
                document=chunks[result.index].document,
                metadata=chunks[result.index].metadata,
                embedding=chunks[result.index].embedding,
                score=result.score,
            )
            for result in results
        ]
        return sorted(ranked_chunks, key=lambda chunk: chunk.score, reverse=True)


class VectorReranker(Reranker):
    def rank(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
        chunks: list[SearchedChunk],
    ) -> list[SearchedChunk]:

        if query_text:
            raise ValueError("VectorReranker doesn't support text")

        if not query_embedding or not all(chunk.embedding for chunk in chunks):
            raise ValueError("The query or chunks must have embedding")

        return sorted(
            [
                SearchedChunk(
                    chunk_id=chunk.chunk_id,
                    document=chunk.document,
                    metadata=chunk.metadata,
                    embedding=chunk.embedding,
                    score=self._cosine_similarity(query_embedding, chunk.embedding),
                )
                for chunk in chunks
            ],
            key=lambda x: x.score,
            reverse=True,
        )

    def _cosine_similarity(self, v1, v2):
        def _dot_product(a, b):
            return sum(x * y for x, y in zip(a, b))

        def _magnitude(v):
            return math.sqrt(sum(x**2 for x in v))

        dot = _dot_product(v1, v2)
        mag1 = _magnitude(v1)
        mag2 = _magnitude(v2)
        if mag1 == 0 or mag2 == 0:
            return 0
        return dot / (mag1 * mag2)


class CrossEncoderReranker(Reranker):

    def __init__(self, model: str, cache_dir: str | None = None):
        if not model:
            raise ValueError("cross-encoder model name is required")

        from sentence_transformers import CrossEncoder

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._model = CrossEncoder(model, cache_dir=cache_dir)

    def rank(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
        chunks: list[SearchedChunk],
    ) -> list[SearchedChunk]:

        if query_embedding:
            raise ValueError("CrossEncoderReranker doesn't need embedding")

        if not query_text:
            raise ValueError("query text is required for CrossEncoderReranker")
        if not chunks:
            return []

        scores = self._model.predict([(query_text, chunk.document) for chunk in chunks])

        return sorted(
            [
                SearchedChunk(
                    chunk_id=chunk.chunk_id,
                    document=chunk.document,
                    metadata=chunk.metadata,
                    embedding=chunk.embedding,
                    score=score,
                )
                for chunk, score in zip(chunks, scores)
            ],
            key=lambda x: x.score,
            reverse=True,
        )
