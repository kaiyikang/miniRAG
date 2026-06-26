from abc import ABC, abstractmethod
from minirag.types import SearchedChunk
import math
from sentence_transformers import CrossEncoder
import os


class Reranker(ABC):
    @abstractmethod
    def rank(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
        chunks: list[SearchedChunk],
    ) -> list[SearchedChunk]:
        """"""


class VectorReranker(Reranker):
    def rank(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
        chunks: list[SearchedChunk],
    ) -> list[SearchedChunk]:

        if not query_embedding or not all([chunk.embedding for chunk in chunks]):
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

    def __ini__(self, model: str, cache_dir=str):
        if not model or not cache_dir:
            raise ValueError("Embedding model name or cache dir can not be found!")

        os.makedirs(cache_dir, exist_ok=True)
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rank(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
        chunks: list[SearchedChunk],
    ) -> list[SearchedChunk]:

        pair = []
        for chunk in chunks:
            pair.append((query_text, chunk.document))
        scores = self._model.predict(pair)
        new_chunks = []
        for chunk in chunks:
