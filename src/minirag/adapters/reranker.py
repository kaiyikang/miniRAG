from minirag.domain.ports import Reranker
from minirag.domain.models import SearchedChunk
import math
import os


class VectorReranker(Reranker):
    def rank(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
        chunks: list[SearchedChunk],
    ) -> list[SearchedChunk]:

        if query_text:
            raise ValueError("VectorReranker doesn't support text")

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

    def __init__(self, model: str, cache_dir: str | None = None):
        from sentence_transformers import CrossEncoder

        if not model:
            raise ValueError("cross-encoder model name is required")

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
