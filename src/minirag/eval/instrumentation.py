from time import perf_counter

from minirag.domain.models import Chunk, SearchedChunk
from minirag.domain.ports import Reranker, VectorStore
from minirag.eval.models import EvalContext, EvalStep


class TraceCollector:
    """Request-scoped storage for evaluator-owned execution details."""

    def __init__(self) -> None:
        self._steps: list[EvalStep] = []

    @property
    def steps(self) -> list[EvalStep]:
        return list(self._steps)

    def reset(self) -> None:
        self._steps.clear()

    def record_chunks(
        self,
        *,
        name: str,
        chunks: list[SearchedChunk],
        query: str | None,
        latency_ms: float,
    ) -> None:
        contexts = [
            EvalContext(
                chunk_id=chunk.chunk_id,
                text=None,
                score=float(chunk.score),
                rank=rank,
            )
            for rank, chunk in enumerate(chunks, start=1)
        ]
        self._steps.append(
            EvalStep(
                name=name,
                attempt=1,
                query=query,
                contexts=contexts,
                latency_ms=latency_ms,
            )
        )


class RecordingVectorStore(VectorStore):
    """A transparent VectorStore decorator that records search results."""

    def __init__(self, inner: VectorStore, trace: TraceCollector):
        self._inner = inner
        self._trace = trace

    def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        return self._inner.add_chunks(chunks)

    def upsert_chunks(self, chunks: list[Chunk]) -> list[str]:
        return self._inner.upsert_chunks(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchedChunk]:
        started_at = perf_counter()
        chunks = self._inner.search(query_embedding, top_k=top_k)
        self._trace.record_chunks(
            name="retrieve",
            chunks=chunks,
            query=None,
            latency_ms=_elapsed_ms(started_at),
        )
        return chunks

    def get_all_chunks(self) -> list[SearchedChunk]:
        return self._inner.get_all_chunks()

    def get_by_ids(self, chunk_ids: list[str]) -> list[SearchedChunk]:
        return self._inner.get_by_ids(chunk_ids)


class RecordingReranker(Reranker):
    """A transparent Reranker decorator that records its complete ranking."""

    def __init__(self, inner: Reranker, trace: TraceCollector):
        self._inner = inner
        self._trace = trace

    def rank(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
        chunks: list[SearchedChunk],
    ) -> list[SearchedChunk]:
        started_at = perf_counter()
        ranked_chunks = self._inner.rank(query_text, query_embedding, chunks)
        self._trace.record_chunks(
            name="rerank",
            chunks=ranked_chunks,
            query=query_text,
            latency_ms=_elapsed_ms(started_at),
        )
        return ranked_chunks


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000
