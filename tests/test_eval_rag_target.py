import queue
from typing import Any

from minirag.domain.models import Chunk, SearchedChunk
from minirag.domain.ports import (
    EmbeddingEngine,
    InferenceEngine,
    InferenceError,
    Reranker,
    VectorStore,
)
from minirag.domain.rag import RAGPipeline
from minirag.eval.instrumentation import (
    RecordingReranker,
    RecordingVectorStore,
    TraceCollector,
)
from minirag.eval.models import EvalCase
from minirag.eval.targets import RagEvalTarget


class FakeEmbedding(EmbeddingEngine):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FakeLLM(InferenceEngine):
    def generate(self, messages, **kwargs) -> dict[str, Any]:
        return {"content": "answer"}


class FailingLLM(InferenceEngine):
    def generate(self, messages, **kwargs) -> dict[str, Any]:
        raise InferenceError("provider failed")


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.chunks = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7)]

    def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        return []

    def upsert_chunks(self, chunks: list[Chunk]) -> list[str]:
        return []

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[SearchedChunk]:
        return self.chunks[:top_k]

    def get_all_chunks(self) -> list[SearchedChunk]:
        return self.chunks

    def get_by_ids(self, chunk_ids: list[str]) -> list[SearchedChunk]:
        return [chunk for chunk in self.chunks if chunk.chunk_id in chunk_ids]


class ReverseReranker(Reranker):
    def rank(self, query_text, query_embedding, chunks):
        return [
            chunk._replace(score=1.0 - index * 0.1)
            for index, chunk in enumerate(reversed(chunks))
        ]


def _chunk(chunk_id: str, score: float) -> SearchedChunk:
    return SearchedChunk(
        chunk_id=chunk_id,
        document=f"text {chunk_id}",
        metadata={"source": chunk_id},
        embedding=[1.0],
        score=score,
    )


def _build_target(
    llm: InferenceEngine | None = None,
    *,
    with_reranker: bool = True,
) -> tuple[RagEvalTarget, RAGPipeline]:
    trace = TraceCollector()
    events = queue.Queue()
    pipeline = RAGPipeline(
        embed=FakeEmbedding(),
        vector_store=RecordingVectorStore(FakeVectorStore(), trace),
        reranker=RecordingReranker(ReverseReranker(), trace) if with_reranker else None,
        llm=llm or FakeLLM(),
        event_queue=events,
    )
    target = RagEvalTarget(
        pipeline,
        trace,
        events,
        retrieve_k=3,
        rerank_k=2,
    )
    return target, pipeline


def test_rag_target_records_candidates_outside_the_pipeline():
    target, _ = _build_target()
    case = EvalCase("case-1", "question", "reference", ["c"])

    run = target.run(case)

    assert run.status == "success"
    assert run.answer == "answer"
    assert [step.name for step in run.trace] == ["transform", "retrieve", "rerank"]

    retrieve_step = run.trace[1]
    assert retrieve_step.query == "question"
    assert [context.chunk_id for context in retrieve_step.contexts] == ["a", "b", "c"]
    assert [context.rank for context in retrieve_step.contexts] == [1, 2, 3]
    assert retrieve_step.contexts[0].text is None
    assert retrieve_step.latency_ms is not None

    rerank_step = run.trace[2]
    assert [context.chunk_id for context in rerank_step.contexts] == ["c", "b", "a"]
    assert rerank_step.contexts[0].score == 1.0
    assert [context.chunk_id for context in run.final_contexts] == ["c", "b"]
    assert run.final_contexts[0].text == "text c"
    assert run.final_contexts[0].score == 1.0
    assert run.latency_ms >= 0


def test_rag_target_resets_trace_and_history_between_cases():
    target, pipeline = _build_target()

    first = target.run(EvalCase("case-1", "first", "reference", ["a"]))
    second = target.run(EvalCase("case-2", "second", "reference", ["b"]))

    assert len(first.trace) == 3
    assert len(second.trace) == 3
    assert second.diagnostics["transformed_query"] == "second"
    assert pipeline._history == []


def test_rag_target_uses_retrieval_ranking_without_a_reranker():
    target, _ = _build_target(with_reranker=False)

    run = target.run(EvalCase("case-1", "question", "reference", ["a"]))

    assert [step.name for step in run.trace] == ["transform", "retrieve"]
    assert [context.chunk_id for context in run.final_contexts] == ["a", "b"]
    assert run.final_contexts[0].score == 0.9


def test_rag_target_normalizes_a_generation_failure():
    target, _ = _build_target(FailingLLM())

    run = target.run(EvalCase("case-1", "question", "reference", ["a"]))

    assert run.status == "error"
    assert run.error_type == "InferenceError"
    assert run.error_message == "generation_failed"
    assert run.answer == "Error: failed to generate a response."
    assert run.final_contexts == []
