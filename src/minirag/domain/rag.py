import queue
import uuid
from time import perf_counter
from typing import Any, ClassVar

from minirag.domain.index import index_chunks
from minirag.domain.models import Answer, RAGEvent, SearchedChunk
from minirag.domain.ports import (
    DocumentSource,
    EmbeddingEngine,
    IdentityTransformer,
    InferenceEngine,
    InferenceError,
    QueryTransformer,
    Reranker,
    VectorStore,
)
from minirag.observability import get_client, observe


class RAGPipeline:
    SYSTEM_MESSAGE: ClassVar[dict[str, str]] = {
        "role": "system",
        "content": "You are a retrieval-based assistant, please answer the question based on the provided context.",
    }

    MAX_HISTORY_MESSAGES = 6

    def __init__(
        self,
        embed: EmbeddingEngine,
        vector_store: VectorStore,
        llm: InferenceEngine,
        reranker: Reranker | None = None,
        source: DocumentSource | None = None,
        query_transformer: QueryTransformer | None = None,
        event_queue: queue.Queue | None = None,
    ):
        # source is only needed to index(); query-only pipelines omit it.
        self._source = source
        self._embed = embed
        self._vstore = vector_store
        self._llm = llm
        self._reranker = reranker
        self._query_transformer = query_transformer or IdentityTransformer()
        self._history: list[dict[str, Any]] = []  # or ChatHistory
        self._events = event_queue or queue.Queue()

    def get_llm(self):
        return self._llm

    def get_embed(self):
        return self._embed

    def _emit(self, event_id: str, step: str, **data: Any) -> None:
        self._events.put(RAGEvent(event_id=event_id, step=step, data=data))
        get_client().update_current_span(metadata={step: data})

    def index(self) -> None:
        if self._source is None:
            raise ValueError("No DocumentSource; construct with source=... to index.")
        index_chunks(self._source.load(), self._embed, self._vstore)

    def clear_history(self):
        self._history = []

    @observe(name="rag_query", capture_input=False)
    def query(self, question: str, retrieve_k: int = 10, rerank_k: int = 5) -> Answer:
        query_id = uuid.uuid4().hex
        # Explicit input keeps `self` (engines, history) out of the trace.
        get_client().update_current_span(input={"question": question})
        self._emit(query_id, "start", question=question)

        # transformation
        transform_started = perf_counter()
        transformed_question = self._query_transformer.transform(question)

        fallback = transformed_question == ""
        if fallback:
            transformed_question = question

        self._emit(
            query_id,
            "transform",
            question=transformed_question,
            original_question=question,
            fallback=fallback,
            latency_ms=_elapsed_ms(transform_started),
        )

        # Dense Retrieval
        ## Embedding query
        embed_started = perf_counter()
        query_embedding = self._embed.embed([transformed_question])[0]
        self._emit(
            query_id,
            "embed",
            latency_ms=_elapsed_ms(embed_started),
        )

        ## Retrieval
        retrieve_started = perf_counter()
        retrieved_chunks = self._vstore.search(
            query_embedding,
            top_k=retrieve_k,
        )

        self._emit(
            query_id,
            "retrieve",
            query=transformed_question,
            chunk_count=len(retrieved_chunks),
            contexts=_snapshot_chunks(retrieved_chunks),
            latency_ms=_elapsed_ms(retrieve_started),
        )

        ## Rerank
        rerank_started = perf_counter()
        if self._reranker is not None:
            ranked_chunks = self._reranker.rank(
                transformed_question,
                None,
                retrieved_chunks,
            )[:rerank_k]
        else:
            ranked_chunks = retrieved_chunks[:rerank_k]

        self._emit(
            query_id,
            "rerank",
            query=transformed_question,
            chunk_count=len(ranked_chunks),
            contexts=_snapshot_chunks(ranked_chunks),
            latency_ms=_elapsed_ms(rerank_started),
        )

        # Augmented
        if not ranked_chunks:
            context = "No relevant context found."
        else:
            context = "\n".join(chunk.document for chunk in ranked_chunks)

        messages = [
            self.SYSTEM_MESSAGE,
            *self._history,
            {
                "role": "user",
                "content": f"Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {question}",
            },
        ]

        # Generation
        generate_started = perf_counter()

        try:
            content = self._llm.generate(
                messages=messages,
                span_name="answer_generation",
            )["content"]
        except (KeyError, TypeError, InferenceError) as exc:
            self._emit(
                query_id,
                "error",
                stage="generate",
                reason="generation_failed",
                error_type=type(exc).__name__,
                latency_ms=_elapsed_ms(generate_started),
            )
            return Answer(
                content="Error: failed to generate a response.",
                sources=[],
                retrieved_chunk_ids=[],
                retrieved_chunks=[],
            )

        self._emit(
            query_id,
            "generate",
            response_preview=content[:200],
            latency_ms=_elapsed_ms(generate_started),
        )

        # Handle History
        self._history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": content},
            ]
        )

        if len(self._history) > self.MAX_HISTORY_MESSAGES:
            self._history = self._history[-self.MAX_HISTORY_MESSAGES :]

        # Final Answer
        answer = Answer(
            content=content,
            sources=[chunk.metadata for chunk in ranked_chunks],
            retrieved_chunk_ids=[chunk.chunk_id for chunk in ranked_chunks],
            retrieved_chunks=[chunk.document for chunk in ranked_chunks],
        )

        self._emit(
            query_id,
            "complete",
            content=answer.content,
            sources=answer.sources,
        )
        return answer


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000


def _snapshot_chunks(chunks: list[SearchedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "score": float(chunk.score),
            "rank": rank,
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]
