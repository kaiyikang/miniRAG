from minirag.domain.index import index_chunks
from minirag.domain.ports import (
    IdentityTransformer,
    QueryTransformer,
    InferenceEngine,
    InferenceError,
    EmbeddingEngine,
    VectorStore,
    Reranker,
    DocumentSource,
)
from minirag.domain.models import Answer, RAGEvent
from typing import Any
from langfuse import observe, get_client
import queue
import uuid


class RAGPipeline:

    SYSTEM_MESSAGE = {
        "role": "system",
        "content": "You are a retrieval-based assistant, please answer the question based on the provided context.",
    }

    MAX_HISTORY_MESSAGES = 6

    def __init__(
        self,
        embed: EmbeddingEngine,
        vector_store: VectorStore,
        llm: InferenceEngine,
        source: DocumentSource | None = None,
        query_transformer: QueryTransformer = IdentityTransformer(),
        event_queue: queue.Queue | None = None,
    ):
        # source is only needed to index_documents(); query-only pipelines omit it.
        self._source = source
        self._embed = embed
        self._vstore = vector_store
        self._llm = llm
        self._query_transformer = query_transformer
        self._history: list[dict[str, Any]] = []  # or ChatHistory
        self._events = event_queue or queue.Queue()

    def get_llm(self):
        return self._llm

    def get_embed(self):
        return self._embed

    def _emit(self, event_id: str, step: str, **data: Any) -> None:
        self._events.put(RAGEvent(event_id=event_id, step=step, data=data))
        get_client().update_current_span(metadata={step: data})

    def index_documents(self, document_dirs: str | list[str]) -> None:
        if self._source is None:
            raise ValueError("No DocumentSource; construct with source=... to index.")
        if isinstance(document_dirs, str):
            document_dirs = [document_dirs]

        for document_dir in document_dirs:
            if not document_dir:
                raise ValueError("Source Document dir can not be found!")

            index_chunks(
                self._source.load(document_dir),
                self._embed,
                self._vstore,
            )

    def clear_history(self):
        self._history = []

    @observe(name="rag_query", capture_input=False)
    def query(self, question: str, retrieve_k: int = 10, rerank_k: int = 5) -> Answer:
        query_id = uuid.uuid4().hex
        # Explicit input keeps `self` (engines, history) out of the trace.
        get_client().update_current_span(input={"question": question})
        self._emit(query_id, "start", question=question)

        # transformation
        transformed_question = self._query_transformer.transform(question)
        if transformed_question == "":
            transformed_question = question
            self._emit(
                query_id,
                "transform",
                question=transformed_question,
                fallback=True,
            )
        else:
            self._emit(
                query_id,
                "transform",
                question=transformed_question,
                fallback=False,
            )

        # Retrieval
        ## Dense retrieval
        query_embedding = self._embed.embed([transformed_question])[0]
        self._emit(query_id, "embed")

        retrieved_chunks = self._vstore.search(query_embedding, top_k=retrieve_k)
        self._emit(query_id, "retrieve", chunk_count=len(retrieved_chunks))

        # placeholder for rerank
        ranked_chunks = retrieved_chunks[:rerank_k]
        self._emit(query_id, "rerank(not yet)", chunk_count=len(ranked_chunks))

        if not ranked_chunks:
            context = "No relevant context found."
        else:
            context = "\n".join([chunk.document for chunk in ranked_chunks])

        # Augmented
        messages = [
            self.SYSTEM_MESSAGE,
            *self._history,
            {
                "role": "user",
                "content": f"Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {question}",
            },
        ]

        # Generation
        try:
            content = self._llm.generate(
                messages=messages, span_name="answer_generation"
            )["content"]
        except (KeyError, TypeError, InferenceError):
            self._emit(query_id, "error", reason="generation_failed")
            return Answer(
                content="Error: failed to generate a response.",
                sources=[],
                retrieved_chunk_ids=[],
                retrieved_chunks=[],
            )
        self._emit(query_id, "generate", response_preview=content[:200])

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
