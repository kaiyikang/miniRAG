from minirag.embedding import EmbeddingEngine
from minirag.vector_store import VectorStore
from minirag.document import Chunker, load_documents, chunk_documents
from minirag.llm_engine import InferenceEngine, InferenceError
from minirag.query_transform import IdentityTransformer, QueryTransformer
from minirag.types import Chunk, Answer, RAGEvent
from minirag.reranker import Reranker
from typing import Any
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
        chunker: Chunker,
        llm: InferenceEngine,
        query_transformer: QueryTransformer = IdentityTransformer(),
        event_queue: queue.Queue | None = None,
    ):
        self._embed = embed
        self._vstore = vector_store
        self._chunker = chunker
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

    def index_documents(self, document_dirs: str | list[str]) -> None:
        if isinstance(document_dirs, str):
            document_dirs = [document_dirs]

        for document_dir in document_dirs:
            if not document_dir:
                raise ValueError("Source Document dir can not be found!")

            docs = load_documents(document_dir)
            if not docs:
                continue

            chunks = chunk_documents(docs, self._chunker)
            embeddings = self._embed.embed([chunk.document for chunk in chunks])
            self._vstore.add_chunks(
                [
                    Chunk(
                        document=chunk.document,
                        metadata=chunk.metadata,
                        embedding=embedding,
                    )
                    for chunk, embedding in zip(chunks, embeddings)
                ]
            )

    def clear_history(self):
        self._history = []

    def query(self, question: str, retrieve_k: int = 10, rerank_k: int = 5) -> Answer:
        query_id = uuid.uuid4().hex
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
            content = self._llm.generate(messages=messages)["content"]
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
