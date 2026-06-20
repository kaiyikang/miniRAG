from minirag.embedding import EmbeddingEngine
from minirag.vector_store import VectorStore
from minirag.document import Chunker, load_documents, chunk_documents
from minirag.llm_engine import InferenceEngine, InferenceError
from minirag.types import Chunk, Answer
from typing import Any


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
    ):
        self._embed = embed
        self._vstore = vector_store
        self._chunker = chunker
        self._llm = llm
        self._history: list[dict[str, Any]] = []  # or ChatHistory

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

    def query(self, question: str, top_k: int = 5) -> Answer:
        # transformation
        transformed_question = question  # rewriting / HyDe / ...

        # Retrieval
        ## Dense retrieval
        question_embedding = self._embed.embed([transformed_question])[0]
        retrieved_chunks = self._vstore.search(question_embedding, top_k=top_k)
        # have to rerank

        if not retrieved_chunks:
            context = "No relevant context found."
        else:
            context = "\n".join([chunk.document for chunk in retrieved_chunks])

        # Augmented
        messages = [
            self.SYSTEM_MESSAGE,
            *self._history,
            {
                "role": "user",
                "content": f"Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {question}",
            },
        ]

        try:
            response = self._llm.generate(messages=messages)["content"]
        except (KeyError, TypeError, InferenceError):
            return Answer(answer="Error: failed to generate a response.", sources=[])
        self._history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        )

        if len(self._history) > self.MAX_HISTORY_MESSAGES:
            self._history = self._history[-self.MAX_HISTORY_MESSAGES :]

        return Answer(
            answer=response, sources=[chunk.metadata for chunk in retrieved_chunks]
        )
