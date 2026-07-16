import chromadb
from chromadb import ClientAPI
from abc import ABC, abstractmethod
from minirag.types import Chunk, SearchedChunk
from hashlib import sha256
from itertools import repeat
from typing import Any, Iterable


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def __init__(self, vector_store_path: str, collection_name: str):
        """Initialize the vector store."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Add chunks to the vector store."""

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[SearchedChunk]:
        """Search for the top-k chunks most similar to the query embedding."""

    @abstractmethod
    def get_all_chunks(self) -> list[SearchedChunk]:
        """Get all chunks"""


class ChromaVectorStore(VectorStore):

    def __init__(
        self,
        vector_store_path: str,
        collection_name: str,
        client: ClientAPI | None = None,
    ):
        vector_store_path = vector_store_path or "default_store"
        collection_name = collection_name or "default_collection"

        _chroma_client = client or chromadb.PersistentClient(path=vector_store_path)
        self._collection = _chroma_client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        if not chunks:
            return []

        ids = [
            sha256(f"{chunk.document}-{chunk.metadata}".encode()).hexdigest()
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=[chunk.document for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
        )

        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchedChunk]:
        if not query_embedding:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["embeddings", "documents", "metadatas", "distances"],
        )

        chunk_ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        embeddings = results["embeddings"][0] if results["embeddings"] else []

        return self._to_searched_chunks(
            chunk_ids, documents, metadatas, embeddings, (1 - d for d in distances)
        )

    def get_by_ids(self, chunk_ids: list[str]) -> list[SearchedChunk]:
        results = self._collection.get(
            ids=chunk_ids, include=["documents", "metadatas", "embeddings"]
        )
        if not results:
            return []

        return self._to_searched_chunks(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["embeddings"],
            repeat(0.0),
        )

    def get_all_chunks(self):
        results = self._collection.get(include=["documents", "metadatas", "embeddings"])
        if not results:
            return []

        return self._to_searched_chunks(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["embeddings"],
            repeat(0.0),
        )

    @staticmethod
    def _to_searched_chunks(
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
        scores: Iterable[float],
    ) -> list[SearchedChunk]:
        return [
            SearchedChunk(
                chunk_id=chunk_id,
                document=document,
                metadata=metadata,
                embedding=embedding,
                score=score,
            )
            for chunk_id, document, metadata, embedding, score in zip(
                chunk_ids, documents, metadatas, embeddings, scores
            )
        ]
