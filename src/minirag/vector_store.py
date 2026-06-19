import chromadb
from chromadb import ClientAPI
from abc import ABC, abstractmethod
from typing import Any, NamedTuple
from minirag.config import get_settings
from minirag.types import Chunk
from hashlib import sha256


class SearchResult(NamedTuple):
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    score: float  # 1 - distance


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
    ) -> list[SearchResult]:
        """Search for the top-k chunks most similar to the query embedding."""


class ChromaVectorStore(VectorStore):

    def __init__(
        self,
        vector_store_path: str,
        collection_name: str,
        client: ClientAPI | None = None,
    ):
        settings = get_settings()
        vector_store_path = (
            vector_store_path or settings.vector_store_path or "default_store"
        )
        collection_name = (
            collection_name or settings.collection_name or "default_collection"
        )

        _chroma_client = client or chromadb.PersistentClient(path=vector_store_path)
        self._collection = _chroma_client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        if not chunks:
            return []

        ids = [sha256(chunk.document.encode()).hexdigest() for chunk in chunks]

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
    ) -> list[SearchResult]:
        if not query_embedding:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        chunk_ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        return [
            SearchResult(
                chunk_id=chunk_id,
                document=document,
                metadata=metadata,
                score=1 - distance,
            )
            for chunk_id, document, metadata, distance in zip(
                chunk_ids, documents, metadatas, distances
            )
        ]
