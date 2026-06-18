import chromadb
from chromadb import ClientAPI
from abc import ABC, abstractmethod
from typing import Any, NamedTuple
from minirag.config import get_settings
from hashlib import sha256


class Chunk(NamedTuple):
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


class SearchResult(NamedTuple):
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # 1 - distance


class VectorStore(ABC):

    @abstractmethod
    def __init__(self, vector_store_path: str, collection_name: str):
        """"""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        """{TODO}"""

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """{TODO}"""


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

        ids = [sha256(chunk.text.encode()).hexdigest() for chunk in chunks]

        self._collection.add(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
        )

        return ids
