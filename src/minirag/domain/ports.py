from abc import ABC, abstractmethod
from typing import Any

from minirag.domain.models import Chunk, SearchedChunk


class DocumentSource(ABC):
    @abstractmethod
    def load(self) -> list[Chunk]:
        """Return all chunks from this source. Location is the adapter's own config."""


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """text -> list of chunk strings"""


class EmbeddingError(Exception):
    """Raised when the LLM embedding request fails."""


class EmbeddingEngine(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""


class InferenceEngine(ABC):
    @abstractmethod
    def generate(
        self,
        messages: str | list[dict[str, Any]],
        *,
        reasoning: bool = True,
        last_response: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
        span_name: str | None = None,
    ) -> dict[str, Any]:
        """Generate a response and return the assistant message dict."""


class InferenceError(Exception):
    """Raised when the LLM inference request fails."""


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


class Reranker(ABC):
    @abstractmethod
    def rank(
        self,
        query_text: str | None,
        query_embedding: list[float] | None,
        chunks: list[SearchedChunk],
    ) -> list[SearchedChunk]:
        """Return candidate chunks ordered from most to least relevant."""


class RerankerError(Exception):
    """Raised when a remote reranking request or response is invalid."""


class QueryTransformer(ABC):
    @abstractmethod
    def transform(self, question: str) -> str:
        """Return the text that should be embedded for retrieval."""


class IdentityTransformer(QueryTransformer):
    def transform(self, question: str) -> str:
        return question
