from typing import NamedTuple, Any
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RAGEvent:
    event_id: str
    step: str
    data: dict[str, Any] = field(default_factory=dict)


class Chunk(NamedTuple):
    document: str
    metadata: dict[str, Any]
    embedding: list[float] | None


class SearchedChunk(NamedTuple):
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float]
    score: float  # 1 - distance


class Answer(NamedTuple):
    content: str
    sources: list[dict[str, Any]]
    retrieved_chunk_ids: list[str]
    retrieved_chunks: list[str]
