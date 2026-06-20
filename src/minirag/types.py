from typing import NamedTuple, Any


class Chunk(NamedTuple):
    document: str
    metadata: dict[str, Any]
    embedding: list[float] | None


class SearchedChunk(NamedTuple):
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    score: float  # 1 - distance


class Answer(NamedTuple):
    answer: str
    sources: list[dict[str, Any]]
