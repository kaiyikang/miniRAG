from typing import NamedTuple, Any


class Chunk(NamedTuple):
    document: str
    metadata: dict[str, Any]
    embedding: list[float] | None
