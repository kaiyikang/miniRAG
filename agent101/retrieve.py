from tool import vector_search, get_docs_by_ids_tool
from minirag.types import SearchedChunk
from dataclasses import dataclass
from typing import List, Dict, Any, Literal


def retrieve(query: str) -> List[SearchedChunk]:
    return vector_search(query, 5)


@dataclass
class SearchAction:
    type: Literal["search"]
    query: str
    top_k: int = 5


@dataclass
class InspectAction:
    type: Literal["inspect"]
    chunk_id: str


@dataclass
class FinishAction:
    type: Literal["finish"]
    reason: str


AgentAction = SearchAction | InspectAction | FinishAction


# Boundary between agent and real world
class RetrievalTools:
    def search(self, query: str, top_k: int) -> list[SearchedChunk]:
        return vector_search(query, top_k)

    def inspect(self, chunk_id: str) -> str:
        chunk = get_docs_by_ids_tool([chunk_id])

        if len(chunk) == 0 or not chunk[0]:
            raise ValueError(f"Document not found: {chunk_id}")
        return chunk.document
