from tool import vector_search, get_docs_by_ids_tool
from minirag.types import SearchedChunk
from dataclasses import dataclass, field
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


@dataclass
class Step:
    action: AgentAction
    observation: Any


@dataclass
class RetrieveState:
    goal: str
    # History steps that can be referenced by the agent
    steps: list[Step] = field(default_factory=list)

    candidate_documents: dict[str, SearchedChunk] = field(default_factory=dict)

    inspected_documents: dict[str, str] = field(default_factory=dict)

    final_document_ids: list[str] = field(default_factory=list)

    finished: bool = False
    finish_reason: str | None = None
