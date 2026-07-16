from tool import vector_search, get_docs_by_ids_tool
from minirag.types import SearchedChunk
from minirag.llm_engine import OpenRouterEngine, InferenceEngine
from minirag.config import get_settings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal
import json

_settings = get_settings()


llm = OpenRouterEngine(
    model=_settings.openrouter_model, api_key=_settings.openrouter_api_key
)


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

AGENT_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "type": {"const": "search"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["type", "query", "top_k"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "type": {"const": "inspect"},
                "chunk_id": {"type": "string"},
            },
            "required": ["type", "chunk_id"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "type": {"const": "finish"},
                "reason": {"type": "string"},
            },
            "required": ["type", "reason"],
            "additionalProperties": False,
        },
    ],
}


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
class RetrieverState:
    goal: str
    # History steps that can be referenced by the agent
    steps: list[Step] = field(default_factory=list)

    candidate_documents: dict[str, SearchedChunk] = field(default_factory=dict)

    inspected_documents: dict[str, str] = field(default_factory=dict)

    final_document_ids: list[str] = field(default_factory=list)

    finished: bool = False
    finish_reason: str | None = None


class LLMRetrieverPolicy:
    def __init__(self, llm: InferenceEngine):
        self._llm = llm

    def decide(self, state: RetrieverState) -> AgentAction:
        prompt = self._build_prompt(state)

        result = self._llm.generate(messages=prompt, schema=AGENT_ACTION_SCHEMA)

        parsed = json.loads(result["content"])

        return self._parse_action(parsed)

    def _parse_action(self, parsed: dict[str, Any]) -> AgentAction:
        action_type = parsed.get("type")
        if not action_type:
            raise ValueError("Can not find type from LLM")

        if action_type == "search":
            return SearchAction(**parsed)
        elif action_type == "inspect":
            return InspectAction(**parsed)
        elif action_type == "finish":
            return FinishAction(**parsed)
        else:
            raise ValueError("Incorrect action type returned from LLM")

    def _build_prompt(self, state: RetrieverState) -> str:
        return f"""
You are a retrieval agent.

Goal:
{state.goal}

Previous steps:
{format_steps(state.steps)}

Candidate documents:
{format_candidates(state.candidate_documents)}

Inspected documents:
{format_inspected(state.inspected_documents)}

Choose exactly one next action:

1. search
   Use when more or different evidence is needed.

2. inspect
   Use when a candidate document appears relevant and must be
   examined in full.

3. finish
   Use when sufficient evidence has been collected, or when
   additional retrieval is unlikely to help.

Do not repeat an identical unsuccessful action.
Return structured JSON only.
"""
