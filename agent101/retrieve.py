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


def format_steps(steps: list[Step]) -> str:
    if not steps:
        return "(none yet)"
    return "\n".join(
        f"Step {i}: {step.action} -> {step.observation}"
        for i, step in enumerate(steps, start=1)
    )


def format_candidates(candidates: dict[str, SearchedChunk]) -> str:
    if not candidates:
        return "(none yet)"
    return "\n".join(
        f"- {chunk_id}: {chunk.document} (score={chunk.score:.3f})"
        for chunk_id, chunk in candidates.items()
    )


def format_inspected(inspected: dict[str, str]) -> str:
    if not inspected:
        return "(none yet)"
    return "\n".join(
        f"- {chunk_id}: {content}" for chunk_id, content in inspected.items()
    )


class RetrieverPolicy:
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


class RetrieverAgent:
    def __init__(
        self, policy: RetrieverPolicy, tools: RetrievalTools, max_steps: int = 8
    ):
        self.policy = policy
        self.tools = tools
        self.max_steps = max_steps

    def run(self, goal: str, verbose: bool = False) -> RetrieverState:
        state = RetrieverState(goal=goal)

        while not state.finished:
            if len(state.steps) >= self.max_steps:
                state.finished = True
                state.finish_reason = "max_steps_reached"
                break

            action = self.policy.decide(state)
            observation = self._execute(action, state)

            if verbose:
                print(f"[step {len(state.steps) + 1}] action={action} observation={observation}")

            state.steps.append(Step(action=action, observation=observation))

        return state

    def _execute(self, action: AgentAction, state: RetrieverState):
        if isinstance(action, SearchAction):
            results = self.tools.search(action.query, action.top_k)

            for result in results:
                state.candidate_documents[result.chunk_id] = result

            return {"status": "ok", "result_count": len(results), "results": results}

        if isinstance(action, InspectAction):
            if action.chunk_id not in state.candidate_documents:
                return {
                    "status": "rejected",
                    "reason": "document_not_in_candidates",
                }
            content = state.candidate_documents[action.chunk_id].document
            state.inspected_documents[action.chunk_id] = content

            return {
                "status": "ok",
                "chunk_id": action.chunk_id,
                "content": content,
            }

        if isinstance(action, FinishAction):
            state.finished = True
            state.finish_reason = action.reason
            state.final_document_ids = list(state.inspected_documents.keys())

            return {"status": "finished", "reason": action.reason}

        raise TypeError(f"Unknown action: {action}")


if __name__ == "__main__":
    import sys

    goal = sys.argv[1] if len(sys.argv) > 1 else "lead climbing?"

    agent = RetrieverAgent(policy=RetrieverPolicy(llm), tools=RetrievalTools())
    final_state = agent.run(goal, verbose=True)

    print("finished:", final_state.finished, "reason:", final_state.finish_reason)
    print("steps taken:", len(final_state.steps))
    for chunk_id in final_state.final_document_ids:
        print("-", chunk_id, ":", final_state.inspected_documents[chunk_id][:100])
