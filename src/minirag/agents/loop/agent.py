from minirag.types import SearchedChunk
from minirag.llm_engine import OpenRouterEngine, InferenceEngine
from minirag.agents.tool import SearchTools
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Literal
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from langfuse import get_client, observe
import json


@dataclass
class SearchAction:
    type: Literal["search"]
    query: str
    method: Literal["vector", "keyword"] = "vector"
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
                "method": {"enum": ["vector", "keyword"]},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["type", "query", "method", "top_k"],
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
    def __init__(self, tools: SearchTools):
        self._tools = tools

    def search(self, query: str, method: str, top_k: int) -> list[SearchedChunk]:
        if method == "keyword":
            return self._tools.keyword_search(query, top_k)
        return self._tools.vector_search(query, top_k)

    def inspect(self, chunk_id: str) -> str:
        chunk = self._tools.get_by_ids([chunk_id])

        if len(chunk) == 0 or not chunk[0]:
            raise ValueError(f"Document not found: {chunk_id}")
        return chunk[0].document


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


@dataclass
class RetrieverLimits:
    max_steps: int = 8
    max_searches: int = 4
    max_inspections: int = 4
    max_same_action: int = 1  # how many times an identical search may repeat


def count_actions(state: RetrieverState, action_type: type) -> int:
    """How many steps so far used this action type."""
    return sum(isinstance(step.action, action_type) for step in state.steps)


def validate_action(
    action: AgentAction,
    state: RetrieverState,
    limits: RetrieverLimits,
) -> str | None:
    """Deterministic gate between policy.decide() and _execute().

    Return a short rejection-reason string (e.g. "duplicate_search") if the
    action breaks a countable budget/guard, or None if it may execute.

    Guards to implement (each is pure counting over state.steps, no LLM):
      - SearchAction: search budget exhausted? (count_actions vs max_searches)
                      identical query already issued? (vs max_same_action)
      - InspectAction: inspection budget exhausted? (vs max_inspections)
                       chunk already in state.inspected_documents?
      - FinishAction / anything else: no deterministic guard -> None
    """
    if isinstance(action, SearchAction):
        if count_actions(state, SearchAction) >= limits.max_searches:
            return "search_budget_exhausted"

        same_query = sum(
            1
            for step in state.steps
            if isinstance(step.action, SearchAction)
            and step.action.query == action.query
            and step.action.method == action.method
        )

        if same_query >= limits.max_same_action:
            return "duplicate_search"

    if isinstance(action, InspectAction):
        if count_actions(state, InspectAction) >= limits.max_inspections:
            return "inspection_budget_exhausted"

        if action.chunk_id in state.inspected_documents:
            return "document_already_inspected"

    return None


class CompletionPolicy:
    def can_finish(self, state: RetrieverState) -> tuple[bool, str]:
        """Return (allowed, reason). Called before FinishAction takes effect.

        Decide the *minimum* bar for stopping — e.g. at least one inspected
        document, or some source diversity. Return (False, "<why_not>") to
        veto (that reason is fed back to the model), or (True, "<why_ok>").
        """
        if len(state.inspected_documents) == 0:
            return False, "no_document_inspected"

        if len(state.inspected_documents) < 2:
            return False, "insufficient_source_diversity"
        return True, "minimum_requirements_satisfied"


class RetrieverPolicy:
    def __init__(self, llm: InferenceEngine):
        self._llm = llm

    @retry(
        retry=retry_if_exception_type(json.JSONDecodeError),
        stop=stop_after_attempt(3),
        reraise=True,
    )
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
   Use when more or different evidence is needed. Choose a method:
   - "vector": semantic similarity. Good default for conceptual questions.
   - "keyword": exact term overlap. Switch to this when vector search misses
     proper nouns, names, or specific terms.

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
        self,
        policy: RetrieverPolicy,
        tools: RetrievalTools,
        limits: RetrieverLimits | None = None,
        completion_policy: CompletionPolicy | None = None,
    ):
        self.policy = policy
        self.tools = tools
        self.limits = limits or RetrieverLimits()
        self.completion_policy = completion_policy or CompletionPolicy()

    @observe(name="retriever_loop")
    def run(self, goal: str, verbose: bool = False) -> RetrieverState:
        get_client().update_current_span(input={"goal": goal})
        state = RetrieverState(goal=goal)

        while not state.finished:
            if len(state.steps) >= self.limits.max_steps:
                state.finished = True
                state.finish_reason = "max_steps_reached"
                break

            observation = self._run_step(state)

            if verbose:
                print(
                    f"[step {len(state.steps)}] action={state.steps[-1].action} observation={observation}"
                )

        get_client().update_current_span(
            output={
                "finished": state.finished,
                "finish_reason": state.finish_reason,
                "final_document_ids": state.final_document_ids,
                "steps": len(state.steps),
            }
        )
        return state

    @observe(name="step")
    def _run_step(self, state: RetrieverState) -> Any:
        action = self.policy.decide(state)

        rejection = validate_action(action, state, self.limits)
        if rejection:
            observation = {"status": "rejected", "reason": rejection}
        else:
            observation = self._execute(action, state)

        get_client().update_current_span(
            input=asdict(action),
            output=self._compact(observation),
            metadata={"status": observation["status"]},
        )

        state.steps.append(Step(action=action, observation=observation))
        return observation

    def _compact(self, observation: dict) -> dict:
        if "result" in observation:
            return {
                **observation,
                "results": [c.chunk_id for c in observation["results"]],
            }
        return observation

    def _execute(self, action: AgentAction, state: RetrieverState) -> dict:
        if isinstance(action, SearchAction):
            results = self.tools.search(action.query, action.method, action.top_k)

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
            # In case the LLM gives the answer directly without any search
            allowed, reason = self.completion_policy.can_finish(state)
            if not allowed:
                return {"status": "rejected", "reason": reason}

            state.finished = True
            state.finish_reason = action.reason
            state.final_document_ids = list(state.inspected_documents.keys())

            return {"status": "finished", "reason": action.reason}

        raise TypeError(f"Unknown action: {action}")


if __name__ == "__main__":
    import sys
    from minirag.config import get_settings
    from minirag.embedding import OpenRouterEmbeddingEngine
    from minirag.vector_store import ChromaVectorStore

    # Dependency Injection
    settings = get_settings()
    embed = OpenRouterEmbeddingEngine(
        model=settings.openrouter_embed_model, api_key=settings.openrouter_api_key
    )
    vstore = ChromaVectorStore(
        vector_store_path=settings.vector_store_path,
        collection_name=settings.collection_name,
    )
    llm = OpenRouterEngine(
        model=settings.openrouter_model, api_key=settings.openrouter_api_key
    )

    goal = sys.argv[1] if len(sys.argv) > 1 else "what is the lead climbing?"

    tools = RetrievalTools(SearchTools(embed, vstore))
    agent = RetrieverAgent(policy=RetrieverPolicy(llm), tools=tools)
    final_state = agent.run(goal, verbose=False)

    # Traces are buffered; flush before the short-lived process exits.
    get_client().flush()

    print("finished:", final_state.finished, "reason:", final_state.finish_reason)
    print("steps taken:", len(final_state.steps))
    for chunk_id in final_state.final_document_ids:
        print("-", chunk_id, ":", final_state.inspected_documents[chunk_id][:100])
