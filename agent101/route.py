from typing import TypedDict, Optional
from state import END


class RouteDecision(TypedDict):
    next_agent: Optional[str]
    reason: str
    exit_reason: Optional[str]


def _go_to(agent_name, reason: str) -> RouteDecision:
    return {"next_agent": agent_name, "reason": reason, "exit_reason": None}


def _stop(reason: str) -> RouteDecision:
    return {"next_agent": END, "reason": reason, "exit_reason": reason}


def _retrieval_exhausted(state):
    return state["retrieval_attempts"] >= state["max_retrieval_attempts"]


def route_next(current_agent: str, state: dict) -> RouteDecision:
    # Global guards
    if state["step"] >= state["max_steps"]:
        return _stop("max_steps_reached")

    if state["verified"]:
        return _stop("answer_verified")

    # Local routes
    if current_agent == "classifier":
        return _go_to("planner", "query_classified")

    if current_agent == "planner":
        return _go_to("query_rewriter", "plan_created")

    if current_agent == "query_rewriter":
        return _go_to("retriever", "query_ready")

    if current_agent == "retriever":
        if state["docs"]:
            return _go_to("reranker", "documents_retrieved")

        if not _retrieval_exhausted(state):
            return _go_to("query_rewriter", "no_documents_found")

        return _go_to("answer", "retrieval_exhausted")

    if current_agent == "reranker":
        if state["reranked_docs"]:
            return _go_to("answer", "documents_reranked")
        return _go_to("query_rewriter", "no_relevant_documents")

    if current_agent == "answer":
        if state["answer"]:
            return _go_to("verifier", "answer_generated")
        return _stop("answer_generation_failed")

    if current_agent == "verifier":
        return _go_to("query_rewriter", "verification_failed")

    raise ValueError(f"Unknown agent: {current_agent}")
