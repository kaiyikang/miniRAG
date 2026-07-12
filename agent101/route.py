from typing import TypedDict, Optional
from state import END


class RouteDecision(TypedDict):
    next_agent: str
    reason: str
    exit_reason: Optional[str]


def go_to(agent_name, reason: str) -> RouteDecision:
    return {"next_agent": agent_name, "reason": reason, "exit_reason": None}


def stop(reason: str) -> RouteDecision:
    return {"next_agent": END, "reason": reason, "exit_reason": reason}


def route_next(current_agent: str, state: dict) -> str:
    # Global guards
    if state["step"] >= state["max_steps"]:
        return stop("max_steps_reached")

    if state["verified"]:
        return stop("answer_verified")

    # Local routes
    if current_agent == "classifier":
        return go_to("planner", "query_classified")

    if current_agent == "planner":
        return go_to("query_rewriter", "plan_created")

    if current_agent == "query_rewriter":
        return go_to("retriever", "query_ready")

    if current_agent == "retriever":
        if state["docs"]:
            return go_to("reranker", "documents_retrieved")
        return go_to("query_rewriter", "no_documents_found")

    if current_agent == "reranker":
        if state["reranked_docs"]:
            return go_to("answer", "documents_reranked")
        return go_to("query_rewriter", "no_relevant_documents")

    if current_agent == "answer":
        if state["answer"]:
            return go_to("verifier", "answer_generated")
        return stop("answer_generation_failed")

    if current_agent == "verifier":
        return go_to("query_rewriter", "verification_failed")

    raise ValueError(f"Unknown agent: {current_agent}")
