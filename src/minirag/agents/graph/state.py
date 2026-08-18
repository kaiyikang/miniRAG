from typing import TypedDict

END = "__end__"


class VerificationState(TypedDict):
    verification_attempts: int
    max_verification_attempts: int
    verification_result: bool | None
    verification_reason: str | None


class RagState(VerificationState):
    # Input from user
    original_query: str
    current_query: str | None

    # Intermediate fields
    classification_reason: str | None
    task_type: str | None
    plan: str | None
    docs: list[dict]
    answer: str | None
    citations: list[str]

    # Process Control for Orchestrator
    step: int  # already done
    max_steps: int
    retrieval_attempts: int
    max_retrieval_attempts: int
    exit_reason: str | None
    verified: bool

    # Debug
    trace: list[dict]
    query_history: list[str]


def create_initial_state(query="explain what is the multi agent RAG") -> RagState:
    return {
        "original_query": query,
        "current_query": None,
        "classification_reason": None,
        "task_type": None,
        "plan": None,
        "docs": [],
        "answer": None,
        "citations": [],
        "verification_attempts": 0,
        "max_verification_attempts": 3,
        "verification_result": None,
        "verification_reason": None,
        "step": 0,
        "max_steps": 20,
        "retrieval_attempts": 0,
        "max_retrieval_attempts": 3,
        "exit_reason": None,
        "verified": False,
        # Debug
        "trace": [],
        "query_history": [],
    }
