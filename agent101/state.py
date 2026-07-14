from typing import TypedDict, List, Optional, Any
from typing import Literal

END = "__end__"


class VerificationState(TypedDict):
    verification_attempts: int
    max_verification_attempts: int
    verification_result: Optional[bool]
    verification_reason: Optional[str]


class RagState(VerificationState):
    # Input from user
    original_query: str
    current_query: Optional[str]

    # Intermediate fields
    classification_reason: Optional[str]
    task_type: Optional[str]
    plan: Optional[str]
    docs: List[str]
    reranked_docs: List[dict]
    answer: Optional[str]

    # Process Control for Orchestrator
    step: int  # already done
    max_steps: int
    retrieval_attempts: int
    max_retrieval_attempts: int
    exit_reason: Optional[str]
    verified: bool

    # Debug
    trace: List[dict]
    query_history: List[str]


def create_initial_state(query="explain what is the multi agent RAG") -> RagState:
    return {
        "original_query": query,
        "current_query": None,
        "classification_reason": None,
        "task_type": None,
        "plan": None,
        "docs": [],
        "reranked_docs": [],
        "answer": None,
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
