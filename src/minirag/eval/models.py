from dataclasses import dataclass, field
from typing import Any, Literal

RunStatus = Literal["success", "error"]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    reference_answer: str
    reference_chunk_ids: list[str]
    answerable: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalContext:
    chunk_id: str
    text: str | None
    score: float | None
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalStep:
    name: str
    attempt: int
    query: str | None
    contexts: list[EvalContext]
    latency_ms: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalRun:
    case_id: str
    target: str
    answer: str | None
    citations: list[str]
    final_contexts: list[EvalContext]
    trace: list[EvalStep]
    status: RunStatus
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
