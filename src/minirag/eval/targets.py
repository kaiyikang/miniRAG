import queue
from dataclasses import replace
from time import perf_counter
from typing import Protocol

from minirag.domain.models import Answer, RAGEvent
from minirag.domain.rag import RAGPipeline
from minirag.eval.instrumentation import TraceCollector
from minirag.eval.models import EvalCase, EvalContext, EvalRun, EvalStep


class EvalTarget(Protocol):
    name: str

    def run(self, case: EvalCase) -> EvalRun: ...


class RagEvalTarget:
    """Run an isolated RAG query and normalize it into an EvalRun."""

    name = "rag"

    def __init__(
        self,
        pipeline: RAGPipeline,
        trace: TraceCollector,
        event_queue: queue.Queue[RAGEvent],
        *,
        retrieve_k: int = 10,
        rerank_k: int = 5,
    ) -> None:
        self._pipeline = pipeline
        self._trace = trace
        self._events = event_queue
        self._retrieve_k = retrieve_k
        self._rerank_k = rerank_k

    def run(self, case: EvalCase) -> EvalRun:
        self._trace.reset()
        _drain_events(self._events)
        self._pipeline.clear_history()

        started_at = perf_counter()
        answer: Answer | None = None
        caught_error: Exception | None = None

        try:
            answer = self._pipeline.query(
                case.question,
                retrieve_k=self._retrieve_k,
                rerank_k=self._rerank_k,
            )
        # An evaluation target converts application failures into data so one bad
        # sample cannot disappear from an experiment run.
        except Exception as exc:  # noqa: BLE001
            caught_error = exc
        finally:
            self._pipeline.clear_history()

        latency_ms = (perf_counter() - started_at) * 1000
        events = _drain_events(self._events)
        error_event = next((event for event in events if event.step == "error"), None)
        transformed_query, fallback = _transform_details(events, case.question)
        trace_steps = _normalized_steps(
            self._trace.steps,
            transformed_query=transformed_query,
            fallback=fallback,
        )

        if caught_error is not None:
            status = "error"
            error_type = type(caught_error).__name__
            error_message = str(caught_error)
        elif error_event is not None:
            status = "error"
            error_type = str(error_event.data.get("error_type", "generation_failed"))
            error_message = str(error_event.data.get("reason", "generation_failed"))
        else:
            status = "success"
            error_type = None
            error_message = None

        return EvalRun(
            case_id=case.case_id,
            target=self.name,
            answer=answer.content if answer is not None else None,
            citations=[],
            final_contexts=_final_contexts(answer, trace_steps),
            trace=trace_steps,
            status=status,
            latency_ms=latency_ms,
            error_type=error_type,
            error_message=error_message,
            diagnostics={
                "transformed_query": transformed_query,
                "fallback": fallback,
            },
        )


def _drain_events(events: queue.Queue[RAGEvent]) -> list[RAGEvent]:
    drained: list[RAGEvent] = []
    while True:
        try:
            drained.append(events.get_nowait())
        except queue.Empty:
            return drained


def _transform_details(
    events: list[RAGEvent], original_question: str
) -> tuple[str, bool]:
    transform_event = next(
        (event for event in events if event.step == "transform"), None
    )
    if transform_event is None:
        return original_question, False
    return (
        str(transform_event.data.get("question", original_question)),
        bool(transform_event.data.get("fallback", False)),
    )


def _normalized_steps(
    recorded_steps: list[EvalStep],
    *,
    transformed_query: str,
    fallback: bool,
) -> list[EvalStep]:
    steps = [
        EvalStep(
            name="transform",
            attempt=1,
            query=transformed_query,
            contexts=[],
            latency_ms=None,
            metadata={"fallback": fallback},
        )
    ]
    steps.extend(
        replace(step, query=transformed_query)
        if step.name == "retrieve" and step.query is None
        else step
        for step in recorded_steps
    )
    return steps


def _final_contexts(
    answer: Answer | None,
    trace_steps: list[EvalStep],
) -> list[EvalContext]:
    if answer is None:
        return []

    final_ranking = next(
        (step for step in reversed(trace_steps) if step.name in {"rerank", "retrieve"}),
        None,
    )
    score_by_id = (
        {context.chunk_id: context.score for context in final_ranking.contexts}
        if final_ranking is not None
        else {}
    )
    return [
        EvalContext(
            chunk_id=chunk_id,
            text=text,
            score=score_by_id.get(chunk_id),
            rank=rank,
            metadata=dict(answer.sources[rank - 1])
            if rank <= len(answer.sources)
            else {},
        )
        for rank, (chunk_id, text) in enumerate(
            zip(answer.retrieved_chunk_ids, answer.retrieved_chunks), start=1
        )
    ]
