import json
from dataclasses import asdict

from minirag.eval.models import (
    EvalCase,
    EvalContext,
    EvalRun,
    EvalStep,
)


def test_eval_case_tags_have_independent_defaults():
    first = EvalCase("1", "q1", "a1", [])
    second = EvalCase("2", "q2", "a2", [])

    assert first.tags == []
    assert second.tags == []
    assert first.tags is not second.tags


def test_eval_run_is_json_serializable():
    context = EvalContext(
        chunk_id="c1",
        text="evidence",
        score=0.8,
        rank=1,
    )
    step = EvalStep(
        name="retrieve",
        attempt=1,
        query="question",
        contexts=[context],
        latency_ms=10.0,
    )
    run = EvalRun(
        case_id="case-1",
        target="rag",
        answer="answer",
        citations=[],
        final_contexts=[context],
        trace=[step],
        status="success",
        latency_ms=20.0,
    )

    payload = json.loads(json.dumps(asdict(run)))

    assert payload["case_id"] == "case-1"
    assert payload["trace"][0]["contexts"][0]["chunk_id"] == "c1"
