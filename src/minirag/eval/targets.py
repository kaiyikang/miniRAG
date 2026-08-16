from typing import Protocol

from minirag.eval.models import EvalCase, EvalRun


class EvalTarget(Protocol):
    name: str

    def run(self, case: EvalCase) -> EvalRun: ...
