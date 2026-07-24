from abc import ABC, abstractmethod
from minirag.llm_engine import InferenceEngine, InferenceError


class QueryTransformer(ABC):
    @abstractmethod
    def transform(self, question: str) -> str:
        """Return the text that should be embedded for retrieval."""


class IdentityTransformer(QueryTransformer):
    def transform(self, question: str) -> str:
        return question


class HyDETransformer(QueryTransformer):

    PROMPT = "Write a short passage that directly answers the question below. Question: {question}\nPassage: "

    def __init__(self, llm: InferenceEngine):
        self._llm = llm

    def transform(self, question: str) -> str:
        try:
            assumed_answer = self._llm.generate(
                self.PROMPT.format(question=question), span_name="hyde_rewrite"
            )["content"]
        except (KeyError, TypeError, InferenceError):
            return ""
        return assumed_answer if assumed_answer else ""
