from minirag.rag import RAGPipeline


class Evaluator:
    def __init__(self, pipeline: RAGPipeline, metrics: list[str] | None = None):
        self._pipeline = pipeline
        self._metrics = metrics or [
            "retrieval_recall@5",
            "answer_f1",
        ]
