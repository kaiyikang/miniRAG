from minirag.rag import RAGPipeline
from typing import NamedTuple
import re
from pathlib import Path
import json
import random
import traceback


class EvalSample(NamedTuple):
    question: str
    expected_answer: str
    expected_chunk_ids: list[str]


class EvalResult(NamedTuple):
    question: str
    expected_answer: str
    actual_answer: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    retrieval_recall: float
    answer_f1: float


def retrieval_recall(
    expected_ids: list[str], retrieved_ids: list[str], k: int = 5
) -> float:
    if not expected_ids:
        return 0.0
    return len(set(expected_ids) & set(retrieved_ids[:k])) / len(expected_ids)


def token_f1(expected_answer: str, actual_answer: str):
    def clean_and_set(text: str) -> set[str]:
        return set(re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).split())

    expected_answer = clean_and_set(expected_answer)
    actual_answer = clean_and_set(actual_answer)

    if not expected_answer or not actual_answer:
        return 0.0

    matched_num = len(expected_answer & actual_answer)
    precision = matched_num / len(actual_answer)
    recall = matched_num / len(expected_answer)
    if not precision + recall:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


class Evaluator:
    def __init__(
        self,
        pipeline: RAGPipeline,
        dataset_dir: str,
        suffix: str = "",
        recall_top_k: int = 5,
    ):
        self._pipeline = pipeline
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        self._dataset = Path(dataset_dir) / "qa_dataset10.jsonl"
        self._eval_results = Path(dataset_dir) / f"eval_results_{suffix}.jsonl"
        self._eval_summary = Path(dataset_dir) / f"eval_summary_{suffix}.json"

        if not self._dataset.exists():
            raise ValueError("Cannot find the qa dataset!")

        self._recall_top_k = recall_top_k

    def _evaluate_sample(self, sample: EvalSample, top_k: int):

        answer = self._pipeline.query(question=sample.question)

        return EvalResult(
            question=sample.question,
            actual_answer=answer.content,
            expected_answer=sample.expected_answer,
            expected_chunk_ids=sample.expected_chunk_ids,
            retrieved_chunk_ids=answer.retrieved_chunk_ids,
            retrieval_recall=retrieval_recall(
                sample.expected_chunk_ids, answer.retrieved_chunk_ids, top_k
            ),
            answer_f1=token_f1(sample.expected_answer, answer.content),
        )

    def evaluate(self, n_samples: int = 20):
        avg_recall = 0.0
        avg_f1 = 0.0
        total = 0
        lines = self._dataset.read_text(encoding="utf-8").splitlines()
        sampled = random.sample(lines, min(n_samples, len(lines)))
        with open(self._eval_results, "w", encoding="utf-8") as out_f:
            for n, sample in enumerate(sampled, 1):
                try:
                    result = self._evaluate_sample(
                        EvalSample(**json.loads(sample)), top_k=self._recall_top_k
                    )
                    out_f.write(json.dumps(result._asdict(), ensure_ascii=False) + "\n")

                    avg_recall += (result.retrieval_recall - avg_recall) / n
                    avg_f1 += (result.answer_f1 - avg_f1) / n
                    total += 1
                    print(f"n = {n}, avg_recall={avg_recall}, avg_f1={avg_f1}")
                except Exception:
                    print(f"error on sample {n}:")
                    traceback.print_exc()
                    continue

        with open(self._eval_summary, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "retrieval_recall@5": avg_recall,
                        "answer_f1": avg_f1,
                        "n_samples": total,
                        "top_k": self._recall_top_k,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
