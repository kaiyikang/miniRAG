from minirag.rag import RAGPipeline
from typing import NamedTuple
import re
from pathlib import Path
import json


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

    matched_num = len(expected_answer & actual_answer)
    precision = matched_num / len(actual_answer)
    recall = matched_num / len(expected_answer)
    if not precision + recall:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


class Evaluator:
    def __init__(self, pipeline: RAGPipeline, dataset_dir: str, recall_top_k: int = 5):
        self._pipeline = pipeline
        self._dataset = Path(dataset_dir) / "qa_dataset.jsonl"
        self._eval_results = Path(dataset_dir) / "eval_results.jsonl"
        self._eval_summary = Path(dataset_dir) / "eval_summary.json"

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

    def evaluate(self):
        limit = 5
        avg_recall = 0.0
        avg_f1 = 0.0
        total = 0
        with (
            open(self._eval_results, "w", encoding="utf-8") as out_f,
            open(self._dataset, "r", encoding="utf-8") as in_f,
        ):
            for n, sample in enumerate(in_f, 1):
                try:
                    result = self._evaluate_sample(
                        EvalSample(**json.loads(sample)), top_k=self._recall_top_k
                    )
                    out_f.write(json.dumps(result._asdict(), ensure_ascii=False) + "\n")

                    avg_recall += (result.retrieval_recall - avg_recall) / n
                    avg_f1 += (result.answer_f1 - avg_f1) / n
                    total += 1
                    print(f"n = {n}, avg_recall={avg_recall}, avg_f1={avg_f1}")
                except Exception as e:
                    print(f"error on sample {n}: {e}")
                    continue
                if n > limit:
                    break

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
