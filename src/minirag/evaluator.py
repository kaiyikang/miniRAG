from minirag.rag import RAGPipeline
from minirag.llm_engine import InferenceEngine, InferenceError
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from typing import NamedTuple
import re
from pathlib import Path
import json
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
    context_relevancy: float | None


class Evaluator:

    def __init__(
        self,
        pipeline: RAGPipeline,
        dataset_dir: str,
        suffix: str = "default",
        recall_top_k: int = 5,
    ):
        self._pipeline = pipeline
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        self._dataset = Path(dataset_dir) / "qa_dataset10.jsonl"
        if not self._dataset.exists():
            raise ValueError("Cannot find the qa dataset!")

        self._eval_results = Path(dataset_dir) / f"eval_results_{suffix}.jsonl"
        self._eval_summary = Path(dataset_dir) / f"eval_summary_{suffix}.json"
        self._recall_top_k = recall_top_k

    def _evaluate_sample(self, sample: EvalSample, top_k: int):

        # RAG
        answer = self._pipeline.query(question=sample.question)

        # Metrics
        retrieval_recall_result = retrieval_recall(
            sample.expected_chunk_ids, answer.retrieved_chunk_ids, top_k
        )
        answer_f1 = token_f1(sample.expected_answer, answer.content)
        context_relevancy_result = context_relevancy(
            sample.question, answer.retrieved_chunks, self._pipeline.get_llm()
        )

        return EvalResult(
            question=sample.question,
            actual_answer=answer.content,
            expected_answer=sample.expected_answer,
            expected_chunk_ids=sample.expected_chunk_ids,
            retrieved_chunk_ids=answer.retrieved_chunk_ids,
            retrieval_recall=retrieval_recall_result,
            answer_f1=answer_f1,
            context_relevancy=context_relevancy_result,
        )

    def evaluate(self):
        avg_recall = 0.0
        avg_f1 = 0.0
        avg_context_relevancy = 0.0
        n_context_relevancy = 0

        total = 0
        lines = self._dataset.read_text(encoding="utf-8").splitlines()
        with open(self._eval_results, "w", encoding="utf-8") as out_f:
            for n, sample in enumerate(lines, 1):
                try:
                    result = self._evaluate_sample(
                        EvalSample(**json.loads(sample)), top_k=self._recall_top_k
                    )
                    out_f.write(json.dumps(result._asdict(), ensure_ascii=False) + "\n")

                    avg_recall += (result.retrieval_recall - avg_recall) / n
                    avg_f1 += (result.answer_f1 - avg_f1) / n

                    if result.context_relevancy is not None:
                        avg_context_relevancy += (
                            result.context_relevancy - avg_context_relevancy
                        ) / (n_context_relevancy + 1)
                        n_context_relevancy += 1

                    total += 1
                    print(
                        f"n = {n}, avg_recall={avg_recall}, avg_f1={avg_f1}, avg_context_relevancy={avg_context_relevancy}"
                    )
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
                        "avg_context_relevancy": avg_context_relevancy,
                        "n_context_relevancy": n_context_relevancy,
                        "top_k": self._recall_top_k,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


def retrieval_recall(
    expected_ids: list[str], retrieved_ids: list[str], k: int = 5
) -> float:
    if not expected_ids:
        return 0.0
    return len(set(expected_ids) & set(retrieved_ids[:k])) / len(expected_ids)


def token_f1(expected_answer: str, actual_answer: str) -> float:
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


def context_relevancy(
    question: str, retrieved_chunks: list[str], llm: InferenceEngine
) -> float | None:

    if not retrieved_chunks:
        return 0.0

    context_relevancy_prompt = """
You are a RAG system evaluator. Please determine which of the following numbered sentences contain information necessary to answer the question.\n\nQuestion:\n{question}\n\nContext sentences (numbered):\n
{sentences}\n\nOutput only a JSON array containing the numbers of the relevant sentences, for example [1, 3, 5]. If no sentences are relevant, output []. Do not output any other text or explanation."""

    _sentence_tokenizer = split_by_sentence_tokenizer()

    sentences = [
        sent
        for chunk in retrieved_chunks
        for sent in _sentence_tokenizer(chunk)
        if sent and not _is_junk(sent)
    ]
    sentences_str: str = "\n".join(
        [f"{idx+1}. {sent}" for idx, sent in enumerate(sentences)]
    )
    num_retry = 0
    while num_retry < 2:
        try:
            content = llm.generate(
                context_relevancy_prompt.format(
                    question=question, sentences=sentences_str
                )
            )["content"]

            results = re.search(r"\[.*\]", content, re.DOTALL)
            if not results:
                raise ValueError("results has no content")

            results = json.loads(results.group())

            if not isinstance(results, list):
                raise ValueError("results must be a list.")

            if len(results) == 0:
                return 0.0
            elif all(1 <= i <= len(sentences) for i in results) and all(
                isinstance(i, int) for i in results
            ):
                print(f"DEBUG: \n{sentences_str}\nresult:\n{results}")
                return len(results) / len(sentences)
            else:
                raise ValueError("Verification failed")
        except (ValueError, KeyError, TypeError, InferenceError):
            num_retry += 1

    return None


def _is_junk(sent: str) -> bool:
    sent = sent.strip()
    if len(sent.split()) < 3:
        return True
    if not re.search(r"[a-zA-Z]{2,}", sent):
        return True
    return False
