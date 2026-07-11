from minirag.rag import RAGPipeline
from minirag.llm_engine import InferenceEngine, InferenceError
from minirag.embedding import EmbeddingEngine
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from typing import NamedTuple
from datetime import datetime
import re
from pathlib import Path
import json
import subprocess
import traceback
import math


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
    answer_relevancy: float | None
    faithfulness: float | None


class Evaluator:

    def __init__(
        self,
        pipeline: RAGPipeline,
        dataset_dir: str,
        suffix: str = "default",
        recall_top_k: int = 5,
        params: dict | None = None,
    ):
        self._pipeline = pipeline
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        self._dataset = Path(dataset_dir) / "qa_dataset10.jsonl"
        if not self._dataset.exists():
            raise ValueError("Cannot find the qa dataset!")

        # One directory per run: eval/runs/<timestamp>_<suffix>/
        run_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_{suffix}"
        self._run_dir = Path(dataset_dir) / "runs" / run_id
        self._run_dir.mkdir(parents=True)  # no exist_ok: a name clash should fail loudly
        self._eval_results = self._run_dir / "results.jsonl"
        self._eval_summary = self._run_dir / "summary.json"
        self._recall_top_k = recall_top_k
        self._params = params or {}

    def _evaluate_sample(self, sample: EvalSample, top_k: int):

        # RAG
        answer = self._pipeline.query(question=sample.question)

        # Metrics
        retrieval_recall_result = cal_retrieval_recall(
            sample.expected_chunk_ids, answer.retrieved_chunk_ids, top_k
        )
        answer_f1 = cal_token_f1(sample.expected_answer, answer.content)
        context_relevancy = cal_context_relevancy(
            sample.question, answer.retrieved_chunks, self._pipeline.get_llm()
        )

        answer_relevancy = cal_question_answer_relevancy(
            sample.question,
            answer.content,
            self._pipeline.get_llm(),
            self._pipeline.get_embed(),
        )

        faithfulness = cal_context_answer_faithfulness(
            query=sample.question,
            context=answer.retrieved_chunks,
            answer=answer.content,
            llm=self._pipeline.get_llm(),
        )

        return EvalResult(
            question=sample.question,
            actual_answer=answer.content,
            expected_answer=sample.expected_answer,
            expected_chunk_ids=sample.expected_chunk_ids,
            retrieved_chunk_ids=answer.retrieved_chunk_ids,
            retrieval_recall=retrieval_recall_result,
            answer_f1=answer_f1,
            context_relevancy=context_relevancy,
            answer_relevancy=answer_relevancy,
            faithfulness=faithfulness,
        )

    def evaluate(self):
        avg_recall = 0.0
        avg_f1 = 0.0
        avg_context_relevancy = 0.0
        n_context_relevancy = 0
        avg_answer_relevancy = 0.0
        n_answer_relevancy = 0
        avg_faithfulness = 0.0
        n_faithfulness = 0

        total = 0
        lines = self._dataset.read_text(encoding="utf-8").splitlines()
        with open(self._eval_results, "w", encoding="utf-8") as out_f:
            for n, sample in enumerate(lines, 1):
                try:
                    result = self._evaluate_sample(
                        EvalSample(**json.loads(sample)), top_k=self._recall_top_k
                    )
                    out_f.write(json.dumps(result._asdict(), ensure_ascii=False) + "\n")

                    # metrics
                    avg_recall += (result.retrieval_recall - avg_recall) / n
                    avg_f1 += (result.answer_f1 - avg_f1) / n

                    if result.context_relevancy is not None:
                        avg_context_relevancy += (
                            result.context_relevancy - avg_context_relevancy
                        ) / (n_context_relevancy + 1)
                        n_context_relevancy += 1

                    if result.answer_relevancy is not None:
                        avg_answer_relevancy += (
                            result.answer_relevancy - avg_answer_relevancy
                        ) / (n_answer_relevancy + 1)
                        n_answer_relevancy += 1

                    if result.faithfulness is not None:
                        avg_faithfulness += (result.faithfulness - avg_faithfulness) / (
                            n_faithfulness + 1
                        )
                        n_faithfulness += 1

                    total += 1
                    print(
                        f"n = {n}, avg_recall={avg_recall}, avg_f1={avg_f1}, avg_context_relevancy={avg_context_relevancy}, avg_n_faithfulness={avg_faithfulness}"
                    )
                except Exception:
                    print(f"error on sample {n}:")
                    traceback.print_exc()
                    continue

        commit, dirty = _git_info()
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": commit,
            "git_dirty": dirty,
            "params": self._params,
            "metrics": {
                "retrieval_recall@5": avg_recall,
                "answer_f1": avg_f1,
                "n_samples": total,
                "avg_context_relevancy": avg_context_relevancy,
                "n_context_relevancy": n_context_relevancy,
                "avg_answer_relevancy": avg_answer_relevancy,
                "n_answer_relevancy": n_answer_relevancy,
                "avg_faithfulness": avg_faithfulness,
                "n_faithfulness": n_faithfulness,
                "top_k": self._recall_top_k,
            },
        }
        with open(self._eval_summary, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))


def cal_retrieval_recall(
    expected_ids: list[str], retrieved_ids: list[str], k: int = 5
) -> float:
    if not expected_ids:
        return 0.0
    return len(set(expected_ids) & set(retrieved_ids[:k])) / len(expected_ids)


def cal_token_f1(expected_answer: str, actual_answer: str) -> float:
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


def cal_context_relevancy(
    question: str, retrieved_chunks: list[str], llm: InferenceEngine
) -> float | None:
    """relevancy: question - retrieved chunk"""
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

    def fn():
        content = llm.generate(
            context_relevancy_prompt.format(question=question, sentences=sentences_str)
        )["content"]

        results = _extract_json_array(content)
        if not results:
            raise ValueError("results has no content")

        results = json.loads(results)

        if not isinstance(results, list):
            raise ValueError("results must be a list.")

        if len(results) == 0:
            return 0.0
        elif all(1 <= i <= len(sentences) for i in results) and all(
            isinstance(i, int) for i in results
        ):
            return len(results) / len(sentences)
        else:
            raise ValueError("Verification failed")

    return _retry_until(fn=fn)


def cal_question_answer_relevancy(
    question: str,
    answer: str,
    llm: InferenceEngine,
    embed: EmbeddingEngine,
    n_questions: int = 3,
) -> float | None:

    perfunctory_judgment_prompt = """You are evaluating whether an answer to a question actually commits to a specific response, or evades/refuses to answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nAn answer is "noncommittal" if it says things like "I don't know", "the context does not provide this information", "I cannot answer this question", or otherwise avoids giving a specific, substantive answer.\n\nOutput only a bool in this exact format: true or false. Do not output any other text or explanation."""

    reverse_inference_prompt = """You are given an answer. Generate {n_questions} distinct questions that this answer could plausibly be responding to. Each generated question must be answerable using only the information in the answer.\n\nAnswer:\n{answer}\n\nOutput only a JSON array of {n_questions} strings, for example ["question 1", "question 2", "question 3"]. Do not output any other text or explanation."""

    if not question:
        return None

    # bad answer will not be counted
    if not answer:
        return 0.0

    # perfunctory judgment
    def _perfunctory_judgement_fn():
        perfunctory_judgment = llm.generate(
            perfunctory_judgment_prompt.format(question=question, answer=answer)
        )["content"]
        if perfunctory_judgment.strip().lower() in {
            "true",
            "1",
            "yes",
        }:
            return 0.0
        return perfunctory_judgment

    perfunctory_judgment = _retry_until(_perfunctory_judgement_fn)
    if perfunctory_judgment == 0.0 or perfunctory_judgment == None:
        return perfunctory_judgment

    # reverse inference
    def _reverse_inference_fn():
        content = llm.generate(
            reverse_inference_prompt.format(n_questions=n_questions, answer=answer)
        )["content"]

        reversed_questions = _extract_json_array(content)
        if not reversed_questions:
            raise ValueError("reversed questions has no content")

        reversed_questions = json.loads(reversed_questions)
        if not isinstance(reversed_questions, list) or not all(
            isinstance(i, str) for i in reversed_questions
        ):
            raise ValueError("reversed questions must be a list and have content")

        return reversed_questions

    reversed_questions = _retry_until(_reverse_inference_fn)
    if not reversed_questions:
        return None

    # Similarity
    question_embed = embed.embed([question])[0]
    reversed_questions_embeds = embed.embed(reversed_questions)
    return sum(
        [_cosine_similarity(question_embed, q) for q in reversed_questions_embeds]
    ) / len(reversed_questions_embeds)


def cal_context_answer_faithfulness(
    query: str,
    context: list[str],
    answer: str,
    llm: InferenceEngine,
) -> float | None:

    def _supported(v) -> bool:
        return str(v).strip() in {"1", "true", "yes"}

    claims = decompose_answer(query, answer, llm)
    if not claims:
        return None
    verified_claims = verify_context_answer(claims, context, llm)
    if not verified_claims:
        return None
    num_verified_claims = len(
        [1 for r in verified_claims if _supported(r.get("verdict"))]
    )
    num_claims = len(claims)
    return num_verified_claims / num_claims


def decompose_answer(
    query: str,
    answer: str,
    llm: InferenceEngine,
) -> list[str] | None:
    prompt = """
Break the following answer into a set of standalone statements (claims).
Rules:
- Each claim states exactly one fact
- Resolve pronouns (he/it/this) to the actual entity, so each claim is understandable on its own
- Only extract facts; ignore filler and pleasantries
- Output a JSON array: ["claim1", "claim2", ...]

Query: {query}
Answer: {answer}
"""

    def fn():
        content = llm.generate(prompt.format(query=query, answer=answer))["content"]
        content = _extract_json_array(content)
        if not content:
            raise ValueError("decomposed answer has no content")

        content = json.loads(content)

        if not isinstance(content, list) or not all(
            isinstance(item, str) for item in content
        ):
            raise ValueError("content cannot be phrased.")

        return content

    claims = _retry_until(fn)

    if not claims:
        return None

    return claims


def verify_context_answer(
    claims: list[str], context: list[str], llm: InferenceEngine
) -> list[dict] | None:
    prompt = """
Given the context, decide whether each statement can be inferred from it.
- Semantic support is enough; exact wording is not required
- If the context doesn't mention it or contradicts it, verdict 0; if it can be inferred, verdict 1
- Output a JSON array in the same order as the input: [{{"claim": "...", "verdict": 0 or 1}}]

Context:
{context}

Statements:
{statements}
"""

    full_context = "\n\n".join(context)
    statements = "\n".join(
        [f"{idx}. {statement}" for idx, statement in enumerate(claims)]
    )

    def fn():

        content = llm.generate(
            prompt.format(context=full_context, statements=statements)
        )["content"]
        content = _extract_json_array(content)
        if not content:
            raise ValueError("The result of the claim has no content")

        content = json.loads(content)

        if not isinstance(content, list) or not all(
            isinstance(item, dict) for item in content
        ):
            raise ValueError("The result of the statement cannot be phrased.")

        return content

    verified_claims = _retry_until(fn)

    if not verified_claims:
        return None

    return verified_claims


def _is_junk(sent: str) -> bool:
    sent = sent.strip()
    if len(sent.split()) < 3:
        return True
    if not re.search(r"[a-zA-Z]{2,}", sent):
        return True
    return False


def _cosine_similarity(v1, v2):
    def _dot_product(a, b):
        return sum(x * y for x, y in zip(a, b))

    def _magnitude(v):
        return math.sqrt(sum(x**2 for x in v))

    dot = _dot_product(v1, v2)
    mag1 = _magnitude(v1)
    mag2 = _magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)


def _retry_until(
    fn, exceptions=(ValueError, KeyError, TypeError, InferenceError), max_attempts=2
):
    for _ in range(max_attempts):
        try:
            return fn()
        except exceptions as e:
            print(repr(e))
            continue
    return None


def _extract_json_array(content: str):
    m = re.search(r"\[.*\]", content, re.DOTALL)
    return m.group() if m else None


def _git_info() -> tuple[str | None, bool | None]:
    """Return (short commit, dirty flag), or (None, None) if git is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except Exception:
        return None, None
