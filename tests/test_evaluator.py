import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from minirag.evaluator import (
    Evaluator,
    EvalResult,
    EvalSample,
    QA_DATASET_FILENAME,
    cal_retrieval_recall as retrieval_recall,
    cal_token_f1 as token_f1,
)
from minirag.types import Answer


class TestRetrievalRecall(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(retrieval_recall(["a", "b"], ["a", "b"]), 1.0)

    def test_partial_match(self):
        self.assertEqual(retrieval_recall(["a", "b", "c"], ["a", "x"]), 1 / 3)

    def test_no_expected(self):
        self.assertEqual(retrieval_recall([], ["a"]), 0.0)

    def test_respects_k(self):
        self.assertEqual(retrieval_recall(["a", "b"], ["a", "b", "c"], k=1), 0.5)


class TestTokenF1(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(token_f1("hello world", "hello world"), 1.0)

    def test_no_overlap(self):
        self.assertEqual(token_f1("hello", "world"), 0.0)

    def test_partial_overlap(self):
        # hello world vs hello there -> precision=1/2, recall=1/2, f1=0.5
        self.assertEqual(token_f1("hello world", "hello there"), 0.5)

    def test_ignores_punctuation_and_case(self):
        self.assertEqual(token_f1("Hello, World!", "hello world"), 1.0)

    def test_empty_actual(self):
        self.assertEqual(token_f1("hello", ""), 0.0)


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        dataset = Path(self.tmpdir) / QA_DATASET_FILENAME
        with open(dataset, "w", encoding="utf-8") as f:
            for i in range(7):
                f.write(json.dumps({"question": f"q{i}", "expected_answer": f"a{i}", "expected_chunk_ids": [f"c{i}"]}, ensure_ascii=False) + "\n")

        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = Answer(
            content="actual answer",
            sources=[],
            retrieved_chunk_ids=["c1"],
            retrieved_chunks=["chunk text"],
        )
        self.evaluator = Evaluator(mock_pipeline, self.tmpdir, recall_top_k=5)

    def test_evaluate_sample(self):
        sample = EvalSample("q1", "a1", ["c1"])
        result = self.evaluator._evaluate_sample(sample, top_k=5)
        self.assertIsInstance(result, EvalResult)
        self.assertEqual(result.question, "q1")
        self.assertEqual(result.actual_answer, "actual answer")

    def test_evaluate_runs_and_writes_files(self):
        self.evaluator.evaluate()
        self.assertTrue(self.evaluator._eval_results.exists())
        self.assertTrue(self.evaluator._eval_summary.exists())

        with open(self.evaluator._eval_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        metrics = summary["metrics"]
        self.assertIn("retrieval_recall@5", metrics)
        self.assertIn("answer_f1", metrics)
        self.assertIn("n_samples", metrics)
        self.assertEqual(metrics["n_samples"], 7)

    def test_evaluate_handles_sample_errors(self):
        self.evaluator._pipeline.query.side_effect = RuntimeError("boom")
        self.evaluator.evaluate()
        # Should still write summary even when all samples error
        self.assertTrue(self.evaluator._eval_summary.exists())
        with open(self.evaluator._eval_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["metrics"]["n_samples"], 0)

    def test_missing_dataset_raises(self):
        # Existing but empty dir: dataset file absent -> ValueError.
        empty_dir = tempfile.mkdtemp()
        with self.assertRaises(ValueError):
            Evaluator(MagicMock(), empty_dir)


if __name__ == "__main__":
    unittest.main()
