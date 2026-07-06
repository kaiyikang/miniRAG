import unittest
from unittest.mock import MagicMock

from minirag.query_transform import IdentityTransformer, HyDETransformer
from minirag.llm_engine import InferenceError


class TestIdentityTransformer(unittest.TestCase):
    def test_returns_question_unchanged(self):
        transformer = IdentityTransformer()
        self.assertEqual(transformer.transform("What is RAG?"), "What is RAG?")


class TestHyDETransformer(unittest.TestCase):
    def test_returns_generated_content(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {"content": "RAG is a hypothetical passage."}

        transformer = HyDETransformer(mock_llm)
        result = transformer.transform("What is RAG?")

        self.assertEqual(result, "RAG is a hypothetical passage.")
        prompt_used = mock_llm.generate.call_args[0][0]
        self.assertIn("What is RAG?", prompt_used)

    def test_inference_error_falls_back_to_empty_string(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = InferenceError("boom")

        transformer = HyDETransformer(mock_llm)
        self.assertEqual(transformer.transform("What is RAG?"), "")

    def test_missing_content_key_falls_back_to_empty_string(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {}

        transformer = HyDETransformer(mock_llm)
        self.assertEqual(transformer.transform("What is RAG?"), "")

    def test_non_dict_response_falls_back_to_empty_string(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = None

        transformer = HyDETransformer(mock_llm)
        self.assertEqual(transformer.transform("What is RAG?"), "")

    def test_empty_content_falls_back_to_empty_string(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {"content": ""}

        transformer = HyDETransformer(mock_llm)
        self.assertEqual(transformer.transform("What is RAG?"), "")


if __name__ == "__main__":
    unittest.main()
