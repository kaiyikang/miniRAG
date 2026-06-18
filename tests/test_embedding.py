import os
import unittest
from unittest.mock import MagicMock, patch

from minirag.embedding import (
    EmbeddingError,
    OpenRouterEmbeddingEngine,
    SentenceTransformerEngine,
)


class TestSentenceTransformerEngine(unittest.TestCase):
    @patch("minirag.embedding.SentenceTransformer")
    def test_init_uses_default_model(self, mock_cls):
        engine = SentenceTransformerEngine()
        mock_cls.assert_called_once()
        self.assertEqual(engine._model, mock_cls.return_value)

    @patch("minirag.embedding.SentenceTransformer")
    def test_init_uses_custom_model_and_cache_dir(self, mock_cls):
        engine = SentenceTransformerEngine(model="custom-model", cache_dir="/tmp/cache")
        mock_cls.assert_called_once_with("custom-model", cache_folder="/tmp/cache")

    @patch("minirag.embedding.SentenceTransformer")
    def test_embed_returns_vectors(self, mock_cls):
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [[0.1, 0.2], [0.3, 0.4]]
        )
        mock_cls.return_value = mock_model

        engine = SentenceTransformerEngine()
        result = engine.embed(["hello", "world"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.assert_called_once_with(["hello", "world"])

    @patch("minirag.embedding.SentenceTransformer")
    def test_embed_empty_list_returns_empty_list(self, mock_cls):
        engine = SentenceTransformerEngine()
        result = engine.embed([])
        self.assertEqual(result, [])


class TestOpenRouterEmbeddingEngine(unittest.TestCase):
    def test_init_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                OpenRouterEmbeddingEngine(api_key=None)
        self.assertIn("OpenRouter API key is required", str(ctx.exception))

    @patch("minirag.embedding.requests.post")
    def test_embed_success(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"data": [{"embedding": [0.1, 0.2]}]}),
        )

        engine = OpenRouterEmbeddingEngine(model="m", api_key="k")
        result = engine.embed(["hello"])

        self.assertEqual(result, [0.1, 0.2])
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k")
        self.assertEqual(kwargs["json"]["model"], "m")
        self.assertEqual(kwargs["json"]["input"], ["hello"])

    @patch("minirag.embedding.requests.post")
    def test_embed_request_exception(self, mock_post):
        import requests

        mock_post.side_effect = requests.ConnectionError("boom")
        engine = OpenRouterEmbeddingEngine(api_key="k")
        with self.assertRaises(EmbeddingError) as ctx:
            engine.embed(["hello"])
        self.assertIn("LLM embedding failed", str(ctx.exception))

    @patch("minirag.embedding.requests.post")
    def test_embed_unexpected_response_format(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"data": []}),
        )
        engine = OpenRouterEmbeddingEngine(api_key="k")
        with self.assertRaises(EmbeddingError) as ctx:
            engine.embed(["hello"])
        self.assertIn("Unexpected response format", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
