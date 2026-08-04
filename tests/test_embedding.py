import unittest
from unittest.mock import MagicMock, patch

from minirag.domain.ports import EmbeddingError
from minirag.adapters.embedder import (
    OpenRouterEmbeddingEngine,
    SentenceTransformerEngine,
)


class TestSentenceTransformerEngine(unittest.TestCase):
    @patch("sentence_transformers.SentenceTransformer")
    def test_init_uses_model_and_cache_dir(self, mock_cls):
        engine = SentenceTransformerEngine(
            model="all-MiniLM-L6-v2", cache_dir="/tmp/cache"
        )
        mock_cls.assert_called_once()
        self.assertEqual(engine._model, mock_cls.return_value)

    @patch("sentence_transformers.SentenceTransformer")
    def test_init_uses_custom_model_and_cache_dir(self, mock_cls):
        engine = SentenceTransformerEngine(model="custom-model", cache_dir="/tmp/cache")
        mock_cls.assert_called_once_with("custom-model", cache_folder="/tmp/cache")

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_returns_vectors(self, mock_cls):
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [[0.1, 0.2], [0.3, 0.4]]
        )
        mock_cls.return_value = mock_model

        engine = SentenceTransformerEngine(
            model="all-MiniLM-L6-v2", cache_dir="/tmp/cache"
        )
        result = engine.embed(["hello", "world"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.assert_called_once_with(["hello", "world"], batch_size=5)

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_empty_list_returns_empty_list(self, mock_cls):
        engine = SentenceTransformerEngine(
            model="all-MiniLM-L6-v2", cache_dir="/tmp/cache"
        )
        result = engine.embed([])
        self.assertEqual(result, [])

    @patch("sentence_transformers.SentenceTransformer")
    def test_init_missing_model_or_cache_dir_raises(self, mock_cls):
        with self.assertRaises(ValueError):
            SentenceTransformerEngine(model="", cache_dir="/tmp/cache")
        with self.assertRaises(ValueError):
            SentenceTransformerEngine(model="m", cache_dir="")


class TestOpenRouterEmbeddingEngine(unittest.TestCase):
    def test_init_missing_api_key_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            OpenRouterEmbeddingEngine(model="m", api_key=None)
        self.assertIn("OpenRouter model and API key are required", str(ctx.exception))

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_success(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"data": [{"embedding": [0.1, 0.2]}]}),
        )

        engine = OpenRouterEmbeddingEngine(model="m", api_key="k")
        result = engine.embed(["hello"])

        self.assertEqual(result, [[0.1, 0.2]])
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k")
        self.assertEqual(kwargs["json"]["model"], "m")
        self.assertEqual(kwargs["json"]["input"], ["hello"])

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_request_exception(self, mock_post):
        import requests

        mock_post.side_effect = requests.ConnectionError("boom")
        engine = OpenRouterEmbeddingEngine(model="m", api_key="k")
        with self.assertRaises(EmbeddingError) as ctx:
            engine.embed(["hello"])
        self.assertIn("LLM embedding failed", str(ctx.exception))

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_respects_custom_batch_size(self, mock_post):
        def _make_response(batch):
            return MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(
                    return_value={"data": [{"embedding": [0.1]} for _ in batch]}
                ),
            )

        mock_post.side_effect = lambda *args, **kwargs: _make_response(
            kwargs["json"]["input"]
        )

        engine = OpenRouterEmbeddingEngine(model="m", api_key="k", batch_size=2)
        result = engine.embed(["a", "b", "c"])

        self.assertEqual(result, [[0.1], [0.1], [0.1]])
        self.assertEqual(mock_post.call_count, 2)

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_count_mismatch_raises(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"data": []}),
        )
        engine = OpenRouterEmbeddingEngine(model="m", api_key="k")
        with self.assertRaises(EmbeddingError) as ctx:
            engine.embed(["hello"])
        self.assertIn("count mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
