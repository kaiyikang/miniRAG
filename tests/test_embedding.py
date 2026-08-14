import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from minirag.adapters.embedder import (
    OpenRouterEmbeddingEngine,
    SentenceTransformerEngine,
)
from minirag.domain.ports import EmbeddingError


class TestSentenceTransformerEngine(unittest.TestCase):
    def _make_engine(self, model="all-MiniLM-L6-v2", cache_dir="/tmp/cache"):
        mock_cls = MagicMock()
        fake_module = SimpleNamespace(SentenceTransformer=mock_cls)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            engine = SentenceTransformerEngine(model=model, cache_dir=cache_dir)
        return engine, mock_cls

    def test_init_uses_model_and_cache_dir(self):
        engine, mock_cls = self._make_engine()
        mock_cls.assert_called_once()
        self.assertEqual(engine._model, mock_cls.return_value)

    def test_init_uses_custom_model_and_cache_dir(self):
        _, mock_cls = self._make_engine(model="custom-model")
        mock_cls.assert_called_once_with("custom-model", cache_folder="/tmp/cache")

    def test_embed_returns_vectors(self):
        engine, mock_cls = self._make_engine()
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [[0.1, 0.2], [0.3, 0.4]]
        )
        mock_cls.return_value = mock_model
        engine._model = mock_model
        result = engine.embed(["hello", "world"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.assert_called_once_with(["hello", "world"], batch_size=5)

    def test_embed_empty_list_returns_empty_list(self):
        engine, _ = self._make_engine()
        result = engine.embed([])
        self.assertEqual(result, [])

    def test_init_missing_model_or_cache_dir_raises(self):
        with self.assertRaises(ValueError):
            SentenceTransformerEngine(model="", cache_dir="/tmp/cache")
        with self.assertRaises(ValueError):
            SentenceTransformerEngine(model="m", cache_dir="")
        with self.assertRaises(ValueError):
            SentenceTransformerEngine(model="m", cache_dir="/tmp/cache", batch_size=0)


class TestOpenRouterEmbeddingEngine(unittest.TestCase):
    def test_init_missing_api_key_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            OpenRouterEmbeddingEngine(model="m", api_key=None)
        self.assertIn("OpenRouter model and API key are required", str(ctx.exception))

    def test_init_rejects_invalid_batch_size_and_timeout(self):
        with self.assertRaisesRegex(ValueError, "batch size"):
            OpenRouterEmbeddingEngine(model="m", api_key="k", batch_size=0)
        with self.assertRaisesRegex(ValueError, "timeout"):
            OpenRouterEmbeddingEngine(model="m", api_key="k", timeout=0)

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_empty_input_does_not_request(self, mock_post):
        engine = OpenRouterEmbeddingEngine(model="m", api_key="k")

        self.assertEqual(engine.embed([]), [])
        mock_post.assert_not_called()

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_success(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={
                    "data": [
                        {
                            "index": 0,
                            "embedding": [0.1, 0.2],
                            "provider_extra": "ignored",
                        }
                    ],
                    "usage": {"prompt_tokens": 1},
                }
            ),
        )

        engine = OpenRouterEmbeddingEngine(model="m", api_key="k", timeout=12)
        result = engine.embed(["hello"])

        self.assertEqual(result, [[0.1, 0.2]])
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k")
        self.assertEqual(kwargs["json"]["model"], "m")
        self.assertEqual(kwargs["json"]["input"], ["hello"])
        self.assertEqual(kwargs["timeout"], 12)

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_restores_input_order_from_response_indexes(self, mock_post):
        mock_post.return_value.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        engine = OpenRouterEmbeddingEngine(model="m", api_key="k")

        result = engine.embed(["first", "second"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])

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
                    return_value={
                        "data": [
                            {"index": index, "embedding": [0.1]}
                            for index, _ in enumerate(batch)
                        ]
                    }
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
        self.assertIn("complete batch", str(ctx.exception))

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_rejects_invalid_json_and_api_error(self, mock_post):
        engine = OpenRouterEmbeddingEngine(model="m", api_key="k")
        mock_post.return_value.json.side_effect = ValueError("bad json")
        with self.assertRaisesRegex(EmbeddingError, "invalid JSON"):
            engine.embed(["hello"])

        mock_post.return_value.json.side_effect = None
        mock_post.return_value.json.return_value = {
            "error": {"message": "unsupported model"}
        }
        with self.assertRaisesRegex(EmbeddingError, "API error"):
            engine.embed(["hello"])

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_rejects_missing_data_envelope(self, mock_post):
        engine = OpenRouterEmbeddingEngine(model="m", api_key="k")

        for payload in ([], {}):
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(EmbeddingError, "embedding response"),
            ):
                mock_post.return_value.json.return_value = payload
                engine.embed(["hello"])

    @patch("minirag.adapters.embedder.requests.post")
    def test_embed_rejects_unusable_vectors(self, mock_post):
        engine = OpenRouterEmbeddingEngine(model="m", api_key="k")
        cases = [
            (
                {"data": [{"index": 0, "embedding": []}]},
                "embedding item",
            ),
            (
                {"data": [{"index": 0, "embedding": [float("inf")]}]},
                "finite vectors",
            ),
            (
                {
                    "data": [
                        {"index": 0, "embedding": [0.1]},
                        {"index": 1, "embedding": [0.2, 0.3]},
                    ]
                },
                "dimensions",
            ),
        ]

        for payload, message in cases:
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(EmbeddingError, message),
            ):
                mock_post.return_value.json.return_value = payload
                texts = ["a", "b"] if len(payload["data"]) == 2 else ["a"]
                engine.embed(texts)


if __name__ == "__main__":
    unittest.main()
