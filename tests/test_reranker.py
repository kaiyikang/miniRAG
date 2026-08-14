import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from minirag.adapters.reranker import (
    CrossEncoderReranker,
    OpenRouterReranker,
    VectorReranker,
)
from minirag.domain.models import SearchedChunk
from minirag.domain.ports import RerankerError


def make_chunks() -> list[SearchedChunk]:
    return [
        SearchedChunk("1", "doc a", {"source": "a"}, [1.0], 0.0),
        SearchedChunk("2", "doc b", {"source": "b"}, [2.0], 0.0),
        SearchedChunk("3", "doc c", {"source": "c"}, [3.0], 0.0),
    ]


class TestOpenRouterReranker(unittest.TestCase):
    def test_requires_configuration_and_positive_timeout(self):
        with self.assertRaises(ValueError):
            OpenRouterReranker("", "key")
        with self.assertRaises(ValueError):
            OpenRouterReranker("model", "")
        with self.assertRaises(ValueError):
            OpenRouterReranker("model", "key", timeout=0)

    @patch("minirag.adapters.reranker.requests.post")
    def test_reranks_and_preserves_chunk_data(self, mock_post):
        mock_post.return_value.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 2, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.9},
            ]
        }
        reranker = OpenRouterReranker("cohere/rerank-4-fast", "key", timeout=12)

        result = reranker.rank("query", None, make_chunks())

        self.assertEqual([chunk.chunk_id for chunk in result], ["2", "3", "1"])
        self.assertEqual([chunk.score for chunk in result], [0.9, 0.5, 0.1])
        self.assertEqual(result[0].metadata, {"source": "b"})
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["documents"], ["doc a", "doc b", "doc c"])
        self.assertEqual(kwargs["json"]["top_n"], 3)
        self.assertEqual(kwargs["timeout"], 12)

    def test_validates_rank_inputs(self):
        reranker = OpenRouterReranker("model", "key")
        with self.assertRaises(ValueError):
            reranker.rank(None, None, make_chunks())
        with self.assertRaises(ValueError):
            reranker.rank("query", [1.0], make_chunks())
        self.assertEqual(reranker.rank("query", None, []), [])

    @patch("minirag.adapters.reranker.requests.post")
    def test_wraps_http_errors(self, mock_post):
        mock_post.side_effect = requests.Timeout("timed out")
        reranker = OpenRouterReranker("model", "key")

        with self.assertRaisesRegex(RerankerError, "request failed"):
            reranker.rank("query", None, make_chunks())

    @patch("minirag.adapters.reranker.requests.post")
    def test_rejects_invalid_or_incomplete_results(self, mock_post):
        reranker = OpenRouterReranker("model", "key")
        invalid_results = [
            [{"index": 0, "relevance_score": 0.5}],
            [
                {"index": 0, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.4},
                {"index": 2, "relevance_score": 0.3},
            ],
            [
                {"index": 0, "relevance_score": float("inf")},
                {"index": 1, "relevance_score": 0.4},
                {"index": 2, "relevance_score": 0.3},
            ],
        ]

        for results in invalid_results:
            with (
                self.subTest(results=results),
                self.assertRaisesRegex(RerankerError, "complete permutation"),
            ):
                mock_post.return_value.json.return_value = {"results": results}
                reranker.rank("query", None, make_chunks())

    @patch("minirag.adapters.reranker.requests.post")
    def test_rejects_invalid_response_envelopes(self, mock_post):
        reranker = OpenRouterReranker("model", "key")
        cases = [
            ([], "expected an object"),
            ({"error": {"message": "bad request"}}, "API error"),
            ({"results": {}}, "results must be a list"),
        ]

        for payload, message in cases:
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(RerankerError, message),
            ):
                mock_post.return_value.json.return_value = payload
                reranker.rank("query", None, make_chunks())

    @patch("minirag.adapters.reranker.requests.post")
    def test_rejects_malformed_result_items(self, mock_post):
        reranker = OpenRouterReranker("model", "key")
        mock_post.return_value.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": "not-a-number"},
                {"index": 1, "relevance_score": 0.4},
                {"index": 2, "relevance_score": 0.3},
            ]
        }

        with self.assertRaisesRegex(RerankerError, "result format"):
            reranker.rank("query", None, make_chunks())


class TestVectorReranker(unittest.TestCase):
    def test_reranks_by_cosine_similarity(self):
        reranker = VectorReranker()
        query_embedding = [1.0, 0.0]
        chunks = [
            SearchedChunk("1", "doc a", {}, [0.0, 1.0], 0.0),  # cos = 0.0
            SearchedChunk("2", "doc b", {}, [1.0, 0.0], 0.0),  # cos = 1.0
            SearchedChunk("3", "doc c", {}, [0.6, 0.8], 0.0),  # cos = 0.6
        ]

        result = reranker.rank(None, query_embedding, chunks)

        assert [r.chunk_id for r in result] == ["2", "3", "1"]
        assert [round(r.score, 6) for r in result] == [1.0, 0.6, 0.0]

    def test_raises_when_query_embedding_missing(self):
        reranker = VectorReranker()
        chunks = [SearchedChunk("1", "doc a", {}, [1.0, 0.0], 0.0)]

        with self.assertRaises(ValueError):
            reranker.rank(None, None, chunks)

    def test_raises_when_query_text_provided(self):
        reranker = VectorReranker()
        chunks = [SearchedChunk("1", "doc a", {}, [1.0, 0.0], 0.0)]

        with self.assertRaises(ValueError):
            reranker.rank("text", [1.0, 0.0], chunks)

    def test_cosine_zero_magnitude(self):
        reranker = VectorReranker()
        chunks = [SearchedChunk("1", "doc a", {}, [0.0, 0.0], 0.0)]
        result = reranker.rank(None, [1.0, 0.0], chunks)
        self.assertEqual(result[0].score, 0.0)


class TestCrossReranker(unittest.TestCase):

    def _make_reranker(self):
        mock_cross_encoder_class = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock_cross_encoder_class.return_value = mock_model
        fake_module = SimpleNamespace(CrossEncoder=mock_cross_encoder_class)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            reranker = CrossEncoderReranker("dummy_model", "/tmp/cache")
        return reranker, mock_model

    def test_cross_encoder_reranker(self):
        reranker, _ = self._make_reranker()
        chunks = [
            SearchedChunk("1", "doc a", {}, [0.0], 0.0),
            SearchedChunk("2", "doc b", {}, [0.0], 0.0),
            SearchedChunk("3", "doc c", {}, [0.0], 0.0),
        ]

        result = reranker.rank("query", None, chunks)
        assert [r.chunk_id for r in result] == ["2", "3", "1"]

    def test_raises_when_query_text_missing(self):
        reranker, _ = self._make_reranker()
        chunks = [SearchedChunk("1", "doc a", {}, [0.0], 0.0)]

        with self.assertRaises(ValueError):
            reranker.rank(None, None, chunks)

    def test_init_missing_model_raises(self):
        with self.assertRaises(ValueError):
            CrossEncoderReranker("")

    def test_raises_when_query_embedding_provided(self):
        reranker, _ = self._make_reranker()
        chunks = [SearchedChunk("1", "doc a", {}, [0.0], 0.0)]

        with self.assertRaises(ValueError):
            reranker.rank("query", [1.0], chunks)

    def test_empty_chunks_returns_empty_list(self):
        reranker, _ = self._make_reranker()
        result = reranker.rank("query", None, [])
        self.assertEqual(result, [])

    def test_init_without_cache_dir_does_not_create_directory(self):
        mock_cross_encoder_class = MagicMock()
        fake_module = SimpleNamespace(CrossEncoder=mock_cross_encoder_class)
        with (
            patch.dict(sys.modules, {"sentence_transformers": fake_module}),
            patch("minirag.adapters.reranker.os.makedirs") as mock_makedirs,
        ):
            reranker = CrossEncoderReranker("dummy_model")

        mock_makedirs.assert_not_called()
        mock_cross_encoder_class.assert_called_once_with(
            "dummy_model", cache_dir=None
        )
        self.assertIs(reranker._model, mock_cross_encoder_class.return_value)
