from unittest.mock import MagicMock, patch
import unittest
from minirag.adapters.reranker import CrossEncoderReranker, VectorReranker
from minirag.domain.models import SearchedChunk


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

    @patch("sentence_transformers.CrossEncoder")
    def test_cross_encoder_reranker(self, mock_cross_encoder_class):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker("dummy_model", "/tmp/cache")
        chunks = [
            SearchedChunk("1", "doc a", {}, [0.0], 0.0),
            SearchedChunk("2", "doc b", {}, [0.0], 0.0),
            SearchedChunk("3", "doc c", {}, [0.0], 0.0),
        ]

        result = reranker.rank("query", None, chunks)
        assert [r.chunk_id for r in result] == ["2", "3", "1"]

    @patch("sentence_transformers.CrossEncoder")
    def test_raises_when_query_text_missing(self, mock_cross_encoder_class):
        reranker = CrossEncoderReranker("dummy_model", "/tmp/cache")
        chunks = [SearchedChunk("1", "doc a", {}, [0.0], 0.0)]

        with self.assertRaises(ValueError):
            reranker.rank(None, None, chunks)

    def test_init_missing_model_raises(self):
        with self.assertRaises(ValueError):
            CrossEncoderReranker("")

    @patch("sentence_transformers.CrossEncoder")
    def test_raises_when_query_embedding_provided(self, mock_cross_encoder_class):
        reranker = CrossEncoderReranker("dummy_model", "/tmp/cache")
        chunks = [SearchedChunk("1", "doc a", {}, [0.0], 0.0)]

        with self.assertRaises(ValueError):
            reranker.rank("query", [1.0], chunks)

    @patch("sentence_transformers.CrossEncoder")
    def test_empty_chunks_returns_empty_list(self, mock_cross_encoder_class):
        reranker = CrossEncoderReranker("dummy_model", "/tmp/cache")
        result = reranker.rank("query", None, [])
        self.assertEqual(result, [])
