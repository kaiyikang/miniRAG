import unittest
from unittest.mock import MagicMock, patch

from llama_index.core import Document

from minirag.document import (
    SlidingWindowChunker,
    SpacyChunker,
    chunk_documents,
    load_documents,
)
from minirag.types import Chunk


class TestSlidingWindowChunker(unittest.TestCase):
    def test_empty_text_returns_empty_list(self):
        chunker = SlidingWindowChunker(chunk_size=10, overlap=2)
        result = chunker.chunk("")
        self.assertEqual(result, [])

    def test_short_text_returns_single_chunk(self):
        chunker = SlidingWindowChunker(chunk_size=10, overlap=2)
        result = chunker.chunk("short")
        self.assertEqual(result, ["short"])

    def test_exact_length_text_returns_single_chunk(self):
        chunker = SlidingWindowChunker(chunk_size=10, overlap=2)
        text = "a" * 10
        result = chunker.chunk(text)
        self.assertEqual(result, [text])

    def test_sliding_window_produces_overlapping_chunks(self):
        chunker = SlidingWindowChunker(chunk_size=10, overlap=3)
        text = "abcdefghijklmnopqrstuvwxyz"
        result = chunker.chunk(text)

        # chunk_size=10, overlap=3, step=7
        # 0-10, 7-17, 14-24, 21-31(26)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "abcdefghij")
        self.assertEqual(result[1], "hijklmnopq")
        self.assertEqual(result[2], "opqrstuvwx")
        self.assertEqual(result[3], "vwxyz")

    def test_overlap_zero_produces_adjacent_chunks(self):
        chunker = SlidingWindowChunker(chunk_size=5, overlap=0)
        text = "abcdefghij"
        result = chunker.chunk(text)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "abcde")
        self.assertEqual(result[1], "fghij")

    def test_invalid_params_raises(self):
        with self.assertRaises(ValueError):
            SlidingWindowChunker(chunk_size=5, overlap=10)


class TestSpacyChunker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunker = SpacyChunker(model_name="en_core_web_md")

    def test_empty_text_returns_empty_list(self):
        result = self.chunker.chunk("")
        self.assertEqual(result, [])

    def test_single_sentence(self):
        text = "This is a simple sentence."
        result = self.chunker.chunk(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "This is a simple sentence.")

    def test_multiple_sentences(self):
        text = "First sentence. Second sentence. Third one."
        result = self.chunker.chunk(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "First sentence.")
        self.assertEqual(result[1], "Second sentence.")
        self.assertEqual(result[2], "Third one.")


class TestChunkerComparison(unittest.TestCase):
    """Compare sliding window vs sentence-based chunking on the same text."""

    SAMPLE_TEXT = (
        "Dense retrieval is a powerful technique. It maps text into vectors. "
        "These vectors live in a high-dimensional space. Similar texts are close together. "
        "This makes search fast and accurate. Chunk size matters for quality."
    )

    @classmethod
    def setUpClass(cls):
        cls.spacy_chunker = SpacyChunker(model_name="en_core_web_md")
        cls.sliding_chunker = SlidingWindowChunker(chunk_size=50, overlap=10)

    def test_chunk_count_comparison(self):
        spacy_chunks = self.spacy_chunker.chunk(self.SAMPLE_TEXT)
        sliding_chunks = self.sliding_chunker.chunk(self.SAMPLE_TEXT)

        # Spacy typically produces fewer chunks because it respects sentence boundaries.
        self.assertLessEqual(len(spacy_chunks), len(sliding_chunks))

    def test_spacy_respects_sentence_boundaries(self):
        spacy_chunks = self.spacy_chunker.chunk(self.SAMPLE_TEXT)
        for chunk in spacy_chunks:
            # Each chunk should end with sentence punctuation.
            self.assertTrue(chunk.endswith((".", "!", "?")))

    def test_sliding_window_has_uniform_length(self):
        sliding_chunks = self.sliding_chunker.chunk(self.SAMPLE_TEXT)
        lengths = [len(c) for c in sliding_chunks]
        # Most chunks should be exactly chunk_size except possibly the last one.
        for length in lengths[:-1]:
            self.assertEqual(length, 50)


class TestLoadDocuments(unittest.TestCase):
    @patch("minirag.document.SimpleDirectoryReader")
    @patch("minirag.document.Path.exists", return_value=True)
    def test_load_documents_from_path(self, mock_exists, mock_reader_cls):
        mock_doc = MagicMock(spec=Document)
        mock_reader = MagicMock()
        mock_reader.load_data.return_value = [mock_doc]
        mock_reader_cls.return_value = mock_reader

        result = load_documents("/fake/path")

        mock_reader_cls.assert_called_once_with(input_dir="/fake/path")
        self.assertEqual(result, [mock_doc])

    def test_load_documents_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_documents("/definitely/does/not/exist")


class TestChunkDocuments(unittest.TestCase):
    def test_chunk_documents_delegates_to_chunker(self):
        chunker = SlidingWindowChunker(chunk_size=10, overlap=2)
        doc = Document(text="abcdefghij", metadata={"file_name": "test.txt"})

        result = chunk_documents([doc], chunker)

        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(c, Chunk) for c in result))
        self.assertEqual(result[0].metadata["file_name"], "test.txt")
        self.assertEqual(result[0].metadata["chunk_idx"], 0)
        self.assertIsNone(result[0].embedding)


if __name__ == "__main__":
    unittest.main()
