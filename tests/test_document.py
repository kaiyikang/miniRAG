import importlib.util
import unittest
from unittest.mock import MagicMock, patch

from llama_index.core import Document

HAS_SPACY = importlib.util.find_spec("spacy") is not None

from minirag.adapters.chunker import (
    SlidingWindowChunker,
    SpacyChunker,
    ParagraphChunker,
)
from minirag.adapters.source_local import (
    MarkdownWithoutFrontmatterReader,
    _chunk_documents as chunk_documents,
    _load_documents as load_documents,
)
from minirag.domain.models import Chunk


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
        chunker = SlidingWindowChunker(chunk_size=3, overlap=1)
        text = "a b c d e f g"  # 7 words
        result = chunker.chunk(text)

        # size=3 words, overlap=1, step=2: [0:3],[2:5],[4:7],[6:7]
        self.assertEqual(result, ["a b c", "c d e", "e f g", "g"])

    def test_overlap_zero_produces_adjacent_chunks(self):
        chunker = SlidingWindowChunker(chunk_size=2, overlap=0)
        text = "a b c d"
        result = chunker.chunk(text)

        self.assertEqual(result, ["a b", "c d"])

    def test_invalid_params_raises(self):
        with self.assertRaises(ValueError):
            SlidingWindowChunker(chunk_size=5, overlap=10)

    def test_words_are_never_split(self):
        # Word-based chunking groups whole words; it never cuts inside one.
        chunker = SlidingWindowChunker(chunk_size=2, overlap=0)
        result = chunker.chunk("hello world foo")

        self.assertEqual(result, ["hello world", "foo"])


@unittest.skipUnless(HAS_SPACY, "spacy not installed")
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


class TestParagraphChunker(unittest.TestCase):
    def test_empty_text_returns_empty_list(self):
        chunker = ParagraphChunker()
        result = chunker.chunk("")
        self.assertEqual(result, [])

    def test_splits_by_double_newline(self):
        chunker = ParagraphChunker()
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        result = chunker.chunk(text)
        self.assertEqual(result, ["Para 1.", "Para 2.", "Para 3."])

    def test_skips_empty_paragraphs(self):
        chunker = ParagraphChunker()
        text = "Para 1.\n\n\n\nPara 2."
        result = chunker.chunk(text)
        self.assertEqual(result, ["Para 1.", "Para 2."])


class TestMarkdownWithoutFrontmatterReader(unittest.TestCase):
    @patch("minirag.adapters.source_local.MarkdownReader")
    def test_strips_frontmatter(self, mock_reader_cls):
        from llama_index.core import Document

        mock_doc = Document(
            text="---\ntitle: X\n---\n\nbody", metadata={"file_path": "f.md"}
        )
        mock_reader = MagicMock()
        mock_reader.load_data.return_value = [mock_doc]
        mock_reader_cls.return_value = mock_reader

        reader = MarkdownWithoutFrontmatterReader()
        result = reader.load_data("f.md")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "body")


class TestSpacyChunkerImportError(unittest.TestCase):
    def test_import_error_raises(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "spacy":
                raise ImportError("No module named 'spacy'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            with self.assertRaises(ImportError):
                SpacyChunker()


@unittest.skipUnless(HAS_SPACY, "spacy not installed")
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

    def test_sliding_window_stays_within_chunk_size(self):
        sliding_chunks = self.sliding_chunker.chunk(self.SAMPLE_TEXT)
        lengths = [len(c) for c in sliding_chunks]
        # No longer exactly chunk_size: boundaries snap back to the nearest
        # space so words aren't split (see test_chunk_boundary_does_not_split_a_word).
        for length in lengths:
            self.assertLessEqual(length, 50)


class TestLoadDocuments(unittest.TestCase):
    @patch("minirag.adapters.source_local.SimpleDirectoryReader")
    @patch("minirag.adapters.source_local.Path.exists", return_value=True)
    def test_load_documents_from_path(self, mock_exists, mock_reader_cls):
        mock_doc = MagicMock(spec=Document)
        mock_reader = MagicMock()
        mock_reader.load_data.return_value = [mock_doc]
        mock_reader_cls.return_value = mock_reader

        result = load_documents("/fake/path")

        kwargs = mock_reader_cls.call_args.kwargs
        self.assertEqual(kwargs["input_dir"], "/fake/path")
        self.assertTrue(kwargs["recursive"])
        self.assertIn(".md", kwargs["file_extractor"])
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
