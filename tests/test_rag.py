import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil
import chromadb
from typing import Any
from minirag.vector_store import ChromaVectorStore
from minirag.rag import RAGPipeline
from minirag.embedding import EmbeddingEngine
from minirag.llm_engine import InferenceEngine
import os
from minirag.document import SlidingWindowChunker


class MockEmbedding(EmbeddingEngine):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class MockLLM(InferenceEngine):
    def __init__(self, answer: str = "RAG is retrieval augmented generation system."):
        self._answer = answer

    def generate(
        self, messages, *, reasoning=True, last_response=None
    ) -> dict[str, Any]:
        return {"content": self._answer}


class TestRagPipeline(unittest.TestCase):

    def setUp(self):
        # vector DB
        self.client = chromadb.EphemeralClient()
        self.store = ChromaVectorStore(
            vector_store_path="",
            collection_name="test_collection",
            client=self.client,
        )

        # Resource
        self.document_dir = tempfile.mkdtemp()

        # RAG
        self.pipeline = RAGPipeline(
            embed=MockEmbedding(),
            vector_store=self.store,
            chunker=SlidingWindowChunker(),
            llm=MockLLM(),
        )

    def tearDown(self):
        if hasattr(self, "client"):
            self.client.delete_collection("test_collection")

        if hasattr(self, "document_dirs"):
            shutil.rmtree(self.document_dir)

        self.pipeline.clear_history()

    def _write_doc(self, filename: str, content: str) -> None:
        path = os.path.join(self.document_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_query_returns_answer_with_sources(self):
        self._write_doc("test.txt", "RAG stands for Retrieval Augmented Generation.")
        self.pipeline.index_documents(self.document_dir)

        answer = self.pipeline.query("What is RAG?")

        self.assertEqual(answer.content, "RAG is retrieval augmented generation system.")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0]["file_name"], "test.txt")

    def test_index_documents_empty_dir_raises(self):
        with self.assertRaises(ValueError):
            self.pipeline.index_documents("")

    @patch("minirag.rag.load_documents")
    def test_index_documents_no_docs_does_not_crash(self, mock_load):
        mock_load.return_value = []
        self.pipeline.index_documents(self.document_dir)

    def test_query_no_retrieved_chunks(self):
        answer = self.pipeline.query("something unrelated")
        self.assertEqual(answer.content, "RAG is retrieval augmented generation system.")
        self.assertEqual(answer.sources, [])

    def test_query_inference_error_handling(self):
        class BadLLM(InferenceEngine):
            def generate(self, messages, *, reasoning=True, last_response=None):
                from minirag.llm_engine import InferenceError
                raise InferenceError("boom")

        pipeline = RAGPipeline(
            embed=MockEmbedding(),
            vector_store=self.store,
            chunker=SlidingWindowChunker(),
            llm=BadLLM(),
        )
        answer = pipeline.query("q")
        self.assertEqual(answer.content, "Error: failed to generate a response.")

    def test_history_trimming(self):
        for i in range(10):
            self._write_doc(f"doc{i}.txt", f"content {i}")
        self.pipeline.index_documents(self.document_dir)

        for i in range(5):
            self.pipeline.query(f"q{i}")

        # MAX_HISTORY_MESSAGES = 6 (3 user + 3 assistant pairs would be 6 messages)
        # After 5 queries, history should have at most 6 messages
        self.assertLessEqual(len(self.pipeline._history), RAGPipeline.MAX_HISTORY_MESSAGES)
