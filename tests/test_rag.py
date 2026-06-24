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

        self.assertEqual(answer.answer, "RAG is retrieval augmented generation system.")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0]["file_name"], "test.txt")
