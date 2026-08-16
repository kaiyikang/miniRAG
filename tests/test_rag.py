import os
import queue
import shutil
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import chromadb

from minirag.adapters.chunker import SlidingWindowChunker
from minirag.adapters.embedder import EmbeddingEngine
from minirag.adapters.hyde import QueryTransformer
from minirag.adapters.llm import InferenceEngine
from minirag.adapters.source_local import LocalMarkdownSource
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.domain.ports import Reranker
from minirag.domain.rag import RAGPipeline


class MockEmbedding(EmbeddingEngine):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class MockLLM(InferenceEngine):
    def __init__(self, answer: str = "RAG is retrieval augmented generation system."):
        self._answer = answer

    def generate(self, messages, **kwargs) -> dict[str, Any]:
        return {"content": self._answer}


class MockReranker(Reranker):
    def __init__(self):
        self.calls = []

    def rank(self, query_text, query_embedding, chunks):
        self.calls.append((query_text, query_embedding, chunks))
        return list(reversed(chunks))


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
        self.reranker = MockReranker()
        self.pipeline = RAGPipeline(
            embed=MockEmbedding(),
            vector_store=self.store,
            source=LocalMarkdownSource(self.document_dir, SlidingWindowChunker()),
            llm=MockLLM(),
            reranker=self.reranker,
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
        self.pipeline.index()

        answer = self.pipeline.query("What is RAG?")

        self.assertEqual(
            answer.content, "RAG is retrieval augmented generation system."
        )
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0]["file_name"], "test.txt")
        self.assertEqual(len(self.reranker.calls), 1)
        self.assertEqual(self.reranker.calls[0][0], "What is RAG?")
        self.assertIsNone(self.reranker.calls[0][1])

    def test_exposes_configured_engines(self):
        self.assertIsInstance(self.pipeline.get_embed(), MockEmbedding)
        self.assertIsInstance(self.pipeline.get_llm(), MockLLM)

    def test_index_no_source_raises(self):
        pipeline = RAGPipeline(
            embed=MockEmbedding(), vector_store=self.store, llm=MockLLM()
        )
        with self.assertRaises(ValueError):
            pipeline.index()

    @patch("minirag.adapters.source_local.LocalMarkdownSource.load")
    def test_index_no_docs_does_not_crash(self, mock_load):
        mock_load.return_value = []
        self.pipeline.index()

    def test_query_no_retrieved_chunks(self):
        answer = self.pipeline.query("something unrelated")
        self.assertEqual(
            answer.content, "RAG is retrieval augmented generation system."
        )
        self.assertEqual(answer.sources, [])

    def test_reranks_all_candidates_before_applying_rerank_limit(self):
        for i in range(3):
            self._write_doc(f"doc{i}.txt", f"content {i}")
        self.pipeline.index()

        answer = self.pipeline.query("query", retrieve_k=3, rerank_k=2)

        self.assertEqual(len(self.reranker.calls), 1)
        self.assertEqual(len(self.reranker.calls[0][2]), 3)
        self.assertEqual(len(answer.retrieved_chunks), 2)

    def test_query_inference_error_handling(self):
        class BadLLM(InferenceEngine):
            def generate(self, messages, **kwargs):
                from minirag.adapters.llm import InferenceError

                raise InferenceError("boom")

        events: queue.Queue = queue.Queue()
        pipeline = RAGPipeline(
            embed=MockEmbedding(),
            vector_store=self.store,
            llm=BadLLM(),
            reranker=MockReranker(),
            event_queue=events,
        )
        answer = pipeline.query("q")
        self.assertEqual(answer.content, "Error: failed to generate a response.")

        error_event = next(e for e in list(events.queue) if e.step == "error")
        self.assertEqual(error_event.data["stage"], "generate")
        self.assertEqual(error_event.data["reason"], "generation_failed")
        self.assertEqual(error_event.data["error_type"], "InferenceError")
        self.assertGreaterEqual(error_event.data["latency_ms"], 0)

    def test_query_transform_success_emits_no_fallback(self):
        class UpperCaseTransformer(QueryTransformer):
            def transform(self, question: str) -> str:
                return question.upper()

        events: queue.Queue = queue.Queue()
        pipeline = RAGPipeline(
            embed=MockEmbedding(),
            vector_store=self.store,
            llm=MockLLM(),
            reranker=MockReranker(),
            query_transformer=UpperCaseTransformer(),
            event_queue=events,
        )

        pipeline.query("what is rag?")

        transform_events = [e for e in list(events.queue) if e.step == "transform"]
        self.assertEqual(len(transform_events), 1)
        self.assertEqual(transform_events[0].data["question"], "WHAT IS RAG?")
        self.assertEqual(transform_events[0].data["original_question"], "what is rag?")
        self.assertFalse(transform_events[0].data["fallback"])
        self.assertGreaterEqual(transform_events[0].data["latency_ms"], 0)

    def test_query_transform_failure_falls_back_to_question(self):
        class BrokenTransformer(QueryTransformer):
            def transform(self, question: str) -> str:
                return ""

        events: queue.Queue = queue.Queue()
        pipeline = RAGPipeline(
            embed=MockEmbedding(),
            vector_store=self.store,
            llm=MockLLM(),
            reranker=MockReranker(),
            query_transformer=BrokenTransformer(),
            event_queue=events,
        )

        pipeline.query("what is rag?")

        transform_events = [e for e in list(events.queue) if e.step == "transform"]
        self.assertEqual(len(transform_events), 1)
        self.assertEqual(transform_events[0].data["question"], "what is rag?")
        self.assertEqual(transform_events[0].data["original_question"], "what is rag?")
        self.assertTrue(transform_events[0].data["fallback"])
        self.assertGreaterEqual(transform_events[0].data["latency_ms"], 0)

    def test_history_trimming(self):
        for i in range(10):
            self._write_doc(f"doc{i}.txt", f"content {i}")
        self.pipeline.index()

        for i in range(5):
            self.pipeline.query(f"q{i}")

        # MAX_HISTORY_MESSAGES = 6 (3 user + 3 assistant pairs would be 6 messages)
        # After 5 queries, history should have at most 6 messages
        self.assertLessEqual(
            len(self.pipeline._history), RAGPipeline.MAX_HISTORY_MESSAGES
        )

    def test_query_events_record_retrieval_and_rerank_candidates(self):
        for i in range(3):
            self._write_doc(f"doc{i}.txt", f"content {i}")
        self.pipeline.index()

        events: queue.Queue = queue.Queue()
        reranker = MockReranker()
        pipeline = RAGPipeline(
            embed=MockEmbedding(),
            vector_store=self.store,
            llm=MockLLM(),
            reranker=reranker,
            event_queue=events,
        )

        pipeline.query("question", retrieve_k=3, rerank_k=2)

        recorded_events = list(events.queue)
        retrieve_event = next(e for e in recorded_events if e.step == "retrieve")
        rerank_event = next(e for e in recorded_events if e.step == "rerank")
        generate_event = next(e for e in recorded_events if e.step == "generate")

        retrieved_contexts = retrieve_event.data["contexts"]
        reranked_contexts = rerank_event.data["contexts"]

        self.assertEqual(retrieve_event.data["query"], "question")
        self.assertEqual(retrieve_event.data["chunk_count"], 3)
        self.assertEqual(len(retrieved_contexts), 3)
        self.assertEqual([c["rank"] for c in retrieved_contexts], [1, 2, 3])
        self.assertTrue(all(isinstance(c["score"], float) for c in retrieved_contexts))

        self.assertEqual(rerank_event.data["query"], "question")
        self.assertEqual(rerank_event.data["chunk_count"], 2)
        self.assertEqual(len(reranked_contexts), 2)
        self.assertEqual(
            [c["chunk_id"] for c in reranked_contexts],
            [c["chunk_id"] for c in reversed(retrieved_contexts)][:2],
        )

        self.assertGreaterEqual(retrieve_event.data["latency_ms"], 0)
        self.assertGreaterEqual(rerank_event.data["latency_ms"], 0)
        self.assertGreaterEqual(generate_event.data["latency_ms"], 0)
