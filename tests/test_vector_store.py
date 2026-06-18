import unittest

import chromadb

from minirag.vector_store import Chunk, ChromaVectorStore, SearchResult


class TestChromaVectorStore(unittest.TestCase):
    def setUp(self):
        self.client = chromadb.Client()
        self.store = ChromaVectorStore(
            vector_store_path="memory",
            collection_name="test_collection",
            client=self.client,
        )

    def tearDown(self):
        self.client.delete_collection("test_collection")

    def test_add_chunks_empty_list_returns_empty_ids(self):
        result = self.store.add_chunks([])
        self.assertEqual(result, [])

    def test_add_chunks_returns_ids(self):
        chunks = [
            Chunk(document="hello world", embedding=[1.0, 0.0], metadata={"source": "a"}),
            Chunk(document="foo bar", embedding=[0.0, 1.0], metadata={"source": "b"}),
        ]

        ids = self.store.add_chunks(chunks)

        self.assertEqual(len(ids), 2)
        self.assertIsInstance(ids[0], str)
        self.assertIsInstance(ids[1], str)

    def test_search_returns_results_ordered_by_relevance(self):
        chunks = [
            Chunk(document="hello world", embedding=[1.0, 0.0], metadata={"source": "a"}),
            Chunk(document="foo bar", embedding=[0.0, 1.0], metadata={"source": "b"}),
        ]
        self.store.add_chunks(chunks)

        results = self.store.search(query_embedding=[1.0, 0.0], top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].document, "hello world")
        self.assertEqual(results[0].metadata, {"source": "a"})
        self.assertGreaterEqual(results[0].score, 0.0)
        self.assertLessEqual(results[0].score, 1.0)
        # The most relevant result should have the highest score.
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_search_empty_query_returns_empty_list(self):
        result = self.store.search(query_embedding=[], top_k=5)
        self.assertEqual(result, [])

    def test_search_empty_collection_returns_empty_list(self):
        results = self.store.search(query_embedding=[1.0, 0.0], top_k=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
