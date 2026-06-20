import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil
import os
import chromadb
from minirag.vector_store import ChromaVectorStore


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

    def tearDown(self):
        if hasattr(self, "client"):
            self.client.delete_collection("test_collection")

        if hasattr(self, "document_dirs"):
            shutil.rmtree(self.document_dir)

    def test_rag_pipeline(self, mock_test):
        pass
