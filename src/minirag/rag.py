from minirag.embedding import SentenceTransformerEngine, EmbeddingEngine
from minirag.vector_store import VectorStore, ChromaVectorStore
from minirag.document import Chunker, load_documents, chunk_documents
from minirag.llm_engine import InferenceEngine
from minirag.types import Chunk, Answer


class RAGPipeline:

    def __init__(
        self,
        embed: EmbeddingEngine,
        vector_store: VectorStore,
        chunker: Chunker,
        llm: InferenceEngine,
        document_dir: str,
    ):
        if not document_dir:
            raise ValueError("Source Document dir can not be found!")
        self._document_dir = document_dir
        self._embed = embed
        self._vstore = vector_store
        self._chunker = chunker
        self._llm = llm
        self._history = ""

    def index_documents(self) -> None:
        docs = load_documents(self._document_dir)
        if not docs:
            return
        chunks = chunk_documents(docs)
        embeddings = self._embed.embed([chunk.document for chunk in chunks])
        self._vstore.add_chunks(
            [
                Chunk(
                    document=chunk.document,
                    metadata=chunk.metadata,
                    embedding=embedding,
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
        )

    def query(self, question: str) -> Answer:
        question_embedding = self._embed.embed(question)[0]
        # R
        retrieved_chunks = self._vstore.search(question_embedding)
        # A
        pass
