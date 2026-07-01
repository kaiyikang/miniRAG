from minirag.config import get_settings
from minirag.rag import RAGPipeline
from minirag.embedding import OpenRouterEmbeddingEngine
from minirag.vector_store import ChromaVectorStore
from minirag.llm_engine import OpenRouterEngine
from minirag.document import SlidingWindowChunker

settings = get_settings()

print(settings)

pipeline = RAGPipeline(
    embed=OpenRouterEmbeddingEngine(
        model=settings.openrouter_embed_model, api_key=settings.openrouter_api_key
    ),
    vector_store=ChromaVectorStore(
        vector_store_path=settings.vector_store_path,
        collection_name=settings.collection_name,
    ),
    chunker=SlidingWindowChunker(chunk_size=512),
    llm=OpenRouterEngine(
        model=settings.openrouter_model, api_key=settings.openrouter_api_key
    ),
)

pipeline.index_documents(settings.documents_dir)
