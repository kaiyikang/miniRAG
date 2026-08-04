from minirag.config import get_settings
from minirag.domain.rag import RAGPipeline
from minirag.adapters.embedder import OpenRouterEmbeddingEngine
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.adapters.llm import OpenRouterEngine
from minirag.adapters.chunker import SlidingWindowChunker
from minirag.adapters.load import LocalMarkdownSource

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
    source=LocalMarkdownSource(SlidingWindowChunker(chunk_size=200)),
    llm=OpenRouterEngine(
        model=settings.openrouter_model, api_key=settings.openrouter_api_key
    ),
)

pipeline.index_documents(settings.documents_dir)
