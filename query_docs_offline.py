from minirag.config import get_settings
from minirag.domain.rag import RAGPipeline
from minirag.adapters.embedder import SentenceTransformerEngine
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.adapters.llm import OpenRouterEngine

settings = get_settings()

print(settings)

pipeline = RAGPipeline(
    embed=SentenceTransformerEngine(
        model=settings.embedding_model, cache_dir=settings.embedding_model_cache_dir
    ),
    vector_store=ChromaVectorStore(
        vector_store_path=settings.vector_store_path,
        collection_name=settings.collection_name,
    ),
    llm=OpenRouterEngine(
        model=settings.openrouter_model, api_key=settings.openrouter_api_key
    ),
)

answer = pipeline.query("What is RAG?")
print(answer.content)
print(answer.sources)
