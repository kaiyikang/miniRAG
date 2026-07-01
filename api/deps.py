from minirag.config import get_settings
from minirag.rag import RAGPipeline
from minirag.embedding import OpenRouterEmbeddingEngine
from minirag.vector_store import ChromaVectorStore
from minirag.llm_engine import OpenRouterEngine
from minirag.document import SlidingWindowChunker

_settings = get_settings()

_embed = OpenRouterEmbeddingEngine(
    model=_settings.openrouter_embed_model,
    api_key=_settings.openrouter_api_key,
)
_vstore = ChromaVectorStore(
    vector_store_path=_settings.vector_store_path,
    collection_name=_settings.collection_name,
)
_chunker = SlidingWindowChunker(chunk_size=512)
_llm = OpenRouterEngine(
    model=_settings.openrouter_model,
    api_key=_settings.openrouter_api_key,
)


def create_pipeline(event_queue) -> RAGPipeline:
    return RAGPipeline(
        embed=_embed,
        vector_store=_vstore,
        chunker=_chunker,
        llm=_llm,
        event_queue=event_queue,
    )
