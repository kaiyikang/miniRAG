from dataclasses import dataclass
from functools import lru_cache
from queue import Queue

from minirag.config import get_settings
from minirag.adapters.embedder import EmbeddingEngine, OpenRouterEmbeddingEngine
from minirag.adapters.llm import InferenceEngine, OpenRouterEngine
from minirag.adapters.hyde import HyDETransformer, QueryTransformer
from minirag.domain.rag import RAGPipeline
from minirag.domain.models import RAGEvent
from minirag.adapters.vector_store import ChromaVectorStore, VectorStore


@dataclass(frozen=True)
class PipelineDependencies:
    embed: EmbeddingEngine
    vector_store: VectorStore
    llm: InferenceEngine
    query_transformer: QueryTransformer


@lru_cache(maxsize=1)
def get_pipeline_dependencies() -> PipelineDependencies:
    """Build and cache the reusable resources shared by API requests."""
    settings = get_settings()
    llm = OpenRouterEngine(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
    )

    return PipelineDependencies(
        embed=OpenRouterEmbeddingEngine(
            model=settings.openrouter_embed_model,
            api_key=settings.openrouter_api_key,
        ),
        vector_store=ChromaVectorStore(
            vector_store_path=settings.vector_store_path,
            collection_name=settings.collection_name,
        ),
        llm=llm,
        query_transformer=HyDETransformer(llm=llm),
    )


def create_pipeline(event_queue: Queue[RAGEvent]) -> RAGPipeline:
    """Create request-scoped pipeline state backed by the request event queue."""
    dependencies = get_pipeline_dependencies()
    return RAGPipeline(
        embed=dependencies.embed,
        vector_store=dependencies.vector_store,
        llm=dependencies.llm,
        query_transformer=dependencies.query_transformer,
        event_queue=event_queue,
    )
