import queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from api import deps
from minirag.adapters.hyde import HyDETransformer


def test_create_pipeline_uses_request_queue_and_current_dependencies():
    settings = SimpleNamespace(
        openrouter_model="test-llm",
        openrouter_embed_model="test-embed",
        openrouter_api_key="test-key",
        vector_store_path="test-store",
        collection_name="test-collection",
    )
    embed = Mock()
    vector_store = Mock()
    llm = Mock()
    events = queue.Queue()
    other_events = queue.Queue()

    deps.get_pipeline_dependencies.cache_clear()
    try:
        with (
            patch("api.deps.get_settings", return_value=settings),
            patch("api.deps.OpenRouterEmbeddingEngine", return_value=embed),
            patch("api.deps.ChromaVectorStore", return_value=vector_store),
            patch("api.deps.OpenRouterEngine", return_value=llm),
        ):
            pipeline = deps.create_pipeline(events)
            other_pipeline = deps.create_pipeline(other_events)

        assert pipeline._events is events
        assert other_pipeline._events is other_events
        assert other_pipeline is not pipeline
        assert pipeline.get_embed() is embed
        assert pipeline.get_llm() is llm
        assert pipeline._vstore is vector_store
        assert isinstance(pipeline._query_transformer, HyDETransformer)
        assert pipeline._query_transformer._llm is llm
    finally:
        deps.get_pipeline_dependencies.cache_clear()
