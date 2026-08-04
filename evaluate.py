from minirag.config import get_settings
from minirag.domain.rag import RAGPipeline
from minirag.adapters.embedder import OpenRouterEmbeddingEngine
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.adapters.llm import OpenRouterEngine
from minirag.adapters.evaluator import Evaluator, QA_DATASET_FILENAME
from minirag.adapters.hyde import HyDETransformer
import queue

settings = get_settings()
q = queue.Queue()

print(settings)

# Knobs for this run, defined once so `params` below can't drift from what's used.
CHUNK_SIZE = 200
TOP_K = 5
SUFFIX = "hyde-glm52"  # human label for the run folder: what variant this tests

llm = OpenRouterEngine(
    model=settings.openrouter_model, api_key=settings.openrouter_api_key
)

pipeline = RAGPipeline(
    embed=OpenRouterEmbeddingEngine(
        model=settings.openrouter_embed_model, api_key=settings.openrouter_api_key
    ),
    vector_store=ChromaVectorStore(
        vector_store_path=settings.vector_store_path,
        collection_name=settings.collection_name,
    ),
    llm=llm,
    query_transformer=HyDETransformer(llm),
    event_queue=q,
)

params = {
    "model": settings.openrouter_model,
    "embed_model": settings.openrouter_embed_model,
    "chunk_size": CHUNK_SIZE,
    "top_k": TOP_K,
    "query_transformer": "HyDE",
    "dataset": QA_DATASET_FILENAME,
}

evaluator = Evaluator(
    pipeline=pipeline,
    dataset_dir="eval",
    suffix=SUFFIX,
    recall_top_k=TOP_K,
    params=params,
)
evaluator.evaluate()
