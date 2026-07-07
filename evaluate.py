from minirag.config import get_settings
from minirag.rag import RAGPipeline
from minirag.embedding import OpenRouterEmbeddingEngine
from minirag.vector_store import ChromaVectorStore
from minirag.llm_engine import OpenRouterEngine
from minirag.document import SlidingWindowChunker
from minirag.evaluator import Evaluator
from minirag.query_transform import HyDETransformer
import queue

settings = get_settings()
q = queue.Queue()

print(settings)


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
    chunker=SlidingWindowChunker(chunk_size=512),
    llm=llm,
    query_transformer=HyDETransformer(llm),
    event_queue=q,
)


evaluator = Evaluator(pipeline=pipeline, dataset_dir="eval")
evaluator.evaluate()
