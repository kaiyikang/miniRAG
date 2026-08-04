from minirag.config import get_settings
from minirag.domain.rag import RAGPipeline
from minirag.adapters.embedder import OpenRouterEmbeddingEngine
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.adapters.llm import OpenRouterEngine
from minirag.adapters.hyde import HyDETransformer
import queue
import threading

settings = get_settings()
print(settings)

q = queue.Queue()
done = threading.Event()


def consumer():
    while True:
        event = q.get()
        print(f"[{event.step}] {event.data}", flush=True)
        if event.step in ("complete", "error"):
            done.set()
            break


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
    query_transformer=HyDETransformer(llm=llm),
    event_queue=q,
)


threading.Thread(target=consumer).start()

answer = pipeline.query("What is lead climbing?")
done.wait()
print(f"Answer: {answer.content}")
