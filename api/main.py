from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import queue
import json
import threading
import uvicorn

from minirag.config import get_settings
from minirag.rag import RAGPipeline
from minirag.embedding import OpenRouterEmbeddingEngine
from minirag.vector_store import ChromaVectorStore
from minirag.llm_engine import OpenRouterEngine
from minirag.document import SlidingWindowChunker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()

_embed = OpenRouterEmbeddingEngine(
    model=settings.openrouter_embed_model,
    api_key=settings.openrouter_api_key,
)
_vstore = ChromaVectorStore(
    vector_store_path=settings.vector_store_path,
    collection_name=settings.collection_name,
)
_chunker = SlidingWindowChunker(chunk_size=512)
_llm = OpenRouterEngine(
    model=settings.openrouter_model,
    api_key=settings.openrouter_api_key,
)


@app.get("/query")
async def query_stream(question: str):
    q = queue.Queue()

    pipeline = RAGPipeline(
        embed=_embed,
        vector_store=_vstore,
        chunker=_chunker,
        llm=_llm,
        event_queue=q,
    )

    def run():
        try:
            pipeline.query(question)
        finally:
            pipeline.clear_history()
            # 如果有其他需要清的状态，在这里加

    threading.Thread(target=run, daemon=True).start()

    def event_stream():
        while True:
            event = q.get()
            payload = {
                "event_id": event.event_id,
                "step": event.step,
                **event.data,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if event.step in ("complete", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
