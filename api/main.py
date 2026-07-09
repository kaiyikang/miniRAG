from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import queue
import json
import threading
import uvicorn

from .deps import create_pipeline, RAGEvent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/query")
async def query_stream(question: str):
    q = queue.Queue()

    pipeline = create_pipeline(q)

    def run():
        try:
            pipeline.query(question)
        except Exception as e:
            q.put(RAGEvent(event_id="unknown", step="error", data={"reason": str(e)}))
        finally:
            pipeline.clear_history()
            # 如果有其他需要清的状态，在这里加

    threading.Thread(target=run, daemon=True).start()

    def event_stream():
        while True:
            try:
                event = q.get(timeout=60)
            except queue.Empty:
                yield f"data: {json.dumps({'step': 'error', 'reason': 'timeout'})}\n\n"
                break
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
