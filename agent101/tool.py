from typing import List, Dict, Any

from minirag.config import get_settings
from minirag.embedding import OpenRouterEmbeddingEngine, EmbeddingError
from minirag.vector_store import ChromaVectorStore
from minirag.types import SearchedChunk
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

CORPUS = [
    {
        "id": "doc_1",
        "title": "RAG Basics",
        "text": "RAG combines retrieval with generation. It first retrieves documents, then generates an answer.",
    },
    {
        "id": "doc_2",
        "title": "Agent Loop",
        "text": "A multi-agent system uses a loop, shared state, routing, and exit conditions.",
    },
    {
        "id": "doc_3",
        "title": "Verifier",
        "text": "A verifier checks whether an answer is supported by the retrieved documents.",
    },
    {
        "id": "doc_4",
        "title": "Vector Search",
        "text": "Vector search retrieves semantically similar documents using embeddings.",
    },
]


class Tool:
    def __init__(self, name, description, run_func):
        self.name = name
        self.description = description
        self.run_func = run_func

    def run(self, **kwargs):
        return self.run_func(**kwargs)


_settings = get_settings()
_embed = OpenRouterEmbeddingEngine(
    model=_settings.openrouter_embed_model, api_key=_settings.openrouter_api_key
)
_vstore = ChromaVectorStore(
    vector_store_path=_settings.vector_store_path,
    collection_name=_settings.collection_name,
)


def simple_search_tool(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    query_words = set(query.lower().split())

    scored_docs = []

    for doc in CORPUS:
        text = (doc["title"] + " " + doc["text"]).lower()
        doc_words = set(text.split())

        score = len(query_words & doc_words)

        scored_docs.append({**doc, "score": score})

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]


@retry(
    retry=retry_if_exception_type(EmbeddingError),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8),
    reraise=True,
)
def _embed_query(query: str) -> list[float]:
    return _embed.embed([query])[0]


def vector_search_tool(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Real tool: embeds the query and searches the Chroma store built by
    scripts/index_docs_online.py (same embedding model, so the vector space matches).
    """

    chunks = vector_search(query, top_k=top_k)

    return [
        {
            "id": chunk.chunk_id,
            "title": chunk.metadata.get("file_name", ""),
            "text": chunk.document,
            "score": chunk.score,
        }
        for chunk in chunks
    ]


def vector_search(query: str, top_k: int = 3) -> List[SearchedChunk]:
    try:
        query_embedding = _embed_query(query)
    except EmbeddingError:
        return []

    chunks = _vstore.search(query_embedding, top_k=top_k)

    return chunks


def get_docs_by_ids_tool(ids: List[str]) -> List[SearchedChunk]:
    """Real tool: fetch documents directly by their chunk ids from the Chroma store."""

    chunks = _vstore.get_by_ids(ids)

    return chunks


if __name__ == "__main__":
    # Test 1
    docs = simple_search_tool("multi agent loop verifier", top_k=2)

    for doc in docs:
        print(doc["id"], doc["title"], doc["score"])

    # Test 2
    TOOLS = {
        "simple_search": Tool(
            name="simple_search",
            description="Search relevant documents from the local corpus.",
            run_func=simple_search_tool,
        )
    }

    docs = TOOLS["simple_search"].run(query="multi agent loop", top_k=2)
    print(docs)
