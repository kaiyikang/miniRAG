from typing import List, Dict, Any
from minirag.domain.ports import EmbeddingEngine, EmbeddingError, VectorStore
from minirag.domain.models import SearchedChunk
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

# Tiny in-memory corpus used only by simple_search (offline keyword baseline).
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


@retry(
    retry=retry_if_exception_type(EmbeddingError),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8),
    reraise=True,
)
def _embed_query(embed: EmbeddingEngine, query: str) -> list[float]:
    return embed.embed([query])[0]


class SearchTools:

    def __init__(self, embed: EmbeddingEngine, vstore: VectorStore):
        self._embed = embed
        self._vstore = vstore

    def simple_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Keyword-overlap baseline over CORPUS. Needs no embed/vstore."""
        query_words = set(query.lower().split())

        scored_docs = []
        for doc in CORPUS:
            text = (doc["title"] + " " + doc["text"]).lower()
            score = len(query_words & set(text.split()))
            scored_docs.append({**doc, "score": score})

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    def vector_search(self, query: str, top_k: int = 3) -> List[SearchedChunk]:
        try:
            query_embedding = _embed_query(self._embed, query)
        except EmbeddingError:
            return []
        return self._vstore.search(query_embedding, top_k=top_k)

    def keyword_search(self, query: str, top_k: int = 3) -> List[SearchedChunk]:
        """Keyword-overlap ranking over the *real* corpus (via get_all_chunks).
        Complements vector_search: catches exact terms / proper nouns that
        embeddings can miss."""
        query_words = set(query.lower().split())

        scored: List[SearchedChunk] = []
        for chunk in self._vstore.get_all_chunks():
            # TODO score this chunk by keyword overlap with query_words,
            # then append chunk._replace(score=<score>) to `scored`.
            # SearchedChunk is a NamedTuple, so _replace returns a copy with the
            # new score. (See simple_search above for an overlap-scoring pattern.)
            pass

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def vector_search_dicts(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Same as vector_search, flattened to plain dicts (workflow agents
        pass docs around as dicts inside RagState)."""
        return [
            {
                "id": chunk.chunk_id,
                "title": chunk.metadata.get("file_name", ""),
                "text": chunk.document,
                "score": chunk.score,
            }
            for chunk in self.vector_search(query, top_k=top_k)
        ]

    def get_by_ids(self, ids: List[str]) -> List[SearchedChunk]:
        return self._vstore.get_by_ids(ids)


if __name__ == "__main__":
    # Offline self-check: simple_search needs neither embed nor vstore.
    tools = SearchTools(embed=None, vstore=None)
    docs = tools.simple_search("multi agent loop verifier", top_k=2)
    assert docs[0]["score"] >= docs[-1]["score"], "results must be sorted by score"
    for doc in docs:
        print(doc["id"], doc["title"], doc["score"])
