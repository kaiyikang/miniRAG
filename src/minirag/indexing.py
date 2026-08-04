from typing import Iterable
from minirag.types import Chunk
from minirag.embedding import EmbeddingEngine
from minirag.vector_store import VectorStore


def index_chunks(
    chunks: Iterable[Chunk], embed: EmbeddingEngine, vstore: VectorStore
) -> list[str]:
    chunks = list(chunks)
    if not chunks:
        return []
    embeddings = embed.embed([c.document for c in chunks])
    return vstore.upsert_chunks(
        [c._replace(embedding=e) for c, e in zip(chunks, embeddings)]
    )
