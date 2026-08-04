import chromadb

from minirag.adapters.embedder import OpenRouterEmbeddingEngine
from minirag.adapters.source_lakehouse import LakehouseSource
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.config import Settings
from minirag.domain.index import index_chunks


def main():
    settings = Settings()

    embed = OpenRouterEmbeddingEngine(
        model=settings.openrouter_embed_model,
        api_key=settings.openrouter_api_key,
    )
    client = chromadb.PersistentClient(settings.vector_store_path)

    vstore = ChromaVectorStore(
        vector_store_path=settings.vector_store_path,
        collection_name=settings.support_collection_name,
        client=client,
    )

    ids = index_chunks(LakehouseSource().load(), embed, vstore)
    print(f"OK published {len(ids)} chunks to '{settings.support_collection_name}' (no JVM)")


if __name__ == "__main__":
    main()
