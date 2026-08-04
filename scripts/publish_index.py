import chromadb

from minirag.config import Settings
from minirag.adapters.embedder import OpenRouterEmbeddingEngine
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.serving.lakehouse import lakehouse_chunks
from minirag.domain.index import index_chunks

COLLECTION_NAME = "support_gemini_embedding_2_v1"


def main():
    settings = Settings()

    embed = OpenRouterEmbeddingEngine(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
    )
    client = chromadb.PersistentClient(settings.vector_store_path)

    vstore = ChromaVectorStore(
        vector_store_path=settings.vector_store_path,
        collection_name=COLLECTION_NAME,
        client=client,
    )

    ids = index_chunks(lakehouse_chunks(), embed, vstore)
    print(f"OK published {len(ids)} chunks to '{COLLECTION_NAME}' (no JVM)")


if __name__ == "__main__":
    main()
