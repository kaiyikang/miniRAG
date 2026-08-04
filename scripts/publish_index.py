import chromadb

from minirag.config import Settings  # ← 你的 settings 模块路径
from minirag.embedding import OpenRouterEmbeddingEngine
from minirag.vector_store import ChromaVectorStore
from minirag.serving.lakehouse import lakehouse_chunks
from minirag.indexing import index_chunks

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
