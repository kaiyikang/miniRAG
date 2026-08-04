import chromadb

from minirag.adapters.embedder import OpenRouterEmbeddingEngine
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.agents.tool import SearchTools
from minirag.serving.lakehouse import TicketContextTool
from minirag.assistant import SupportAssistant
from minirag.config import Settings

CHROMA_PATH, COLLECTION = "data/chroma", "support"

QUESTIONS = [
    "What does error code E104 mean?",  # docs
    "What is the status of ticket T-102?",  # ticket
    "Ticket T-104 reported E301 - what happened and what does the guide recommend?",  # mixed
]


def build_assistant() -> SupportAssistant:

    settings = Settings()

    embed = OpenRouterEmbeddingEngine(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
    )

    vstore = ChromaVectorStore(
        CHROMA_PATH, COLLECTION, client=chromadb.PersistentClient(CHROMA_PATH)
    )
    return SupportAssistant(TicketContextTool(), SearchTools(embed, vstore))


def main() -> None:
    assistant = build_assistant()
    for q in QUESTIONS:
        r = assistant.route(q)
        print("=" * 60)
        print("Q:", q, "| route:", r["route"])
        for t in r["tickets"]:
            print(
                "  ticket:",
                t and (t["ticket_id"], t["status"], t["critical_event_count"]),
            )
        for d in r["docs"]:
            print(f"  doc[{d['score']:.2f}]:", d["text"][:70].replace(chr(10), " "))


if __name__ == "__main__":
    main()
