import re

import chromadb

from minirag.adapters.embedder import OpenRouterEmbeddingEngine
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.agents.ticket_tool import TicketContextTool
from minirag.agents.tool import SearchTools
from minirag.config import Settings

QUESTIONS = [
    "What does error code E104 mean?",  # docs
    "What is the status of ticket T-102?",  # ticket
    "Ticket T-104 reported E301 - what happened and what does the guide recommend?",  # mixed
]

TICKET_RE = re.compile(r"\bT-\d+\b")
# Deliberately excludes generic words like "what"/"how"/"should" that appear in
# pure ticket questions ("What is the status of ticket T-102?") and would
# misroute them to "mixed". Keep only terms that signal genuine doc intent.
DOC_INTENT = (
    "mean",
    "recommend",
    "fix",
    "cause",
    "guide",
    "procedure",
)


class SupportAssistant:

    def __init__(self, ticket_tool, search_tools):
        self._tickets = ticket_tool
        self._search = search_tools

    def route(self, question: str) -> dict:
        ids = TICKET_RE.findall(question)
        wants_docs = any(w in question.lower() for w in DOC_INTENT)

        label = "mixed" if (ids and wants_docs) else ("ticket" if ids else "docs")

        result = {"route": label, "tickets": [], "docs": []}
        if ids:
            result["tickets"] = [self._tickets.get_context(t) for t in ids]
        if label in ("docs", "mixed"):
            result["docs"] = self._search.vector_search_dicts(question, top_k=3)

        return result


def build_assistant() -> SupportAssistant:
    settings = Settings()

    # Must match the embedder used by publish_index.py, or the query embeddings
    # won't align with the indexed collection.
    embed = OpenRouterEmbeddingEngine(
        model=settings.openrouter_embed_model,
        api_key=settings.openrouter_api_key,
    )

    vstore = ChromaVectorStore(
        settings.vector_store_path,
        settings.support_collection_name,
        client=chromadb.PersistentClient(settings.vector_store_path),
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
