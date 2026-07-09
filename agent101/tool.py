from typing import List, Dict, Any

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
