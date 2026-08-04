import re

TICKET_RE = re.compile(r"\bT-\d+\b")
DOC_INTENT = (
    "mean",
    "how",
    "what",
    "recommend",
    "fix",
    "cause",
    "guide",
    "should",
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
