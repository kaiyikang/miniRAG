state = {
    "query": "what is RAG?",
    "task_type": None,
    "docs": [],
    "answer": None,
    "verified": False,
}


def classify_agent(state):
    query = state["query"]

    if "code" in query or "implementation" in query:
        task_type = "implementation"
    elif "what is" in query or "explanation" in query:
        task_type = "conceptual"
    else:
        task_type = "general"

    return {"task_type": task_type}


update = classify_agent(state)
state.update(update)
print(state)
