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


def apply_update(state, update, allowed_keys):
    for key, value in update.items():
        if key in allowed_keys:
            state[key] = value
    return state


update = classify_agent(state)
# verified must not be updated!
# classify_agent can only update the task_type
# state.update(update)
state = apply_update(state=state, update=update, allowed_keys={"task_type"})

print(state)
