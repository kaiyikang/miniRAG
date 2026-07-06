from utils import call_llm_json

state = {
    "query": "explain the RAG in code",
    "task_type": None,
    "docs": [],
    "answer": None,
    "verified": False,
}


# agent is not a prompt
# it is the role + input + output schema + allowed updates + behavior (do what)


class Agent:
    def __init__(self, name, allowed_update_keys, run_func):
        self.name = name
        self.allowed_updated_keys = allowed_update_keys
        self.run_func = run_func

    def run(self, state):
        return self.run_func(state)


def apply_update(state, update, allowed_keys):
    for key, value in update.items():
        if key in allowed_keys:
            state[key] = value
    return state


def classify_agent_func(state):
    query = state["query"]

    if "code" in query or "implementation" in query:
        task_type = "implementation"
    elif "what is" in query or "explanation" in query:
        task_type = "conceptual"
    else:
        task_type = "general"

    return {"task_type": task_type}


def llm_classifier_agent_func(state):
    prompt = f"""

your are question classifier.

User query: {state["query"]}

Only return JSON:
{{
  "task_type": "conceptual | implementation | debugging | retrieval_needed",
  "reason_summary": "the reason with one sentence"
}}
"""

    return call_llm_json(prompt)


classifier = Agent(
    name="classifier",
    allowed_update_keys={"task_type"},
    run_func=llm_classifier_agent_func,
)
update = classifier.run(state)
state = apply_update(
    state=state, update=update, allowed_keys=classifier.allowed_updated_keys
)

print(state)
