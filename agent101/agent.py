from utils import call_llm_json

state = {
    "query": "explain the RAG in code",
    "task_type": None,
    "docs": [],
    "answer": None,
    "verified": False,
}


class Agent:
    def __init__(
        self, name, role, input_keys, output_keys, allowed_update_keys, run_func
    ):
        self.name = name
        self.role = role
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.allowed_updated_keys = allowed_update_keys
        self.run_func = run_func

    def run(self, state):
        local_inputs = {key: state.get(key) for key in self.input_keys}
        return self.run_func(local_inputs)


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


def retriever_agent_func(state):
    query = state["query"]
    # docs = retrieve(query)
    docs = ["get reference from db"]

    return {"docs": docs}


classifier_agent = Agent(
    name="classifier",
    role="Classify the user query",
    input_keys={"query"},
    output_keys={"task_type", "reason_summary"},
    allowed_update_keys={"task_type", "reason_summary"},
    run_func=llm_classifier_agent_func,
)

retriever_agent = Agent(
    name="retriever",
    role="Retrieve relevant documents",
    input_keys={"query"},
    output_keys={"docs"},
    allowed_update_keys={"docs"},
    run_func=retriever_agent_func,
)


update = classifier_agent.run(state)
state = apply_update(
    state=state, update=update, allowed_keys=classifier_agent.allowed_updated_keys
)
print(state)

update = retriever_agent.run(state)
state = apply_update(
    state=state, update=update, allowed_keys=retriever_agent.allowed_updated_keys
)
print(state)
