from utils import call_llm_json
from typing import TypedDict, List, Optional, Any


class RagState(TypedDict):
    # Input from user
    original_query: str
    current_query: str

    # Intermediate fields
    classification_reason: Optional[str]
    task_type: Optional[str]
    plan: Optional[str]
    docs: List[str]
    reranked_docs: List[dict]
    answer: Optional[str]
    verification_result: Optional[bool]
    verification_reason: Optional[str]

    # Process Control for Orchestrator
    next_agent: str
    step: int
    max_steps: int
    exit_reason: Optional[str]
    verified: bool

    # Debug
    trace: List[dict]
    query_history: List[str]


def create_initial_state(query="从代码角度解释 multi-agent RAG") -> RagState:
    return {
        "original_query": query,
        "current_query": "what is multi-agent RAG implementation loop",
        "classification_reason": None,
        "task_type": None,
        "plan": None,
        "docs": [],
        "reranked_docs": [],
        "answer": None,
        "verification_result": None,
        "verification_reason": None,
        "next_agent": "classifier",
        "step": 0,
        "max_steps": 6,
        "exit_reason": None,
        "verified": False,
        # Debug
        "trace": [],
        "query_history": [],
    }


class Agent:
    def __init__(
        self, name, role, input_keys, output_schema, allowed_update_keys, run_func
    ):
        self.name = name
        self.role = role
        self.input_keys = input_keys
        self.output_schema = output_schema

        self.allowed_update_keys = allowed_update_keys
        self.run_func = run_func

    def run(self, state):
        # limit input
        local_input = {key: state.get(key) for key in self.input_keys}

        output = self.run_func(local_input)

        # limit output
        self._validate_output(output, self.output_schema)

        return output

    def _validate_output(self, output, schema):
        for key, expected_type in schema.items():
            if key not in output:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(output[key], expected_type):
                raise TypeError(
                    f"Key {key} should be {expected_type}, got {type(output[key])}"
                )
        return True


def apply_update(state, update, agent: Agent):
    safe_update = {}

    for key, value in update.items():
        if key in agent.allowed_update_keys:
            safe_update[key] = value
        else:
            print(f"Ignore unauthorized key from {agent.name}: {key}")

    state.update(safe_update)
    return state


def classify_agent_func(state):
    query = state["original_query"]

    if "code" in query or "implementation" in query:
        task_type = "implementation"
    elif "what is" in query or "explanation" in query:
        task_type = "conceptual"
    else:
        task_type = "general"

    return {
        "task_type": task_type,
        "classification_reason": "Classified by simple keyword rules.",
    }


def llm_classifier_agent_func(local_input):
    prompt = f"""

your are question classifier.

User query: {local_input["original_query"]}

Only return JSON:
{{
  "task_type": "conceptual | implementation | debugging | retrieval_needed",
  "classification_reason": "the reason with one sentence"
}}
"""

    return call_llm_json(prompt)


def llm_answer_agent_func(local_input):
    query = local_input["original_query"]
    docs = local_input["docs"]
    prompt = f"""
your are question answer.

User query: 
{query}

With the context: 
{docs}

Only return JSON:
{{
    "answer": "here is the answer",
    "citations": ["docs1"]
}}
"""
    return call_llm_json(prompt)


def verifier_agent_func(local_input):
    answer = local_input["answer"]
    citations = local_input["citations"]

    supported = bool(answer) and len(citations) > 0

    return {
        "verification_result": supported,
        "verification_reason": "Demo verifier, answer must exist and citations must not be empty.",
    }


def retriever_agent_func(local_input):
    query = local_input["original_query"]

    docs = [
        {"id": "doc_1", "text": "RAG means retrieval augmented generation."},
        {
            "id": "doc_2",
            "text": "In RAG, the system retrieves relevant documents before generating an answer.",
        },
        {
            "id": "doc_3",
            "text": "RAG can be implemented as retrieve, generate, and verify steps.",
        },
    ]

    return {"docs": docs}


classifier_agent = Agent(
    name="classifier",
    role="Classify the user query",
    input_keys={"original_query"},
    output_schema={"task_type": str, "classification_reason": str},
    allowed_update_keys={"task_type", "classification_reason"},
    run_func=llm_classifier_agent_func,
)

retriever_agent = Agent(
    name="retriever",
    role="Retrieve relevant documents",
    input_keys={"original_query"},
    output_schema={"docs": list},
    allowed_update_keys={"docs"},
    run_func=retriever_agent_func,
)

answer_agent = Agent(
    name="answer",
    role="Generate an answer based on retrieved docs",
    input_keys={"original_query", "docs"},
    output_schema={"answer": str, "citations": list},
    allowed_update_keys={"answer", "citations"},
    run_func=llm_answer_agent_func,
)

verifier_agent = Agent(
    name="verifier",
    role="Verify whether the answer is supported by documents",
    input_keys={"original_query", "docs", "answer", "citations"},
    output_schema={"verification_result": bool, "verification_reason": str},
    allowed_update_keys={"verification_result"},
    run_func=verifier_agent_func,
)

if __name__ == "__main__":
    state = create_initial_state()

    update = classifier_agent.run(state)
    state = apply_update(state=state, update=update, agent=classifier_agent)
    print(state)

    update = retriever_agent.run(state)
    state = apply_update(state=state, update=update, agent=retriever_agent)
    print(state)

    update = answer_agent.run(state)
    state = apply_update(state=state, update=update, agent=answer_agent)
    print(state)

    update = verifier_agent.run(state)
    state = apply_update(state=state, update=update, agent=verifier_agent)
    print(state)
