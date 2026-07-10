from utils import call_llm_json
from state import create_initial_state, RagState
from tool import Tool, simple_search_tool

TOOLS = {
    "simple_search": Tool(
        name="simple_search",
        description="Search relevant documents from the local corpus.",
        run_func=simple_search_tool,
    )
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
User query: {local_input["original_query"]}

Only return JSON:
{{
  "task_type": "conceptual | implementation | debugging | retrieval_needed",
  "classification_reason": "the reason with one sentence"
}}
"""

    return call_llm_json(prompt)


classifier_agent = Agent(
    name="classifier",
    role="Classify the user query",
    input_keys={"original_query"},
    output_schema={"task_type": str, "classification_reason": str},
    allowed_update_keys={"task_type", "classification_reason"},
    run_func=llm_classifier_agent_func,
)


def create_plan_agent_func(local_input: dict):
    query = local_input["original_query"]
    task_type = local_input["task_type"]

    if task_type == "implementation":
        plan = (
            "1. Define shared state;"
            "2. Define agent contract;"
            "3. Implement retrieval and response;"
            "4. Add validation;"
            "5. Add routing and exit conditions."
        )
    else:
        plan = (
            "1. Understand the user's question."
            "2. Search for relevant information."
            "3. Organize the answer."
        )
    return {"plan": plan}


planner_agent = Agent(
    name="planner",
    role="Break down the user's question into an execution plan",
    input_keys={"original_query", "task_type"},
    output_schema={"plan": str},
    allowed_update_keys={"plan"},
    run_func=create_plan_agent_func,
)


def rewrite_query_agent_func(local_input: dict) -> dict:
    query = local_input["original_query"]
    prompt = f"""
You rewrite a user query into a clearer, keyword-rich search query.

User query:
{query}

Only return JSON:
{{
    "current_query": "the rewritten search query"
}}
"""
    return call_llm_json(prompt)


rewrite_query_agent = Agent(
    name="rewrite_query",
    role="Rewrite the query for better retrieval",
    input_keys={"original_query"},
    output_schema={"current_query": str},
    allowed_update_keys={"current_query"},
    run_func=rewrite_query_agent_func,
)


def llm_answer_agent_func(local_input):
    query = local_input["original_query"]
    docs = local_input["docs"]
    prompt = f"""
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


answer_agent = Agent(
    name="answer",
    role="Generate an answer based on retrieved docs",
    input_keys={"original_query", "docs"},
    output_schema={"answer": str, "citations": list},
    allowed_update_keys={"answer", "citations"},
    run_func=llm_answer_agent_func,
)


def retriever_agent_func(local_input):
    query = local_input["original_query"]

    docs = TOOLS["simple_search"].run(query=query, top_k=2)

    return {"docs": docs}


retriever_agent = Agent(
    name="retriever",
    role="Retrieve relevant documents",
    input_keys={"original_query"},
    output_schema={"docs": list},
    allowed_update_keys={"docs"},
    run_func=retriever_agent_func,
)


def rerank_documents_agent_func(local_input: dict) -> dict:
    query = local_input["original_query"]
    docs = local_input["docs"]
    prompt = f"""
You grade each document's relevance to the user query on a 1-5 scale:
5 ESSENTIAL    - answer is impossible without it (direct answer, definition, or prerequisite).
4 CONTRIBUTING - supplies something a complete answer needs alongside other docs.
3 SUPPORTING   - on topic and plausibly useful, but answer is likely complete without it.
2 TANGENTIAL   - same domain/terminology, no concrete contribution.
1 UNRELATED    - no meaningful connection.

User query:
{query}

Documents:
{docs}

Grade every document, then return them sorted by score (highest first).
Only return JSON:
{{
    "reranked_docs": [
        {{"id": "doc_1", "score": 5}}
    ]
}}
"""
    return call_llm_json(prompt)


reranker_agent = Agent(
    name="reranker",
    role="Re-ranking of retrieved documents",
    input_keys={
        "original_query",
        "docs",
    },
    output_schema={"reranked_docs": list},
    allowed_update_keys={"reranked_docs"},
    run_func=rerank_documents_agent_func,
)


def verifier_agent_func(local_input):
    query = local_input["original_query"]
    docs = local_input["docs"]
    answer = local_input["answer"]
    prompt = f"""
You check whether the answer is supported by the documents for the given query.

User query:
{query}

Documents:
{docs}

Answer:
{answer}

Only return JSON:
{{
    "verification_result": true,
    "verification_reason": "the reason with one sentence"
}}
"""
    return call_llm_json(prompt)


verifier_agent = Agent(
    name="verifier",
    role="Verify whether the answer is supported by documents",
    input_keys={"original_query", "docs", "answer", "citations"},
    output_schema={"verification_result": bool, "verification_reason": str},
    allowed_update_keys={"verification_result", "verification_reason"},
    run_func=verifier_agent_func,
)


def apply_update(state, updated_state, agent: Agent):
    safe_update = {}

    for key, value in updated_state.items():
        if key in agent.allowed_update_keys:
            safe_update[key] = value
        else:
            raise ValueError(
                f"Agent '{agent.name}' tried to update " f"forbidden keys: {key}"
            )

    state.update(safe_update)
    return state


def run_agent_once(state: RagState, agent_name: str) -> RagState:
    agent = AGENTS[agent_name]

    updated_state = agent.run(state)

    state = apply_update(state=state, updated_state=updated_state, agent=agent)

    state["trace"].append(
        {
            "step": state["step"],
            "agent": agent_name,
            "output_keys": list(updated_state.keys()),
        }
    )

    print(state["trace"])

    state["step"] += 1
    return state


END = "__end__"


def route_next(current_agent: str, state: dict) -> str:
    routes = {
        "classifier": "planner",
        "planner": "query_rewriter",
        "query_rewriter": "retriever",
        "retriever": "reranker",
        "reranker": "answer",
        "answer": "verifier",
        "verifier": END,
    }
    return routes[current_agent]


AGENTS = {
    "classifier": classifier_agent,
    "planner": planner_agent,
    "query_rewriter": rewrite_query_agent,
    "retriever": retriever_agent,
    "reranker": reranker_agent,
    "answer": answer_agent,
    "verifier": verifier_agent,
}

if __name__ == "__main__":
    state = create_initial_state()
    next_agent = route_next("retriever", state)
    print(next_agent)
    # state = run_agent_once(state, "classifier")
    # state = run_agent_once(state, "planner")
    # state = run_agent_once(state, "query_rewriter")
    # state = run_agent_once(state, "retriever")
    # state = run_agent_once(state, "reranker")
    # state = run_agent_once(state, "answer")
    # state = run_agent_once(state, "verifier")
    # print(state["answer"])
    # print(state["verification_result"])
    # print(state["verified"])
