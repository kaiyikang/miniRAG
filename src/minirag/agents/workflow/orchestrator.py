from functools import partial

from minirag.agents.workflow.utils import call_llm_json
from minirag.agents.workflow.state import create_initial_state, RagState, END
from minirag.agents.workflow.route import route_next
from minirag.agents.tool import SearchTools
from minirag.llm_engine import InferenceEngine, OpenRouterEngine


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


def llm_classifier_agent_func(local_input, llm: InferenceEngine):
    messages = f"""
User query: {local_input["original_query"]}

Only return JSON:
{{
  "task_type": "conceptual | implementation | debugging | retrieval_needed",
  "classification_reason": "the reason with one sentence"
}}
"""

    return llm.generate(messages=messages)


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
    query_history = "\n".join(
        f"{idx}. {v}" for idx, v in enumerate(local_input["query_history"])
    )
    verification_reason = local_input["verification_reason"]
    reason = (
        f"The previous attempt failed because: {verification_reason}. Adjust the query to specifically address this gap."
        if verification_reason
        else ""
    )
    prompt = f"""
Rewrite into a well-formed, semantically explicit question/statement. Don't repeat the history queries.
{reason}

History queries:
{query_history}

User query:
{query}

Only return JSON:
{{
    "current_query": "the rewritten search query"
}}
"""
    return call_llm_json(prompt)


rewrite_query_agent = Agent(
    name="query_rewriter",
    role="Rewrite the query for better retrieval",
    input_keys={"original_query", "query_history", "verification_reason"},
    output_schema={"current_query": str},
    allowed_update_keys={"current_query"},
    run_func=rewrite_query_agent_func,
)


def llm_answer_agent_func(local_input):
    query = local_input["original_query"]
    last_chance = (
        local_input["verification_attempts"] >= local_input["max_verification_attempts"]
    )
    chunks = local_input["docs"]
    docs_ids = [chunk["id"] for chunk in chunks]
    docs_texts = [chunk["text"] for chunk in chunks]

    context = "".join(f"{idx}. {text}\n" for (idx, text) in zip(docs_ids, docs_texts))
    prompt_last_chance = (
        "This is the final attempt. If the documents above don't fully support a complete answer, say so explicitly instead of presenting an unverified claim as fact."
        if last_chance
        else ""
    )

    prompt = f"""
User query:
{query}

With the context:
{context}

{prompt_last_chance}

Only return JSON:
{{
    "answer": "here is the answer",
    "citations": ["<id copied from the documents above>"]
}}
"""
    answer_and_citations = call_llm_json(prompt)
    # Verification for true citations
    answer_and_citations["citations"] = [
        c for c in answer_and_citations["citations"] if c in docs_ids
    ]
    return answer_and_citations


answer_agent = Agent(
    name="answer",
    role="Generate an answer based on retrieved docs",
    input_keys={
        "original_query",
        "docs",
        "verification_attempts",
        "max_verification_attempts",
    },
    output_schema={"answer": str, "citations": list},
    allowed_update_keys={"answer", "citations"},
    run_func=llm_answer_agent_func,
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


def retriever_agent_func(local_input, tools: SearchTools):
    query = local_input.get("current_query") or local_input.get("original_query")
    docs = tools.vector_search_dicts(query, top_k=2)
    return {"docs": docs}


def get_retrieval_query_agent(tools: SearchTools, llm: InferenceEngine):
    return Agent(
        name="retriever",
        role="Retrieve relevant documents",
        input_keys={"original_query", "current_query"},
        output_schema={"docs": list},
        allowed_update_keys={"docs"},
        run_func=partial(retriever_agent_func, tools=tools, llm=llm),
    )


def build_agents(tools: SearchTools, llm: InferenceEngine) -> dict:
    return {
        "classifier": classifier_agent,
        "planner": planner_agent,
        "query_rewriter": rewrite_query_agent,
        "retriever": get_retrieval_query_agent(tools=tools, llm=llm),
        "reranker": reranker_agent,
        "answer": answer_agent,
        "verifier": verifier_agent,
    }


def apply_agent_update(state, updated_state, agent: Agent):
    safe_update = {}

    for key, value in updated_state.items():
        if key in agent.allowed_update_keys:
            safe_update[key] = value
        else:
            raise ValueError(
                f"Agent '{agent.name}' tried to update forbidden keys: {key}"
            )

    state.update(safe_update)
    return state


def apply_system_update(state, agent: Agent):

    state["step"] += 1

    if agent.name == "query_rewriter":
        state["query_history"].append(state["current_query"])

    if agent.name == "retriever":
        state["retrieval_attempts"] += 1

    if agent.name == "verifier":
        state["verification_attempts"] += 1
        state["verified"] = compute_verified(state)

    return state


def run_agent(state: RagState, agents: dict) -> RagState:

    current_agent = "classifier"

    while current_agent != END:
        if current_agent not in agents:
            raise ValueError(f"Agent is not registered: {current_agent}")

        agent = agents[current_agent]

        updated_state = agent.run(state)

        state = apply_agent_update(state, updated_state, agent)
        state = apply_system_update(state, agent)

        # Get the next routed agent
        decision = route_next(current_agent, state)

        state["trace"].append(
            {
                "step": state["step"],
                "agent": current_agent,
                "next_agent": decision["next_agent"],
                "route_reason": decision["reason"],
            }
        )

        # Apply the decision
        current_agent = decision["next_agent"]

        if decision["exit_reason"] is not None:
            state["exit_reason"] = decision["exit_reason"]

    return state


def compute_verified(state: dict) -> bool:
    result = state.get("verification_result")

    if result is None:
        return False

    return bool(result)


if __name__ == "__main__":
    from minirag.config import get_settings
    from minirag.embedding import OpenRouterEmbeddingEngine
    from minirag.vector_store import ChromaVectorStore

    settings = get_settings()
    embed = OpenRouterEmbeddingEngine(
        model=settings.openrouter_embed_model, api_key=settings.openrouter_api_key
    )
    vstore = ChromaVectorStore(
        vector_store_path=settings.vector_store_path,
        collection_name=settings.collection_name,
    )
    llm = OpenRouterEngine(
        model=settings.openrouter_model, api_key=settings.openrouter_api_key
    )

    tools = SearchTools(embed, vstore)
    agents = build_agents(tools, llm)

    state = create_initial_state("Explain the Java Exception")
    state = run_agent(state, agents)
    print(state["answer"])
    print(state["verification_result"])
    print(state["verified"])
