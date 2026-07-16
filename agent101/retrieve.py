from tool import vector_search_tool


def retrieve(query: str) -> list[str]:
    return vector_search_tool(query, 5)
