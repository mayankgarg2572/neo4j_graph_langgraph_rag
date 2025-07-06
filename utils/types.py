from typing import List, TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    graph_context: str

