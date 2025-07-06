from langgraph.graph import StateGraph, END
from utils.types import GraphState
from pipeline import node_router   # tiny helper that wraps route_question()
from retrieval.retrieve import retrieve, web_search
from retrieval.grading import grade_documents, decide_to_generate
from graph.search import graph_search
from generation.generate import generate
from evaluation.decision import grade_generation_v_documents_and_question

def build_graph():
    print("In function: build_graph")
    g = StateGraph(GraphState)

    g.set_conditional_entry_point(
        node_router.route_question,
        {"websearch": "websearch", "retrieve": "retrieve", "graphrag": "graphrag"},
    )

    g.add_node("retrieve", retrieve)
    g.add_node("websearch", web_search)
    g.add_node("grade_documents", grade_documents)
    g.add_node("generate", generate)
    g.add_node("graphrag", graph_search)

    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"websearch": "websearch", "generate": "generate"},
    )
    g.add_edge("websearch", "generate")
    g.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {"useful": END, "not useful": "websearch", "not supported": "generate"},
    )

    return g.compile()
