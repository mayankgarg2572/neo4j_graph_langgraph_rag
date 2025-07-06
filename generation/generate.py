from composite_chain import composite_chain


def generate(state):
    """
    Generate answer using RAG on retrieved documents and graph context

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state.get("documents", [])
    graph_context = state.get("graph_context", "")

    print("In function generate with args, question:", question, 
          "documents:", documents, "graph_context:", graph_context)

    # Composite RAG generation
    generation = composite_chain.invoke(
        {"question": question, "context": documents, "graph_context": graph_context}
    )
    print("Final generated response:", generation)
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "graph_context": graph_context,
    }

