from schema.generator import GraphSchema
from graph.graph_resources import get_resources
from langchain.schema import Document

def graph_search(state, docs, schema: GraphSchema):
    print("In function graph_search with args, schema:", schema, "docs:", docs, "state:", state)
    if not (schema.allowed_nodes and schema.allowed_relationships):
        return {"graph_context": "No graph schema available", "question": state["question"]}

    nodes, rels, props = (
        schema.allowed_nodes,
        schema.allowed_relationships,
        schema.node_properties,
    )
    _, graph_rag_chain = get_resources(nodes, rels, props, docs)
    result = graph_rag_chain.invoke({"query": state["question"]})
    return {"graph_context": result.get("result", ""), "question": state["question"]}
