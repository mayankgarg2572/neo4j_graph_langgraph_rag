from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from utils.runtime import registry

### Search
web_search_tool = TavilySearchResults(k=3)

def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("In function web_search with args, state:", state)
    question = state["question"]
    documents = state.get("documents", [])  # Use get() with a default empty list

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}



def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("In function retrieve with args, state:", state)
    question = state["question"]

    # Retrieval
    retriever   = registry.retriever
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
