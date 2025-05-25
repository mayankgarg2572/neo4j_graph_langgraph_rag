from graph_resources import get_resources
from question_router import question_router
from composite_chain import composite_chain
from hallucination_grader import hallucination_grader
from retrieval_grader import retrieval_grader
from answer_grader import answer_grader
from llm_config import SCHEMA_LLM 


# Imports
import json
from dotenv import load_dotenv
import streamlit as st

from langchain_community.tools.tavily_search import TavilySearchResults

# State Setup
from typing_extensions import TypedDict
from typing import List

# Doc Splitters
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Vector Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 

# Graph imports
from langgraph.graph import END, START, StateGraph


from langchain_google_genai import ChatGoogleGenerativeAI


from pydantic import BaseModel
from typing import List
from pydantic import BaseModel


import torch

torch.classes.__path__ = [] # add this line to manually set it to empty.

# Constants for UI
PAGE_TITLE = "Advanced RAG"
PAGE_ICON = "🔎"
FILE_UPLOAD_PROMPT = "Upload your Text file here"
FILE_UPLOAD_TYPE = ".txt"

from langchain_community.tools.tavily_search import TavilySearchResults

### Search
web_search_tool = TavilySearchResults(k=3)

load_dotenv()  # Load environment variables from .env file

### State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        graph_context: results from graph search
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    graph_context: str


# Setup the UI configuration
def setup_ui():
    """
    Configures the Streamlit app page settings and displays the main title.
    """
    st.set_page_config(
        page_title=PAGE_TITLE, 
        page_icon=PAGE_ICON
    )
    st.header("",divider='blue')
    st.title(f"{PAGE_ICON} :blue[_{PAGE_TITLE}_] | Text File Search")
    st.header("",divider='blue')


def ask_question(user_file):
    """
    Allows the user to ask a question about the uploaded file and displays the result.
    """    
    if user_file is None:
        return
    st.divider()
    question = st.text_input('Please enter your question:', placeholder = "Which year was Marty transported to?", disabled=not user_file)
    # Test
    from pprint import pprint
    if question:
        with st.spinner('Please wait...'):
            app = create_graph()

            for output in app.stream({"question": question}):
                for key, value in output.items():
                    st.info(f"Finished running: {key}:")
                    st.divider()
                    pprint(f"Finished running: {key}:")
            # result = graph.invoke(input={"question": question})
            st.info(value["generation"])
            st.divider()


def split_the_user_file(user_file):
    """
    Adaptively split the user file based on its structure
    """
    content = user_file.read().decode().strip()
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Check if data appears to be structured (tab-separated)
    tab_separated_lines = sum(1 for line in lines if '\t' in line and len(line.split('\t')) >= 2)
    
    if tab_separated_lines / max(len(lines), 1) > 0.7:  # 70% threshold
        # Structured data: split by lines
        print("Using line-based splitting for structured data")
        doc_splits = [Document(page_content=line) for line in lines]
    else:
        # Unstructured data: use recursive character splitting
        print("Using recursive character splitting for unstructured data")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        doc_splits = splitter.create_documents([content])
    
    return doc_splits


def get_vector_retriever(doc_splits):
    """
    Creates a retriever from the uploaded file by splitting it into chunks and inserting embeddings into a vector database.
    """
    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
        )
    )

    return vectorstore.as_retriever()


class GraphSchema(BaseModel):
    allowed_nodes: List[str]
    node_properties: List[str]
    allowed_relationships: List[str]

# # This is your LLM (can be Gemini or OpenAI)

# def design_neo4j_graph_schema(state: Dict) -> Dict:
#     """
#     Dynamically generate GraphRAG schema from input documents.

#     Args:
#         state: Graph state dict

#     Returns:
#         state with added keys: allowed_nodes, node_properties, allowed_relationships
#     """
#     schema_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

#     print("---DESIGNING GRAPH SCHEMA---")
#     # documents: List[Document] = state["documents"]
#     global doc_splits

#     schema_prompt = (
#         "You are a graph schema expert. Analyze the following documents and extract:\n"
#         "- The most appropriate node types (entity categories)\n"
#         "- Node properties (metadata fields)\n"
#         "- Possible relationships between entities\n\n"
#         "Return ONLY valid JSON in this structure:\n"
#     "{\n"
#     '  "allowed_nodes": ["Entity1", "Entity2"],\n'
#     '  "node_properties": ["title", "summary"],\n'
#     '  "allowed_relationships": ["AUTHORED", "MENTIONS"]\n'
#     "}\n\n"
#     "Ensure all keys are present and all values are lists of strings.\n"
#     "Do NOT include any extra text or explanations.\n\n"
#     "Documents:\n"
#         + "\n\n".join([doc.page_content[:1000] for doc in doc_splits[:5]])  # limit to first few chunks
#     )

#     raw_response = schema_llm.invoke(schema_prompt)
    
#     # Parse JSON safely
#     try:
#         parsed_json = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
#         schema = GraphSchema(**parsed_json)
#     except (json.JSONDecodeError, ValidationError) as e:
#         print("---SCHEMA VALIDATION FAILED---")
#         print(e)
#         return {
#             **state,
#             "allowed_nodes": [],
#             "node_properties": [],
#             "allowed_relationships": [],
#         }

#     return {
#         **state,
#         "allowed_nodes": schema.allowed_nodes,
#         "node_properties": schema.node_properties,
#         "allowed_relationships": schema.allowed_relationships
#     }


def generate_schema_once(doc_splits):
    """Generate schema once and cache it"""
    schema_llm = SCHEMA_LLM
    
    schema_prompt = (
        "You are a graph schema expert. Analyze the following documents and extract:\n"
        "- The most appropriate node types (entity categories)\n"
        "- Node properties (metadata fields)\n" 
        "- Possible relationships between entities\n\n"
        "Return ONLY valid JSON in this structure:\n"
        "{\n"
        '  "allowed_nodes": ["Entity1", "Entity2"],\n'
        '  "node_properties": ["title", "summary"],\n'
        '  "allowed_relationships": ["AUTHORED", "MENTIONS"]\n'
        "}\n\n"
        "No need to provide any explanations or additional text.\n\n"
        "Ensure all keys are present and all values are lists of strings as in above example.\n\n"
        "Must ensure you just return a valid JSON object having structure similar to the above example with no additional text.\n\n"
        "Documents:\n"
        + "\n\n".join([doc.page_content for doc in doc_splits])
    )
    
    try:
        raw_response = schema_llm.invoke(schema_prompt)
        print("LLM response for graph database schema:", raw_response.content if hasattr(raw_response, 'content') else raw_response)
        parsed_json = json.loads(raw_response.content if hasattr(raw_response, 'content') else raw_response)
        return GraphSchema(**parsed_json)
    except Exception as e:
        print(f"Schema generation failed: {e}")
        return GraphSchema(
            allowed_nodes=[],
            node_properties=[],
            allowed_relationships=[]
        )


def handle_file_upload(user_file):
    """
    Handles the uploaded text file, splits it into chunks, and inserts embeddings into a vector database.
    """
    if user_file is None:
        return
    
    # Split the user file into manageable chunks
    global doc_splits, graph_schema
    doc_splits = split_the_user_file(user_file)

    # Generate schema once during upload
    graph_schema = generate_schema_once(doc_splits)

    vector_retiever = get_vector_retriever(doc_splits)
    
    st.success("Text file embeddings were successfully inserted into VectorDB")

    return vector_retiever


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


# Generate answer by invoking composite_chain on retrieved documents and graph context
def generate(state):
    """
    Generate answer using RAG on retrieved documents and graph context

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", [])
    graph_context = state.get("graph_context", "")

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


# Grade documents using retrieval_grader to determine relevance to the question, If any of the doc is irrelevant, we will set web_search flag
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])  # Use get() with a default empty list

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}


### Conditional edge
def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])

    if source["datasource"] == "graphrag":
        print("---TRYING GRAPH SEARCH---")
        graph_result = graph_search({"question": question})
        if graph_result["graph_context"] != "No results found in the graph database.":
            return "graphrag"
        else:
            print("---NO RESULTS IN GRAPH, FALLING BACK TO VECTORSTORE---")
            return "retrieve"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO VECTORSTORE RAG---")
        return "retrieve"
    elif source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def graph_search(state):
    """
    Perform GraphRAG search using Neo4j

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with graph search results
    """
    print("---GRAPH SEARCH---")

    # nodes = state.get("allowed_nodes", [])
    # rels = state.get("allowed_relationships", [])
    # props = state.get("node_properties", [])

    # Use the cached schema instead of generating new one
    global graph_schema
    if 'graph_schema' not in globals() or len(graph_schema.allowed_nodes) == 0 or len(graph_schema.allowed_relationships) == 0 or len(graph_schema.node_properties) == 0:

        return {"graph_context": "No graph schema available", "question": state["question"]}
    
    nodes = graph_schema.allowed_nodes
    rels = graph_schema.allowed_relationships  
    props = graph_schema.node_properties

    _, graph_rag_chain = get_resources(nodes, rels, props, doc_splits)  # heavy objects are global
    # result       = rag_chain.invoke({"query": state["query"]})
    question = state["question"]

    # Use the graph_rag_chain to perform the search
    result = graph_rag_chain.invoke({"query": question})

    # Extract the relevant information from the result
    # Adjust this based on what graph_rag_chain returns
    graph_context = result.get("result", "")

    # You might want to combine this with existing documents or keep it separate
    return {"graph_context": graph_context, "question": question}


### Conditional edge
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = grade = score.get("score", "").lower()

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def is_valid_schema(state):
    """
    Checks if the graph schema is valid.

    Args:
        state (dict): The current graph state

    Returns:
        bool: True if the schema is valid, False otherwise
    """
    print("---CHECK GRAPH SCHEMA VALIDITY---")
    allowed_nodes = state.get("allowed_nodes", [])
    node_properties = state.get("node_properties", [])
    allowed_relationships = state.get("allowed_relationships", [])

    # Check if all required fields are present and non-empty
    return {
        is_valid_schema: isinstance(allowed_nodes, list) and allowed_nodes
        and isinstance(node_properties, list) and node_properties
        and isinstance(allowed_relationships, list) and allowed_relationships and len(allowed_nodes)> 0 and len(node_properties) > 0 and len(allowed_relationships) > 0
    }


def create_graph(): 
    """
    Creates and configures the state graph for handling queries and generating answers.
    """
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(GraphState)

    # Define the nodes
    # workflow.add_node("neo4j_graph_schema", design_neo4j_graph_schema)  # design graph schema
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("graphrag", graph_search)
    # Set conditional entry point

    # Add edges
    # workflow.add_edge(START, "neo4j_graph_schema")
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "retrieve": "retrieve",
            "graphrag": "graphrag",
        },
    )
    # workflow.add_conditional_edges(
    #     "neo4j_graph_schema",
    #     route_question,
    #     {
    #         "websearch": "websearch",
    #         "retrieve": "retrieve",
    #         "graphrag": "graphrag",
    #     },
    # )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("graphrag", "generate")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    # Compile
    app = workflow.compile()
    return app


def search_online(state: GraphState):
    """
    Searches online for additional context if the answer cannot be generated locally.
    """    
    question = state["question"]
    documents = state["documents"]
    tavily_client = TavilySearchResults(k=2)
    response = tavily_client.invoke({"query": question})
    results = "\n".join([element["content"] for element in response])
    results = Document(page_content=results)
    if documents is not None:
        documents.append(results)
    else:
        documents = [results]
    return {"documents": documents, "question": question}

# Main function to orchestrate the app
def main():
    # Setup the UI
    setup_ui()
    
    # File uploader
    user_file = st.file_uploader(FILE_UPLOAD_PROMPT, type=FILE_UPLOAD_TYPE)
    
    # Handle the file upload
    global retriever
    retriever = handle_file_upload(user_file)
    

    # Ask the question
    ask_question(user_file)

if __name__ == "__main__":
    load_dotenv()
    main()
