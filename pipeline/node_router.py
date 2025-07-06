

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import MAIN_LLM
from graph.search import graph_search

prompt = PromptTemplate(
    template="""You are an expert at routing a user question to the most appropriate data source. 
    You have three options:
    1. 'vectorstore': Use for questions about LLM agents, prompt engineering, and adversarial attacks.
    2. 'graphrag': Use for questions that involve relationships between entities, such as authors, papers, and topics, or when the question requires understanding connections between concepts.
    3. 'web_search': Use for all other questions or when current information is needed.

    You do not need to be stringent with the keywords in the question related to these topics. 
    Choose the most appropriate option based on the nature of the question.

    Return a JSON with a single key 'datasource' and no preamble or explanation. 
    The value should be one of: 'vectorstore', 'graphrag', or 'web_search'.
    
    Question to route: 
    {question}""",
    input_variables=["question"],
)

question_router = prompt | MAIN_LLM | JsonOutputParser()


def route_question(state):
    print("In function route_question with args, state:", state)
    question = state["question"]
    source = question_router.invoke({"question": question})
    print("Suggested source by LLM: ", source["datasource"])

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
