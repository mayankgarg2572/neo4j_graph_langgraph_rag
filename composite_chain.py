from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import MAIN_LLM

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context from a vector store and a graph database to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Vector Context: {context} 
    Graph Context: {graph_context}
    Answer: 
    """,
    input_variables=["question", "context", "graph_context"],
)

composite_chain = prompt | MAIN_LLM | StrOutputParser()