### Retrieval Grader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI
from llm_config import MAIN_LLM


# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

prompt = PromptTemplate(
    template="""You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     
    Here is the retrieved document: 
    {document}
    
    Here is the user question: 
    {question}
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | MAIN_LLM | JsonOutputParser()

# # Example usage of the retrieval grader
# question = "Do we have articles that talk about Prompt Engineering?"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(
#     f'Is our answer relevant to the question asked: {retrieval_grader.invoke({"question": question, "document": doc_txt})}'
# )