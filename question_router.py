from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import JsonOutputParser


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",  temperature=0.0)

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

question_router = prompt | llm | JsonOutputParser()