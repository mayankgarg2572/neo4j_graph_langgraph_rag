from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# # Fast model for general tasks (GraphTransformer, QA chains)
# MAIN_LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

# # More capable model for complex reasoning (schema design)
# SCHEMA_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)



from groq import Groq
from langchain_groq import ChatGroq

MAIN_LLM = ChatGroq(temperature=0,
                      model_name="Llama3-8b-8192",)

SCHEMA_LLM = ChatGroq(temperature=0,
                      model_name="Llama3-8b-8192",)

# from langchain_together import ChatTogether

# MAIN_LLM = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo",)
# SCHEMA_LLM = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo",)


# from langchain_cohere import ChatCohere

# # Define the Cohere LLM
# MAIN_LLM = ChatCohere(model="command-r-03-2025", temperature=0.0)
# SCHEMA_LLM = ChatCohere(model="command-r-03-2025", temperature=0.0)


# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )

# MAIN_LLM = ChatHuggingFace(llm=llm)
# SCHEMA_LLM = ChatHuggingFace(llm=llm)
