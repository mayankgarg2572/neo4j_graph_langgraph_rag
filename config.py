"""Central place for constants and environment bootstrap."""
from dotenv import load_dotenv
import os

load_dotenv()

PAGE_TITLE = "Advanced RAG"
PAGE_ICON = "🔎"
FILE_UPLOAD_PROMPT = "Upload your Text file here"
FILE_UPLOAD_TYPE = ".txt"

# Show the model actually in use
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
MAIN_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    api_key=os.getenv("GOOGLE_API_KEY"),  # Set to None to use the environment variable
    )
SCHEMA_LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY"),)