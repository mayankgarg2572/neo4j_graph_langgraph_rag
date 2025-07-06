from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_retriever(docs):
    print("In function build_retriever with args, docs:", docs)
    vs = FAISS.from_documents(
        docs,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
    )
    return vs.as_retriever()
