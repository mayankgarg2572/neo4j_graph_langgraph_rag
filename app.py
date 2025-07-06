import streamlit as st
from ingestion.file_splitter import split_file
from ingestion.vector_store import build_retriever
from ui.core import setup_page, ask_question
from schema.generator import generate_schema
from pipeline.workflow import build_graph
from utils.runtime import registry

def main() -> None:
    print("In function main")
    setup_page()

    upload = st.file_uploader("Upload your Text file here", type=".txt")
    if not upload:
        return

    with st.spinner("Processing file…"):
        docs = split_file(upload.read())
        # retriever = build_retriever(docs)
        # schema = generate_schema(docs)
        registry.retriever = build_retriever(docs)
        registry.graph_schema = generate_schema(docs)

    question = ask_question(upload)
    if not question:
        return

    with st.spinner("Running RAG pipeline…"):
        app = build_graph()
        for out in app.stream({"question": question}):
            key, value = next(iter(out.items()))
            st.info(f"Finished stage: {key}")
        st.success(value["generation"])


if __name__ == "__main__":
    main()
