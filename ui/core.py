import streamlit as st
from config import PAGE_ICON, PAGE_TITLE

def setup_page() -> None:
    print("In function setup_page")
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.header("", divider="blue")
    st.title(f"{PAGE_ICON} :blue[_{PAGE_TITLE}_] | Text File Search")
    st.header("", divider="blue")


def ask_question(file_uploaded) -> str | None:
    """Return question string or None."""
    print("In function ask_question with args, file_uploaded:", file_uploaded)
    if not file_uploaded:
        return None
    return st.text_input(
        "Please enter your question:",
        placeholder="Which year was Marty transported to?",
        disabled=not file_uploaded,
    )
