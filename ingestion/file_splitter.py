from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_file(buffer: bytes) -> list[Document]:
    content = buffer.decode().strip()
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    print("In function split_file with args, lines(buffer):", lines)

    ts_lines = sum(1 for l in lines if "\t" in l and len(l.split("\t")) >= 2)
    structured = ts_lines / max(len(lines), 1) > 0.7

    if structured:
        return [Document(page_content=l) for l in lines]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    return splitter.create_documents([content])
