# document_utils.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    loader = PyPDFLoader(path)
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_documents(pages)
    return chunks