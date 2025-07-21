import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from document_utils import load_and_chunk_pdf
from vectorstore import create_vector_store

if __name__ == "__main__":
    chunks = load_and_chunk_pdf(r"C:\Users\kanth\Desktop\multi-document-comparator-agent\data\doc1.pdf")
    vectordb = create_vector_store(chunks)
    print(f"Vector store created with {vectordb.index.ntotal} vectors.")
