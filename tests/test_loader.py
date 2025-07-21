import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from document_utils import load_and_chunk_pdf

if __name__ == "__main__":
    chunks = load_and_chunk_pdf(r"C:\Users\kanth\Desktop\multi-document-comparator-agent\data\doc1.pdf")
    print(f"Loaded {len(chunks)} chunks.")
    print("\nFirst chunk content:\n")
    print(chunks[0].page_content[:500])
