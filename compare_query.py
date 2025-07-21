import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from document_utils import load_and_chunk_pdf
from vectorstore import create_vector_store

def main():
    # Load and chunk both PDFs
    chunks1 = load_and_chunk_pdf(r"C:\Users\kanth\Desktop\multi-document-comparator-agent\data\doc1.pdf")
    chunks2 = load_and_chunk_pdf(r"C:\Users\kanth\Desktop\multi-document-comparator-agent\data\doc2.pdf")
    
    all_chunks = chunks1 + chunks2
    
    # Create a combined vector store
    vectordb = create_vector_store(all_chunks)
    print(f"Vector store created with {vectordb.index.ntotal} vectors.")
    
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        results = vectordb.similarity_search(query, k=3)  # top 3 chunks
        
        print("\nTop relevant chunks:")
        for i, doc in enumerate(results):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:500])  # print first 500 chars

if __name__ == "__main__":
    main()
