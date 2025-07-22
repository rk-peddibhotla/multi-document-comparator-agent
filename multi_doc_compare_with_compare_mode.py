from langchain_ollama import OllamaLLM
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def load_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    return chunks

def main():
    
    chunks1 = load_and_chunk(r"C:\Users\kanth\Desktop\multi-document-comparator-agent\data\doc1.pdf")
    chunks2 = load_and_chunk(r"C:\Users\kanth\Desktop\multi-document-comparator-agent\data\doc2.pdf")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    
    vectorstore1 = FAISS.from_documents(chunks1, embedding_model)
    vectorstore2 = FAISS.from_documents(chunks2, embedding_model)

    llm = OllamaLLM(model="gemma:2b")

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        
        compare_keywords = ["compare", "difference", "contrast", "how about", "which is longer", "which is better"]
        if any(word in query.lower() for word in compare_keywords):
            
            docs1 = vectorstore1.similarity_search(query, k=3)
            docs2 = vectorstore2.similarity_search(query, k=3)

            context1 = "\n\n".join([d.page_content for d in docs1])
            context2 = "\n\n".join([d.page_content for d in docs2])

            prompt = (
                f"Given the following information from Document 1:\n{context1}\n\n"
                f"And the following information from Document 2:\n{context2}\n\n"
                f"Answer this comparison question: {query}\n"
                "Be clear about the differences and similarities."
            )
            response = llm.invoke(prompt)
            print("\nComparison Answer:\n", response)
            print("\n" + "="*50 + "\n")
        else:
            
            all_chunks = chunks1 + chunks2
            combined_vectorstore = FAISS.from_documents(all_chunks, embedding_model)
            relevant_docs = combined_vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {query}\nAnswer:"
            response = llm.invoke(prompt)
            print("\nAnswer:\n", response)
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()