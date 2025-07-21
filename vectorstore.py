from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def create_vector_store(docs, model_name="all-MiniLM-L6-v2"):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore
