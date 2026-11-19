
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

if __name__ == "__main__":

    # Step 1: Load documents
    docs = load_all_documents("data")

    # Step 2: Initialize FAISS vector store
    store = FaissVectorStore("faiss_store")

    # Step 3: Check whether FAISS index exists
    faiss_path = os.path.join("faiss_store", "faiss.index")

    if os.path.exists(faiss_path):
        print("[INFO] Existing FAISS index found. Loading...")
        store.load()
    else:
        print("[INFO] FAISS index not found. Creating new index...")
        store.build_from_documents(docs)
        print("[INFO] FAISS index created successfully!")

    # Step 4: Initialize RAG search
    rag_search = RAGSearch()

    # Step 5: Ask question
    query = "What is attention mechanism?"

    summary = rag_search.search_and_summarize(query, top_k=3)

    print("Summary:", summary)

