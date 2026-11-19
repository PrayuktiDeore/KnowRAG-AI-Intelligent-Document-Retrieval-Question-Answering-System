
import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-20b"
    ):

        self.vectorstore = FaissVectorStore(
            persist_dir, embedding_model
        )

        # Load existing FAISS vector store
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents

            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)

        else:
            self.vectorstore.load()

        # Load Groq API key
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file"
            )

        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model=llm_model,
            temperature=0.2
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")


    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        # Step 1: Retrieve relevant chunks
        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        # Step 2: Extract retrieved text
        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        # Step 3: Create prompt
        prompt = f"""
        Answer the following query using the provided context.

        Query: {query}

        Context:
        {context}

        Provide a clear and accurate answer.
        """

        # Step 4: Generate response
        response = self.llm.invoke(prompt)

        return response.content


# Example usage
if __name__ == "__main__":

    rag_search = RAGSearch()

    query = "What is attention mechanism?"

    summary = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print("Summary:", summary)
