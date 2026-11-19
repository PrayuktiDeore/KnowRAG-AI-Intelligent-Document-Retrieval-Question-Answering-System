# 🧠 KnowRAG AI
### Intelligent Document Retrieval & Question Answering System

KnowRAG AI is an AI-powered document intelligence system built using **Retrieval-Augmented Generation (RAG)**. It allows users to retrieve relevant information from multiple documents and generate context-aware answers using Large Language Models (LLMs).

The system combines semantic search, vector embeddings, FAISS vector storage, and Groq-powered LLM generation to answer questions based on the content of uploaded documents.

## 🚀 Features

- **Multi-Format Document Loading:** Extracts content from PDF, TXT, CSV, Excel, Word, and JSON files.
- **Intelligent Text Chunking:** Uses Recursive Character Text Splitter to divide large documents into manageable chunks.
- **Semantic Embeddings:** Generates 384-dimensional embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- **Vector Database:** Uses FAISS for efficient similarity search and retrieval.
- **Persistent Storage:** Saves vector indexes and metadata for reuse without rebuilding the entire database.
- **Context-Aware Q&A:** Retrieves relevant document chunks and provides context-based answers.
- **LLM Integration:** Uses Groq API with GPT-OSS-20B for natural-language response generation.

## 🏗️ Project Architecture

```text
KnowRAG-AI/
│
├── data/
│   ├── pdf/
│   └── text_files/
│
├── src/
│   ├── data_loader.py
│   ├── embedding.py
│   ├── vectorstore.py
│   └── search.py
│
├── faiss_store/
│   ├── faiss.index
│   └── metadata.pkl
│
├── app.py
├── .env
├── .gitignore
└── README.md
```


