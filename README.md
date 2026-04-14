# Retrieval-Augmented-Generation
A modular Retrieval-Augmented Generation (RAG) chatbot that combines vector search, keyword-based retrieval, and external tools (web search, OpenAI, and local LLMs) into a unified conversational agent. The system uses FAISS for dense embeddings, BM25 for lexical matching, and cross-encoder reranking to improve retrieval quality and answer relevance.

Built with a FastAPI backend and a lightweight HTML/JavaScript chat interface, the project enables real-time question answering over custom documents with optional fallback to external knowledge sources when internal retrieval confidence is low. It also includes short-term conversational memory to maintain context across interactions, making responses more coherent and context-aware.

This project demonstrates a full end-to-end pipeline for hybrid information retrieval, tool-augmented reasoning, and production-style API deployment of a RAG system.
