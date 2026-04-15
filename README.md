1. Project Overview
   This project implements a multi-step adaptive RAG agent that dynamically selects the best strategy to answer a query.
   The agent is currently implemented in a single module (rag_agent.py) for developement speed, and can be modularized into components(retrieval, reranking, generation, tools)
   ** rag_agent.py   → core brain
   ** api_server.py  → API gateway
   ** chat.html      → UI client
   
2. Pipeline Overview
   2.1 Hybrid Retrieval :
      Dense retrieval (FAISS + embeddings)
      Sparse retrieval (BM25)
   It produces a diverse candidate set
   2.2 Re-ranking:
      Cosine similarity (fast filtering)
      Cross-encoder (precision ranking)
   2.3 Confidence Estimation
      Based on embedding similarity between query and top chubks
      Determines whether retrieved context is reliable
   2.4 Decision Layer (Agent logic)
      Dynamically chooses : High confidence (RAG) and Lowe confidence (fallback strategy)
    
3. General Picture: 
                                                   --> High Confidence : RAG            
   Query --> Hybrid Retrieval --> Confidence Score:                                        --> OpenAI    
                                                   --> Low Confidence : Tool Routing Layer:--> Web Search
                                                                                           --> Local LLM                 

4. Architecture : Local LLM via Ollama (phi3)
                  OpenAI (gpt-4o-mini)
                  Web Search (DuckDuckGo API) 
                  Document-based retrieval (RAG pipeline)
                  Optional web-augmented retrieval
                  FastAPI backend for serving responses
                  Simple HTML frontend interface

6. Setup
      pip install -r requirements.txt

8. OpenAI Integration : The agent supports OpenAI as a fallback and primary generation tool.
    6.1 Behaiviour : Automatically validates API key at startup
                     Uses OpenAI when: Confidence is low or Local model is insufficient
                     Falls back to local model if OpenAI fails
    6.2 Setup : export OPENAI_API_KEY=your_key
                 

9. Ollama setup
    4.1 Isntall ollama (https://ollama.com)
    4.2 Pull reqirements in Terminal
        ollama pull phi3
        ollama pull nomic-embed-text


10. Run API

uvicorn api_server:app --reload
    Server runs at : http://127.0.0.1:8000
    
9. Open frontend

open frontend/chat.html

10. Configuration 
    Settings are centralized in : config.py
    It includes :
        Model selection (phi3)
        Chunking strategy
        Retrieval parameters
        Optional web search setings

11. Future Improvements: 
    Streaming responses
    Better ranking(re-ranking models)
    Web seach integration
    Docekr deployement

12. Memory : The agent maintains conversational memory:
        Stores (query, answer) pairs
        Injects past interactions into future context
        Enables multi-turn reasoning    
        
13. Data Generation Pipeline
        The project includes a custom data ingestion pipeline to build the knowledge base used by the RAG system.  
        Sources: Wikipedia (conceptual and foundational knowledge) and arXiv (research-level content)
        Build the Corpus : run python -m scripts/build_corpus.py



Author : 
Hamed Hosseinpour
AI Engineer    
