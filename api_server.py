from fastapi import FastAPI
from pydantic import BaseModel
from rag_agent import agent_run, build_rag, AgentMemory
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = build_rag("data/docs")
agent_memory = AgentMemory(max_len=50)


#--------model----------------
class QueryRequest(BaseModel):
    query:str

class QueryResponse(BaseModel):
    query :str
    answer : str


#----------Endpoints------------

@app.get("/health")
def health():
    return{"status":"ok", "message": "Agent is alive."}


@app.get("/")
def root():
    return {"message": "RAG API is running..."}


@app.post("/ask",response_model = QueryResponse)
def ask_agent(request: QueryRequest):
    answer = agent_run(store, request.query, agent_memory)
    return{"query": request.query, "answer": answer}


@app.get("/debug")
def debug():
    return{
        "memory_size": len(agent_memory.memory),
        "sample_memory": agent_memory.memory[-3:] if agent_memory.memory else []
    }