import os
import numpy as np # type: ignore
import faiss # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sentence_transformers import CrossEncoder # type: ignore
from openai import OpenAI  # type: ignore
#from transformers import pipeline # type: ignore 
import requests # type: ignore
import random
from rank_bm25 import BM25Okapi
import math
import os
import pickle
import nltk # type: ignore
from nltk.tokenize import sent_tokenize # type: ignore
import config # type: ignore



data_path = "data/docs"
embed_model_name = "all-MiniLM-L6-v2"
llm_model =  "gpt-4o-mini"
#cross_encoder_model = 


#init
embed_model = SentenceTransformer("BAAI/bge-small-en")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



all_chunks = []
TOOL_STATUS = {}
query_embedding_cache = {}
bm25 = None # we set it where we read the whole documents.

def validate_openai():
    global USE_OPENAI, client

    key = os.getenv("OPENAI_API_KEY")

    if not key:
        print("No OpenAI key found")
        USE_OPENAI = False
        client = None
        return False

    client = OpenAI(api_key=key)

    try:
        client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        print("OpenAI key WORKS")
        USE_OPENAI = True
        return True

    except Exception as e:
        print(f"OpenAI key INVALID: {e}")
        USE_OPENAI = False
        return False



#ingestion
def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file),"r", encoding ="utf-8") as f:
            docs.append(f.read())
    return docs


#chunking

#in case we have big paragraphs or sentences.
def fallback_chunk(text, chunk_size=150, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def chunk_text(text, chunk_size=150, overlap =2):
    sentences = sent_tokenize(text)

    if len(sentences)<5:
        print("Using fallback chunking.")
        return fallback_chunk(text)

    chunks=[]
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks






#vecot store
class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts =[]
        self.embeddings = None

        
    def add(self, embeddings, texts):
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

        if self.embeddings is None:
           self.embeddings=embeddings
        else:
            self.embeddings = np.vstack([self.embeddings,embeddings])
           
   
    #we modify the search toreturn store
    def search (self, query_embedding, k=3):
        D, I = self.index.search(np.array([query_embedding]).astype('float32'),k)

        results = []
        for idx, dist in zip(I[0],D[0]):
            results.append((dist, self.texts[idx]))

        return results    

#Embedding
def embed_chunks(chunks):
    return embed_model.encode(chunks)



def get_query_embedding(query):
    if query in query_embedding_cache:
        return query_embedding_cache[query]
    emb = embed_model.encode([query])[0]  # compute embedding
    query_embedding_cache[query] = emb    # store in cache
    return emb

#extract confidence
def retrieve_with_scores(store, query):
    query_embedding = get_query_embedding(query)
    return store.search(query_embedding, k=10)

#Confidence Metric : lowest disctance =the best


def compute_confidence_cross(query, chunks):
    pairs = [(query, c) for c in chunks]
    scores = cross_encoder.predict(pairs)
    #normalize 
    if  scores.size ==0:
        confidence = 0.0
    elif scores.size==1:
         confidence= 1/(1 + math.exp(-scores[0]))
    else:
        sorted_scores = np.sort(scores)[::-1] #descending
        confidence= 1/(1+math.exp(-(sorted_scores[0]-sorted_scores[1])))
       
    # we could aslo use the differenc of the two best scores
    #score = sorted(scores, reverse=True)
    #confidence = scores[0]-scores[1] #big gap = stromng match, small gap = ambiguous
    return confidence


def build_rag_in(folder):
    chunks =[]
    print("Loading Documnets...")
    docs = load_documents(folder)

    print("Chunking...")
    for doc in docs:
        chunks.extend(chunk_text(doc))

    print(f"Total chunks: {len(chunks)}")
    # print(f"Number of documents: {len(docs)}")

    print ("Creating Embeddings...")
    embeddings = embed_chunks(chunks)  

    print("Building vecot store...") 
    store = VectorStore(len(embeddings[0]))
    store.add(embeddings, chunks)


    return store

def build_rag(folder):
    global bm25, all_chunks
    ans = input("Use the existing embedin? (y/n)")
    if ans.lower() == "y":
        try :
            with open("vector_space.pkl","rb") as f:
                store = pickle.load(f)
                print("Loaded existing vector store.")
        except Exception as es :
                print("Failed to load, building new store...")
                store = build_rag_in(folder)
                with open("vector_space.pkl","wb") as f:
                    pickle.dump(store,f)
                    print("Saved vector store.")
    else:
        store = build_rag_in(folder)
        with open("vector_space.pkl","wb") as f:
            pickle.dump(store,f)
            print("Saved vector store.")
    print("Rebuilding BM25 + chunks...")

    docs = load_documents(folder)

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    bm25 = BM25Okapi([c.lower().split() for c in all_chunks])

    print(f"BM25 ready with {len(all_chunks)} chunks.")

    return store




#Clean Pipeline : modular,testable and scalable
def retrieve (store, query):
    query_embedding = get_query_embedding(query)
    return store.search(query_embedding,k=15)

def rerank_chunks(query, chunks, top_k=5):
    query_embedding = get_query_embedding(query)
    chunks_embedgings = embed_model.encode(chunks)
    scores = []
    for i , emb in enumerate(chunks_embedgings):
        score = np.dot(query_embedding,emb) #coside-like : inds similarity
        scores.append((score, chunks[i]))
    #sort by score descending
    scores.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scores[:top_k]]

def rerank_chunks_cosine(store, query, chunks, top_k=10):
    query_emb = get_query_embedding(query)
#    chunk_indices = [store.texts.index(c) for c in chunks]
    chunk_indices = [store.texts.index(c) if c in store.texts else 0 for c in chunks]
    chunks_embs = store.embeddings[chunk_indices]

    scores = np.dot(chunks_embs, query_emb) #cosine like if normalized
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def rerank_cross_encoder(query, chunks, top_k=5):
    pairs = [(query, chunk) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    scored = list(zip(scores, chunks)) # combines scores and chunks
    scored.sort(reverse=True, key=lambda x:x[0])

    return [c for _, c in scored[:top_k]]

#clean HYbrid retrieval
def hybrid_retrieve(store, bm25, chunks, query, k=10):
    #BM25
    tokeniezd_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokeniezd_query)
    bm25_top_indices = np.argsort(bm25_scores)[-50:]

    #e=Embedding
    query_embedding = get_query_embedding(query)
    D,I = store.index.search(np.array([query_embedding]),50)
    emb_top_indices = I[0]
    
    # merge candidates
    candidate_indices = set(bm25_top_indices) | set(emb_top_indices)
    candidates = [chunks[i] for i in candidate_indices]
    return candidates






def build_context(chunks):
    return "\n\n".join(chunks)
    

#generate
def generate_answer_openai(prompt):
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


def generate_answer_local(prompt):
    try:
        print(f"[DEBUG] Prompt length: {len(prompt)}")
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model" : "phi3",
                "prompt" : prompt,
                "stream" : False
            },
            timeout=180
        )
        return response.json().get("response", "")
    except Exception as e:
        print(f"[Ollama ERROR]: {e}")
        return "Local model not available."


def generate_answer(query, context):
    prompt = f"""

You are a helpful AI assistant.
    
Answer the question based only on the provided context.
If unsure, give your best possible answer based on the context.

Context:
{context[:1500]}

Question:
{query}

Answer:
"""
    if USE_OPENAI:
        try: 
            return generate_answer_openai(prompt)
        except Exception as e:
            print(f"Switching to Local. OpenAI Error : {str(e)}")
            return generate_answer_local(prompt)
    else : 
            return generate_answer_local(prompt)


class AgentMemory:
    def __init__(self,max_len=20):
        self.memory = [] #list of (query,context)
        self.max_len = max_len

    def add(self, query, context):
        self.memory.append((query,context))
        if len(self.memory)> self.max_len:
            self.memory = self.memory[-self.max_len:]   

    def get_context(self):
        """
        Combine memory into a single string for RAG/LLM context. 
        Optional: summarization can be added here if memory grows too long.
        """
        context_lines=[]
        for q, a in self.memory:
            context_lines.append(f"Q:{q}\n A:{a}")
        return "\n\n".join(context_lines)
    
 
def combine_memory_context(agent_memory, current_context):
        if agent_memory.memory: #empty
           context_with_memory = agent_memory.get_context() + "\n\n" + current_context 
        else: context_with_memory  =current_context
        return context_with_memory


def is_bad_answer(answer):
    bad_patterns = [
        "i do not know",
        "not sure",
        "no information",
        "not in the context",
        ""
    ]
    answer_lower = answer.lower()
    return any(p in answer.lower for p in bad_patterns)



#we add a web tool



def check_tool_health():
    status = {
        "openai": False,
        "web": False,
        "local": False
    }

    # OpenAI
    if USE_OPENAI:
        status["openai"] = True

    # Web
    try:
        r = requests.get(
            "https://duckduckgo.com",
            timeout=5
        )
        status["web"] = r.ok
    except Exception as e:
        print(f"[HealthCheck] Web unavailable: {e}")
        status["web"] = False

    # Local (Ollama)
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if r.status_code == 200:
            status["local"] = True
    except Exception as e:
        print(f"[HealthCheck] Local model unavailable: {e}")

    return status




def tool_web_search(query):
    print("[Tool]: Web Search")

    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url, timeout=10).json()

        if "AbstractText" in response and response["AbstractText"]:
            return response["AbstractText"]

        elif "RelatedTopics" in response and len(response["RelatedTopics"]) > 0:
            return response["RelatedTopics"][0].get("Text", "")

        else:
            return "No useful web result found."

    except Exception as e:
        return f"Web search failed: {str(e)}"




#Multi-step agent
def agent_run(store, query,agent_memory):
    # step 1: retrieve candidate chunks
    candidates = hybrid_retrieve(store, bm25, all_chunks, query)
   # candidates = candidates[:20]

    # step 2: compute confidence based on embedding similarity only
    top_chunks = rerank_chunks_cosine(store, query, candidates, top_k=5)

   # confidence : highest cosine similarity of top chunks
    query_embd = embed_model.encode([query])[0]
    top_embds = embed_model.encode(top_chunks)

    sim = [np.dot(query_embd,c_embd) for c_embd in top_embds]
    confidence = max(sim) if sim else 0.0
   # confidence = compute_confidence_cross(query, candidates[:5])
    print(f"[Confidence]: {confidence:.3f}")

    #step 3: build context from top chunks + memory (RAG)
    if confidence > 0.2 and top_chunks:
        print (f"[Agent decision]: RAG")
        cosine_top = rerank_chunks_cosine(store, query,candidates,top_k=10)
        reranked = rerank_cross_encoder(query,cosine_top, top_k=3)
       # reranked = top_chunks
        current_context = build_context(reranked)
        #prepend memory safely
        context_with_memory = combine_memory_context(agent_memory,current_context)
    #step4 : web search
    else :
        print("[Agent decision]: LOW CONFIDENCE ROUTING")
        context_with_memory = None
        if USE_OPENAI and TOOL_STATUS.get("openai"):
            print("[ROUTING]: OpenAI fallback")
            try: 
                context_with_memory = generate_answer_openai(query)
            except Exception as e:
                print(f"[OpenAI failed]: {e}")
        if not context_with_memory and TOOL_STATUS.get("web"):
            print("[Routing]: Web tool")

            web_result = tool_web_search(query)
            web_context = f"""
            Web search result:
            {web_result}

            Use this information to answer the question.
            """

            context_with_memory = combine_memory_context(agent_memory, web_context)




        if not context_with_memory and TOOL_STATUS.get("local"):
            print("[Routing]: Local LLM fallback")

            fallback = build_context(candidates[:5])
            context_with_memory = combine_memory_context(agent_memory, fallback)

        # 4. Absolute fallback
        if not context_with_memory:
            print("[Routing]: NO TOOLS AVAILABLE")

            return "No tools are available (OpenAI, Web, Local all offline)."

    answer = generate_answer(query, context_with_memory)

    agent_memory.add(query, answer)
    return answer   


def initialize_rag():
    global agent_memory, store, TOOL_STATUS

    # validate OpenAI FIRST
    validate_openai()

    agent_memory = AgentMemory(max_len=50)
    store = build_rag(data_path)

    TOOL_STATUS = check_tool_health()

    print(f"[DEBUG] USE_OPENAI: {USE_OPENAI}")
    print(f"[DEBUG] TOOL_STATUS: {TOOL_STATUS}")

    return store, agent_memory




# main
if __name__ == "__main__":
    store , agent_memory = initialize_rag()
    print("\n RAG system ready, ASk question! \n")

    while True:
        query = input("Ask:")
        if query.lower() in ["exit", "quit"]:
            break
        answer = agent_run(store,query, agent_memory)
        print("\n Answer:\n", answer)
        print("\n"+ "-"*50 + "\n")
              


