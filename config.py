import os

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "docs")

# ===== MODEL PROVIDERS =====
USE_OPENAI = True   # auto-validated later

# ===== MODELS =====
OLLAMA_MODEL = "phi3"
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "BAAI/bge-small-en"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ===== API KEYS =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===== RAG SETTINGS =====
CHUNK_SIZE = 150
CHUNK_OVERLAP = 2

# ===== RETRIEVAL =====
TOP_K = 10
RERANK_TOP_K = 5

# ===== AGENT =====
CONFIDENCE_THRESHOLD = 0.2

# ===== TOOLS =====
ENABLE_WEB = True
WEB_API = "https://api.duckduckgo.com/"

# ===== SERVER =====
HOST = "0.0.0.0"
PORT = 8000
