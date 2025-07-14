# config.py

import os

# --- PDF Configuration ---
PDF_URL = "https://www.aetnabetterhealth.com/content/dam/aetna/medicaid/illinois/pdf/ABHIL_Member_Handbook.pdf"
LOCAL_PDF_PATH = "ABHIL_Member_Handbook.pdf"

# --- ChromaDB Configuration ---
CHROMA_DB_DIR = "./chroma_db" # Directory to store ChromaDB persistent data

# --- Chunking Strategy Parameters ---
CHUNK_SIZE = 1000   # Max characters per chunk. ~750 tokens for English.
CHUNK_OVERLAP = 100 # Overlap between chunks to maintain context

# --- OpenAI Model Configuration ---
EMBEDDING_MODEL_NAME = "text-embedding-ada-002" 
CHAT_MODEL_NAME = "gpt-3.5-turbo" # 'gpt-4o' for higher quality, but more expensive
TEMPERATURE = 0.7 # Controls randomness of LLM output (0.0 for deterministic, 1.0 for creative)

# --- Reranking Configuration (Optional) ---
# If you want to use Cohere Rerank, uncomment the line below and ensure COHERE_API_KEY is set in .env
# RERANK_MODEL_NAME = "rerank-english-v3.0"
# RERANK_TOP_N = 3 # Number of top documents to return after reranking