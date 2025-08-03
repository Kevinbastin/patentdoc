import os
import faiss
import json
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_PATH = os.path.join(BASE_DIR, "models", "models", "phi-3-mini-4k-instruct-q4.gguf")
INDEX_PATH = os.path.join(BASE_DIR, "data", "bigpatent_tiny", "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "data", "bigpatent_tiny", "faiss_metadata.json")

# === Load models ONCE (do NOT reload in every function call) ===
llm = Llama(model_path=LLM_PATH, n_ctx=4096, n_threads=4)  # Increase n_threads for faster CPU generation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Load FAISS index and metadata ===
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

def generate_claims_from_abstract(abstract: str, top_k: int = 3) -> str:
    """
    Generate 5 patent claims (1 independent, 4 dependent) based on input abstract and similar prior art.
    """
    try:
        # Encode abstract to embedding
        query_embedding = embedding_model.encode([abstract], convert_to_numpy=True)
        D, I = index.search(query_embedding, top_k)
    except Exception as e:
        return f"Error during FAISS search: {str(e)}"
    
    prior_abstracts = "\n".join(
        f"- {metadata[idx]['abstract']}" for idx in I[0] if idx < len(metadata)
    )
    prompt = f"""You are a helpful patent assistant.
Based on the invention abstract and prior art below, generate 5 patent claims:
1 independent claim and 4 dependent claims. Format as a numbered list.

Abstract:
{abstract}

Prior Art:
{prior_abstracts}

Claims:"""
    try:
        output = llm(prompt=prompt, max_tokens=512, temperature=0.7)
        return output["choices"][0]["text"].strip()
    except Exception as e:
        return f"Error during model inference: {str(e)}"
