import faiss
import numpy as np
import json
import jsonlines
from sentence_transformers import SentenceTransformer

# Load BIGPATENT data
print("🔄 Loading BIGPATENT data...")
with open("data/bigpatent_tiny/bigpatent_c.jsonl", "r") as f:
    data = list(jsonlines.Reader(f))

# Load sentence embedding model
print("🧠 Generating embeddings with SentenceTransformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional output
texts = [item.get("abstract", "") for item in data]
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Create FAISS index
print("📦 Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "data/bigpatent_tiny/faiss.index")

# Save metadata
print("💾 Saving metadata...")
metadata = [{"abstract": item.get("abstract", "[Missing abstract]"),
             "background": item.get("background", "")} for item in data]

with open("data/bigpatent_tiny/faiss_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ FAISS index and metadata saved.")
