# match_drawings.py
import json
import torch
from sentence_transformers import SentenceTransformer, util

# Load the claim (could be passed dynamically later)
claim = input("🔍 Enter a patent claim to match drawings:\n> ")

# Load captions
with open("data/drawing_captions.json", "r") as f:
    captions = json.load(f)

# Load model
print("🤖 Loading embedding model (MiniLM)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode
claim_embedding = model.encode(claim, convert_to_tensor=True)
caption_embeddings = model.encode(captions, convert_to_tensor=True)

# Compute similarity
cos_scores = util.pytorch_cos_sim(claim_embedding, caption_embeddings)[0]
top_results = torch.topk(cos_scores, k=3)

print("\n📸 Top 3 matched drawing captions:\n")
for score, idx in zip(top_results.values, top_results.indices):
    print(f"{captions[idx]} (Score: {score:.4f})")
