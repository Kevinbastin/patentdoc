# cpc_classifier.py (LOCAL DistilBERT version)
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import os

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Load CPC label dataset (you can create your own if not available)
CPC_FILE = "data/cpc_labels.json"  # Make sure to place a small CPC code-label list here

if not os.path.exists(CPC_FILE):
    raise FileNotFoundError("❌ CPC label file not found. Please provide 'data/cpc_labels.json'.")

with open(CPC_FILE, "r") as f:
    cpc_data = json.load(f)  # Format: [{"code": "G06F", "description": "Electrical digital data processing"}, ...]

# Precompute CPC embeddings
def encode(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

cpc_embeddings = [(item["code"], item["description"], encode(item["description"])) for item in cpc_data]

# Predict the best match
def classify_cpc(abstract: str):
    abstract_embedding = encode(abstract)
    similarities = [(code, desc, cosine_similarity([abstract_embedding], [embed])[0][0])
                    for code, desc, embed in cpc_embeddings]

    best_match = max(similarities, key=lambda x: x[2])
    return f"[{best_match[0]}] - {best_match[1]}\nReason: Highest semantic similarity score ({best_match[2]:.3f})"
