from datasets import load_dataset
import json
import os

# Load category 'c' from BIGPATENT dataset
print("🔄 Loading BIGPATENT category c...")
dataset = load_dataset("big_patent", "c", split="train")

# Take a smaller subset for faster local use (adjust as needed)
MAX_SAMPLES = 10000
dataset = dataset.select(range(MAX_SAMPLES))

# Output directory and file
output_dir = "data/bigpatent_tiny"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "bigpatent_c.jsonl")

# Save in JSON Lines format
print(f"💾 Saving {MAX_SAMPLES} records to {output_path}...")
with open(output_path, "w") as f:
    for record in dataset:
        json.dump(record, f)
        f.write("\n")

print("✅ Dataset saved successfully!")

