import os
from datasets import load_dataset

# 🐍 --- Script to Download BigPatent Dataset (Corrected) --- 🐍

# Specify the dataset name and the desired configuration
dataset_name = "big_patent"
config_name = 'a'

try:
    print(f"⬇️  Starting download for '{dataset_name}' with configuration '{config_name}'...")
    print("This will take a while depending on your internet connection.")

    # ✅ MODIFIED LINE: Added 'trust_remote_code=True'
    # This is now required for datasets that have a loading script.
    dataset = load_dataset(dataset_name, config_name, trust_remote_code=True)

    print("\n✅ Download and loading complete!")
    print("Dataset information:")
    print(dataset)

    # Optional: Print the path where the dataset is cached
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache/huggingface/datasets')
    print(f"\n💾 Dataset cached at: {cache_dir}")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("Please check your internet connection, available disk space, and library versions.")