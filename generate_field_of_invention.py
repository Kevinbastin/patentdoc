from llama_cpp import Llama

# Path to your local GGUF model (adjust if needed)
LLM_PATH = "/home/urk23cs7081/patentdoc-copilot/models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load the model once
llm = Llama(model_path=LLM_PATH, n_threads=4)  # Increase n_threads if your CPU supports more

def generate_field_of_invention(abstract: str) -> str:
    """
    Generates the 'Field of the Invention' section from an abstract using a local Phi-3 model.
    """
    prompt = (
        "You are a patent assistant. Given the invention abstract below, "
        "write the 'Field of the Invention' section for a US patent using formal technical language.\n\n"
        f"Abstract:\n{abstract.strip()}\n\n"
        "Field of the Invention:"
    )

    response = llm(prompt, max_tokens=250, temperature=0.5)
    return response["choices"][0]["text"].strip()

# CLI for testing locally
if __name__ == "__main__":
    print("\n📥 Enter the invention abstract:")
    abstract = input(" > ")

    print("\n🛠️ Generating 'Field of the Invention'...\n")
    result = generate_field_of_invention(abstract)

    print("\n📘 Field of the Invention:\n")
    print(result)
