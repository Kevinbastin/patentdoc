from llama_cpp import Llama

# Path to your locally downloaded Phi-3 model (.gguf file)
LLM_PATH = "/home/urk23cs7081/patentdoc-copilot/models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load the model ONCE at module level (not inside the function)
llm = Llama(model_path=LLM_PATH, n_ctx=4096, n_threads=4)  # Increase n_threads if your CPU has more cores

def generate_summary(abstract: str) -> str:
    """
    Generates the 'Summary of Accompanying Drawings' section for a U.S. patent
    using a local Phi-3 model and an invention abstract.
    """
    prompt = (
        "You are a helpful assistant that writes a professional 'Summary of the Accompanying Drawings' section "
        "for a U.S. patent application based on the invention abstract provided below.\n\n"
        f"Invention Abstract:\n{abstract.strip()}\n\n"
        "Summary of Accompanying Drawings:"
    )
    try:
        response = llm(prompt=prompt, max_tokens=300, temperature=0.6)
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"❌ LLM inference failed: {str(e)}"

# Optional CLI interface for manual testing
if __name__ == "__main__":
    print("📥 Enter the invention abstract:")
    abstract = input("> ").strip()
    print("\n🖼️ Generating drawing summary...\n")
    result = generate_summary(abstract)
    print("\n📑 Summary of Accompanying Drawings:\n")
    print(result)
