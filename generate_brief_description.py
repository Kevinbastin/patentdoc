from llama_cpp import Llama

# Path to your GGUF model
LLM_PATH = "/home/urk23cs7081/patentdoc-copilot/models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load model ONCE globally
llm = Llama(model_path=LLM_PATH, n_ctx=4096, n_threads=4)  # Use more threads if your CPU supports it

def generate_brief_description(abstract: str, drawing_summary: str) -> str:
    prompt = (
        "You are a helpful assistant who writes the 'Brief Description of the Drawings' section for a U.S. patent.\n"
        "Use the given abstract and drawing summary to write a concise and formal brief description.\n\n"
        f"Abstract:\n{abstract.strip()}\n\n"
        f"Drawing Summary:\n{drawing_summary.strip()}\n\n"
        f"Brief Description of the Drawings:"
    )
    try:
        # Lower max_tokens for faster, but increase if needed
        response = llm(prompt=prompt, max_tokens=300, temperature=0.7)
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"❌ LLM inference failed: {str(e)}"

if __name__ == "__main__":
    abstract = input("Abstract:\n> ").strip()
    drawing_summary = input("Drawing summary:\n> ").strip()
    print("\nGenerating brief description...")
    print(generate_brief_description(abstract, drawing_summary))
