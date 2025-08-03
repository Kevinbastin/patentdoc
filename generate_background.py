from llama_cpp import Llama

# Path to your GGUF model
LLM_PATH = "./models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load the model ONCE (set n_threads to your CPU logical cores, and n_ctx to handle long prompts)
llm = Llama(model_path=LLM_PATH, n_ctx=4096, n_threads=4)  # Increase n_threads if you have more CPU cores

def generate_background_locally(abstract: str) -> str:
    """
    Generate the 'Background of the Invention' section for a patent application based on the abstract.
    """
    prompt = f"""Write a detailed 'Background of the Invention' section for a patent application based on the following abstract.

Invention Abstract:
{abstract.strip()}

The background should:
- Explain the current state of the field,
- Identify limitations or problems with existing technologies,
- Show how this invention solves or improves upon them.

Use formal, technical language and keep it within ~3 paragraphs.
"""
    try:
        response = llm(prompt=prompt, max_tokens=300, temperature=0.6)
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"❌ LLM inference failed: {str(e)}"

if __name__ == "__main__":
    print("📥 Enter the invention abstract:")
    abstract = input("> ").strip()

    print("\n🧠 Generating 'Background of the Invention'...\n")
    background = generate_background_locally(abstract)

    print("\n📘 Background of the Invention:\n")
    print(background)
