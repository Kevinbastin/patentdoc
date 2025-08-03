from llama_cpp import Llama

# Path to your local Phi-3 model
LLM_PATH = "/home/urk23cs7081/patentdoc-copilot/models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load the model once
llm = Llama(model_path=LLM_PATH)

def generate_title_from_abstract(abstract: str) -> str:
    prompt = prompt = f"""You are a patent title generator.
Given the following abstract, write a concise patent title in under 12 words.

Abstract:
{abstract}

Title:"""
    response = llm(prompt=prompt, max_tokens=40, temperature=0.5)
    # Return only the title, no abstract, no header
    return response["choices"][0]["text"].strip()

# Example CLI usage for manual testing
if __name__ == "__main__":
    print("📥 Enter the invention abstract:")
    abstract = input("> ").strip()
    title = generate_title_from_abstract(abstract)
    print("\nGenerated Title:\n")
    print(title)
