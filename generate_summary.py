from llama_cpp import Llama

# Path to your local Phi-3 GGUF model
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load the model
llm = Llama(model_path=LLM_PATH)

def summarize_abstract(abstract: str) -> str:
    prompt = f"""You are a summarization expert.
Summarize the following invention abstract in 3-4 clear, informative sentences.

Abstract:
{abstract}

Summary:"""

    response = llm(prompt, max_tokens=200, temperature=0.7)
    return response["choices"][0]["text"].strip()
