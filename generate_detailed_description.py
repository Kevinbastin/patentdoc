from llama_cpp import Llama

LLM_PATH = "/home/urk23cs7081/patentdoc-copilot/models/models/phi-3-mini-4k-instruct-q4.gguf"
llm = Llama(model_path=LLM_PATH, n_ctx=4096, n_threads=4)

def generate_detailed_description(abstract, claims, drawing_summary):
    prompt = f"""You are a patent assistant AI. Using the provided abstract, claims, and drawing summary, 
generate a formal and technically detailed 'Detailed Description of the Invention' section for a U.S. patent. 
Include a step-by-step explanation in 3-5 paragraphs.

Abstract:
{abstract}

Claims:
{claims}

Drawing Summary:
{drawing_summary}

Detailed Description of the Invention:"""
    try:
        response = llm(prompt=prompt, max_tokens=600, temperature=0.7)
        return response["choices"][0]["text"].strip()
    except Exception as e:
        # Return error as string, not None
        return f"❌ LLM inference failed: {str(e)}"
