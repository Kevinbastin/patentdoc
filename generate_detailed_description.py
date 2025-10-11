from llama_cpp import Llama
import re

LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"
llm = Llama(model_path=LLM_PATH, n_ctx=4096, n_threads=4)

def generate_detailed_description(abstract, claims, drawing_summary):
    """
    Generate a formal, USPTO-compliant detailed description section.
    
    Args:
        abstract: Brief summary of the invention
        claims: Patent claims defining scope of protection
        drawing_summary: Description of figures/drawings
    
    Returns:
        Formatted detailed description text
    """
    
    prompt = f"""You are an expert patent attorney drafting a U.S. utility patent application according to USPTO guidelines (35 U.S.C. § 112).

Generate a comprehensive "DETAILED DESCRIPTION OF THE INVENTION" section that follows standard patent format.

REQUIREMENTS:
- Use formal, technical language with precise terminology
- Include specific reference numerals when discussing components (e.g., "element 10", "component 102")
- Organize into clear paragraphs with logical flow
- Start with background context and problem statement
- Describe the invention's structure and components systematically
- Explain operational principles and method steps
- Include embodiments and variations
- Use present tense ("comprises", "includes", "is configured to")
- Write 5-8 substantial paragraphs (minimum 800 words)

INPUT MATERIALS:

ABSTRACT:
{abstract}

CLAIMS:
{claims}

BRIEF DESCRIPTION OF DRAWINGS:
{drawing_summary}

---

Generate the DETAILED DESCRIPTION OF THE INVENTION section now, following this structure:

DETAILED DESCRIPTION OF THE INVENTION

[Para 1: Field and Background - describe the technical field and existing problems]
[Para 2: Summary of Solution - overview of how the invention addresses these problems]
[Para 3-5: Structural Description - detailed description of components, with reference numerals]
[Para 6-7: Operation and Method - explain how the invention works and operational steps]
[Para 8: Embodiments and Variations - alternative implementations and modifications]

Begin the detailed description:"""

    try:
        response = llm(
            prompt=prompt, 
            max_tokens=1500,  # Increased for more comprehensive output
            temperature=0.6,   # Slightly lower for more consistent, formal output
            top_p=0.9,
            repeat_penalty=1.15,
            stop=["USER:", "ASSISTANT:", "\n\n\n\n"]  # Prevent unwanted continuations
        )
        
        generated_text = response["choices"][0]["text"].strip()
        
        # Post-process the output
        formatted_text = post_process_description(generated_text)
        
        return formatted_text
        
    except Exception as e:
        return f"❌ LLM inference failed: {str(e)}\n\nPlease check model path and system resources."


def post_process_description(text):
    """
    Clean and format the generated description to match patent standards.
    """
    # Remove any repeated section headers
    text = re.sub(r'(DETAILED DESCRIPTION.*?\n){2,}', 'DETAILED DESCRIPTION OF THE INVENTION\n\n', text, flags=re.IGNORECASE)
    
    # Ensure proper paragraph spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Add section header if missing
    if not text.upper().startswith('DETAILED DESCRIPTION'):
        text = "DETAILED DESCRIPTION OF THE INVENTION\n\n" + text
    
    # Format reference numerals consistently (e.g., "element 10" or "component 102")
    text = re.sub(r'\b(\w+)\s+(\d{1,3})\b', r'\1 \2', text)
    
    return text


def generate_with_context(abstract, claims, drawing_summary, field_of_invention="", background=""):
    """
    Enhanced version that accepts additional context for more accurate generation.
    """
    context_section = ""
    if field_of_invention:
        context_section += f"\nFIELD OF INVENTION:\n{field_of_invention}\n"
    if background:
        context_section += f"\nBACKGROUND:\n{background}\n"
    
    prompt = f"""You are an expert patent attorney drafting a U.S. utility patent application.

Generate a comprehensive "DETAILED DESCRIPTION OF THE INVENTION" section following USPTO standards.
{context_section}

ABSTRACT:
{abstract}

CLAIMS:
{claims}

DRAWINGS:
{drawing_summary}

Write a formal, technically precise detailed description (800-1200 words) with:
- Background and technical field
- Problem statement and prior art limitations
- Summary of the inventive solution
- Detailed structural description with reference numerals
- Operational principles and method steps
- Alternative embodiments and variations
- Advantages and benefits

DETAILED DESCRIPTION OF THE INVENTION:"""

    try:
        response = llm(
            prompt=prompt,
            max_tokens=1800,
            temperature=0.6,
            top_p=0.9,
            repeat_penalty=1.15
        )
        
        return post_process_description(response["choices"][0]["text"].strip())
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Example usage
if __name__ == "__main__":
    sample_abstract = """A smart irrigation system comprising soil moisture sensors, 
    a microcontroller unit, and wireless communication modules. The system automatically 
    adjusts water delivery based on real-time soil conditions and weather data."""
    
    sample_claims = """1. An irrigation system comprising: a plurality of soil moisture 
    sensors; a central processing unit configured to receive sensor data; and a valve 
    control mechanism responsive to the processing unit."""
    
    sample_drawings = """Figure 1 shows an overview of the irrigation system. Figure 2 
    illustrates the sensor array configuration. Figure 3 depicts the control flow diagram."""
    
    result = generate_detailed_description(sample_abstract, sample_claims, sample_drawings)
    print(result)