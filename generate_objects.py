from llama_cpp import Llama
import re


# Path to your local Phi-3 model
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"


# Load the model once
llm = Llama(model_path=LLM_PATH, n_ctx=2048, n_threads=4)


def clean_objects(text):
    """Clean and format the generated objects section according to patent standards."""
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove any repeated section headers
    text = re.sub(r'(OBJECTS OF THE INVENTION.*?\n){2,}', 'OBJECTS OF THE INVENTION\n\n', text, flags=re.IGNORECASE)
    
    # Remove duplicate "primary object" paragraphs
    lines = text.split('\n')
    seen_primary = False
    cleaned_lines = []
    skip_until_blank = False
    
    for line in lines:
        if 'primary object' in line.lower() and seen_primary:
            skip_until_blank = True
            continue
        if skip_until_blank:
            if line.strip() == '':
                skip_until_blank = False
            continue
        if 'primary object' in line.lower():
            seen_primary = True
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Add section header if missing
    if not text.upper().startswith('OBJECTS OF THE INVENTION'):
        text = "OBJECTS OF THE INVENTION\n\n" + text
    
    # Ensure proper spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Standardize bullet points
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '• ', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    
    # Remove === markers
    text = text.replace('===', '').strip()
    
    return text.strip()


def generate_objects_of_invention(abstract, max_attempts=3):
    """
    Generate OBJECTS OF THE INVENTION section from abstract.
    
    Args:
        abstract: Brief summary of the invention
        max_attempts: Number of generation attempts if needed
    
    Returns:
        Formatted objects section (without line numbers for Streamlit display)
    """
    
    prompt = f"""You are an expert patent attorney drafting a U.S. utility patent application.

Generate a professional "OBJECTS OF THE INVENTION" section for this invention.

INVENTION ABSTRACT:
{abstract}

REQUIREMENTS:
1. Start with EXACTLY ONE paragraph: "The primary object of the present invention is to provide..."
2. Follow with EXACTLY ONE line: "More specifically, the invention is aimed to:"
3. List 5-7 specific objectives as bullet points
4. Use technical, formal patent language
5. Focus on problems solved and benefits achieved
6. Each objective should be 1-2 lines maximum
7. Keep objectives concise and specific
8. DO NOT repeat the primary object paragraph
9. DO NOT use markdown headers or formatting

EXAMPLE FORMAT:
The primary object of the present invention is to provide a comprehensive system for monitoring and managing water consumption habits with enhanced accuracy and user engagement.

More specifically, the invention is aimed to:
• Provide real-time monitoring of water temperature and consumption levels
• Enable automated hydration tracking and goal management
• Deliver personalized hydration reminders and notifications
• Facilitate seamless wireless communication with mobile devices
• Ensure optimal water temperature maintenance through intelligent control
• Enhance user engagement through intuitive status indicators

Generate only the objects content (no section header, no explanations):"""

    for attempt in range(max_attempts):
        response = llm(
            prompt=prompt,
            max_tokens=600,
            temperature=0.3 if attempt == 0 else 0.5 + (attempt * 0.1),
            stop=["BRIEF DESCRIPTION", "SUMMARY", "BACKGROUND", "DETAILED", "\n\n\n\n"],
            top_p=0.9,
            repeat_penalty=1.15
        )
        
        raw_text = response["choices"][0]["text"].strip()
        cleaned_text = clean_objects(raw_text)
        
        # Check if we got reasonable output
        if len(cleaned_text) > 100 and "primary object" in cleaned_text.lower():
            return cleaned_text
    
    # If all attempts fail, return the last attempt
    return cleaned_text


# Example CLI usage for manual testing
if __name__ == "__main__":
    print("=" * 70)
    print("         PATENT OBJECTS OF INVENTION GENERATOR")
    print("=" * 70)
    print("\n📥 Enter the invention abstract (press Enter twice to finish):")
    print("-" * 70)
    
    lines = []
    while True:
        line = input()
        if line.strip() == "" and lines:
            break
        if line.strip() != "":
            lines.append(line)
    
    abstract = " ".join(lines).strip()
    
    if not abstract:
        print("\n❌ No abstract provided. Exiting.")
        exit(1)
    
    print("\n⏳ Generating OBJECTS OF THE INVENTION section...")
    print("-" * 70)
    
    result = generate_objects_of_invention(abstract)
    
    print("\n📋 GENERATED OUTPUT:")
    print("=" * 70)
    print(result)
    print("=" * 70)
    print("\n✅ Generation complete!")
