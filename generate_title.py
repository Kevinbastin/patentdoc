from llama_cpp import Llama
import re

# Path to your local Phi-3 model
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load the model once
llm = Llama(model_path=LLM_PATH, n_ctx=2048, n_threads=4)

def clean_title(title: str) -> str:
    """Clean and format the generated title according to patent standards."""
    # Remove common prefixes that LLMs might add
    title = re.sub(r'^(Title:|Patent Title:|Generated Title:)\s*', '', title, flags=re.IGNORECASE)
    
    # Remove quotes if present
    title = title.strip('"\'')
    
    # Remove any trailing periods (patents titles don't use periods)
    title = title.rstrip('.')
    
    # Capitalize appropriately (title case, but keep technical terms)
    # Patent titles typically use all caps or title case
    title = title.strip()
    
    # Remove extra whitespace
    title = ' '.join(title.split())
    
    return title

def validate_title(title: str) -> tuple[bool, str]:
    """Validate if the title meets patent standards."""
    word_count = len(title.split())
    
    if word_count > 15:
        return False, f"Title too long ({word_count} words). Patent titles should be 5-12 words."
    elif word_count < 3:
        return False, f"Title too short ({word_count} words). Be more descriptive."
    elif title.endswith('.'):
        return False, "Patent titles should not end with a period."
    
    return True, "Valid"

def generate_title_from_abstract(abstract: str, max_attempts: int = 3) -> dict:
    """
    Generate a patent title from an abstract.
    
    Args:
        abstract: The patent abstract text
        max_attempts: Number of regeneration attempts if validation fails
        
    Returns:
        dict with 'title', 'valid', and 'message' keys
    """
    
    # Enhanced prompt with patent-specific instructions
    prompt = f"""You are an expert patent attorney specializing in patent title drafting.

INSTRUCTIONS:
- Create a concise, descriptive patent title (5-12 words maximum)
- Use technical terminology appropriate to the invention
- Focus on the key innovation, not generic descriptions
- Follow USPTO format: describe WHAT it is and WHAT it does
- Do NOT use words like "improved", "novel", "innovative", "efficient"
- Do NOT include articles (a, an, the) at the beginning unless necessary
- Do NOT use periods or ending punctuation
- Use title case or all caps format

GOOD EXAMPLES:
- METHOD AND APPARATUS FOR WIRELESS POWER TRANSMISSION
- ARTIFICIAL INTELLIGENCE SYSTEM FOR MEDICAL DIAGNOSIS
- BIODEGRADABLE POLYMER COMPOSITION FOR PACKAGING
- QUANTUM COMPUTING ARCHITECTURE WITH ERROR CORRECTION

Abstract:
{abstract}

Generate only the patent title (no explanations, no prefix):"""

    for attempt in range(max_attempts):
        response = llm(
            prompt=prompt,
            max_tokens=50,
            temperature=0.3 if attempt == 0 else 0.5 + (attempt * 0.1),
            stop=["\n\n", "Abstract:", "Explanation:"],
            top_p=0.9,
            repeat_penalty=1.1
        )
        
        raw_title = response["choices"][0]["text"].strip()
        cleaned_title = clean_title(raw_title)
        
        is_valid, message = validate_title(cleaned_title)
        
        if is_valid:
            return {
                "title": cleaned_title,
                "valid": True,
                "message": "Title generated successfully",
                "word_count": len(cleaned_title.split()),
                "attempt": attempt + 1
            }
    
    # If all attempts fail, return the best attempt with warning
    return {
        "title": cleaned_title,
        "valid": False,
        "message": message,
        "word_count": len(cleaned_title.split()),
        "attempt": max_attempts
    }

def format_title_variants(title: str) -> dict:
    """Generate different formatting variants of the title."""
    return {
        "uppercase": title.upper(),
        "title_case": title.title(),
        "sentence_case": title.capitalize()
    }

# Example CLI usage for manual testing
if __name__ == "__main__":
    print("=" * 70)
    print("         PATENT TITLE GENERATOR")
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
    
    print("\n⏳ Generating patent title...")
    print("-" * 70)
    
    result = generate_title_from_abstract(abstract)
    
    print("\n📋 GENERATED TITLE:")
    print("=" * 70)
    
    if result["valid"]:
        print(f"✅ Status: Valid")
        print(f"📊 Word Count: {result['word_count']}")
        print(f"🔄 Attempts: {result['attempt']}")
    else:
        print(f"⚠️  Status: {result['message']}")
        print(f"📊 Word Count: {result['word_count']}")
    
    print("\n" + "─" * 70)
    print(result["title"])
    print("─" * 70)
    
    # Show formatting variants
    variants = format_title_variants(result["title"])
    print("\n📝 FORMATTING OPTIONS:")
    print("-" * 70)
    print(f"UPPERCASE:     {variants['uppercase']}")
    print(f"Title Case:    {variants['title_case']}")
    print(f"Sentence case: {variants['sentence_case']}")
    print("=" * 70)