from llama_cpp import Llama
import re
from typing import Dict, List

# Path to your local GGUF model
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load the model once with optimized settings
llm = Llama(
    model_path=LLM_PATH,
    n_ctx=2048,
    n_threads=4,
    n_batch=512
)

def extract_technical_domain(abstract: str) -> str:
    """Extract the primary technical domain from the abstract."""
    common_domains = [
        "computer science", "electrical engineering", "mechanical engineering",
        "biotechnology", "pharmaceutical", "telecommunications", "artificial intelligence",
        "machine learning", "medical devices", "semiconductor", "software",
        "chemical engineering", "materials science", "automotive", "aerospace"
    ]
    
    abstract_lower = abstract.lower()
    for domain in common_domains:
        if domain in abstract_lower:
            return domain
    return "technology"

def clean_field_text(text: str) -> str:
    """Clean and format the generated field of invention text."""
    # Remove common prefixes
    text = re.sub(r'^(Field of the Invention:|Field of Invention:|This invention relates to)\s*', '', text, flags=re.IGNORECASE)
    
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Ensure proper sentence structure
    text = text.strip()
    
    # Capitalize first letter
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    
    # Ensure it ends with a period
    if text and not text.endswith('.'):
        text += '.'
    
    return text

def validate_field_text(text: str) -> Dict[str, any]:
    """Validate the field of invention text against patent standards."""
    issues = []
    word_count = len(text.split())
    sentence_count = len(re.findall(r'[.!?]+', text))
    
    # Check length (typically 1-3 sentences, 20-100 words)
    if word_count < 15:
        issues.append("Too brief. Field of Invention should be more descriptive (15-100 words).")
    elif word_count > 150:
        issues.append("Too lengthy. Field of Invention should be concise (15-100 words).")
    
    # Check for proper sentence structure
    if sentence_count == 0:
        issues.append("Missing proper sentence ending.")
    
    # Check for required phrases
    standard_phrases = [
        "relates to", "pertains to", "concerns", "directed to",
        "field of", "area of", "relates generally to"
    ]
    has_standard_phrase = any(phrase in text.lower() for phrase in standard_phrases)
    
    if not has_standard_phrase:
        issues.append("Consider using standard patent language (e.g., 'relates to', 'pertains to').")
    
    # Check for marketing language (should be avoided)
    marketing_words = ["revolutionary", "groundbreaking", "innovative", "novel", "unique", "best"]
    found_marketing = [word for word in marketing_words if word in text.lower()]
    if found_marketing:
        issues.append(f"Avoid marketing language: {', '.join(found_marketing)}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "word_count": word_count,
        "sentence_count": sentence_count
    }

def generate_field_of_invention(abstract: str, max_attempts: int = 2) -> Dict[str, any]:
    """
    Generates the 'Field of the Invention' section from an abstract using a local Phi-3 model.
    
    Args:
        abstract: The patent abstract text
        max_attempts: Number of generation attempts if validation fails
        
    Returns:
        Dictionary containing the generated field text and metadata
    """
    
    technical_domain = extract_technical_domain(abstract)
    
    # Enhanced prompt with patent-specific instructions and examples
    prompt = f"""You are an expert patent attorney drafting a US utility patent application.

TASK: Write ONLY the "Field of the Invention" section based on the abstract below.

REQUIREMENTS:
1. Use formal, technical language (third person, present tense)
2. Start with standard phrases like:
   - "The present invention relates to..."
   - "This invention pertains to..."
   - "The present disclosure relates generally to..."
3. Be concise: 1-3 sentences, 20-80 words
4. State the technical field broadly, then narrow to specific area
5. Do NOT include technical details, claims, or advantages
6. Do NOT use marketing language (novel, innovative, etc.)
7. Do NOT repeat the abstract verbatim

GOOD EXAMPLES:
- "The present invention relates generally to wireless communication systems, and more particularly to methods and apparatus for improving signal transmission in 5G networks."
- "This invention pertains to the field of medical imaging, specifically to enhanced MRI scanning techniques for early disease detection."
- "The present disclosure relates to artificial intelligence systems, and more specifically to machine learning models for natural language processing."

ABSTRACT:
{abstract.strip()}

Now write ONLY the Field of the Invention section (no headings, no explanations):"""

    best_result = None
    
    for attempt in range(max_attempts):
        response = llm(
            prompt=prompt,
            max_tokens=200,
            temperature=0.4 if attempt == 0 else 0.6,
            stop=["\n\n", "Background:", "Summary:", "BACKGROUND"],
            top_p=0.9,
            repeat_penalty=1.15
        )
        
        raw_text = response["choices"][0]["text"].strip()
        cleaned_text = clean_field_text(raw_text)
        validation = validate_field_text(cleaned_text)
        
        result = {
            "text": cleaned_text,
            "valid": validation["valid"],
            "issues": validation["issues"],
            "word_count": validation["word_count"],
            "sentence_count": validation["sentence_count"],
            "attempt": attempt + 1,
            "technical_domain": technical_domain
        }
        
        if validation["valid"]:
            return result
        
        # Keep track of best attempt
        if best_result is None or len(validation["issues"]) < len(best_result["issues"]):
            best_result = result
    
    return best_result

def format_for_patent(field_text: str, paragraph_number: str = "[0001]") -> str:
    """Format the field text with standard patent paragraph numbering."""
    return f"{paragraph_number} {field_text}"

def generate_variations(abstract: str) -> List[str]:
    """Generate multiple variations of the field of invention."""
    variations = []
    
    # Variation 1: Broad to narrow
    prompt1 = f"""Write a Field of Invention that starts broad then narrows. 
Abstract: {abstract}
Start with "The present invention relates generally to..." """
    
    # Variation 2: Specific focus
    prompt2 = f"""Write a Field of Invention focusing on the specific application.
Abstract: {abstract}
Start with "This invention pertains to..." """
    
    for prompt in [prompt1, prompt2]:
        response = llm(prompt=prompt, max_tokens=150, temperature=0.5, stop=["\n\n"])
        cleaned = clean_field_text(response["choices"][0]["text"].strip())
        variations.append(cleaned)
    
    return variations

# CLI for testing locally
if __name__ == "__main__":
    print("=" * 80)
    print("         PATENT FIELD OF INVENTION GENERATOR")
    print("=" * 80)
    print("\n📥 Enter the invention abstract (press Enter twice to finish):")
    print("-" * 80)
    
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
    
    print("\n🛠️  Generating 'Field of the Invention'...")
    print("-" * 80)
    
    result = generate_field_of_invention(abstract)
    
    print("\n📘 FIELD OF THE INVENTION")
    print("=" * 80)
    
    # Display validation status
    if result["valid"]:
        print("✅ Status: Valid")
    else:
        print("⚠️  Status: Needs Review")
        print("\n🔍 Issues Found:")
        for issue in result["issues"]:
            print(f"   • {issue}")
    
    print(f"\n📊 Statistics:")
    print(f"   • Word Count: {result['word_count']}")
    print(f"   • Sentences: {result['sentence_count']}")
    print(f"   • Technical Domain: {result['technical_domain']}")
    print(f"   • Generation Attempts: {result['attempt']}")
    
    print("\n" + "─" * 80)
    print(result["text"])
    print("─" * 80)
    
    print("\n📄 FORMATTED FOR PATENT (with paragraph numbering):")
    print("-" * 80)
    print(format_for_patent(result["text"]))
    print("-" * 80)
    
    # Optional: Generate variations
    print("\n🔄 Generate alternative versions? (y/n): ", end="")
    if input().lower() == 'y':
        print("\n⏳ Generating variations...")
        variations = generate_variations(abstract)
        print("\n📝 ALTERNATIVE VERSIONS:")
        print("=" * 80)
        for i, var in enumerate(variations, 1):
            print(f"\nVariation {i}:")
            print("-" * 80)
            print(var)
            print("-" * 80)
    
    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)