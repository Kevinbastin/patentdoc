from llama_cpp import Llama
import re
from typing import Dict, List, Tuple

# Path to your GGUF model
LLM_PATH = "./models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load the model ONCE with optimized settings
llm = Llama(
    model_path=LLM_PATH,
    n_ctx=4096,
    n_threads=4,
    n_batch=512,
    verbose=False
)

def identify_technical_problems(abstract: str) -> List[str]:
    """Extract potential technical problems from the abstract."""
    problem_keywords = [
        "inefficient", "slow", "complex", "difficult", "expensive",
        "limited", "inadequate", "unreliable", "inaccurate", "problem",
        "challenge", "drawback", "disadvantage", "lack", "unable"
    ]
    
    problems = []
    sentences = re.split(r'[.!?]+', abstract)
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in problem_keywords):
            problems.append(sentence.strip())
    
    return problems

def structure_background_sections(text: str) -> Dict[str, str]:
    """Identify and structure the background into logical sections."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Try to categorize paragraphs
    sections = {
        "state_of_art": "",
        "problems": "",
        "need": ""
    }
    
    for para in paragraphs:
        para_lower = para.lower()
        if any(word in para_lower for word in ["current", "existing", "conventional", "traditional", "typically"]):
            sections["state_of_art"] = para
        elif any(word in para_lower for word in ["problem", "limitation", "drawback", "disadvantage", "difficulty"]):
            sections["problems"] = para
        elif any(word in para_lower for word in ["need", "desire", "would be", "therefore"]):
            sections["need"] = para
    
    return sections

def clean_background_text(text: str) -> str:
    """Clean and format the generated background text."""
    # Remove common prefixes/headers
    text = re.sub(r'^(Background of the Invention:|Background:|BACKGROUND)\s*', '', text, flags=re.IGNORECASE)
    
    # Remove paragraph numbering if LLM added it
    text = re.sub(r'^\[\d+\]\s*', '', text, flags=re.MULTILINE)
    
    # Ensure proper paragraph separation
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove extra spaces
    text = re.sub(r' +', ' ', text)
    
    # Ensure paragraphs start with capital letters
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para and not para[0].isupper():
            para = para[0].upper() + para[1:]
        if para and not para.endswith('.'):
            para += '.'
        if para:
            cleaned_paragraphs.append(para)
    
    return '\n\n'.join(cleaned_paragraphs)

def validate_background(text: str) -> Dict[str, any]:
    """Validate background section against patent drafting standards."""
    issues = []
    warnings = []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    word_count = len(text.split())
    
    # Check length (USPTO typically expects 200-800 words, 2-5 paragraphs)
    if word_count < 150:
        issues.append("Background too brief. Should be 200-800 words for adequate context.")
    elif word_count > 1000:
        warnings.append("Background is lengthy (>1000 words). Consider condensing to 200-800 words.")
    
    # Check paragraph count
    if len(paragraphs) < 2:
        issues.append("Background should have 2-5 paragraphs covering: state of art, problems, and need.")
    elif len(paragraphs) > 6:
        warnings.append("Background has many paragraphs. Consider consolidating to 2-5 paragraphs.")
    
    # Check for required elements
    text_lower = text.lower()
    
    has_prior_art = any(phrase in text_lower for phrase in [
        "existing", "current", "conventional", "traditional", "prior art",
        "known", "typical", "commonly", "previously"
    ])
    
    has_problems = any(phrase in text_lower for phrase in [
        "problem", "limitation", "drawback", "disadvantage", "difficulty",
        "challenge", "suffer", "inadequate", "inefficient"
    ])
    
    has_need = any(phrase in text_lower for phrase in [
        "need", "desire", "would be", "therefore", "accordingly",
        "desirable", "beneficial", "advantageous"
    ])
    
    if not has_prior_art:
        issues.append("Missing discussion of prior art/existing technology.")
    
    if not has_problems:
        issues.append("Should identify problems or limitations with existing technology.")
    
    if not has_need:
        warnings.append("Consider adding statement of need or desire for improvement.")
    
    # Check for prohibited content
    prohibited_phrases = [
        "this invention", "our invention", "the present invention solves",
        "we developed", "we created", "my invention"
    ]
    
    for phrase in prohibited_phrases:
        if phrase in text_lower:
            issues.append(f"Avoid describing your own invention in Background. Found: '{phrase}'")
    
    # Check for proper tone (should be objective, not promotional)
    promotional_words = ["revolutionary", "breakthrough", "unprecedented", "amazing", "incredible"]
    found_promotional = [word for word in promotional_words if word in text_lower]
    if found_promotional:
        issues.append(f"Use objective language. Avoid: {', '.join(found_promotional)}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "paragraph_count": len(paragraphs),
        "has_prior_art": has_prior_art,
        "has_problems": has_problems,
        "has_need": has_need
    }

def generate_background_locally(abstract: str, max_attempts: int = 2) -> Dict[str, any]:
    """
    Generate the 'Background of the Invention' section for a patent application.
    
    Args:
        abstract: The patent abstract text
        max_attempts: Number of generation attempts if validation fails
        
    Returns:
        Dictionary containing the generated background and metadata
    """
    
    # Enhanced prompt with specific patent drafting guidelines
    prompt = f"""You are an expert patent attorney drafting the Background section of a US utility patent application.

TASK: Write the "Background of the Invention" section based on the abstract below.

CRITICAL REQUIREMENTS:
1. Use third person, present tense (avoid "we", "our", "I")
2. Do NOT describe YOUR invention - only describe existing technology
3. Structure: 3-4 paragraphs covering:
   - Paragraph 1: Current state of the technical field
   - Paragraph 2: Existing approaches/technologies and their operation
   - Paragraph 3: Problems, limitations, or drawbacks with current solutions
   - Paragraph 4 (optional): Statement of need for improvement

4. Use objective, technical language (avoid promotional words)
5. Length: 250-600 words total
6. Each paragraph: 3-6 sentences
7. Do NOT include solutions or describe the claimed invention
8. Do NOT use phrases like "the present invention", "this invention", "our solution"

STANDARD PATENT PHRASES TO USE:
- "Existing systems typically..."
- "Conventional methods suffer from..."
- "Prior art solutions have limitations including..."
- "Current approaches face challenges such as..."
- "Therefore, a need exists for..."
- "It would be desirable to have..."

GOOD EXAMPLE STRUCTURE:
Paragraph 1: "In the field of [domain], systems commonly employ [existing approach]. These systems typically operate by [brief description of how they work]."

Paragraph 2: "Conventional methods for [task] generally utilize [technology X]. For example, traditional approaches involve [specific technique], which requires [components/steps]."

Paragraph 3: "However, existing solutions suffer from several limitations. First, [problem 1]. Additionally, [problem 2]. Furthermore, [problem 3]."

Paragraph 4: "Therefore, a need exists for [improvement area] that addresses these limitations while maintaining [important features]."

INVENTION ABSTRACT:
{abstract.strip()}

Now write ONLY the Background of the Invention section (no headings, no mention of your invention):"""

    best_result = None
    
    for attempt in range(max_attempts):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=800,
                temperature=0.5 if attempt == 0 else 0.65,
                stop=["Summary of the Invention", "SUMMARY", "Brief Description"],
                top_p=0.92,
                repeat_penalty=1.1
            )
            
            raw_text = response["choices"][0]["text"].strip()
            cleaned_text = clean_background_text(raw_text)
            validation = validate_background(cleaned_text)
            
            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "word_count": validation["word_count"],
                "paragraph_count": validation["paragraph_count"],
                "has_prior_art": validation["has_prior_art"],
                "has_problems": validation["has_problems"],
                "has_need": validation["has_need"],
                "attempt": attempt + 1
            }
            
            # If valid or better than previous attempts
            if validation["valid"]:
                return result
            
            if best_result is None or len(validation["issues"]) < len(best_result["issues"]):
                best_result = result
                
        except Exception as e:
            return {
                "text": "",
                "valid": False,
                "issues": [f"LLM inference failed: {str(e)}"],
                "warnings": [],
                "word_count": 0,
                "paragraph_count": 0,
                "attempt": attempt + 1
            }
    
    return best_result if best_result else {
        "text": "",
        "valid": False,
        "issues": ["Failed to generate valid background"],
        "warnings": [],
        "word_count": 0,
        "paragraph_count": 0,
        "attempt": max_attempts
    }

def format_with_paragraph_numbers(text: str, start_num: int = 2) -> str:
    """Format background with USPTO paragraph numbering (typically starts at [0002])."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    formatted = []
    
    for i, para in enumerate(paragraphs):
        para_num = f"[{start_num + i:04d}]"
        formatted.append(f"{para_num} {para}")
    
    return '\n\n'.join(formatted)

def analyze_background_quality(text: str) -> Dict[str, any]:
    """Provide detailed quality analysis of the background section."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    analysis = {
        "structure_score": 0,
        "content_score": 0,
        "language_score": 0,
        "recommendations": []
    }
    
    text_lower = text.lower()
    
    # Structure analysis
    if 2 <= len(paragraphs) <= 5:
        analysis["structure_score"] += 30
    if 200 <= len(text.split()) <= 800:
        analysis["structure_score"] += 20
    
    # Content analysis
    if "existing" in text_lower or "conventional" in text_lower:
        analysis["content_score"] += 20
    if "problem" in text_lower or "limitation" in text_lower:
        analysis["content_score"] += 20
    if "need" in text_lower or "desirable" in text_lower:
        analysis["content_score"] += 10
    
    # Language analysis
    if not any(word in text_lower for word in ["we", "our", "i", "my"]):
        analysis["language_score"] += 25
    if not any(word in text_lower for word in ["revolutionary", "amazing", "incredible"]):
        analysis["language_score"] += 25
    
    # Recommendations
    total_score = analysis["structure_score"] + analysis["content_score"] + analysis["language_score"]
    
    if analysis["structure_score"] < 40:
        analysis["recommendations"].append("Improve structure: aim for 3-4 paragraphs, 250-600 words")
    if analysis["content_score"] < 40:
        analysis["recommendations"].append("Add more content: describe prior art, problems, and needs")
    if analysis["language_score"] < 40:
        analysis["recommendations"].append("Use more formal, objective language")
    
    analysis["total_score"] = total_score
    analysis["grade"] = "Excellent" if total_score >= 90 else "Good" if total_score >= 70 else "Needs Improvement"
    
    return analysis

# CLI for testing locally
if __name__ == "__main__":
    print("=" * 85)
    print("         PATENT BACKGROUND OF INVENTION GENERATOR")
    print("=" * 85)
    print("\n📥 Enter the invention abstract (press Enter twice to finish):")
    print("-" * 85)
    
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
    
    print("\n🧠 Generating 'Background of the Invention'...")
    print("-" * 85)
    
    result = generate_background_locally(abstract)
    
    if not result["text"]:
        print("\n❌ ERROR:")
        for issue in result["issues"]:
            print(f"   {issue}")
        exit(1)
    
    print("\n📘 BACKGROUND OF THE INVENTION")
    print("=" * 85)
    
    # Display validation status
    if result["valid"]:
        print("✅ Status: Valid - Meets USPTO Standards")
    else:
        print("⚠️  Status: Needs Review")
    
    print(f"\n📊 Statistics:")
    print(f"   • Word Count: {result['word_count']} words")
    print(f"   • Paragraphs: {result['paragraph_count']}")
    print(f"   • Generation Attempts: {result['attempt']}")
    
    print(f"\n📋 Content Check:")
    print(f"   • Prior Art Discussion: {'✓' if result['has_prior_art'] else '✗'}")
    print(f"   • Problems Identified: {'✓' if result['has_problems'] else '✗'}")
    print(f"   • Need Statement: {'✓' if result['has_need'] else '✗'}")
    
    if result["issues"]:
        print(f"\n🔍 Issues Found ({len(result['issues'])}):")
        for issue in result["issues"]:
            print(f"   • {issue}")
    
    if result["warnings"]:
        print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
        for warning in result["warnings"]:
            print(f"   • {warning}")
    
    print("\n" + "─" * 85)
    print("PLAIN TEXT:")
    print("─" * 85)
    print(result["text"])
    print("─" * 85)
    
    print("\n📄 FORMATTED WITH USPTO PARAGRAPH NUMBERING:")
    print("─" * 85)
    formatted = format_with_paragraph_numbers(result["text"])
    print(formatted)
    print("─" * 85)
    
    # Quality analysis
    print("\n📈 QUALITY ANALYSIS:")
    print("-" * 85)
    quality = analyze_background_quality(result["text"])
    print(f"Overall Grade: {quality['grade']} ({quality['total_score']}/100)")
    print(f"   • Structure: {quality['structure_score']}/50")
    print(f"   • Content: {quality['content_score']}/50")
    print(f"   • Language: {quality['language_score']}/50")
    
    if quality["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in quality["recommendations"]:
            print(f"   • {rec}")
    
    print("\n" + "=" * 85)
    print("Generation complete! Copy the formatted version for your patent application.")
    print("=" * 85)