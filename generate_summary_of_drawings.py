from llama_cpp import Llama
import re
from typing import Dict, List, Tuple

# Path to your locally downloaded Phi-3 model (.gguf file)
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"

# Load the model ONCE at module level
llm = Llama(
    model_path=LLM_PATH,
    n_ctx=4096,
    n_threads=4,
    n_batch=512,
    verbose=False
)

def estimate_figure_count(abstract: str) -> int:
    """Estimate reasonable number of figures based on invention complexity."""
    word_count = len(abstract.split())
    
    # Check for system/method indicators
    has_system = any(word in abstract.lower() for word in ["system", "device", "apparatus"])
    has_method = any(word in abstract.lower() for word in ["method", "process", "steps"])
    has_components = any(word in abstract.lower() for word in ["components", "modules", "parts", "elements"])
    
    # Base figure count
    base_figures = 3
    
    # Adjust based on complexity
    if word_count > 150:
        base_figures += 1
    if has_system and has_method:
        base_figures += 2  # Need system diagram + flowchart
    if has_components:
        base_figures += 1  # Component detail views
    
    # Cap at reasonable number
    return min(base_figures, 8)

def suggest_figure_types(abstract: str) -> List[Dict[str, str]]:
    """Suggest appropriate figure types based on invention nature."""
    suggestions = []
    abstract_lower = abstract.lower()
    
    # Always start with overview
    suggestions.append({
        "type": "overview",
        "description": "block diagram illustrating an overview of the system/invention"
    })
    
    # System/Device figures
    if any(word in abstract_lower for word in ["system", "device", "apparatus", "hardware"]):
        suggestions.append({
            "type": "system",
            "description": "schematic diagram showing components and their connections"
        })
        suggestions.append({
            "type": "detail",
            "description": "detailed view of key components or subsystems"
        })
    
    # Method/Process figures
    if any(word in abstract_lower for word in ["method", "process", "steps", "algorithm"]):
        suggestions.append({
            "type": "flowchart",
            "description": "flowchart depicting the method steps or process flow"
        })
    
    # Software/Computing figures
    if any(word in abstract_lower for word in ["software", "computer", "processor", "algorithm", "data"]):
        suggestions.append({
            "type": "architecture",
            "description": "functional block diagram of the software architecture"
        })
    
    # UI/Display figures
    if any(word in abstract_lower for word in ["display", "interface", "screen", "user"]):
        suggestions.append({
            "type": "interface",
            "description": "illustration of user interface or display screens"
        })
    
    # Physical structure figures
    if any(word in abstract_lower for word in ["structure", "assembly", "mechanical", "physical"]):
        suggestions.append({
            "type": "cross-section",
            "description": "cross-sectional view showing internal structure"
        })
        suggestions.append({
            "type": "perspective",
            "description": "perspective view of the assembled device"
        })
    
    return suggestions

def clean_drawing_description(text: str) -> str:
    """Clean and format the drawing descriptions."""
    # Remove section headers
    text = re.sub(r'^(Brief Description of the Drawings:|Brief Description:|BRIEF DESCRIPTION)\s*', '', text, flags=re.IGNORECASE)
    
    # Split into individual figure descriptions
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Ensure proper capitalization
        if line and not line[0].isupper():
            line = line[0].upper() + line[1:]
        
        # Ensure ends with period
        if line and not line.endswith(('.', ';')):
            line += '.'
        
        cleaned_lines.append(line)
    
    return '\n\n'.join(cleaned_lines)

def validate_drawing_descriptions(text: str) -> Dict[str, any]:
    """Validate drawing descriptions against USPTO standards."""
    issues = []
    warnings = []
    
    # Split into individual figure descriptions
    fig_pattern = r'FIG(?:URE)?\.?\s*\d+[A-Z]?'
    figures = re.findall(fig_pattern, text, re.IGNORECASE)
    
    # Check if figures are present
    if not figures:
        issues.append("No figure references found (e.g., 'FIG. 1', 'FIGURE 1').")
    
    # Check figure numbering sequence
    fig_numbers = []
    for fig in figures:
        match = re.search(r'\d+', fig)
        if match:
            fig_numbers.append(int(match.group()))
    
    if fig_numbers:
        # Check for sequential numbering
        expected = list(range(1, max(fig_numbers) + 1))
        if sorted(set(fig_numbers)) != expected:
            warnings.append("Figure numbers should be sequential (1, 2, 3, ...).")
        
        # Check minimum figures
        if len(set(fig_numbers)) < 2:
            warnings.append("Most patents have at least 2-3 figures. Consider adding more views.")
    
    # Check for proper format
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    for i, line in enumerate(lines):
        # Each line should start with "FIG." or "FIGURE"
        if not re.match(r'^FIG(?:URE)?\.?\s*\d+', line, re.IGNORECASE):
            issues.append(f"Line {i+1} doesn't start with figure reference (e.g., 'FIG. 1').")
        
        # Should contain descriptive verbs
        descriptive_verbs = ["illustrates", "shows", "depicts", "is a", "represents", "displays"]
        if not any(verb in line.lower() for verb in descriptive_verbs):
            warnings.append(f"Line {i+1} should use descriptive verbs (illustrates, shows, depicts, etc.).")
        
        # Should end with period
        if not line.endswith('.'):
            issues.append(f"Line {i+1} should end with a period.")
    
    # Check length
    word_count = len(text.split())
    if word_count < 30:
        issues.append("Description too brief. Each figure should have adequate description.")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "figure_count": len(set(fig_numbers)),
        "word_count": word_count
    }

def generate_drawing_descriptions(abstract: str, num_figures: int = None, max_attempts: int = 2) -> Dict[str, any]:
    """
    Generate the 'Brief Description of the Drawings' section for a U.S. patent.
    
    Args:
        abstract: The patent abstract text
        num_figures: Desired number of figures (auto-estimated if None)
        max_attempts: Number of generation attempts if validation fails
        
    Returns:
        Dictionary containing generated descriptions and metadata
    """
    
    # Estimate figures if not provided
    if num_figures is None:
        num_figures = estimate_figure_count(abstract)
    
    # Get figure type suggestions
    fig_suggestions = suggest_figure_types(abstract)
    
    # Build suggestions text
    suggestions_text = "\n".join([
        f"   - {fig['type'].title()}: {fig['description']}"
        for fig in fig_suggestions[:num_figures]
    ])
    
    # Enhanced prompt with USPTO standards
    prompt = f"""You are an expert patent attorney drafting the "Brief Description of the Drawings" section for a U.S. patent application.

TASK: Write brief descriptions for {num_figures} figures based on the abstract below.

CRITICAL USPTO REQUIREMENTS:
1. Use the exact format: "FIG. 1 is a [type] illustrating/showing [what it depicts]."
2. Number figures sequentially: FIG. 1, FIG. 2, FIG. 3, etc.
3. Use standard diagram types:
   - "block diagram"
   - "schematic diagram"
   - "flowchart"
   - "perspective view"
   - "cross-sectional view"
   - "functional diagram"
   - "detailed view"

4. Use descriptive verbs: "illustrates", "shows", "depicts", "is a diagram of"
5. Be concise: 1-2 sentences per figure maximum
6. Each description ends with a period
7. Focus on WHAT is shown, not WHY or HOW it works
8. Use present tense

GOOD EXAMPLES:
- "FIG. 1 is a block diagram illustrating an overview of the wireless communication system according to embodiments of the invention."
- "FIG. 2 is a schematic diagram showing the internal components of the transmitter module."
- "FIG. 3 is a flowchart depicting the steps of the signal processing method."
- "FIG. 4 is a detailed view of the antenna array configuration."

FIGURE TYPE SUGGESTIONS for this invention:
{suggestions_text}

INVENTION ABSTRACT:
{abstract.strip()}

Now write ONLY the figure descriptions (no section heading, no explanations):"""

    best_result = None
    
    for attempt in range(max_attempts):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=400,
                temperature=0.4 if attempt == 0 else 0.6,
                stop=["Detailed Description", "DETAILED DESCRIPTION", "Summary of"],
                top_p=0.9,
                repeat_penalty=1.1
            )
            
            raw_text = response["choices"][0]["text"].strip()
            cleaned_text = clean_drawing_description(raw_text)
            validation = validate_drawing_descriptions(cleaned_text)
            
            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "figure_count": validation["figure_count"],
                "word_count": validation["word_count"],
                "suggested_figures": fig_suggestions[:num_figures],
                "attempt": attempt + 1
            }
            
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
                "figure_count": 0,
                "word_count": 0,
                "suggested_figures": [],
                "attempt": attempt + 1
            }
    
    return best_result if best_result else {
        "text": "",
        "valid": False,
        "issues": ["Failed to generate valid descriptions"],
        "warnings": [],
        "figure_count": 0,
        "word_count": 0,
        "suggested_figures": [],
        "attempt": max_attempts
    }

def format_with_section_header(text: str, start_paragraph: int = None) -> str:
    """Format with section header and optional paragraph numbering."""
    header = "BRIEF DESCRIPTION OF THE DRAWINGS\n\n"
    
    if start_paragraph:
        lines = [l.strip() for l in text.split('\n\n') if l.strip()]
        formatted_lines = []
        for i, line in enumerate(lines):
            para_num = f"[{start_paragraph + i:04d}]"
            formatted_lines.append(f"{para_num} {line}")
        return header + '\n\n'.join(formatted_lines)
    
    return header + text

def create_figure_checklist(suggested_figures: List[Dict[str, str]]) -> str:
    """Create a checklist of figures to draw."""
    checklist = "FIGURE DRAWING CHECKLIST:\n" + "=" * 70 + "\n\n"
    
    for i, fig in enumerate(suggested_figures, 1):
        checklist += f"□ FIG. {i} - {fig['type'].upper()}\n"
        checklist += f"  Description: {fig['description']}\n"
        checklist += f"  Requirements: Clear labels, reference numerals, professional appearance\n\n"
    
    checklist += "\nREMINDER:\n"
    checklist += "- Use reference numerals (10, 12, 14, etc.) to label components\n"
    checklist += "- Keep drawings simple and clear\n"
    checklist += "- Use black ink on white paper (or digital equivalent)\n"
    checklist += "- Follow USPTO drawing standards (37 CFR 1.84)\n"
    
    return checklist

# Optional CLI interface for manual testing
if __name__ == "__main__":
    print("=" * 80)
    print("         BRIEF DESCRIPTION OF THE DRAWINGS GENERATOR")
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
    
    # Ask for number of figures
    print("\n🔢 How many figures do you need? (press Enter for auto-suggestion): ", end="")
    num_input = input().strip()
    num_figures = int(num_input) if num_input.isdigit() else None
    
    print("\n🖼️  Generating drawing descriptions...")
    print("-" * 80)
    
    result = generate_drawing_descriptions(abstract, num_figures)
    
    if not result["text"]:
        print("\n❌ ERROR:")
        for issue in result["issues"]:
            print(f"   {issue}")
        exit(1)
    
    print("\n📑 BRIEF DESCRIPTION OF THE DRAWINGS")
    print("=" * 80)
    
    # Display validation status
    if result["valid"]:
        print("✅ Status: Valid - Meets USPTO Standards")
    else:
        print("⚠️  Status: Needs Review")
    
    print(f"\n📊 Statistics:")
    print(f"   • Number of Figures: {result['figure_count']}")
    print(f"   • Word Count: {result['word_count']}")
    print(f"   • Generation Attempts: {result['attempt']}")
    
    if result["issues"]:
        print(f"\n🔍 Issues Found ({len(result['issues'])}):")
        for issue in result["issues"]:
            print(f"   • {issue}")
    
    if result["warnings"]:
        print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
        for warning in result["warnings"]:
            print(f"   • {warning}")
    
    print("\n" + "─" * 80)
    print("PLAIN TEXT:")
    print("─" * 80)
    print(result["text"])
    print("─" * 80)
    
    print("\n📄 FORMATTED WITH SECTION HEADER:")
    print("─" * 80)
    formatted = format_with_section_header(result["text"])
    print(formatted)
    print("─" * 80)
    
    # Show figure checklist
    if result["suggested_figures"]:
        print("\n" + "=" * 80)
        checklist = create_figure_checklist(result["suggested_figures"])
        print(checklist)
        print("=" * 80)
    
    print("\n💡 NEXT STEPS:")
    print("-" * 80)
    print("1. Review the figure descriptions above")
    print("2. Use the checklist to create actual drawings")
    print("3. Ensure each drawing has reference numerals matching the detailed description")
    print("4. Follow USPTO drawing standards (37 CFR 1.84)")
    print("5. Consider hiring a professional patent illustrator")
    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)