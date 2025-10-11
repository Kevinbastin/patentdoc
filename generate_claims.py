import os
import faiss
import json
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import re
import textwrap
from datetime import datetime


# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_PATH = os.path.join(BASE_DIR, "models", "models", "phi-3-mini-4k-instruct-q4.gguf")
INDEX_PATH = os.path.join(BASE_DIR, "data", "bigpatent_tiny", "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "data", "bigpatent_tiny", "faiss_metadata.json")

# === Load models ONCE ===
llm = Llama(model_path=LLM_PATH, n_ctx=8192, n_threads=4)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Load FAISS index and metadata ===
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)


def is_claim_complete(claim_text: str) -> bool:
    """Check if a claim appears to be complete"""
    if len(claim_text.strip()) < 50:
        return False
    if not claim_text.strip().endswith('.'):
        return False
    
    incomplete_endings = ['wherein', 'comprising', 'and', 'or', 'the', 'a', 'an', 'with', 'to', 'for', 'that']
    last_word = claim_text.strip().rstrip('.').split()[-1].lower()
    if last_word in incomplete_endings:
        return False
    return True


def generate_claims_from_abstract(abstract: str, top_k: int = 3) -> str:
    """
    Generate USPTO-style claims with proper formatting matching Indian Patent Office style
    """
    
    # Get prior art for context
    try:
        query_embedding = embedding_model.encode([abstract], convert_to_numpy=True)
        D, I = index.search(query_embedding, top_k)
        prior_abstracts = "\n".join(
            f"Prior Art {i+1}: {metadata[idx]['abstract'][:180].strip()}..."
            for i, idx in enumerate(I[0]) if idx < len(metadata)
        )
    except Exception as e:
        prior_abstracts = ""

    # Extract key components from abstract
    components = extract_components_from_abstract(abstract)
    
    # Generate independent claim with structured format
    independent_claim = generate_independent_claim_structured(abstract, components, prior_abstracts)
    
    # Generate dependent claims (claims 2-9)
    dependent_claims = []
    for i in range(2, 10):
        dep_claim = generate_dependent_claim(i, abstract, independent_claim, components)
        dependent_claims.append(dep_claim)
    
    # Generate method claim (claim 10)
    method_claim = generate_method_claim(abstract, independent_claim)
    
    # Format all claims in proper USPTO/Indian Patent Office style
    formatted_claims = format_claims_patent_office_style(
        independent_claim, 
        dependent_claims, 
        method_claim
    )
    
    return formatted_claims


def extract_components_from_abstract(abstract: str) -> list:
    """Extract key components/modules from the abstract"""
    components = []
    
    # Look for common patterns
    patterns = [
        r'(\w+\s+(?:module|unit|component|sensor|system|interface|engine|processor|controller|device|structure|line|tube|pipe|absorber|condenser))',
        r'comprising[:\s]+([^.]+)',
        r'includes?\s+([^.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, abstract, re.IGNORECASE)
        components.extend(matches)
    
    # Clean and deduplicate
    components = [c.strip() for c in components if len(c.strip()) > 5]
    components = list(dict.fromkeys(components))[:10]
    
    # If not enough components found, extract noun phrases
    if len(components) < 4:
        words = abstract.split()
        for i in range(len(words)-1):
            if words[i].lower() in ['a', 'an', 'the'] and len(words[i+1]) > 3:
                components.append(words[i+1])
    
    return components[:10]


def generate_independent_claim_structured(abstract: str, components: list, prior_art: str) -> dict:
    """Generate independent claim with proper sub-element structure"""
    
    # Determine device/system name
    device_match = re.search(r'A\s+(.+?)\s+(?:comprising|for|system|device)', abstract, re.IGNORECASE)
    if device_match:
        device_name = device_match.group(1).strip()
    else:
        device_name = "system"
    
    prompt = f"""You are a patent attorney. Write ONE independent apparatus claim in proper patent format.

INVENTION: {abstract}

REQUIREMENTS:
1. Start with: "A [device/system] for [purpose], comprising:"
2. List 4-6 key elements WITHOUT letters (no a., b., c.)
3. Each element on a new line, indented
4. Use semicolons between elements
5. Last element ends with a period
6. Use "configured to", "operable to" for functions

FORMAT EXAMPLE:
A loop heat pipe system for cooling electronics, comprising:
a heat absorber configured to absorb heat from an electronic component;
a condenser configured to release heat and condense vapour;
a vapour line connecting the heat absorber to the condenser; and
a capillary tube connecting the condenser to the heat absorber, wherein the capillary tube creates a pressure differential.

NOW WRITE THE INDEPENDENT CLAIM:
A"""

    try:
        output = llm(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.8,
            repeat_penalty=1.2,
            stop=["2.", "\n\n\n", "Claim 2"]
        )
        
        claim_text = "A" + output["choices"][0]["text"].strip()
        
        # Parse into preamble and elements
        if 'comprising:' in claim_text.lower():
            parts = re.split(r'comprising:', claim_text, flags=re.IGNORECASE)
            preamble = parts[0].strip() + ', comprising:'
            elements_text = parts[1].strip()
            
            # Split elements by semicolon or newline
            elements = [e.strip() for e in re.split(r';|\n', elements_text) if e.strip()]
            
            if not elements or len(elements) < 3:
                elements = create_fallback_elements_unlettered(components)
            
            return {
                'preamble': preamble,
                'elements': elements,
                'device_name': device_name
            }
        else:
            elements = create_fallback_elements_unlettered(components)
            return {
                'preamble': f"A {device_name} for performing operations, comprising:",
                'elements': elements,
                'device_name': device_name
            }
            
    except Exception as e:
        elements = create_fallback_elements_unlettered(components)
        return {
            'preamble': f"A {device_name} for performing operations, comprising:",
            'elements': elements,
            'device_name': device_name
        }


def create_fallback_elements_unlettered(components: list) -> list:
    """Create structured elements WITHOUT letters (a., b., c.)"""
    elements = []
    
    for i, comp in enumerate(components[:5]):
        comp_clean = comp.strip()
        if i == len(components[:5]) - 1:
            # Last element
            elements.append(f"{comp_clean} configured to perform operations related to the system.")
        else:
            elements.append(f"{comp_clean} configured to interact with the system;")
    
    return elements


def generate_dependent_claim(claim_num: int, abstract: str, indep_claim: dict, components: list) -> str:
    """Generate a dependent claim"""
    
    device_name = indep_claim.get('device_name', 'system')
    
    # Determine which claim to depend on (most depend on claim 1)
    depends_on = 1
    if claim_num in [4, 6]:
        depends_on = claim_num - 1
    
    # Select element to enhance
    element_index = (claim_num - 2) % len(components)
    component = components[element_index].strip() if components else "component"
    
    prompt = f"""Write ONE short dependent claim in patent format.

INDEPENDENT CLAIM: {indep_claim.get('preamble', '')}

Write claim {claim_num} that depends on claim {depends_on}.

FORMAT:
The [device] as claimed in claim {depends_on}, wherein [one specific limitation].

EXAMPLE:
The system as claimed in claim 1, wherein the capillary tube has a length ranging from 0.5 to 2 meters.

NOW WRITE (only the text, no number):
The"""

    try:
        output = llm(
            prompt=prompt,
            max_tokens=256,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.2,
            stop=["\n\n", f"{claim_num+1}.", "Claim"]
        )
        
        claim_text = "The" + output["choices"][0]["text"].strip()
        
        # Ensure it references the correct parent claim
        if f"claim {depends_on}" not in claim_text.lower():
            claim_text = f"The {device_name} as claimed in claim {depends_on}, wherein the {component} is configured to provide enhanced functionality."
        
        # Ensure ends with period
        if not claim_text.endswith('.'):
            claim_text += '.'
        
        return claim_text
        
    except Exception as e:
        return f"The {device_name} as claimed in claim {depends_on}, wherein the {component} provides additional functionality."


def generate_method_claim(abstract: str, indep_claim: dict) -> str:
    """Generate a method claim (claim 10)"""
    
    device_name = indep_claim.get('device_name', 'system')
    
    prompt = f"""Write ONE method claim in patent format.

DEVICE: {indep_claim.get('preamble', '')}

Write a method claim with these steps (no letters):

FORMAT:
A method for [purpose] using [device], comprising the steps of:
[step 1];
[step 2];
[step 3];
[step 4]; and
[step 5].

EXAMPLE:
A method for cooling electronics using a loop heat pipe system, comprising the steps of:
absorbing heat from an electronic component via a heat absorber;
evaporating a working fluid to generate vapour;
transporting the vapour to a condenser;
condensing the vapour into liquid; and
directing the liquid through a capillary tube to create a pressure differential.

NOW WRITE THE METHOD CLAIM:
A method for"""

    try:
        output = llm(
            prompt=prompt,
            max_tokens=512,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.2,
            stop=["\n\n\n", "11.", "Claim 11"]
        )
        
        claim_text = "A method for" + output["choices"][0]["text"].strip()
        
        # Ensure ends with period
        if not claim_text.endswith('.'):
            claim_text += '.'
        
        return claim_text
        
    except Exception as e:
        return f"A method for using the {device_name} as claimed in claim 1, comprising steps of operating the system according to its intended purpose."


def format_claims_patent_office_style(independent_claim: dict, dependent_claims: list, method_claim: str) -> str:
    """
    Format claims in proper Indian Patent Office / USPTO style with:
    - Line numbers on the right margin (every 5 lines)
    - Proper indentation
    - No lettering for sub-elements in apparatus claims
    - "We Claim" header
    - Date footer
    """
    
    output = "Claims\n\n"
    output += "We Claim\n\n"
    
    line_counter = 1
    
    # === CLAIM 1 (Independent Apparatus Claim) ===
    preamble = independent_claim['preamble']
    elements = independent_claim['elements']
    
    # Write claim number and preamble
    output += f"1. {preamble}"
    
    # Add line number after preamble (every 5 lines)
    if line_counter % 5 == 0:
        output += f"{line_counter:>5}"
    output += "\n"
    line_counter += 1
    
    # Write each element with proper indentation and wrapping
    for i, element in enumerate(elements):
        element = element.strip()
        
        # Check if this is the last element
        is_last = (i == len(elements) - 1)
        
        # Add "and" before last element if it doesn't have it
        if is_last and not element.startswith('and '):
            if element.endswith(';'):
                element = element.rstrip(';').strip() + '.'
            element = 'and ' + element
        
        # Wrap long elements at 65 characters
        wrapped_lines = textwrap.wrap(
            element,
            width=65,
            break_long_words=False,
            break_on_hyphens=False
        )
        
        for j, line in enumerate(wrapped_lines):
            if j == 0:
                output += f"   {line}"
            else:
                output += f"   {line}"
            
            # Add line number every 5 lines
            if line_counter % 5 == 0:
                output += f"{line_counter:>5}"
            output += "\n"
            line_counter += 1
    
    output += "\n"
    
    # === CLAIMS 2-9 (Dependent Claims) ===
    for i, dep_claim in enumerate(dependent_claims, start=2):
        claim_text = f"{i}. {dep_claim}"
        
        # Wrap at 65 characters with continuation indentation
        wrapped_lines = textwrap.wrap(
            claim_text,
            width=65,
            break_long_words=False,
            break_on_hyphens=False,
            subsequent_indent='   '
        )
        
        for line in wrapped_lines:
            output += line
            
            # Add line number every 5 lines
            if line_counter % 5 == 0:
                output += f"{line_counter:>5}"
            output += "\n"
            line_counter += 1
        
        output += "\n"
    
    # === CLAIM 10 (Method Claim) ===
    claim_text = f"10. {method_claim}"
    
    # Wrap method claim
    wrapped_lines = textwrap.wrap(
        claim_text,
        width=65,
        break_long_words=False,
        break_on_hyphens=False,
        subsequent_indent='   '
    )
    
    for line in wrapped_lines:
        output += line
        
        if line_counter % 5 == 0:
            output += f"{line_counter:>5}"
        output += "\n"
        line_counter += 1
    
    output += "\n"
    
    # === ADD DATE FOOTER ===
    current_date = datetime.now()
    month_name = current_date.strftime("%B")
    output += f"Dated this {current_date.day} day of {month_name} {current_date.year}\n"
    
    return output


def post_process_claims(claims_text: str) -> str:
    """Legacy function - returns the properly formatted claims"""
    return claims_text


def validate_claims(claims_text: str) -> bool:
    """Check if generated claims meet minimum quality standards"""
    if not claims_text or len(claims_text) < 200:
        return False
    
    # Check for We Claim header
    if 'We Claim' not in claims_text and 'WE CLAIM' not in claims_text:
        return False
    
    # Check for numbered claims
    claim_numbers = re.findall(r'^\s*(\d+)\.', claims_text, re.MULTILINE)
    if len(claim_numbers) < 5:
        return False
    
    # Check that claim 1 contains "comprising"
    if 'comprising' not in claims_text.lower()[:800]:
        return False
    
    return True


def regenerate_with_stricter_prompt(abstract: str, prior_art: str) -> str:
    """Fallback generation if main function fails"""
    return generate_claims_from_abstract(abstract)


def generate_detailed_claims(abstract: str, top_k: int = 3) -> dict:
    """
    Generate complete claims with apparatus and method claims
    """
    claims = generate_claims_from_abstract(abstract, top_k)
    
    return {
        "apparatus_claims": claims,
        "method_claims": "",  # Already included in main claims
        "total_claims": claims
    }


# === Example Usage ===
if __name__ == "__main__":
    sample_abstract = """A loop heat pipe system for ultra-low temperature cooling of high heat flux electronics. The system comprises a heat absorber configured to absorb heat from electronic components with a wick structure, a condenser configured to release heat and condense vapour into liquid, a vapour line connecting the heat absorber to the condenser, and a capillary tube connecting the condenser to the heat absorber. The capillary tube creates a pressure differential and throttles the working fluid, resulting in significant temperature reduction in the absorber. The system operates passively without mechanical pumps and can achieve temperature reductions of at least 40 degrees Celsius compared to conventional systems."""
    
    print("=== GENERATING PATENT OFFICE STYLE CLAIMS ===\n")
    claims = generate_claims_from_abstract(sample_abstract)
    print(claims)
    
    print("\n=== VALIDATION ===")
    print(f"Valid: {validate_claims(claims)}")