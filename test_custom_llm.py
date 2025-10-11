#!/usr/bin/env python3
"""Test the custom LLM wrapper"""

from custom_llm import CustomLlamaCpp

print("Testing Custom LlamaCpp wrapper...")

try:
    llm = CustomLlamaCpp(
        model_path="/app/models/models/phi-3-mini-4k-instruct-q4.gguf",
        n_ctx=512,
        n_threads=2,
        max_tokens=50
    )
    
    print("\n✅ LLM initialized successfully!")
    
    # Test call
    print("\nTesting LLM call...")
    response = llm("Say 'hello world'")
    print(f"Response: {response}")
    
    print("\n✅ Custom LLM wrapper works!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
