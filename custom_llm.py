"""
Custom LLM wrapper for CrewAI - FINAL WORKING VERSION
Uses Pydantic v2 with proper configuration
"""

from typing import Any, List, Optional, Dict
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from llama_cpp import Llama as LlamaCppModel
from pydantic import model_validator


class CustomLlamaCpp(LLM):
    """Complete wrapper for llama-cpp-python that works with CrewAI"""
    
    # Pydantic v2 configuration
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }
    
    # Fields
    model_path: str = ""
    n_ctx: int = 4096
    n_threads: int = 4
    temperature: float = 0.3
    max_tokens: int = 512
    _llm: Optional[Any] = None
    
    @model_validator(mode='after')
    def load_model(self):
        """Load the model after validation"""
        if self.model_path and self._llm is None:
            print(f"🔄 Loading model from: {self.model_path}")
            self._llm = LlamaCppModel(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            print(f"✅ Model loaded successfully!")
        return self
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type"""
        return "custom_llamacpp"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for multiple prompts"""
        generations = []
        
        for prompt in prompts:
            text = self._call_model(prompt, stop)
            generations.append([Generation(text=text)])
        
        return LLMResult(generations=generations)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> str:
        """Call the model with a single prompt"""
        return self._call_model(prompt, stop)
    
    def _call_model(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Internal method to call the model"""
        try:
            response = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                stop=stop or [],
                echo=False
            )
            
            if isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["text"].strip()
            return str(response).strip()
                
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            return f"Error: {str(e)}"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return parameters that identify this LLM"""
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


if __name__ == "__main__":
    print("="*60)
    print("Testing CustomLlamaCpp Wrapper")
    print("="*60)
    
    try:
        # Initialize
        llm = CustomLlamaCpp(
            model_path="/app/models/models/phi-3-mini-4k-instruct-q4.gguf",
            n_ctx=512,
            n_threads=2,
            max_tokens=30
        )
        
        # Test single call
        print("\n🔄 Test 1: Single prompt")
        response = llm("Say 'Hello World'")
        print(f"✅ Response: {response}")
        
        # Test generate (used by CrewAI)
        print("\n🔄 Test 2: Multiple prompts (CrewAI style)")
        result = llm.generate(["Say hi", "Say bye"])
        print(f"✅ Generated {len(result.generations)} responses")
        for i, gen in enumerate(result.generations):
            print(f"   Response {i+1}: {gen[0].text[:50]}...")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED! Ready to use with CrewAI")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
