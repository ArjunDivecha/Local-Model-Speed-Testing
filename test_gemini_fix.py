#!/usr/bin/env python3
"""
Test script to verify the Gemini FinishReason fix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_llm_benchmarker import UnifiedBenchmarker

def test_gemini_fix():
    """Test the Gemini FinishReason fix"""
    print("Testing Gemini FinishReason fix...")
    
    # Initialize benchmarker
    benchmarker = UnifiedBenchmarker()
    
    # Test specific models
    models_to_test = [
        ("google", "models/gemini-2.5-flash"),
        ("google", "models/gemini-2.5-pro")
    ]
    
    for provider, model in models_to_test:
        print(f"\nTesting {provider}/{model}...")
        
        try:
            # Create provider
            provider_instance = benchmarker.create_provider(provider)
            if not provider_instance:
                print(f"✗ Failed to create provider for {provider}")
                continue
                
            # Test with a simple prompt and higher token limit
            response = provider_instance.generate_response(
                prompt="Write a simple Python function that adds two numbers.",
                model=model,
                max_tokens=200  # Higher limit to allow some content
            )
            
            if response.success:
                print(f"✓ {provider}/{model}: Success - {len(response.content)} chars")
                print(f"  Tokens: {response.tokens_generated}, Time: {response.generation_time:.2f}s")
            else:
                print(f"✗ {provider}/{model}: Failed - {response.error_message}")
                
        except Exception as e:
            print(f"✗ {provider}/{model}: Exception - {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_gemini_fix()