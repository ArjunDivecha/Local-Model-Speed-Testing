#!/usr/bin/env python3
"""
Quick test of the benchmarker with a few models to verify fixes
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_llm_benchmarker import UnifiedBenchmarker

def test_quick_benchmark():
    """Run a quick benchmark with just a few models to test fixes"""
    
    print("=== Quick Benchmark Test ===")
    
    benchmarker = UnifiedBenchmarker()
    
    # Override the models list to test just a few key ones
    original_providers = benchmarker.providers.copy()
    
    # Test just the models that were failing
    benchmarker.providers["google"]["models"] = [
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash"
    ]
    
    # Reduce other providers for faster testing
    benchmarker.providers["openai"]["models"] = ["gpt-4o-mini"]
    benchmarker.providers["anthropic"]["models"] = ["claude-3-5-haiku-20241022"]
    
    # Disable other providers to make it faster
    for provider in ["groq", "xai", "deepseek", "cerebras"]:
        if provider in benchmarker.providers:
            benchmarker.providers[provider]["models"] = []
    
    print("Running quick benchmark with Gemini + 2 other models...")
    
    try:
        results = benchmarker.run()
        
        print(f"\n✅ Benchmark completed!")
        print(f"Results shape: {results.shape if results is not None else 'None'}")
        
        if results is not None and not results.empty:
            print("\nModels tested:")
            for model in results['model'].tolist():
                success = results[results['model'] == model]['success'].iloc[0]
                status = "✅" if success else "❌"
                print(f"  {status} {model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original providers
        benchmarker.providers = original_providers

if __name__ == "__main__":
    test_quick_benchmark()