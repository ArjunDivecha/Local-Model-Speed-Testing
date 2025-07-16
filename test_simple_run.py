#!/usr/bin/env python3
"""
Simple test with minimal models to verify core functionality
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_llm_benchmarker import UnifiedBenchmarker, PROVIDERS

def test_simple_run():
    """Test with minimal set of models"""
    
    print("=== Simple Benchmarker Test ===")
    
    # Create a minimal config with only fast/reliable models
    minimal_config = {
        'benchmark_prompt': "Write a simple Python function to add two numbers.",  # Shorter prompt
        'token_limit': 500  # Lower token limit for faster completion
    }
    
    benchmarker = UnifiedBenchmarker(minimal_config)
    
    # Temporarily modify the providers to use only reliable models
    # Save original
    original_providers = PROVIDERS.copy()
    
    try:
        # Override with minimal set
        PROVIDERS["openai"]["models"] = ["gpt-4o-mini"]  # One fast OpenAI model
        PROVIDERS["google"]["models"] = ["models/gemini-2.5-flash"]  # One Gemini model with our fix
        
        # Disable problematic providers
        for provider in ["anthropic", "groq", "xai", "deepseek", "cerebras"]:
            if provider in PROVIDERS:
                PROVIDERS[provider]["models"] = []
        
        print("Testing with minimal set: gpt-4o-mini + gemini-2.5-flash")
        
        start_time = time.time()
        execution_summary = benchmarker.run_complete_benchmark()
        end_time = time.time()
        
        print(f"\n✅ Benchmark completed in {end_time - start_time:.1f} seconds!")
        
        if execution_summary and execution_summary['status'] == 'completed_with_results':
            print(f"Status: {execution_summary['status']}")
            print(f"Models tested: {execution_summary['total_models_attempted']}")
            print(f"Successful: {execution_summary['successful_benchmarks']}")
            print(f"Failed: {execution_summary['failed_benchmarks']}")
            print(f"Success rate: {execution_summary['success_rate']:.1f}%")
            
            if execution_summary.get('output_files'):
                files = execution_summary['output_files']
                print(f"Excel files: {len(files.get('excel', []))}")
                print(f"Chart files: {len(files.get('charts', []))}")
            
            return True
        else:
            print(f"❌ Benchmark failed: {execution_summary.get('status', 'Unknown status')}")
            if execution_summary.get('errors'):
                for error in execution_summary['errors'][:3]:  # Show first 3 errors
                    print(f"  Error: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original providers
        PROVIDERS.clear()
        PROVIDERS.update(original_providers)

if __name__ == "__main__":
    test_simple_run()