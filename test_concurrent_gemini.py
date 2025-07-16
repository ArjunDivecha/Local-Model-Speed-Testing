#!/usr/bin/env python3
"""
Test Gemini models under concurrent execution like in the benchmarker
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_llm_benchmarker import GoogleProvider
from dotenv import load_dotenv

load_dotenv()

def test_single_model(model_name, test_id):
    """Test a single model - simulates benchmarker behavior"""
    print(f"[{test_id}] Starting {model_name}...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    config = {"base_url": None}
    provider = GoogleProvider(api_key, config)
    
    # Use the same prompt as benchmarker
    prompt = """Write a script that connects to Interactive Brokers API, retrieves real-time options data for SPY, calculates implied volatility skew, and generates an alert when the skew exceeds historical 90th percentile. Include reconnection logic and handle all possible API errors explicitly."""
    
    try:
        response = provider.generate_response(prompt, model_name, max_tokens=10000)
        
        if response.success:
            print(f"[{test_id}] ✅ {model_name}: {len(response.content)} chars, {response.tokens_generated} tokens")
            return True
        else:
            print(f"[{test_id}] ❌ {model_name}: {response.error_message}")
            return False
            
    except Exception as e:
        print(f"[{test_id}] ❌ {model_name} Exception: {e}")
        return False

def test_concurrent_gemini():
    """Test concurrent execution like the benchmarker does"""
    
    print("=== Testing Concurrent Gemini Execution ===")
    
    # Models to test concurrently 
    models_to_test = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]
    
    print(f"Testing {len(models_to_test)} models concurrently...")
    
    # Use same concurrent settings as benchmarker
    max_workers = min(5, len(models_to_test))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks at once (same as benchmarker)
        future_to_model = {
            executor.submit(test_single_model, model, i): model
            for i, model in enumerate(models_to_test)
        }
        
        # Collect results as they complete
        results = {}
        for future in future_to_model:
            model = future_to_model[future]
            try:
                success = future.result(timeout=60)  # 60 second timeout per model
                results[model] = success
            except Exception as e:
                print(f"❌ {model} failed with exception: {e}")
                results[model] = False
    
    # Report results
    print("\n=== Results ===")
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model}: {status}")
    
    success_count = sum(results.values())
    print(f"\nOverall: {success_count}/{len(models_to_test)} models succeeded")
    
    if success_count < len(models_to_test):
        print("\n⚠️  Some models failed in concurrent execution - this explains the benchmarker issue!")
    else:
        print("\n✅ All models succeeded - issue might be elsewhere")

if __name__ == "__main__":
    test_concurrent_gemini()