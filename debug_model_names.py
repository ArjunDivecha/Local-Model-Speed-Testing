#!/usr/bin/env python3
"""
Debug model name handling in the benchmarker
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_llm_benchmarker import UnifiedBenchmarker

# Check what models are actually configured
benchmarker = UnifiedBenchmarker()

from unified_llm_benchmarker import PROVIDERS

print("Configured Google models:")
for model in PROVIDERS['google']['models']:
    print(f"  - {model}")

print(f"\nTesting with token limit: {benchmarker.token_limit}")

# Test the exact models from the config
print("\nTesting configured models:")
provider_instance = benchmarker.create_provider("google")

for model in PROVIDERS['google']['models']:
    print(f"\n🧪 Testing {model}...")
    
    try:
        response = provider_instance.generate_response(
            prompt="Write a simple hello world function in Python.",
            model=model,
            max_tokens=benchmarker.token_limit
        )
        
        if response.success:
            print(f"✅ SUCCESS: {len(response.content)} chars, {response.tokens_generated} tokens")
        else:
            print(f"❌ FAILED: {response.error_message}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()