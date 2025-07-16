#!/usr/bin/env python3

import sys
sys.path.append('.')
from unified_llm_benchmarker import UnifiedBenchmarker

# Test actual benchmarking functionality
benchmarker = UnifiedBenchmarker()

# Test with a simple prompt to verify the engine works
test_prompt = 'Write a simple hello world function in Python.'
benchmarker.benchmark_prompt = test_prompt

# Get first available model
available_models = benchmarker.get_available_models()
if available_models:
    provider, model = available_models[0]
    print(f'Testing benchmark engine with {provider}/{model}...')
    
    result = benchmarker.benchmark_model(provider, model)
    
    print(f'Success: {result.success}')
    print(f'Generation time: {result.generation_time:.3f}s')
    print(f'Tokens per second: {result.tokens_per_second:.2f}')
    print(f'Time to first token: {result.time_to_first_token:.3f}s')
    print(f'Tokens generated: {result.tokens_generated}')
    print(f'Cost: ${result.cost:.4f}')
    print(f'Response length: {len(result.response_content)} chars')
    
    if result.success:
        print('✅ Benchmark engine working correctly!')
    else:
        print(f'❌ Benchmark failed: {result.error_message}')
else:
    print('No available models to test')