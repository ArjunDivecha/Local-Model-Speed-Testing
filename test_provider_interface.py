#!/usr/bin/env python3
"""
Test script for the unified provider interface implementation.
"""

from unified_llm_benchmarker import UnifiedBenchmarker

def test_provider_interface():
    """Test the provider interface implementation."""
    print("Testing Unified Provider Interface...")
    
    # Initialize benchmarker
    benchmarker = UnifiedBenchmarker()
    
    # Test API key validation
    print("\n1. API Key Validation:")
    api_status = benchmarker.validate_api_keys()
    for provider, status in api_status.items():
        status_symbol = "✓" if status else "✗"
        print(f"   {status_symbol} {provider}: {'Available' if status else 'Not available'}")
    
    # Test available models
    print("\n2. Available Models:")
    available_models = benchmarker.get_available_models()
    if available_models:
        for provider, model in available_models[:5]:  # Show first 5 models
            print(f"   - {provider}/{model}")
        if len(available_models) > 5:
            print(f"   ... and {len(available_models) - 5} more models")
    else:
        print("   No models available (no valid API keys)")
    
    # Test provider creation
    print("\n3. Provider Creation:")
    for provider_name in ["openai", "anthropic", "google", "groq"]:
        provider = benchmarker.create_provider(provider_name)
        if provider:
            print(f"   ✓ {provider_name}: {provider.__class__.__name__} created successfully")
        else:
            print(f"   ✗ {provider_name}: Failed to create provider")
    
    # Test benchmark method (if any providers are available)
    print("\n4. Benchmark Test:")
    if available_models:
        provider_name, model = available_models[0]
        print(f"   Testing benchmark with {provider_name}/{model}...")
        
        # Use a simple prompt for testing
        test_config = {"benchmark_prompt": "Write a simple hello world function in Python."}
        test_benchmarker = UnifiedBenchmarker(test_config)
        
        try:
            result = test_benchmarker.benchmark_model(provider_name, model)
            if result.success:
                print(f"   ✓ Benchmark successful:")
                print(f"     - Generation time: {result.generation_time:.2f}s")
                print(f"     - Tokens/sec: {result.tokens_per_second:.2f}")
                print(f"     - Cost: ${result.cost:.4f}")
                print(f"     - Response length: {len(result.response_content)} chars")
            else:
                print(f"   ✗ Benchmark failed: {result.error_message}")
        except Exception as e:
            print(f"   ✗ Benchmark error: {str(e)}")
    else:
        print("   Skipping benchmark test (no available models)")
    
    print("\nProvider interface test completed!")

if __name__ == "__main__":
    test_provider_interface()