#!/usr/bin/env python3
"""Test script to verify configuration and API key management functionality."""

from unified_llm_benchmarker import UnifiedBenchmarker

def test_configuration():
    """Test the configuration and API key management functionality."""
    print("Testing Unified LLM Benchmarker Configuration...")
    
    # Initialize benchmarker
    benchmarker = UnifiedBenchmarker()
    
    # Test API key validation status
    print("\n=== API Key Validation Status ===")
    validation_status = benchmarker.validate_api_keys()
    for provider, is_valid in validation_status.items():
        status = "✓ Valid" if is_valid else "✗ Invalid/Missing"
        print(f"{provider}: {status}")
    
    # Test available models
    print("\n=== Available Models ===")
    available_models = benchmarker.get_available_models()
    print(f"Total available models: {len(available_models)}")
    
    for provider, model in available_models[:10]:  # Show first 10
        print(f"  {provider}: {model}")
    
    if len(available_models) > 10:
        print(f"  ... and {len(available_models) - 10} more models")
    
    # Test provider configuration
    print("\n=== Provider Configuration Test ===")
    test_providers = ["openai", "anthropic", "google"]
    for provider in test_providers:
        config = benchmarker.get_provider_config(provider)
        if config:
            print(f"✓ {provider}: Configuration loaded successfully")
            print(f"  Models: {len(config.get('models', []))} available")
            print(f"  Has API key: {'Yes' if config.get('api_key') else 'No'}")
        else:
            print(f"✗ {provider}: Configuration failed to load")
    
    print("\n=== Configuration Test Complete ===")

if __name__ == "__main__":
    test_configuration()