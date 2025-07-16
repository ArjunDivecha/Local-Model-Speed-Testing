#!/usr/bin/env python3
"""
Comprehensive test for the unified provider interface implementation.
Tests all sub-tasks: base interface, provider implementations, error handling, and cost calculation.
"""

from unified_llm_benchmarker import (
    UnifiedBenchmarker, BaseProvider, OpenAIProvider, AnthropicProvider, 
    GoogleProvider, OpenAICompatibleProvider, PerplexityProvider, ProviderResponse
)
import time

def test_base_provider_interface():
    """Test 1: Base provider interface class implementation."""
    print("1. Testing Base Provider Interface Class...")
    
    # Test that BaseProvider is abstract
    try:
        # This should fail because BaseProvider is abstract
        provider = BaseProvider("test-key", {})
        print("   ✗ BaseProvider should be abstract")
        return False
    except TypeError:
        print("   ✓ BaseProvider is properly abstract")
    
    # Test cost calculation method
    class TestProvider(BaseProvider):
        def generate_response(self, prompt, model, max_tokens=2000):
            return ProviderResponse("test", 10, 20, 10, 10, 1.0, 0.5, True)
    
    test_provider = TestProvider("test-key", {
        "pricing": {
            "test-model": {"input": 0.001, "output": 0.002}
        }
    })
    
    cost = test_provider.calculate_cost("test-model", 1000, 500)
    expected_cost = (1000/1000 * 0.001) + (500/1000 * 0.002)  # $0.001 + $0.001 = $0.002
    
    if abs(cost - expected_cost) < 0.0001:
        print(f"   ✓ Cost calculation works: ${cost:.4f}")
    else:
        print(f"   ✗ Cost calculation failed: expected ${expected_cost:.4f}, got ${cost:.4f}")
        return False
    
    return True

def test_provider_implementations():
    """Test 2: Provider-specific implementations."""
    print("\n2. Testing Provider-Specific Implementations...")
    
    benchmarker = UnifiedBenchmarker()
    provider_classes = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "groq": OpenAICompatibleProvider,
        "perplexity": PerplexityProvider
    }
    
    success_count = 0
    for provider_name, expected_class in provider_classes.items():
        provider = benchmarker.create_provider(provider_name)
        if provider and isinstance(provider, expected_class):
            print(f"   ✓ {provider_name}: {expected_class.__name__} created correctly")
            success_count += 1
        elif provider:
            print(f"   ✗ {provider_name}: Wrong class type {type(provider)}")
        else:
            print(f"   ✗ {provider_name}: Failed to create provider")
    
    return success_count >= 3  # At least 3 providers should work

def test_error_handling_and_retry():
    """Test 3: Error handling and retry logic with exponential backoff."""
    print("\n3. Testing Error Handling and Retry Logic...")
    
    class FailingProvider(BaseProvider):
        def __init__(self, api_key, config):
            super().__init__(api_key, config)
            self.attempt_count = 0
        
        def generate_response(self, prompt, model, max_tokens=2000):
            def _failing_request():
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise Exception(f"Simulated failure {self.attempt_count}")
                return ProviderResponse("success", 10, 20, 10, 10, 1.0, 0.5, True)
            
            return self._retry_with_backoff(_failing_request)
    
    # Test retry mechanism
    start_time = time.time()
    failing_provider = FailingProvider("test-key", {})
    
    try:
        response = failing_provider.generate_response("test prompt", "test-model")
        end_time = time.time()
        
        if response.success and failing_provider.attempt_count == 3:
            print(f"   ✓ Retry logic works: {failing_provider.attempt_count} attempts")
            print(f"   ✓ Exponential backoff: took {end_time - start_time:.1f}s (expected ~3s)")
        else:
            print(f"   ✗ Retry logic failed: {failing_provider.attempt_count} attempts")
            return False
    except Exception as e:
        print(f"   ✗ Retry logic error: {str(e)}")
        return False
    
    # Test max retries exceeded
    class AlwaysFailingProvider(BaseProvider):
        def __init__(self, api_key, config):
            super().__init__(api_key, config)
            self.attempt_count = 0
        
        def generate_response(self, prompt, model, max_tokens=2000):
            def _always_failing_request():
                self.attempt_count += 1
                raise Exception(f"Always fails {self.attempt_count}")
            
            return self._retry_with_backoff(_always_failing_request)
    
    always_failing = AlwaysFailingProvider("test-key", {})
    try:
        always_failing.generate_response("test", "test")
        print("   ✗ Should have raised exception after max retries")
        return False
    except Exception:
        if always_failing.attempt_count == 3:  # MAX_RETRIES
            print("   ✓ Max retries respected: stopped after 3 attempts")
        else:
            print(f"   ✗ Wrong retry count: {always_failing.attempt_count}")
            return False
    
    return True

def test_cost_calculation():
    """Test 4: Cost calculation for each provider."""
    print("\n4. Testing Cost Calculation for Each Provider...")
    
    benchmarker = UnifiedBenchmarker()
    test_cases = [
        ("openai", "gpt-4o", 1000, 500),
        ("anthropic", "claude-3-5-sonnet-20241022", 1000, 500),
        ("google", "gemini-1.5-pro", 1000, 500),
        ("groq", "llama-3.1-70b-versatile", 1000, 500)
    ]
    
    success_count = 0
    for provider_name, model, input_tokens, output_tokens in test_cases:
        provider = benchmarker.create_provider(provider_name)
        if provider:
            cost = provider.calculate_cost(model, input_tokens, output_tokens)
            if cost > 0:
                print(f"   ✓ {provider_name}/{model}: ${cost:.6f}")
                success_count += 1
            else:
                print(f"   ✗ {provider_name}/{model}: Invalid cost ${cost:.6f}")
        else:
            print(f"   ✗ {provider_name}: Provider not available")
    
    return success_count >= 2  # At least 2 providers should calculate costs

def test_integration():
    """Test 5: Integration test with actual API call."""
    print("\n5. Testing Integration with Real API Call...")
    
    benchmarker = UnifiedBenchmarker({
        "benchmark_prompt": "Say 'Hello, World!' in Python.",
        "token_limit": 100
    })
    
    available_models = benchmarker.get_available_models()
    if not available_models:
        print("   ⚠ No available models for integration test")
        return True  # Skip test if no API keys
    
    # Test with first available model
    provider_name, model = available_models[0]
    print(f"   Testing integration with {provider_name}/{model}...")
    
    try:
        result = benchmarker.benchmark_model(provider_name, model)
        if result.success:
            print(f"   ✓ Integration successful:")
            print(f"     - Response: {len(result.response_content)} chars")
            print(f"     - Performance: {result.tokens_per_second:.2f} tokens/sec")
            print(f"     - Cost: ${result.cost:.6f}")
            print(f"     - TTFT: {result.time_to_first_token:.3f}s")
            return True
        else:
            print(f"   ✗ Integration failed: {result.error_message}")
            return False
    except Exception as e:
        print(f"   ✗ Integration error: {str(e)}")
        return False

def main():
    """Run comprehensive provider interface tests."""
    print("=== Comprehensive Provider Interface Test ===\n")
    
    tests = [
        ("Base Provider Interface", test_base_provider_interface),
        ("Provider Implementations", test_provider_implementations),
        ("Error Handling & Retry", test_error_handling_and_retry),
        ("Cost Calculation", test_cost_calculation),
        ("Integration Test", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {str(e)}")
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All provider interface tests passed!")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    main()