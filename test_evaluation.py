#!/usr/bin/env python3
"""
Test script for AI-powered quality evaluation functionality.
"""

import os
from unified_llm_benchmarker import UnifiedBenchmarker, BenchmarkResult

def test_evaluation():
    """Test the AI-powered quality evaluation functionality."""
    
    # Create a sample benchmark result for testing
    sample_result = BenchmarkResult(
        model="test-model",
        provider="test-provider",
        platform="test-platform",
        generation_time=2.5,
        tokens_per_second=50.0,
        time_to_first_token=0.5,
        tokens_generated=125,
        total_tokens=200,
        cost=0.01,
        response_content="""
def binary_search(arr, target):
    \"\"\"
    Implements binary search algorithm on a sorted array.
    
    Args:
        arr: Sorted list of elements
        target: Element to search for
    
    Returns:
        Index of target if found, -1 otherwise
    \"\"\"
    if not arr:
        return -1
    
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
""",
        success=True
    )
    
    # Initialize benchmarker
    benchmarker = UnifiedBenchmarker()
    
    print("Testing AI-powered quality evaluation...")
    print(f"Available providers: {list(benchmarker.available_providers.keys())}")
    print(f"Anthropic available: {benchmarker.available_providers.get('anthropic', False)}")
    
    # Test reference response generation
    print("\n1. Testing reference response generation...")
    reference_response = benchmarker.generate_reference_response()
    if reference_response:
        print(f"✓ Reference response generated ({len(reference_response)} characters)")
        print(f"Preview: {reference_response[:200]}...")
    else:
        print("✗ Reference response generation failed")
    
    # Test quality evaluation
    print("\n2. Testing quality evaluation...")
    evaluation_result = benchmarker.evaluate_response_quality(sample_result, reference_response)
    
    if evaluation_result:
        print(f"✓ Evaluation completed for {evaluation_result.model}")
        print(f"  - Correctness: {evaluation_result.correctness_score:.1f}/10")
        print(f"  - Completeness: {evaluation_result.completeness_score:.1f}/10")
        print(f"  - Code Quality: {evaluation_result.code_quality_score:.1f}/10")
        print(f"  - Readability: {evaluation_result.readability_score:.1f}/10")
        print(f"  - Error Handling: {evaluation_result.error_handling_score:.1f}/10")
        print(f"  - Overall Score: {evaluation_result.overall_score:.1f}/10")
        
        if evaluation_result.pros:
            print(f"  - Pros: {evaluation_result.pros}")
        if evaluation_result.cons:
            print(f"  - Cons: {evaluation_result.cons}")
        if evaluation_result.summary:
            print(f"  - Summary: {evaluation_result.summary}")
    else:
        print("✗ Evaluation failed (returned None)")
    
    # Test with failed benchmark result
    print("\n3. Testing evaluation with failed benchmark...")
    failed_result = BenchmarkResult(
        model="failed-model",
        provider="test-provider",
        platform="test-platform",
        generation_time=0.0,
        tokens_per_second=0.0,
        time_to_first_token=0.0,
        tokens_generated=0,
        total_tokens=0,
        cost=0.0,
        response_content="",
        success=False,
        error_message="API call failed"
    )
    
    failed_evaluation = benchmarker.evaluate_response_quality(failed_result)
    if failed_evaluation:
        print(f"✓ Failed evaluation handled: {failed_evaluation.overall_score:.1f}/10")
        print(f"  - Summary: {failed_evaluation.summary}")
    else:
        print("✓ Failed evaluation properly returned None")
    
    # Test evaluation failure when Anthropic is not available
    print("\n4. Testing evaluation failure without Anthropic...")
    # Temporarily disable Anthropic
    original_anthropic_status = benchmarker.available_providers.get('anthropic', False)
    benchmarker.available_providers['anthropic'] = False
    
    no_anthropic_evaluation = benchmarker.evaluate_response_quality(sample_result)
    if no_anthropic_evaluation is None:
        print("✓ Evaluation properly failed when Claude Opus not available")
    else:
        print("✗ Evaluation should have failed when Claude Opus not available")
    
    # Restore Anthropic status
    benchmarker.available_providers['anthropic'] = original_anthropic_status
    
    print("\n✅ All evaluation tests completed!")

if __name__ == "__main__":
    test_evaluation()