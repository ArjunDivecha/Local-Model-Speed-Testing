#!/usr/bin/env python3
"""
Test Excel generation functionality for the unified LLM benchmarker.
"""

import os
import sys
import tempfile
import pandas as pd
from datetime import datetime
from unified_llm_benchmarker import UnifiedBenchmarker, BenchmarkResult, EvaluationResult


def create_test_data():
    """Create sample benchmark and evaluation results for testing."""
    benchmark_results = [
        BenchmarkResult(
            model="gpt-4o",
            provider="openai",
            platform="openai",
            generation_time=2.5,
            tokens_per_second=45.2,
            time_to_first_token=0.3,
            tokens_generated=113,
            total_tokens=150,
            cost=0.0375,
            response_content="def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            success=True
        ),
        BenchmarkResult(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            platform="anthropic",
            generation_time=3.1,
            tokens_per_second=38.7,
            time_to_first_token=0.4,
            tokens_generated=120,
            total_tokens=160,
            cost=0.048,
            response_content="def binary_search(sorted_list, target):\n    \"\"\"Binary search implementation.\"\"\"\n    left, right = 0, len(sorted_list) - 1\n    \n    while left <= right:\n        mid = left + (right - left) // 2\n        \n        if sorted_list[mid] == target:\n            return mid\n        elif sorted_list[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1",
            success=True
        ),
        BenchmarkResult(
            model="gemini-2.5-flash",
            provider="google",
            platform="google",
            generation_time=1.8,
            tokens_per_second=55.6,
            time_to_first_token=0.2,
            tokens_generated=100,
            total_tokens=135,
            cost=0.000135,
            response_content="def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    \n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    \n    return -1",
            success=True
        ),
        BenchmarkResult(
            model="failed-model",
            provider="test",
            platform="test",
            generation_time=0.0,
            tokens_per_second=0.0,
            time_to_first_token=0.0,
            tokens_generated=0,
            total_tokens=0,
            cost=0.0,
            response_content="",
            success=False,
            error_message="API connection failed"
        )
    ]
    
    evaluation_results = [
        EvaluationResult(
            model="gpt-4o",
            correctness_score=9.0,
            completeness_score=8.5,
            code_quality_score=8.8,
            readability_score=9.2,
            error_handling_score=7.5,
            overall_score=8.6,
            pros=["Excellent structure", "Clear variable names", "Proper algorithm implementation"],
            cons=["Could use more error handling", "Missing docstring"],
            summary="High-quality implementation with good performance"
        ),
        EvaluationResult(
            model="claude-3-5-sonnet-20241022",
            correctness_score=9.5,
            completeness_score=9.0,
            code_quality_score=9.2,
            readability_score=9.5,
            error_handling_score=8.0,
            overall_score=9.0,
            pros=["Comprehensive docstring", "Excellent readability", "Proper overflow handling"],
            cons=["Minor optimization opportunity"],
            summary="Excellent implementation following best practices"
        ),
        EvaluationResult(
            model="gemini-2.5-flash",
            correctness_score=8.5,
            completeness_score=8.0,
            code_quality_score=8.2,
            readability_score=8.8,
            error_handling_score=7.0,
            overall_score=8.1,
            pros=["Clean implementation", "Good variable naming"],
            cons=["Missing docstring", "Limited error handling"],
            summary="Solid implementation with room for improvement"
        )
    ]
    
    return benchmark_results, evaluation_results


def test_excel_generation():
    """Test the Excel generation functionality."""
    print("Testing Excel generation functionality...")
    
    # Create test benchmarker instance
    benchmarker = UnifiedBenchmarker()
    
    # Create test data
    benchmark_results, evaluation_results = create_test_data()
    
    # Process and aggregate data
    print("Processing test data...")
    df = benchmarker.process_and_aggregate_data(benchmark_results, evaluation_results)
    
    print(f"Generated DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print("Columns:", list(df.columns))
    
    # Test Excel generation
    print("Generating Excel file...")
    excel_path = benchmarker.generate_excel_output(df)
    
    if excel_path and os.path.exists(excel_path):
        print(f"✓ Excel file generated successfully: {excel_path}")
        
        # Verify file contents
        try:
            # Read the main sheet
            main_df = pd.read_excel(excel_path, sheet_name='Benchmark Results')
            print(f"✓ Main sheet contains {len(main_df)} rows")
            
            # Check for expected columns
            expected_columns = ['model', 'provider', 'generation_time', 'tokens_per_second', 'overall_quality_score', 'cost']
            missing_columns = [col for col in expected_columns if col not in main_df.columns]
            if missing_columns:
                print(f"⚠ Missing expected columns: {missing_columns}")
            else:
                print("✓ All expected columns present")
            
            # Read other sheets
            try:
                summary_df = pd.read_excel(excel_path, sheet_name='Summary')
                print(f"✓ Summary sheet contains {len(summary_df)} rows")
            except Exception as e:
                print(f"⚠ Could not read Summary sheet: {e}")
            
            try:
                perf_df = pd.read_excel(excel_path, sheet_name='Performance Comparison')
                print(f"✓ Performance sheet contains {len(perf_df)} rows")
            except Exception as e:
                print(f"⚠ Could not read Performance sheet: {e}")
            
            try:
                quality_df = pd.read_excel(excel_path, sheet_name='Quality Analysis')
                print(f"✓ Quality sheet contains {len(quality_df)} rows")
            except Exception as e:
                print(f"⚠ Could not read Quality sheet: {e}")
            
            # Check sorting (should be sorted by composite_score descending)
            if 'composite_score' in main_df.columns and len(main_df) > 1:
                successful_rows = main_df[main_df['success'] == True]
                if len(successful_rows) > 1:
                    is_sorted = successful_rows['composite_score'].is_monotonic_decreasing
                    if is_sorted:
                        print("✓ Results are properly sorted by composite score")
                    else:
                        print("⚠ Results may not be properly sorted")
            
            print(f"✓ Excel generation test completed successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error reading Excel file: {e}")
            return False
    else:
        print(f"✗ Excel file generation failed")
        return False


def test_empty_data_handling():
    """Test Excel generation with empty data."""
    print("\nTesting empty data handling...")
    
    benchmarker = UnifiedBenchmarker()
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    excel_path = benchmarker.generate_excel_output(empty_df)
    
    if not excel_path:
        print("✓ Empty data handled correctly (no file generated)")
        return True
    else:
        print("⚠ Unexpected behavior with empty data")
        return False


def test_formatting_requirements():
    """Test that Excel output meets formatting requirements."""
    print("\nTesting formatting requirements...")
    
    benchmarker = UnifiedBenchmarker()
    benchmark_results, evaluation_results = create_test_data()
    df = benchmarker.process_and_aggregate_data(benchmark_results, evaluation_results)
    
    excel_path = benchmarker.generate_excel_output(df)
    
    if not excel_path or not os.path.exists(excel_path):
        print("✗ Could not generate Excel file for formatting test")
        return False
    
    try:
        # Check file naming convention
        filename = os.path.basename(excel_path)
        expected_pattern = "unified_benchmark_results_"
        if filename.startswith(expected_pattern) and filename.endswith('.xlsx'):
            print("✓ File naming convention correct")
        else:
            print(f"⚠ File naming may not follow convention: {filename}")
        
        # Read and check data organization
        main_df = pd.read_excel(excel_path, sheet_name='Benchmark Results')
        
        # Check for required columns per requirements
        required_columns = [
            'model', 'provider', 'platform',  # 4.1: model name, provider, platform
            'generation_time', 'tokens_per_second', 'time_to_first_token',  # 4.2: performance metrics
            'overall_quality_score',  # 4.3: quality scores
            'cost'  # 4.4: cost estimates
        ]
        
        missing_required = [col for col in required_columns if col not in main_df.columns]
        if missing_required:
            print(f"✗ Missing required columns: {missing_required}")
            return False
        else:
            print("✓ All required columns present per requirements 4.1-4.4")
        
        # Check sorting (requirement 4.6)
        if 'final_rank' in main_df.columns:
            is_ranked = main_df['final_rank'].equals(pd.Series(range(1, len(main_df) + 1)))
            if is_ranked:
                print("✓ Results properly ranked (requirement 4.6)")
            else:
                print("⚠ Ranking may not be correct")
        
        print("✓ Formatting requirements test completed")
        return True
        
    except Exception as e:
        print(f"✗ Error checking formatting requirements: {e}")
        return False


def main():
    """Run all Excel generation tests."""
    print("=" * 60)
    print("UNIFIED LLM BENCHMARKER - EXCEL GENERATION TESTS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic Excel generation
    if test_excel_generation():
        tests_passed += 1
    
    # Test 2: Empty data handling
    if test_empty_data_handling():
        tests_passed += 1
    
    # Test 3: Formatting requirements
    if test_formatting_requirements():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"EXCEL GENERATION TESTS COMPLETED: {tests_passed}/{total_tests} passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("✓ All Excel generation tests passed!")
        return True
    else:
        print(f"✗ {total_tests - tests_passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)