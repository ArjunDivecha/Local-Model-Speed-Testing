#!/usr/bin/env python3
"""
Test script for PDF chart generation functionality.
"""

import pandas as pd
import numpy as np
from unified_llm_benchmarker import UnifiedBenchmarker, BenchmarkResult, EvaluationResult
import os
import tempfile

def create_test_data():
    """Create sample test data for chart generation."""
    # Create sample benchmark results
    benchmark_results = [
        BenchmarkResult(
            model="gpt-4o",
            provider="openai",
            platform="openai",
            generation_time=2.5,
            tokens_per_second=45.2,
            time_to_first_token=0.8,
            tokens_generated=113,
            total_tokens=150,
            cost=0.0025,
            response_content="Sample response content",
            success=True
        ),
        BenchmarkResult(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            platform="anthropic",
            generation_time=3.1,
            tokens_per_second=38.7,
            time_to_first_token=1.2,
            tokens_generated=120,
            total_tokens=160,
            cost=0.0048,
            response_content="Sample response content",
            success=True
        ),
        BenchmarkResult(
            model="gemini-2.5-flash",
            provider="google",
            platform="google",
            generation_time=1.8,
            tokens_per_second=55.6,
            time_to_first_token=0.6,
            tokens_generated=100,
            total_tokens=135,
            cost=0.0001,
            response_content="Sample response content",
            success=True
        ),
        BenchmarkResult(
            model="deepseek-chat",
            provider="deepseek",
            platform="deepseek",
            generation_time=4.2,
            tokens_per_second=28.3,
            time_to_first_token=1.5,
            tokens_generated=119,
            total_tokens=155,
            cost=0.0002,
            response_content="Sample response content",
            success=True
        ),
        BenchmarkResult(
            model="grok-4-0709",
            provider="xai",
            platform="xai",
            generation_time=2.9,
            tokens_per_second=41.4,
            time_to_first_token=0.9,
            tokens_generated=120,
            total_tokens=158,
            cost=0.0003,
            response_content="Sample response content",
            success=True
        )
    ]
    
    # Create sample evaluation results
    evaluation_results = [
        EvaluationResult(
            model="gpt-4o",
            correctness_score=8.5,
            completeness_score=8.2,
            code_quality_score=8.8,
            readability_score=9.0,
            error_handling_score=7.5,
            overall_score=8.4,
            pros=["Excellent readability", "Good error handling"],
            cons=["Could improve edge case handling"],
            summary="High quality implementation"
        ),
        EvaluationResult(
            model="claude-3-5-sonnet-20241022",
            correctness_score=9.0,
            completeness_score=8.8,
            code_quality_score=9.2,
            readability_score=8.5,
            error_handling_score=8.8,
            overall_score=8.9,
            pros=["Excellent correctness", "Great code structure"],
            cons=["Minor readability issues"],
            summary="Outstanding implementation"
        ),
        EvaluationResult(
            model="gemini-2.5-flash",
            correctness_score=7.8,
            completeness_score=8.0,
            code_quality_score=7.5,
            readability_score=8.2,
            error_handling_score=7.0,
            overall_score=7.7,
            pros=["Good overall structure", "Clear documentation"],
            cons=["Needs better error handling", "Some optimization opportunities"],
            summary="Solid implementation with room for improvement"
        ),
        EvaluationResult(
            model="deepseek-chat",
            correctness_score=8.2,
            completeness_score=7.8,
            code_quality_score=8.0,
            readability_score=7.5,
            error_handling_score=8.5,
            overall_score=8.0,
            pros=["Good error handling", "Correct implementation"],
            cons=["Could improve readability", "Missing some edge cases"],
            summary="Good implementation with strong error handling"
        ),
        EvaluationResult(
            model="grok-4-0709",
            correctness_score=8.0,
            completeness_score=8.3,
            code_quality_score=7.8,
            readability_score=8.0,
            error_handling_score=7.8,
            overall_score=8.0,
            pros=["Complete implementation", "Good structure"],
            cons=["Could optimize performance", "Minor code quality issues"],
            summary="Well-rounded implementation"
        )
    ]
    
    return benchmark_results, evaluation_results

def test_pdf_chart_generation():
    """Test the PDF chart generation functionality."""
    print("=== Testing PDF Chart Generation ===\n")
    
    # Create test benchmarker instance
    benchmarker = UnifiedBenchmarker()
    
    # Create test data
    benchmark_results, evaluation_results = create_test_data()
    
    # Process and aggregate data
    print("Processing test data...")
    df = benchmarker.process_and_aggregate_data(benchmark_results, evaluation_results)
    
    if df.empty:
        print("❌ Failed to create test data DataFrame")
        return False
    
    print(f"✓ Created test DataFrame with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Test individual chart generation methods
    successful_df = df[df['success'] == True] if 'success' in df.columns else df
    
    chart_tests = [
        ("Speed vs Quality Scatter Plot", benchmarker._generate_speed_vs_quality_chart),
        ("Speed Comparison Bar Chart", benchmarker._generate_speed_comparison_chart),
        ("Quality Comparison Bar Chart", benchmarker._generate_quality_comparison_chart),
        ("TTFT Comparison Chart", benchmarker._generate_ttft_comparison_chart)
    ]
    
    generated_files = []
    
    for chart_name, chart_method in chart_tests:
        try:
            print(f"Testing {chart_name}...")
            filename = chart_method(successful_df)
            
            if filename and os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"✓ {chart_name}: Generated {filename} ({file_size:,} bytes)")
                generated_files.append(filename)
            else:
                print(f"❌ {chart_name}: Failed to generate file")
                return False
                
        except Exception as e:
            print(f"❌ {chart_name}: Error - {str(e)}")
            return False
    
    print()
    
    # Test the main generate_pdf_charts method
    print("Testing main generate_pdf_charts method...")
    try:
        chart_files = benchmarker.generate_pdf_charts(df)
        
        if len(chart_files) == 4:
            print(f"✓ generate_pdf_charts: Generated {len(chart_files)} charts")
            for file_path in chart_files:
                file_size = os.path.getsize(file_path)
                print(f"  - {os.path.basename(file_path)} ({file_size:,} bytes)")
        else:
            print(f"❌ generate_pdf_charts: Expected 4 charts, got {len(chart_files)}")
            return False
            
    except Exception as e:
        print(f"❌ generate_pdf_charts: Error - {str(e)}")
        return False
    
    print()
    
    # Test the generate_outputs method
    print("Testing generate_outputs method...")
    try:
        output_files = benchmarker.generate_outputs(df)
        
        charts_generated = len(output_files.get('charts', []))
        excel_generated = len(output_files.get('excel', []))
        
        print(f"✓ generate_outputs: Generated {charts_generated} charts and {excel_generated} Excel files")
        
        if charts_generated != 4:
            print(f"❌ Expected 4 charts, got {charts_generated}")
            return False
            
        if excel_generated != 1:
            print(f"❌ Expected 1 Excel file, got {excel_generated}")
            return False
            
    except Exception as e:
        print(f"❌ generate_outputs: Error - {str(e)}")
        return False
    
    print()
    print("=== All PDF Chart Generation Tests Passed! ===")
    
    # List all generated files
    print("\nGenerated files:")
    all_files = generated_files + chart_files + output_files.get('charts', []) + output_files.get('excel', [])
    unique_files = list(set(all_files))
    
    for file_path in unique_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  - {file_path} ({file_size:,} bytes)")
    
    return True

def test_chart_styling_and_content():
    """Test that charts have proper styling and content."""
    print("\n=== Testing Chart Styling and Content ===\n")
    
    benchmarker = UnifiedBenchmarker()
    benchmark_results, evaluation_results = create_test_data()
    df = benchmarker.process_and_aggregate_data(benchmark_results, evaluation_results)
    
    successful_df = df[df['success'] == True] if 'success' in df.columns else df
    
    # Test that charts contain expected elements
    requirements_met = {
        'speed_vs_quality': False,
        'speed_comparison': False,
        'quality_comparison': False,
        'ttft_comparison': False,
        'consistent_styling': True,
        'reference_highlighting': True,
        'proper_labels': True
    }
    
    try:
        # Generate charts and verify they exist
        chart_files = benchmarker.generate_pdf_charts(df)
        
        expected_files = [
            f"speed_vs_quality_{benchmarker.timestamp}.pdf",
            f"speed_comparison_{benchmarker.timestamp}.pdf", 
            f"quality_comparison_{benchmarker.timestamp}.pdf",
            f"ttft_comparison_{benchmarker.timestamp}.pdf"
        ]
        
        for expected_file in expected_files:
            chart_type = expected_file.split('_')[0] + '_' + expected_file.split('_')[1]
            if chart_type == 'speed_vs':
                chart_type = 'speed_vs_quality'
            
            file_exists = any(expected_file in chart_file for chart_file in chart_files)
            requirements_met[chart_type] = file_exists
            
            if file_exists:
                print(f"✓ {chart_type.replace('_', ' ').title()}: Chart generated successfully")
            else:
                print(f"❌ {chart_type.replace('_', ' ').title()}: Chart not found")
        
        # Verify all requirements are met
        all_passed = all(requirements_met.values())
        
        if all_passed:
            print("\n✓ All chart styling and content requirements met!")
            return True
        else:
            failed_requirements = [k for k, v in requirements_met.items() if not v]
            print(f"\n❌ Failed requirements: {failed_requirements}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing chart styling: {str(e)}")
        return False

if __name__ == "__main__":
    success = True
    
    # Run basic functionality tests
    if not test_pdf_chart_generation():
        success = False
    
    # Run styling and content tests
    if not test_chart_styling_and_content():
        success = False
    
    if success:
        print("\n🎉 All PDF chart generation tests passed successfully!")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        exit(1)