#!/usr/bin/env python3
"""
Test data processing and aggregation functionality for the Unified LLM Benchmarker.
"""

import unittest
import pandas as pd
from unified_llm_benchmarker import UnifiedBenchmarker, BenchmarkResult, EvaluationResult


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing and aggregation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmarker = UnifiedBenchmarker()
        
        # Create sample benchmark results
        self.sample_benchmark_results = [
            BenchmarkResult(
                model="gpt-4o",
                provider="openai",
                platform="openai",
                generation_time=2.5,
                tokens_per_second=45.2,
                time_to_first_token=0.3,
                tokens_generated=113,
                total_tokens=150,
                cost=0.0025,
                response_content="def binary_search(arr, target):\n    # Implementation here",
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
                cost=0.0048,
                response_content="def binary_search(sorted_list, target):\n    # Better implementation",
                success=True
            ),
            BenchmarkResult(
                model="gemini-1.5-pro",
                provider="google",
                platform="google",
                generation_time=0.0,
                tokens_per_second=0.0,
                time_to_first_token=0.0,
                tokens_generated=0,
                total_tokens=0,
                cost=0.0,
                response_content="",
                success=False,
                error_message="API timeout"
            ),
            # Duplicate model with different provider (for deduplication testing)
            BenchmarkResult(
                model="gpt-4o",
                provider="openrouter",
                platform="openrouter",
                generation_time=2.8,
                tokens_per_second=42.1,
                time_to_first_token=0.35,
                tokens_generated=118,
                total_tokens=155,
                cost=0.0030,
                response_content="def binary_search(array, value):\n    # Another implementation",
                success=True
            )
        ]
        
        # Create sample evaluation results
        self.sample_evaluation_results = [
            EvaluationResult(
                model="gpt-4o",
                correctness_score=8.5,
                completeness_score=9.0,
                code_quality_score=8.0,
                readability_score=8.5,
                error_handling_score=7.5,
                overall_score=8.3,
                pros=["Clear implementation", "Good error handling"],
                cons=["Could be more efficient"],
                summary="Solid implementation with good practices"
            ),
            EvaluationResult(
                model="claude-3-5-sonnet-20241022",
                correctness_score=9.0,
                completeness_score=9.5,
                code_quality_score=9.0,
                readability_score=9.0,
                error_handling_score=8.5,
                overall_score=9.0,
                pros=["Excellent structure", "Comprehensive error handling", "Clear documentation"],
                cons=["Minor optimization opportunity"],
                summary="Excellent implementation following best practices"
            )
        ]
    
    def test_normalize_model_name(self):
        """Test model name normalization."""
        test_cases = [
            ("gpt-4o", "gpt-4o"),
            ("models/gemini-1.5-pro", "gemini-1.5-pro"),
            ("anthropic/claude-3-5-sonnet-20241022", "claude-3-5-sonnet"),
            ("meta-llama/llama-3.1-405b-instruct", "llama-3.1-405b"),
            ("deepseek-chat-v1.5", "deepseek-chat"),
            ("", "unknown"),
            (None, "unknown")
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = self.benchmarker.normalize_model_name(input_name)
                self.assertEqual(result, expected)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create results with missing/invalid values
        benchmark_with_missing = [
            BenchmarkResult(
                model="",  # Missing model name
                provider="openai",
                platform="",  # Missing platform
                generation_time=-1.0,  # Invalid negative time
                tokens_per_second=-5.0,  # Invalid negative rate
                time_to_first_token=0.0,
                tokens_generated=-10,  # Invalid negative tokens
                total_tokens=0,
                cost=-0.01,  # Invalid negative cost
                response_content="",
                success=True
            )
        ]
        
        evaluation_with_missing = [
            EvaluationResult(
                model="",
                correctness_score=15.0,  # Out of range (should be 1-10)
                completeness_score=-2.0,  # Out of range
                code_quality_score=5.0,
                readability_score=8.0,
                error_handling_score=7.0,
                overall_score=6.6,
                pros=[],
                cons=[],
                summary=""
            )
        ]
        
        processed_benchmark, processed_evaluation = self.benchmarker.handle_missing_values(
            benchmark_with_missing, evaluation_with_missing
        )
        
        # Check benchmark result processing
        result = processed_benchmark[0]
        self.assertEqual(result.model, "unknown")
        self.assertEqual(result.platform, "openai")  # Should use provider as fallback
        self.assertEqual(result.generation_time, 0.0)
        self.assertEqual(result.tokens_per_second, 0.0)
        self.assertEqual(result.tokens_generated, 0)
        self.assertEqual(result.cost, 0.0)
        
        # Check evaluation result processing
        eval_result = processed_evaluation[0]
        self.assertEqual(eval_result.model, "unknown")
        self.assertEqual(eval_result.correctness_score, 10.0)  # Clamped to max
        self.assertEqual(eval_result.completeness_score, 1.0)  # Clamped to min
        self.assertIsNotNone(eval_result.summary)
    
    def test_deduplicate_results(self):
        """Test result deduplication."""
        # Use sample results that include duplicates
        deduplicated_benchmark, deduplicated_evaluation = self.benchmarker.deduplicate_results(
            self.sample_benchmark_results, self.sample_evaluation_results
        )
        
        # Should have 3 unique models (gpt-4o deduplicated, claude, gemini)
        self.assertEqual(len(deduplicated_benchmark), 3)
        
        # Check that the better performing gpt-4o was kept (higher tokens_per_second)
        gpt4_results = [r for r in deduplicated_benchmark if "gpt-4o" in r.model.lower()]
        self.assertEqual(len(gpt4_results), 1)
        self.assertEqual(gpt4_results[0].tokens_per_second, 45.2)  # Should keep the better one
    
    def test_aggregate_results(self):
        """Test result aggregation into DataFrame."""
        df = self.benchmarker.aggregate_results(
            self.sample_benchmark_results, self.sample_evaluation_results
        )
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.sample_benchmark_results))
        
        # Check required columns exist
        required_columns = [
            'model', 'provider', 'platform', 'normalized_name',
            'generation_time', 'tokens_per_second', 'time_to_first_token',
            'tokens_generated', 'total_tokens', 'cost',
            'correctness_score', 'completeness_score', 'code_quality_score',
            'readability_score', 'error_handling_score', 'overall_quality_score',
            'success', 'evaluation_available', 'cost_per_token', 'efficiency_score'
        ]
        
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        
        # Check that successful results have rankings
        successful_rows = df[df['success'] == True]
        if len(successful_rows) > 0:
            self.assertIn('speed_rank', df.columns)
            self.assertIn('quality_rank', df.columns)
            self.assertIn('overall_rank', df.columns)
    
    def test_sort_and_rank_results(self):
        """Test result sorting and ranking."""
        df = self.benchmarker.aggregate_results(
            self.sample_benchmark_results, self.sample_evaluation_results
        )
        
        sorted_df = self.benchmarker.sort_and_rank_results(df, sort_by='tokens_per_second')
        
        # Check that results are sorted correctly
        self.assertIn('final_rank', sorted_df.columns)
        self.assertEqual(len(sorted_df), len(df))
        
        # Successful results should come first
        successful_indices = sorted_df[sorted_df['success'] == True].index
        failed_indices = sorted_df[sorted_df['success'] == False].index
        
        if len(successful_indices) > 0 and len(failed_indices) > 0:
            self.assertTrue(all(s < f for s in successful_indices for f in failed_indices))
    
    def test_process_and_aggregate_data_complete_pipeline(self):
        """Test the complete data processing pipeline."""
        final_df = self.benchmarker.process_and_aggregate_data(
            self.sample_benchmark_results, self.sample_evaluation_results
        )
        
        # Check that pipeline completed successfully
        self.assertIsInstance(final_df, pd.DataFrame)
        self.assertGreater(len(final_df), 0)
        
        # Check that all processing steps were applied
        self.assertIn('normalized_name', final_df.columns)
        self.assertIn('final_rank', final_df.columns)
        self.assertIn('composite_score', final_df.columns)
        
        # Check that successful results are ranked higher
        if len(final_df[final_df['success'] == True]) > 0:
            successful_ranks = final_df[final_df['success'] == True]['final_rank']
            failed_ranks = final_df[final_df['success'] == False]['final_rank']
            
            if len(failed_ranks) > 0:
                self.assertTrue(successful_ranks.max() < failed_ranks.min())
    
    def test_empty_results_handling(self):
        """Test handling of empty result lists."""
        empty_df = self.benchmarker.aggregate_results([], [])
        self.assertTrue(empty_df.empty)
        
        sorted_empty = self.benchmarker.sort_and_rank_results(empty_df)
        self.assertTrue(sorted_empty.empty)
        
        processed_empty = self.benchmarker.process_and_aggregate_data([], [])
        self.assertTrue(processed_empty.empty)


if __name__ == "__main__":
    unittest.main()