# Design Document

## Overview

The unified LLM benchmarker will be a single Python script that combines benchmarking and evaluation into one streamlined process. It will test multiple LLM providers, evaluate response quality using an AI judge, and generate focused outputs for analysis without creating unnecessary intermediate files.

## Architecture

### High-Level Flow
```
1. Configuration & Setup
   ↓
2. API Key Validation
   ↓
3. Model Collection & Filtering
   ↓
4. Benchmark Execution (Performance + Content)
   ↓
5. AI-Powered Quality Evaluation
   ↓
6. Data Aggregation & Analysis
   ↓
7. Output Generation (Charts + Excel)
```

### Core Components

#### 1. Configuration Manager
- Loads environment variables and API keys
- Manages provider configurations and model lists
- Handles benchmark prompt and evaluation criteria
- Validates required dependencies

#### 2. Provider Interface
- Unified interface for all LLM providers
- Handles provider-specific API formats and authentication
- Manages retries and error handling
- Calculates costs based on current pricing

#### 3. Benchmark Engine
- Executes performance testing for each model
- Measures generation time, tokens/sec, and TTFT
- Captures model responses for quality evaluation
- Handles timeouts and failures gracefully

#### 4. Quality Evaluator
- Uses Google Gemini as AI judge
- Generates reference response for comparison
- Scores models on defined criteria (1-10 scale)
- Processes evaluation results into structured data

#### 5. Data Processor
- Aggregates performance and quality metrics
- Handles missing data with appropriate defaults
- Normalizes model names for consistent matching
- Calculates derived metrics and rankings

#### 6. Output Generator
- Creates four PDF charts with consistent styling
- Generates comprehensive Excel file with all metrics
- Handles file naming with timestamps
- Provides clear success/failure feedback

## Components and Interfaces

### BenchmarkResult Class
```python
@dataclass
class BenchmarkResult:
    model: str
    provider: str
    platform: str
    generation_time: float
    tokens_per_second: float
    time_to_first_token: float
    tokens_generated: int
    total_tokens: int
    cost: float
    response_content: str
    success: bool
    error_message: Optional[str] = None
```

### EvaluationResult Class
```python
@dataclass
class EvaluationResult:
    model: str
    correctness_score: float
    completeness_score: float
    code_quality_score: float
    readability_score: float
    error_handling_score: float
    overall_score: float
    pros: List[str]
    cons: List[str]
    summary: str
```

### UnifiedBenchmarker Class
```python
class UnifiedBenchmarker:
    def __init__(self, config: Dict[str, Any])
    def validate_api_keys(self) -> Dict[str, bool]
    def collect_available_models(self) -> List[Tuple[str, str]]
    def benchmark_model(self, provider: str, model: str) -> BenchmarkResult
    def evaluate_response_quality(self, benchmark_result: BenchmarkResult, reference_response: str) -> EvaluationResult
    def generate_outputs(self, results: List[Tuple[BenchmarkResult, EvaluationResult]])
    def run_complete_benchmark(self) -> None
```

## Data Models

### Provider Configuration
```python
PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "o3"],
        "api_key_env": "OPENAI_API_KEY",
        "pricing": {...}
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "models": ["qwen-3-235b-a22b", "qwen-3-32b", "llama-3.3-70b", "llama3.1-8b"],
        "api_key_env": "CEREBRAS_API_KEY",
        "pricing": {
            "qwen-3-235b-a22b": {"input": 0.60, "output": 0.60},
            "qwen-3-32b": {"input": 0.30, "output": 0.30},
            "llama-3.3-70b": {"input": 0.40, "output": 0.40},
            "llama3.1-8b": {"input": 0.10, "output": 0.10}
        }
    },
    # ... other providers
}
```

### Evaluation Criteria
```python
EVALUATION_CRITERIA = [
    "Correctness",
    "Completeness", 
    "Code Quality",
    "Readability",
    "Error Handling"
]
```

### Output File Structure
```
unified_benchmark_results_YYYY-MM-DD_HH-MM-SS.xlsx
speed_vs_quality_YYYY-MM-DD_HH-MM-SS.pdf
speed_comparison_YYYY-MM-DD_HH-MM-SS.pdf
quality_comparison_YYYY-MM-DD_HH-MM-SS.pdf
ttft_comparison_YYYY-MM-DD_HH-MM-SS.pdf
```

## Error Handling

### API Failures
- Implement exponential backoff retry mechanism (3 attempts)
- Log specific error messages for debugging
- Continue processing other models when one fails
- Track success/failure rates in final summary

### Missing Data
- Use "N/A" for unavailable metrics
- Exclude failed models from charts but note in Excel
- Provide default values for missing evaluation scores
- Handle partial results gracefully

### File Generation
- Validate output directory permissions
- Handle disk space issues
- Provide clear error messages for file creation failures
- Confirm successful file generation with full paths

## Testing Strategy

### Unit Testing
- Test each provider interface independently
- Mock API responses for consistent testing
- Validate data processing and aggregation logic
- Test error handling scenarios

### Integration Testing
- Test complete workflow with limited model set
- Validate output file generation and format
- Test with missing API keys and failed responses
- Verify chart generation and Excel formatting

### Performance Testing
- Measure execution time with multiple models
- Test parallel processing capabilities
- Validate memory usage with large result sets
- Test timeout handling for slow APIs

## Security Considerations

### API Key Management
- Load keys from environment variables only
- Never log or expose API keys in output
- Validate key format before making requests
- Handle missing keys gracefully

### Data Privacy
- Don't store model responses permanently
- Clear sensitive data from memory after processing
- Use secure HTTP connections for all API calls
- Respect provider rate limits and terms of service

## Performance Optimizations

### Parallel Processing
- Use ThreadPoolExecutor for concurrent API calls
- Limit concurrent requests to respect rate limits
- Process evaluation in parallel where possible
- Optimize chart generation for large datasets

### Memory Management
- Stream large responses instead of loading entirely
- Clear processed data to free memory
- Use generators for large result sets
- Optimize pandas operations for Excel generation

### Caching
- Cache reference response to avoid repeated API calls
- Store provider configurations in memory
- Reuse chart styling and formatting objects
- Cache compiled regex patterns for parsing