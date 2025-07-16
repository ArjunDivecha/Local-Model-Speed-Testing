# Project Structure

## Root Directory Layout

```
unified-llm-benchmarker/
├── .env                              # API keys and environment variables
├── .kiro/                           # Kiro configuration and specs
│   ├── specs/unified-llm-benchmarker/  # Project specifications
│   └── steering/                    # AI assistant guidance rules
├── __pycache__/                     # Python bytecode cache
├── unified_llm_benchmarker.py       # Main application module
└── test_*.py                        # Test suite files
```

## Core Files

### Main Application
- **`unified_llm_benchmarker.py`**: Single-file application containing all classes and functionality
  - Provider implementations (OpenAI, Anthropic, Google, etc.)
  - Benchmark engine and evaluation logic
  - Data processing and output generation
  - Configuration management

### Test Suite
- **`test_config.py`**: Configuration and API key validation tests
- **`test_provider_interface.py`**: Provider interface and creation tests
- **`test_provider_comprehensive.py`**: Comprehensive provider functionality tests
- **`test_benchmark_engine.py`**: Benchmark execution and metrics tests
- **`test_evaluation.py`**: AI evaluation and scoring tests
- **`test_api_connections.py`**: API connectivity and authentication tests
- **`test_benchmark_comprehensive.py`**: End-to-end benchmark workflow tests

### Configuration
- **`.env`**: Environment variables for API keys (not committed to version control)
- **`.kiro/specs/`**: Project requirements, design, and task specifications
- **`.kiro/steering/`**: AI assistant guidance and project conventions

## Code Organization Patterns

### Class Hierarchy
```
BaseProvider (ABC)
├── OpenAIProvider
├── AnthropicProvider  
├── GoogleProvider
├── PerplexityProvider
└── OpenAICompatibleProvider (Groq, DeepSeek, XAI, OpenRouter)
```

### Data Classes
- **`BenchmarkResult`**: Performance metrics and response data
- **`EvaluationResult`**: Quality scores and AI judge feedback
- **`ProviderResponse`**: Standardized API response format

### Main Controller
- **`UnifiedBenchmarker`**: Orchestrates the entire benchmark workflow

## Output Structure

Generated files follow timestamp naming convention:
```
unified_benchmark_results_YYYY-MM-DD_HH-MM-SS.xlsx
speed_vs_quality_YYYY-MM-DD_HH-MM-SS.pdf
speed_comparison_YYYY-MM-DD_HH-MM-SS.pdf
quality_comparison_YYYY-MM-DD_HH-MM-SS.pdf
ttft_comparison_YYYY-MM-DD_HH-MM-SS.pdf
```

## Development Conventions

- **Single File Architecture**: All functionality consolidated in one module for simplicity
- **Test-Driven Structure**: Comprehensive test coverage with focused test files
- **Environment-Based Configuration**: API keys and settings via environment variables
- **Dataclass Usage**: Structured data representation with type hints
- **Abstract Base Classes**: Unified interfaces for provider implementations
- **Error Handling**: Graceful failure handling with detailed logging