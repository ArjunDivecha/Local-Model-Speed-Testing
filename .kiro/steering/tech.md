# Technology Stack

## Core Technologies

- **Python 3.x**: Primary programming language
- **Environment Management**: `.env` files with `python-dotenv` for API key management
- **Data Processing**: `pandas` for data manipulation and Excel export
- **Visualization**: `matplotlib` and `seaborn` for PDF chart generation
- **HTTP Requests**: `requests` library for API communication
- **Concurrency**: `ThreadPoolExecutor` for parallel API calls

## LLM Provider SDKs

- **OpenAI**: Official `openai` Python client
- **Google**: `google.generativeai` for Gemini models
- **Anthropic**: Direct REST API calls via `requests`
- **Others**: OpenAI-compatible endpoints for Groq, DeepSeek, XAI, OpenRouter, Perplexity

## Key Dependencies

```python
# Core libraries
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
requests>=2.28.0
python-dotenv>=0.19.0

# LLM Provider SDKs
openai>=1.0.0
google-generativeai>=0.3.0

# Data classes and type hints
dataclasses (built-in)
typing (built-in)
```

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Testing
```bash
# Run configuration tests
python test_config.py

# Test provider interfaces
python test_provider_interface.py

# Run comprehensive tests
python test_provider_comprehensive.py

# Test benchmark engine
python test_benchmark_engine.py
```

### Execution
```bash
# Run full benchmark
python unified_llm_benchmarker.py

# Run with custom configuration
python unified_llm_benchmarker.py --config custom_config.json
```

## Architecture Patterns

- **Abstract Base Classes**: `BaseProvider` for unified provider interface
- **Data Classes**: Structured data with `@dataclass` decorator
- **Factory Pattern**: Provider creation based on configuration
- **Retry Pattern**: Exponential backoff for API failures
- **Streaming**: Real-time token processing for performance metrics