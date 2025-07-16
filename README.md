# Unified LLM Benchmarker

A comprehensive Python tool for benchmarking and evaluating multiple Large Language Model (LLM) providers in a single unified process. This tool measures both performance metrics (speed, tokens/sec, time-to-first-token) and response quality using AI-powered evaluation.

## 🚀 Features

### Multi-Provider Support
- **OpenAI**: GPT-4o, GPT-4o-mini, O3, O4-mini, GPT-3.5-turbo
- **Anthropic**: Claude Sonnet 4, Claude Opus 4, Claude 3.5 Sonnet, Claude 3.5 Haiku
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Cerebras**: Qwen-3-235B-A22B, Qwen-3-32B, Llama-3.3-70B, Llama3.1-8B
- **Groq**: Moonshot AI Kimi K2 Instruct
- **XAI**: Grok-4-0709
- **DeepSeek**: DeepSeek Chat, DeepSeek Coder, DeepSeek Reasoner

### Comprehensive Metrics
- **Performance**: Generation time, tokens per second, time to first token
- **Quality**: AI-powered evaluation using Claude Opus as judge
- **Cost**: Real-time API usage cost calculation
- **Reliability**: Success rates and error tracking

### Rich Output Formats
- **Excel Reports**: Comprehensive data with multiple sheets
- **PDF Charts**: Speed vs Quality, Performance comparisons, TTFT analysis
- **Detailed Logging**: Complete execution logs with timestamps

## 📋 Requirements

### Dependencies
```bash
pip install pandas matplotlib seaborn requests openai google-generativeai cerebras-cloud-sdk python-dotenv openpyxl numpy
```

### API Keys Required
Create a `.env` file in the project directory with your API keys:

```env
# Required API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
CEREBRAS_API_KEY=csk-...
GROQ_API_KEY=gsk_...
XAI_API_KEY=xai-...
DEEPSEEK_API_KEY=sk-...
```

**Note**: Missing API keys will cause those providers to be skipped automatically.

## 🏃‍♂️ Quick Start

### Basic Usage
```bash
python unified_llm_benchmarker.py
```

This will:
1. Validate all available API keys
2. Benchmark all available models
3. Evaluate response quality using AI judge
4. Generate comprehensive reports

### Expected Output Files
```
unified_benchmark_results_YYYY-MM-DD_HH-MM-SS.xlsx
speed_vs_quality_YYYY-MM-DD_HH-MM-SS.pdf
speed_comparison_YYYY-MM-DD_HH-MM-SS.pdf
quality_comparison_YYYY-MM-DD_HH-MM-SS.pdf
ttft_comparison_YYYY-MM-DD_HH-MM-SS.pdf
```

## 📊 Performance Benchmarks

### Speed Leaders (Tokens/Second)
Based on recent benchmarks:

1. **Cerebras Llama3.1-8B**: ~1,650 t/s
2. **Cerebras Llama-3.3-70B**: ~1,620 t/s
3. **Cerebras Qwen-3-32B**: ~1,300 t/s
4. **Cerebras Qwen-3-235B-A22B**: ~700 t/s
5. **Groq Moonshot Kimi**: ~200 t/s

### Quality Leaders (AI Judge Score /10)
1. **OpenAI O4-mini**: 9.8/10
2. **Anthropic Claude 3.5 Sonnet**: 9.6/10
3. **OpenAI O3**: 9.4/10
4. **Anthropic Claude 3.5 Haiku**: 9.2/10
5. **DeepSeek Coder**: 9.2/10

## 🔧 Configuration

### Custom Benchmark Prompt
Edit the `DEFAULT_BENCHMARK_PROMPT` in `unified_llm_benchmarker.py`:

```python
DEFAULT_BENCHMARK_PROMPT = """
Your custom prompt here...
"""
```

### Evaluation Criteria
Modify the `EVALUATION_CRITERIA` list to change quality assessment focus:

```python
EVALUATION_CRITERIA = [
    "Correctness",
    "Completeness", 
    "Code Quality",
    "Readability",
    "Error Handling"
]
```

### Provider Configuration
Each provider can be configured in the `PROVIDERS` dictionary:

```python
"cerebras": {
    "base_url": "https://api.cerebras.ai/v1",
    "models": ["qwen-3-235b-a22b", "qwen-3-32b", "llama-3.3-70b", "llama3.1-8b"],
    "api_key_env": "CEREBRAS_API_KEY",
    "pricing": {
        "qwen-3-235b-a22b": {"input": 0.60, "output": 0.60},
        # ... other models
    }
}
```

## 📈 Understanding the Results

### Excel Report Sheets
- **Benchmark Results**: Complete metrics for all models
- **Summary**: High-level overview and top performers
- **Performance Comparison**: Speed and cost focused analysis
- **Quality Analysis**: AI evaluation scores and summaries

### PDF Charts
- **Speed vs Quality**: Scatter plot showing performance trade-offs
- **Speed Comparison**: Bar chart of tokens per second
- **Quality Comparison**: Bar chart of AI judge scores
- **TTFT Comparison**: Time to first token analysis

### Key Metrics Explained
- **Tokens/Second**: Raw generation speed
- **Time to First Token (TTFT)**: Responsiveness measure
- **Quality Score**: AI judge evaluation (1-10 scale)
- **Cost**: Estimated API usage cost per request
- **Success Rate**: Reliability percentage

## 🛠️ Architecture

### Core Components
1. **Provider Interface**: Unified API abstraction
2. **Benchmark Engine**: Performance measurement
3. **Quality Evaluator**: AI-powered assessment
4. **Data Processor**: Results aggregation
5. **Output Generator**: Report creation

### Error Handling
- **Exponential Backoff**: 3 retry attempts for failed requests
- **Graceful Degradation**: Continue with available providers
- **Comprehensive Logging**: Detailed error tracking
- **Partial Results**: Handle incomplete data sets

## 🔍 Troubleshooting

### Common Issues

**API Key Errors**
```
❌ Provider xyz: Invalid API key format
```
- Verify API key format in `.env` file
- Check provider documentation for key format

**Model Not Found**
```
❌ Model abc does not exist or you do not have access
```
- Verify model name spelling
- Check provider account access/credits

**Rate Limiting**
```
❌ Rate limit exceeded
```
- The tool includes automatic retry with backoff
- Consider running with fewer concurrent requests

### Debug Mode
Enable detailed logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Adding New Providers
1. Create a new provider class inheriting from `BaseProvider`
2. Implement the `generate_response` method
3. Add provider configuration to `PROVIDERS` dictionary
4. Update the `create_provider` method

### Example Provider Implementation
```python
class NewProvider(BaseProvider):
    def generate_response(self, prompt: str, model: str, max_tokens: int = 2000) -> ProviderResponse:
        # Implementation here
        pass
```

## 📝 License

This project is open source. Please ensure you comply with all provider terms of service when using their APIs.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and API
- **Anthropic** for Claude models
- **Google** for Gemini models
- **Cerebras** for ultra-fast inference
- **Groq**, **XAI**, **DeepSeek** for additional model access

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review provider documentation for API-specific issues
3. Ensure all dependencies are properly installed
4. Verify API keys are correctly configured

---

**Note**: This tool makes API calls that may incur costs. Monitor your usage and set appropriate limits with each provider.