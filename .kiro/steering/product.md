# Product Overview

## Unified LLM Benchmarker

A comprehensive Python tool for benchmarking and evaluating multiple LLM providers in a single unified process. The tool measures both performance metrics (speed, tokens/sec, time-to-first-token) and response quality using AI-powered evaluation.

### Key Features

- **Multi-Provider Support**: Tests OpenAI, Anthropic, Google, Groq, Perplexity, DeepSeek, XAI, and OpenRouter models
- **Unified Interface**: Single command execution for complete benchmarking and evaluation workflow
- **AI-Powered Quality Assessment**: Uses Google Gemini as an AI judge to score responses on multiple criteria
- **Visual Analytics**: Generates PDF charts comparing speed vs quality, performance metrics, and TTFT
- **Comprehensive Reporting**: Exports detailed Excel files with all metrics and analysis
- **Cost Tracking**: Calculates API usage costs based on current provider pricing
- **Robust Error Handling**: Graceful failure handling with retry logic and detailed logging

### Target Users

- Developers comparing LLM providers for integration decisions
- Researchers analyzing model performance and quality trade-offs
- Data analysts requiring comprehensive metrics for custom analysis
- System administrators managing LLM infrastructure and costs