# CLAUDE.md - Local Model Speed Testing Project

This file provides guidance to Claude Code when working with the Local Model Speed Testing benchmarking system.

## PROJECT OVERVIEW

**Multi-Provider LLM Benchmarking Tool** - A comprehensive Python-based system that evaluates performance and quality across multiple LLM providers including OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, and XAI Grok.

### Key Architecture
- **Main Engine**: `unified_llm_benchmarker.py` (2423 lines) - Core benchmarking system
- **Configuration**: `.env` file with API keys for all providers
- **Evaluation Model**: Claude Opus 4 (`claude-opus-4-20250514`) used as reference judge
- **Output**: Excel reports + PDF visualizations with performance metrics

## CRITICAL TECHNICAL CONTEXT

### Gemini API Error Handling
The Google Gemini provider requires special error handling for `finish_reason=2` (MAX_TOKENS):

```python
# Gemini response extraction pattern (already implemented)
content = ""
if response.candidates and len(response.candidates) > 0:
    candidate = response.candidates[0]
    if hasattr(candidate, 'content') and candidate.content:
        if hasattr(candidate.content, 'parts') and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    content += part.text
```

**Finish Reason Values**: STOP=1, MAX_TOKENS=2, SAFETY=3, RECITATION=4

### Reference Model Architecture
- **Evaluator**: Claude Opus 4 judges all responses (not included in test results)
- **Test Subjects**: All other configured models
- **No Circular Evaluation**: Prevents using the same model for both testing and evaluation

## FILE STRUCTURE

### Input Files
- `.env` - API keys for all providers (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- `unified_llm_benchmarker.py` - Main benchmarking engine

### Output Files
- `unified_benchmark_results_{timestamp}.xlsx` - Multi-sheet Excel report
- `speed_vs_quality_{timestamp}.pdf` - Performance vs quality scatter plot
- `speed_comparison_{timestamp}.pdf` - Speed comparison bar chart  
- `quality_comparison_{timestamp}.pdf` - Quality scores bar chart
- `ttft_comparison_{timestamp}.pdf` - Time to first token comparison

### Test/Debug Files
- `test_gemini_fix.py` - Gemini error handling verification
- `verify_reference_fix.py` - Reference model validation
- `debug_gemini_pro.py` - Gemini API response analysis

## PROVIDER CONFIGURATIONS

### Tested Models by Provider
- **OpenAI**: gpt-4o, gpt-4o-mini, o1-preview, o1-mini
- **Anthropic**: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- **Google**: gemini-2.5-pro, gemini-1.5-flash
- **Groq**: llama-3.3-70b-versatile, llama-3.1-8b-instant
- **DeepSeek**: deepseek-chat, deepseek-coder
- **XAI**: grok-2-latest, grok-2-vision-1212

### Reference/Evaluation Model
- **Claude Opus 4**: `claude-opus-4-20250514` - Used only for quality evaluation

## PERFORMANCE OPTIMIZATION

### System Requirements
- **Hardware**: Optimized for M4 Max Mac (128GB RAM)
- **Parallelization**: Uses concurrent.futures for parallel model testing
- **Memory**: Efficient handling of large response datasets

### Processing Flow
1. **Configuration Load**: API keys and provider setup
2. **Parallel Benchmarking**: Concurrent testing across providers
3. **Quality Evaluation**: Claude Opus 4 judges all responses
4. **Data Aggregation**: Combine metrics into unified dataset
5. **Report Generation**: Excel + PDF outputs with visualizations

## ERROR HANDLING STANDARDS

### API Failure Management
```python
# Robust error handling pattern
try:
    response = provider.generate_response(prompt)
    if response and len(response.strip()) > 0:
        return response
    else:
        return None
except Exception as e:
    print(f"❌ {provider} error: {e}")
    return None
```

### Data Quality Assurance
- Missing evaluations → Quality scores set to 0.0
- Failed API calls → Marked as unsuccessful with error logging
- Empty responses → Filtered out of final analysis

## CHART GENERATION NOTES

### Legend Standards
- **Provider-based legends only** (no "Reference" entries in charts)
- **Clean color coding** by provider organization
- **No Gemini reference highlighting** (fixed issue)

### Chart Types
- Scatter plots for speed vs quality analysis
- Bar charts for direct metric comparisons
- PDF format for publication-ready outputs

## COMMON TASKS

### Running Full Benchmark
```bash
python unified_llm_benchmarker.py
```

### Testing Specific Provider
```bash
python test_gemini_fix.py  # Test Gemini error handling
python verify_reference_fix.py  # Verify reference model setup
```

### Debugging API Issues
```bash
python debug_gemini_pro.py  # Analyze Gemini response structure
```

## DEVELOPMENT GUIDELINES

### API Key Management
- Store all keys in `.env` file (never commit to repo)
- Handle missing keys gracefully with informative error messages
- Support for 8 different provider APIs

### Adding New Providers
1. Create provider class with `generate_response()` method
2. Add API key to `.env` template
3. Include in provider configurations list
4. Test error handling for provider-specific issues

### Quality Evaluation
- Always use Claude Opus 4 as the evaluation model
- Maintain separation between evaluator and test subjects
- Provide structured scoring criteria (correctness, completeness, code quality)

## KNOWN ISSUES & SOLUTIONS

### Gemini MAX_TOKENS Error
- **Issue**: `response.text` fails when finish_reason=2
- **Solution**: Extract content from `response.candidates[0].content.parts`
- **Status**: ✅ Fixed and tested

### Reference Model Confusion
- **Issue**: Charts showing "Reference (Gemini)" labels
- **Solution**: Use Claude Opus 4 for evaluation only, not in test results
- **Status**: ✅ Fixed and verified

## TESTING VERIFICATION

Run verification tests after any changes:
```bash
python verify_reference_fix.py  # Comprehensive verification
python test_reference_model.py  # Reference model validation
```

Expected output:
- ✅ Reference response from Claude Opus 4
- ✅ No "Reference (Gemini)" in chart legends
- ✅ Proper error handling for all providers

---

**Last Updated**: 2025-07-16  
**Version**: 1.1  
**Status**: All critical fixes implemented and verified