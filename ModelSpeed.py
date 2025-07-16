"""
ModelSpeed.py - Comprehensive Multi-Provider LLM Benchmarking Tool
==================================================================

This script extends the existing LM Studio/Ollama benchmarking system to include
external API-based models like Kimi, OpenAI, Anthropic, and any other provider.
It provides a unified interface for testing both local and cloud models.

INPUT FILES:
    None (connects to local APIs and external cloud services)

OUTPUT FILES:
    - benchmark_results_[timestamp].txt: Formatted table of results
    - benchmark_results_[timestamp].json: Raw data in JSON format
    - benchmark_outputs_[timestamp].html: HTML with model outputs
    - benchmark_stats_[timestamp].xlsx: Excel file with data quality checks
    - performance_overview_[timestamp].pdf: PDF performance graphs
    - api_cost_report_[timestamp].xlsx: API usage and cost tracking

API Configuration:
    Set these environment variables before running:
    - GROQ_API_KEY: Groq API key (for Kimi via Groq)
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic API key
    - Any other provider keys as needed

Version History:
- 1.0 (2024-07-14): Initial version with multi-provider support
"""

import os
import time
import requests
import json
import openpyxl
import matplotlib.pyplot as plt
from datetime import datetime
from jinja2 import Template
import tabulate
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import openai # Added for OpenAI benchmark

# Load environment variables from .env file
load_dotenv()

# ======= CONFIGURATION PARAMETERS =======
BENCHMARK_PROMPT = (
    "Given a CSV with stock returns, implement a pairs trading strategy that: "
    "1) Uses OPTICS clustering to find stock pairs, "
    "2) Tests for cointegration, "
    "3) Generates trading signals based on z-score deviations, "
    "4) Calculates performance metrics including Sharpe ratio and maximum drawdown. "
    "Each step must be a separate function with comprehensive docstrings. /Nothink"
)

TOKEN_COUNTS = [1000]  # Reduced for testing
ENABLE_LM_STUDIO = False  # Turned off
ENABLE_OLLAMA = False     # Turned off

# External API providers configuration - Updated with correct pricing (Jan 2025)
EXTERNAL_PROVIDERS = {
    "kimi": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["moonshotai/kimi-k2-instruct"],
        "api_key_env": "GROQ_API_KEY",
        "pricing": {
            "input_cost_per_1m": 1.00,   # $1.00 per 1M input tokens
            "output_cost_per_1m": 3.00   # $3.00 per 1M output tokens
        }
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "o4-mini", "o3", "gpt-4o-mini", "gpt-3.5-turbo"],
        "api_key_env": "OPENAI_API_KEY",
        "pricing": {
            # Model-specific pricing handled in calculate_cost function
            "gpt-4o": {"input_cost_per_1m": 2.50, "output_cost_per_1m": 10.00},
            "gpt-4o-mini": {"input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60},
            "gpt-3.5-turbo": {"input_cost_per_1m": 0.50, "output_cost_per_1m": 1.50},
            "o3": {"input_cost_per_1m": 2.00, "output_cost_per_1m": 8.00},
            "o4-mini": {"input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60}
        }
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
        "api_key_env": "ANTHROPIC_API_KEY",
        "pricing": {
            # Model-specific pricing
            "claude-sonnet-4-20250514": {"input_cost_per_1m": 3.00, "output_cost_per_1m": 15.00},
            "claude-opus-4-20250514": {"input_cost_per_1m": 15.00, "output_cost_per_1m": 75.00},
            "claude-3-5-sonnet-20241022": {"input_cost_per_1m": 3.00, "output_cost_per_1m": 15.00},
            "claude-3-5-haiku-20241022": {"input_cost_per_1m": 0.80, "output_cost_per_1m": 4.00}
        }
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "api_key_env": "GEMINI_API_KEY",
        "pricing": {
            "combined_cost_per_1k": 0.001  # $0.001 per 1K combined tokens
        }
    },
    "grok": {
        "base_url": "https://api.x.ai/v1",
        "models": ["grok-4-0709"],
        "api_key_env": "XAI_API_KEY",
        "pricing": {
            "combined_cost_per_1k": 0.002  # $0.002 per 1K combined tokens
        }
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "api_key_env": "DEEPSEEK_API_KEY",
        "pricing": {
            "combined_cost_per_1k": 0.0014  # $0.0014 per 1K combined tokens
        }
    }
}

# =====================================

# SECTION: API Key Management
def get_api_key(provider: str) -> Optional[str]:
    """Retrieve API key from environment variables."""
    env_var = EXTERNAL_PROVIDERS[provider]["api_key_env"]
    return os.getenv(env_var)

# SECTION: External API Benchmarking
def benchmark_external_api(provider: str, model: str, prompt: str, token_count: int) -> Optional[Dict[str, Any]]:
    """Benchmark external API models."""
    config = EXTERNAL_PROVIDERS[provider]
    api_key = get_api_key(provider)
    
    if not api_key:
        print(f"❌ Skipping {provider} - API key not found (expected env var: {config['api_key_env']})")
        return None
    
    print(f"\n=== Benchmarking {provider.upper()} Model: {model} ===")
    print(f"🔑 API Key found for {provider}: {'Yes' if api_key else 'No'}")
    print(f"🌐 Base URL: {config['base_url']}")
    
    try:
        if provider == "kimi":
            return benchmark_kimi(api_key, model, prompt, token_count)
        elif provider == "openai":
            return benchmark_openai(api_key, model, prompt, token_count)
        elif provider == "anthropic":
            return benchmark_anthropic(api_key, model, prompt, token_count)
        elif provider == "gemini":
            return benchmark_gemini(api_key, model, prompt, token_count)
        elif provider == "grok":
            return benchmark_grok(api_key, model, prompt, token_count)
        elif provider == "deepseek":
            return benchmark_deepseek(api_key, model, prompt, token_count)
        else:
            print(f"❌ Unknown provider: {provider}")
            return None
    except Exception as e:
        print(f"❌ Exception benchmarking {provider}: {e}")
        return None

def benchmark_kimi(api_key: str, model: str, prompt: str, max_tokens: int) -> Optional[Dict[str, Any]]:
    """Benchmark Kimi API via Groq."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False
    }
    
    print(f"🚀 Making request to Kimi API via Groq...")
    print(f"📍 URL: {url}")
    print(f"🤖 Model: {model}")
    print(f"📊 Max tokens: {max_tokens}")
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        end_time = time.time()
        
        print(f"📈 Response status: {response.status_code}")
        print(f"⏱️  Response time: {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            output_text = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            total_tokens = usage.get('total_tokens', 0)
            
            print(f"✅ Kimi API via Groq call successful")
            print(f"📝 Generated {usage.get('completion_tokens', 0)} tokens")
            
            return {
                "model": f"kimi-{model}",
                "platform": "kimi",
                "max_tokens": max_tokens,
                "generation_time": end_time - start_time,
                "time_to_first_token": "N/A",
                "tokens_per_second": usage.get('completion_tokens', len(output_text.split())) / (end_time - start_time),
                "tokens_generated": usage.get('completion_tokens', len(output_text.split())),
                "prompt_tokens": usage.get('prompt_tokens', 0),
                "total_tokens": total_tokens,
                "stop_reason": result['choices'][0].get('finish_reason', 'Unknown'),
                "arch": 'Cloud API',
                "quantization": 'Cloud API',
                "output_text": output_text,
                "cost": calculate_cost("kimi", model=model, 
                                      input_tokens=usage.get('prompt_tokens', 0),
                                      output_tokens=usage.get('completion_tokens', len(output_text.split())),
                                      total_tokens=total_tokens)
            }
        else:
            print(f"❌ Kimi API via Groq error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.Timeout:
        print(f"❌ Kimi API via Groq timeout after 60 seconds")
        return None
    except Exception as e:
        print(f"❌ Kimi API via Groq exception: {e}")
        return None

def benchmark_openai(api_key: str, model: str, prompt: str, token_count: int) -> Optional[Dict[str, Any]]:
    """Benchmark OpenAI models with proper handling for O-series models."""
    openai.api_key = api_key
    
    print(f"📤 Request details:")
    print(f"   Model: {model}")
    print(f"   Prompt length: {len(prompt)} chars")
    
    # Use different parameter for O-series models
    if model.startswith(('o1', 'o3', 'o4')):
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": token_count
        }
        print(f"   Max completion tokens: {token_count}")
    else:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": token_count
        }
        print(f"   Max tokens: {token_count}")
    
    start_time = time.time()
    try:
        response = openai.chat.completions.create(**params)
        end_time = time.time()
        
        generation_time = end_time - start_time
        content = response.choices[0].message.content or ""
        tokens_generated = response.usage.completion_tokens if response.usage else len(content.split())
        total_tokens = response.usage.total_tokens if response.usage else len(content.split()) + len(prompt.split())
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        input_tokens = response.usage.prompt_tokens if response.usage else len(prompt.split())
        output_tokens = response.usage.completion_tokens if response.usage else len(content.split())
        cost = calculate_cost("openai", model=model, 
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens)
        
        print(f"📥 Response received:")
        print(f"   Status: ✅ Success")
        print(f"   Generation time: {generation_time:.2f}s")
        print(f"   Tokens generated: {tokens_generated}")
        print(f"   Tokens/sec: {tokens_per_second:.1f}")
        print(f"   Total tokens used: {total_tokens}")
        print(f"   Content preview: {content[:100]}...")
        
        return {
            "model": model,
            "platform": "OpenAI",
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "tokens_generated": tokens_generated,
            "total_tokens": total_tokens,
            "cost": cost,
            "success": True,
            "content": content  # Add the actual response content
        }
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return None

def benchmark_anthropic(api_key: str, model: str, prompt: str, max_tokens: int) -> Optional[Dict[str, Any]]:
    """Benchmark Anthropic Claude API."""
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    print(f"🚀 Making request to Anthropic API...")
    print(f"📍 URL: {url}")
    print(f"🤖 Model: {model}")
    print(f"📊 Max tokens: {max_tokens}")
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        end_time = time.time()
        
        print(f"📈 Response status: {response.status_code}")
        print(f"⏱️  Response time: {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            output_text = result['content'][0]['text']
            usage = result.get('usage', {})
            total_tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            
            print(f"✅ Anthropic API call successful")
            print(f"📝 Generated {usage.get('output_tokens', 0)} tokens")
            
            return {
                "model": f"anthropic-{model}",
                "platform": "anthropic",
                "max_tokens": max_tokens,
                "generation_time": end_time - start_time,
                "time_to_first_token": "N/A",
                "tokens_per_second": usage.get('output_tokens', len(output_text.split())) / (end_time - start_time),
                "tokens_generated": usage.get('output_tokens', len(output_text.split())),
                "prompt_tokens": usage.get('input_tokens', 0),
                "total_tokens": total_tokens,
                "stop_reason": result.get('stop_reason', 'Unknown'),
                "arch": 'Cloud API',
                "quantization": 'Cloud API',
                "output_text": output_text,
                "cost": calculate_cost("anthropic", model=model,
                                      input_tokens=usage.get('input_tokens', 0),
                                      output_tokens=usage.get('output_tokens', len(output_text.split())),
                                      total_tokens=total_tokens)
            }
        else:
            print(f"❌ Anthropic API error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.Timeout:
        print(f"❌ Anthropic API timeout after 60 seconds")
        return None
    except Exception as e:
        print(f"❌ Anthropic API exception: {e}")
        return None

def benchmark_gemini(api_key: str, model: str, prompt: str, token_count: int) -> Optional[Dict[str, Any]]:
    """Benchmark Gemini models with correct API format."""
    print(f"📤 Request details:")
    print(f"   Model: {model}")
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   Max output tokens: {token_count}")
    
    # Gemini uses URL parameter authentication and different request format
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": token_count,
            "temperature": 0.7
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        end_time = time.time()
        
        print(f"📤 Response status: {response.status_code}")
        print(f"⏱️  Response time: {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            # Extract content from Gemini response format
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                
                # Try different ways to extract content
                content = None
                if 'content' in candidate and 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                    content = candidate['content']['parts'][0].get('text', '')
                elif 'content' in candidate and 'text' in candidate['content']:
                    content = candidate['content']['text']
                elif 'text' in candidate:
                    content = candidate['text']
                
                # If no content found, check if there's a finish reason
                if not content:
                    finish_reason = candidate.get('finishReason', 'Unknown')
                    if finish_reason == 'MAX_TOKENS':
                        content = "[Response truncated due to max tokens limit]"
                    else:
                        print(f"❌ No content found in response. Finish reason: {finish_reason}")
                        return None
                
                # Calculate metrics
                generation_time = end_time - start_time
                tokens_generated = len(content.split())  # Rough estimate
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                total_tokens = len(prompt.split()) + tokens_generated
                cost = calculate_cost("gemini", total_tokens=total_tokens)
                
                print(f"✅ Success! Generated {tokens_generated} tokens in {generation_time:.2f}s")
                return {
                    "model": model,
                    "platform": "Gemini",
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "tokens_generated": tokens_generated,
                    "total_tokens": total_tokens,
                    "cost": cost,
                    "success": True,
                    "output_text": content
                }
            else:
                print("❌ No candidates in response")
                return None
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def benchmark_grok(api_key: str, model: str, prompt: str, token_count: int) -> Optional[Dict[str, Any]]:
    """Benchmark Grok models with xAI API."""
    print(f"📤 Request details:")
    print(f"   Model: {model}")
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   Max tokens: {token_count}")
    
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": token_count,
        "temperature": 0.7
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        end_time = time.time()
        
        print(f"📤 Response status: {response.status_code}")
        print(f"⏱️  Response time: {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Calculate metrics
            generation_time = end_time - start_time
            tokens_generated = data.get("usage", {}).get("completion_tokens", len(content.split()))
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            total_tokens = data.get("usage", {}).get("total_tokens", len(prompt.split()) + tokens_generated)
            cost = calculate_cost("grok", total_tokens=total_tokens)
            
            print(f"✅ Success! Generated {tokens_generated} tokens in {generation_time:.2f}s")
            return {
                "model": model,
                "platform": "Grok",
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
                "tokens_generated": tokens_generated,
                "total_tokens": total_tokens,
                "cost": cost,
                "success": True,
                "output_text": content
            }
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def benchmark_deepseek(api_key: str, model: str, prompt: str, token_count: int) -> Optional[Dict[str, Any]]:
    """Benchmark DeepSeek models with OpenAI-compatible API."""
    print(f"📤 Request details:")
    print(f"   Model: {model}")
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   Max tokens: {token_count}")
    
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": token_count,
        "temperature": 0.7
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        end_time = time.time()
        
        print(f"📤 Response status: {response.status_code}")
        print(f"⏱️  Response time: {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Calculate metrics
            generation_time = end_time - start_time
            tokens_generated = data.get("usage", {}).get("completion_tokens", len(content.split()))
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            total_tokens = data.get("usage", {}).get("total_tokens", len(prompt.split()) + tokens_generated)
            cost = calculate_cost("deepseek", total_tokens=total_tokens)
            
            print(f"✅ Success! Generated {tokens_generated} tokens in {generation_time:.2f}s")
            return {
                "model": model,
                "platform": "DeepSeek", 
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
                "tokens_generated": tokens_generated,
                "total_tokens": total_tokens,
                "cost": cost,
                "success": True,
                "output_text": content
            }
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def calculate_cost(provider: str, model: Optional[str] = None, input_tokens: int = 0, output_tokens: int = 0, total_tokens: Optional[int] = None) -> float:
    """Calculate API cost based on tokens used with accurate provider-specific pricing."""
    if provider not in EXTERNAL_PROVIDERS:
        return 0.0
    
    pricing = EXTERNAL_PROVIDERS[provider]["pricing"]
    
    # Handle combined pricing (Gemini, Grok, DeepSeek)
    if "combined_cost_per_1k" in pricing:
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * pricing["combined_cost_per_1k"]
    
    # Handle separate input/output pricing (Kimi, OpenAI, Anthropic)
    if model and model in pricing:
        # Model-specific pricing (OpenAI, Anthropic)
        model_pricing = pricing[model]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input_cost_per_1m"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output_cost_per_1m"]
        return input_cost + output_cost
    elif "input_cost_per_1m" in pricing:
        # Provider-level pricing (Kimi)
        input_cost = (input_tokens / 1_000_000) * pricing["input_cost_per_1m"]
        output_cost = (output_tokens / 1_000_000) * pricing["output_cost_per_1m"]
        return input_cost + output_cost
    
    # Fallback to total tokens if input/output breakdown not available
    if total_tokens and "combined_cost_per_1k" in pricing:
        return (total_tokens / 1000) * pricing["combined_cost_per_1k"]
    
    return 0.0

# SECTION: Main Benchmark Runner
def run_benchmarks():
    """Main function to run all benchmarks."""
    print("=== Starting ModelSpeed Benchmarks ===")
    print(f"Benchmark prompt: \"{BENCHMARK_PROMPT}\"")
    print(f"Testing with token counts: {TOKEN_COUNTS}")
    print(f"LM Studio enabled: {ENABLE_LM_STUDIO} | Ollama enabled: {ENABLE_OLLAMA}")
    
    # Check environment variables
    print("\n🔍 Checking API keys...")
    for provider, config in EXTERNAL_PROVIDERS.items():
        api_key_env = config['api_key_env']
        api_key = os.getenv(api_key_env)
        status = "✅ Found" if api_key else "❌ Missing"
        print(f"  {provider}: {status} (env var: {api_key_env})")
    
    # Collect all models
    all_models = []
    
    print("\n📋 Collecting models...")
    # External API models only
    for provider, config in EXTERNAL_PROVIDERS.items():
        api_key = get_api_key(provider)
        if api_key:
            for model in config['models']:
                all_models.append((model, provider))
                print(f"  ✅ Added: {provider}/{model}")
        else:
            print(f"  ❌ Skipped {provider} models (no API key)")
    
    print(f"\n📊 Total models to test: {len(all_models)}")
    if not all_models:
        print("❌ No models available for benchmarking - please set API keys")
        print("\nRequired environment variables:")
        for provider, config in EXTERNAL_PROVIDERS.items():
            print(f"  - {config['api_key_env']} (for {provider})")
        return
    
    # Run benchmarks
    all_results = []
    all_outputs = []
    
    total_tests = len(TOKEN_COUNTS) * len(all_models)
    current_test = 0
    
    print(f"\n🚀 Starting {total_tests} benchmark tests...")
    
    for token_count in TOKEN_COUNTS:
        print(f"\n🎯 Testing with {token_count} tokens...")
        for model_info, platform in all_models:
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] Testing {platform}/{model_info}")
            
            result = benchmark_external_api(platform, model_info, BENCHMARK_PROMPT, token_count)
            if result:
                all_results.append(result)
                
                # Handle different content key names for compatibility
                content = "No content available"
                if "content" in result:
                    content = result["content"]
                elif "output_text" in result:
                    content = result["output_text"]
                elif "output" in result:
                    content = result["output"]
                
                all_outputs.append({
                    "model": result["model"],
                    "platform": result["platform"],
                    "output": content
                })
                print(f"✅ Test {current_test} completed successfully")
            else:
                print(f"❌ Test {current_test} failed")
    
    print(f"\n📈 Benchmark Summary:")
    print(f"  Total tests run: {total_tests}")
    print(f"  Successful tests: {len(all_results)}")
    print(f"  Failed tests: {total_tests - len(all_results)}")
    
    if all_results:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"\n💾 Saving results with timestamp: {timestamp}")
        
        # Save all outputs
        save_results_to_txt(all_results, BENCHMARK_PROMPT, timestamp)
        save_results_to_json(all_results, timestamp)
        save_outputs_to_html(all_outputs, BENCHMARK_PROMPT, timestamp, all_results)
        save_results_to_xlsx(all_results, timestamp)
        save_performance_pdf(all_results, timestamp)
        save_api_cost_report(all_results, timestamp)
        
        print("✅ All files saved successfully!")
    else:
        print("❌ No results collected - check API keys and error messages above")

# SECTION: Output Saving Functions
def save_results_to_txt(results: List[Dict[str, Any]], prompt: str, timestamp: str):
    """Save results to TXT file."""
    output_file = f"benchmark_results_{timestamp}.txt"
    with open(output_file, "w") as f:
        f.write(f"=== ModelSpeed Benchmark Results ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prompt: \"{prompt}\"\n\n")
        
        table_data = []
        for r in results:
            table_data.append([
                r["model"],
                r["platform"],
                f"{r['generation_time']:.2f}s",
                f"{r['tokens_per_second']:.2f}",
                r["tokens_generated"],
                r["total_tokens"],
                f"${r['cost']:.4f}"
            ])
        
        table = tabulate.tabulate(
            table_data,
            headers=["Model", "Platform", "Gen Time", "Tokens/Sec", "Tokens Gen", "Total Tokens", "Cost"],
            tablefmt="grid"
        )
        f.write(table)
    print(f"Benchmark results saved to {output_file}")

def save_results_to_json(all_results: List[Dict[str, Any]], timestamp: str):
    """Save raw results to JSON."""
    with open(f"benchmark_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw JSON saved to benchmark_results_{timestamp}.json")

def save_outputs_to_html(all_outputs: List[Dict[str, Any]], prompt: str, timestamp: str, all_results: List[Dict[str, Any]]):
    """Save model outputs to HTML."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ModelSpeed Benchmark Outputs</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .prompt { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .model-output { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .model-name { font-weight: bold; color: #2c5282; margin-bottom: 10px; }
            .platform { color: #718096; margin-bottom: 10px; }
            .output-text { white-space: pre-wrap; background: #f5f5f5; padding: 10px; }
            .stats { font-size: 0.9em; color: #555; margin: 10px 0; }
            table { border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>ModelSpeed Benchmark Results</h1>
        <div class="prompt">
            <h3>Prompt:</h3>
            <p>{{ prompt }}</p>
        </div>
        
        <h2>Model Outputs:</h2>
        {% for output in outputs %}
        <div class="model-output">
            <div class="model-name">{{ output.model }}</div>
            <div class="platform">Platform: {{ output.platform }}</div>
            <div class="output-text">{{ output.output }}</div>
        </div>
        {% endfor %}
        
        <h2>Performance Summary:</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Platform</th>
                <th>Generation Time (s)</th>
                <th>Tokens/Second</th>
                <th>Total Tokens</th>
                <th>Cost (USD)</th>
            </tr>
            {% for result in results %}
            <tr>
                <td>{{ result.model }}</td>
                <td>{{ result.platform }}</td>
                <td>{{ "%.2f"|format(result.generation_time) }}</td>
                <td>{{ "%.2f"|format(result.tokens_per_second) }}</td>
                <td>{{ result.total_tokens }}</td>
                <td>${{ "%.4f"|format(result.cost) }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """
    
    template = Template(html_template)
    html_content = template.render(outputs=all_outputs, prompt=prompt, results=all_results)
    
    with open(f"benchmark_outputs_{timestamp}.html", "w") as f:
        f.write(html_content)
    print(f"Model outputs saved to benchmark_outputs_{timestamp}.html")

def save_results_to_xlsx(all_results: List[Dict[str, Any]], timestamp: str):
    """Save results to XLSX with data quality checks."""
    output_file = f"benchmark_stats_{timestamp}.xlsx"
    wb = openpyxl.Workbook()
    
    # Data sheet
    ws = wb.active
    if ws is not None:
        ws.title = "Results"
        headers = ["Model", "Platform", "Gen Time (s)", "Tokens/Sec", "Tokens Gen", "Total Tokens", "Cost (USD)"]
        ws.append(headers)
        
        for r in all_results:
            ws.append([
                r["model"],
                r["platform"],
                r["generation_time"],
                r["tokens_per_second"],
                r["tokens_generated"],
                r["total_tokens"],
                f"{r['cost']:.4f}"
            ])
    
    wb.save(output_file)
    print(f"XLSX stats saved to {output_file}")

def save_performance_pdf(all_results: List[Dict[str, Any]], timestamp: str):
    """Save performance overview to PDF."""
    if not all_results:
        return
    
    models = [r['model'] for r in all_results]
    tps = [r['tokens_per_second'] for r in all_results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(models, tps)
    ax.set_ylabel('Tokens per Second')
    ax.set_title('ModelSpeed Performance Overview')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(f"performance_overview_{timestamp}.pdf")
    plt.close()
    print(f"PDF graph saved to performance_overview_{timestamp}.pdf")

def save_api_cost_report(all_results: List[Dict[str, Any]], timestamp: str):
    """Save API cost report to XLSX."""
    output_file = f"api_cost_report_{timestamp}.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    
    if ws is not None:
        ws.title = "API Costs"
        
        ws.append(["Model", "Platform", "Total Tokens", "Cost (USD)"])
        total_cost = 0.0
        
        for r in all_results:
            ws.append([r["model"], r["platform"], r["total_tokens"], f"{r['cost']:.4f}"])
            total_cost += r['cost']
        
        ws.append([])
        ws.append(["Total Cost", "", "", f"${total_cost:.4f}"])
    
    wb.save(output_file)
    print(f"API cost report saved to {output_file}")

if __name__ == "__main__":
    run_benchmarks() 