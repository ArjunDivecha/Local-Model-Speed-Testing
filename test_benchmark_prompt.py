#!/usr/bin/env python3
"""
Test the actual benchmark prompt that's causing Gemini failures
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_llm_benchmarker import GoogleProvider

load_dotenv()

def test_benchmark_prompt():
    """Test with the actual benchmark prompt"""
    
    print("=== Testing Real Benchmark Prompt ===")
    
    # The actual prompt from the benchmarker
    benchmark_prompt = """Write a script that connects to Interactive Brokers API, retrieves real-time options data for SPY, calculates implied volatility skew, and generates an alert when the skew exceeds historical 90th percentile. Include reconnection logic and handle all possible API errors explicitly."""
    
    print(f"Prompt length: {len(benchmark_prompt)} characters")
    print(f"Prompt: {benchmark_prompt}")
    
    # Configure API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No GEMINI_API_KEY found")
        return
    
    genai.configure(api_key=api_key)
    
    # Models that are failing
    failing_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash"
    ]
    
    for model_name in failing_models:
        print(f"\n🔍 Testing {model_name} with benchmark prompt...")
        
        # Test using the GoogleProvider class (same as benchmarker)
        config = {"base_url": None}
        provider = GoogleProvider(api_key, config)
        
        response = provider.generate_response(benchmark_prompt, model_name, max_tokens=10000)
        
        print(f"Success: {response.success}")
        if response.success:
            print(f"Content length: {len(response.content)}")
            print(f"Content preview: {response.content[:300]}...")
        else:
            print(f"Error: {response.error_message}")
        
        # Also test with direct API to see if it's a safety issue
        print(f"\n🔍 Testing {model_name} directly...")
        try:
            # Map model names
            model_mapping = {
                "gemini-2.5-flash": "gemini-1.5-flash-latest",
                "gemini-2.5-pro": "gemini-1.5-pro-latest",
            }
            actual_model = model_mapping.get(model_name, model_name)
            
            # Use very permissive safety settings
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            model_instance = genai.GenerativeModel(
                model_name=actual_model,
                safety_settings=safety_settings
            )
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=10000,
                temperature=0.7
            )
            
            response = model_instance.generate_content(
                benchmark_prompt,
                generation_config=generation_config
            )
            
            print(f"Finish reason: {response.candidates[0].finish_reason if response.candidates else 'No candidates'}")
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                finish_reason_value = finish_reason.value if hasattr(finish_reason, 'value') else int(finish_reason)
                
                print(f"Finish reason value: {finish_reason_value}")
                if finish_reason_value == 3:
                    print("❌ Content blocked by safety filters!")
                    safety_ratings = getattr(candidate, 'safety_ratings', [])
                    for rating in safety_ratings:
                        print(f"  Category: {getattr(rating, 'category', 'UNKNOWN')}, Blocked: {getattr(rating, 'blocked', False)}")
                elif finish_reason_value == 4:
                    print("❌ Content blocked due to recitation!")
                elif finish_reason_value == 2:
                    print("⚠️ Hit max tokens limit")
                elif finish_reason_value == 1:
                    print("✅ Normal completion")
                
                # Try to extract content
                content = ""
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                content += part.text
                
                if content:
                    print(f"✅ Content extracted: {len(content)} chars")
                    print(f"Preview: {content[:200]}...")
                else:
                    print("❌ No content extracted")
                    
        except Exception as e:
            print(f"❌ Direct API error: {e}")

if __name__ == "__main__":
    test_benchmark_prompt()