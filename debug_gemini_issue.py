#!/usr/bin/env python3
"""
Debug the specific Gemini issue happening in the benchmarker
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_llm_benchmarker import GoogleProvider

load_dotenv()

def test_gemini_models():
    """Test the specific models that are failing"""
    
    print("=== Testing Gemini Models Directly ===")
    
    # Configure API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No GEMINI_API_KEY found")
        return
    
    genai.configure(api_key=api_key)
    
    # Test prompt from benchmarker
    prompt = """Write a Python function that implements a simple binary search algorithm.

Requirements:
- Function should take a sorted list and a target value as parameters
- Return the index of the target if found, -1 if not found
- Include proper error handling and edge cases
- Add docstring with examples"""
    
    # Models that are failing
    failing_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash"
    ]
    
    for model_name in failing_models:
        print(f"\n🔍 Testing {model_name}...")
        
        # Test using the GoogleProvider class
        config = {"base_url": None}
        provider = GoogleProvider(api_key, config)
        
        response = provider.generate_response(prompt, model_name, max_tokens=10000)
        
        print(f"Success: {response.success}")
        if response.success:
            print(f"Content length: {len(response.content)}")
            print(f"Content preview: {response.content[:200]}...")
        else:
            print(f"Error: {response.error_message}")
        
        # Also test direct API call to compare
        print(f"\n🔍 Testing {model_name} directly with API...")
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
                prompt,
                generation_config=generation_config
            )
            
            print(f"Finish reason: {response.candidates[0].finish_reason if response.candidates else 'No candidates'}")
            print(f"Has text: {hasattr(response, 'text')}")
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                print(f"Content available: {hasattr(candidate, 'content') and candidate.content}")
                if hasattr(candidate, 'content') and candidate.content:
                    print(f"Parts available: {hasattr(candidate.content, 'parts') and candidate.content.parts}")
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for i, part in enumerate(candidate.content.parts):
                            print(f"  Part {i}: text={hasattr(part, 'text')}, text_len={len(part.text) if hasattr(part, 'text') and part.text else 0}")
            
            # Try to extract content
            content = ""
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                content += part.text
            
            if content:
                print(f"✅ Content extracted: {len(content)} chars")
                print(f"Preview: {content[:100]}...")
            else:
                print("❌ No content extracted")
                
        except Exception as e:
            print(f"❌ Direct API error: {e}")

if __name__ == "__main__":
    test_gemini_models()