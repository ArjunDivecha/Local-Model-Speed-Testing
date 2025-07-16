#!/usr/bin/env python3
"""
Direct test of Gemini 2.5 Pro to prove it works
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Testing Gemini 2.5 Pro directly...")

# Test with different token limits
test_cases = [
    {"max_tokens": 2000, "description": "High token limit"},
    {"max_tokens": 1000, "description": "Medium token limit"},
    {"max_tokens": 500, "description": "Lower token limit"},
    {"max_tokens": 200, "description": "Very low token limit"}
]

for test in test_cases:
    print(f"\n--- {test['description']} ({test['max_tokens']} tokens) ---")
    
    try:
        # Create model
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        
        # Configure generation
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=test['max_tokens'],
            temperature=0.7,
        )
        
        # Generate content
        start_time = time.time()
        response = model.generate_content(
            "Write a Python function that implements a binary search algorithm. Include proper error handling and docstring.",
            generation_config=generation_config
        )
        end_time = time.time()
        
        print(f"⏱️  Generation time: {end_time - start_time:.2f} seconds")
        print(f"🔍 Finish reason: {response.candidates[0].finish_reason}")
        print(f"📊 Finish reason value: {response.candidates[0].finish_reason.value}")
        
        # Try to get content
        content = ""
        
        # Method 1: From parts
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            content += part.text
                            
        if content:
            print(f"✅ SUCCESS: Got {len(content)} characters from parts")
            print(f"📝 Content preview: {content[:200]}...")
            
            # Get token info if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                print(f"🔢 Tokens: {input_tokens} input, {output_tokens} output")
        else:
            print("❌ FAILED: No content from parts")
            
        # Method 2: Try response.text
        try:
            text_content = response.text
            print(f"✅ response.text also works: {len(text_content)} characters")
        except Exception as e:
            print(f"❌ response.text failed: {e}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("CONCLUSION:")
print("If any test case shows SUCCESS, then Gemini 2.5 Pro works fine.")
print("The issue is in the benchmarker's handling, not the model itself.")
print("="*60)