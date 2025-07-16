#!/usr/bin/env python3
"""
Debug script to understand Gemini response structure
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Test with a simple prompt that might hit token limits
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Use very low max_tokens to force MAX_TOKENS finish reason
generation_config = genai.types.GenerationConfig(
    max_output_tokens=10,  # Very low to force truncation
    temperature=0.7,
)

response = model.generate_content(
    "Write a comprehensive Python function that implements a binary search algorithm with detailed documentation and error handling.",
    generation_config=generation_config
)

print("=== Response Debug Info ===")
print(f"Response type: {type(response)}")
print(f"Has text attr: {hasattr(response, 'text')}")
print(f"Has candidates: {hasattr(response, 'candidates')}")

if hasattr(response, 'candidates') and response.candidates:
    candidate = response.candidates[0]
    print(f"Candidate type: {type(candidate)}")
    print(f"Finish reason: {candidate.finish_reason}")
    print(f"Finish reason type: {type(candidate.finish_reason)}")
    print(f"Finish reason value: {candidate.finish_reason.value if hasattr(candidate.finish_reason, 'value') else 'no value'}")
    print(f"Finish reason name: {candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else 'no name'}")
    
    # Check what values are available
    print("\n=== Available FinishReason Values ===")
    finish_reason_class = type(candidate.finish_reason)
    print(f"FinishReason class: {finish_reason_class}")
    print(f"FinishReason module: {finish_reason_class.__module__}")
    
    # Try to enumerate values
    if hasattr(finish_reason_class, '__members__'):
        print("Available values:")
        for name, value in finish_reason_class.__members__.items():
            print(f"  {name}: {value.value}")
    
    # Check content
    print("\n=== Content Debug ===")
    print(f"Has content: {hasattr(candidate, 'content')}")
    if hasattr(candidate, 'content') and candidate.content:
        print(f"Content type: {type(candidate.content)}")
        print(f"Has parts: {hasattr(candidate.content, 'parts')}")
        if hasattr(candidate.content, 'parts'):
            print(f"Parts count: {len(candidate.content.parts)}")
            for i, part in enumerate(candidate.content.parts):
                print(f"  Part {i}: {type(part)}")
                if hasattr(part, 'text'):
                    print(f"    Text: {part.text[:100]}...")

# Test response.text access
print("\n=== Response.text Test ===")
try:
    text = response.text
    print(f"Response text: {text[:100]}...")
except Exception as e:
    print(f"Error accessing response.text: {e}")