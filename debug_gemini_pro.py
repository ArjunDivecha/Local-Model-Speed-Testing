#!/usr/bin/env python3
"""
Debug script to understand why gemini-2.5-pro fails
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Test with gemini-2.5-pro which should map to gemini-1.5-pro-latest
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Use very low max_tokens to force MAX_TOKENS finish reason
generation_config = genai.types.GenerationConfig(
    max_output_tokens=10000,  # Very low to force truncation
    temperature=0.7,
)

response = model.generate_content(
    "Write a simple Python function that adds two numbers.",
    generation_config=generation_config
)

print("=== Response Debug Info ===")
print(f"Finish reason: {response.candidates[0].finish_reason}")
print(f"Finish reason value: {response.candidates[0].finish_reason.value}")

print("\n=== Content Extraction Test ===")
content = ""

# Try method 1: From parts
if response.candidates and len(response.candidates) > 0:
    candidate = response.candidates[0]
    if hasattr(candidate, 'content') and candidate.content:
        if hasattr(candidate.content, 'parts') and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    content += part.text
                    print(f"Part text: {part.text}")

print(f"Content from parts: '{content}'")

# Try method 2: response.text
try:
    text = response.text
    print(f"Response.text: '{text}'")
except Exception as e:
    print(f"Error accessing response.text: {e}")

print(f"\nContent available: {len(content) > 0}")
print(f"Content length: {len(content)}")

# Test if it's just empty content
if not content:
    print("No content extracted - this explains the error!")
else:
    print(f"Content extracted successfully: {content[:100]}...")

# Check all parts more thoroughly
print("\n=== Detailed Parts Analysis ===")
if response.candidates and len(response.candidates) > 0:
    candidate = response.candidates[0]
    print(f"Candidate content: {candidate.content}")
    if hasattr(candidate, 'content') and candidate.content:
        print(f"Content parts: {candidate.content.parts}")
        if hasattr(candidate.content, 'parts') and candidate.content.parts:
            for i, part in enumerate(candidate.content.parts):
                print(f"Part {i}: {part}")
                print(f"  Type: {type(part)}")
                print(f"  Has text: {hasattr(part, 'text')}")
                if hasattr(part, 'text'):
                    print(f"  Text value: '{part.text}'")
                    print(f"  Text len: {len(part.text) if part.text else 'None'}")