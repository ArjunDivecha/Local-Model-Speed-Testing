#!/usr/bin/env python3
"""
Check which Gemini models are actually available
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Available Gemini models:")
for model in genai.list_models():
    print(f"- {model.name}: {model.display_name}")

# Test specific models
test_models = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash", 
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-flash-latest"
]

for model_name in test_models:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Test")
        print(f"✓ {model_name}: Works")
    except Exception as e:
        print(f"✗ {model_name}: Error - {e}")