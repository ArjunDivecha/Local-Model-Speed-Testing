#!/usr/bin/env python3
"""
Test the models/ prefix fix for Gemini models
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_llm_benchmarker import GoogleProvider
from dotenv import load_dotenv

load_dotenv()

def test_models_prefix_fix():
    """Test the models with the models/ prefix that was causing failures"""
    
    print("=== Testing models/ prefix fix ===")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No GEMINI_API_KEY found")
        return
    
    config = {"base_url": None}
    provider = GoogleProvider(api_key, config)
    
    # Test the specific models that were failing
    failing_models = [
        "models/gemini-2.5-pro",     # This was failing in benchmarker
        "models/gemini-2.5-flash",   # This was failing in benchmarker
        "models/gemini-1.5-pro-latest",
        "models/gemini-1.5-flash-latest"
    ]
    
    prompt = "Write a simple Python function to add two numbers."
    
    for model_name in failing_models:
        print(f"\n🔍 Testing {model_name}...")
        
        response = provider.generate_response(prompt, model_name, max_tokens=1000)
        
        if response.success:
            print(f"✅ SUCCESS: {len(response.content)} chars")
            print(f"   Preview: {response.content[:100]}...")
        else:
            print(f"❌ FAILED: {response.error_message}")

if __name__ == "__main__":
    test_models_prefix_fix()