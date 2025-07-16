#!/usr/bin/env python3
"""
API Connection Tester
====================

Simple script to test API connections for the ModelSpeed benchmarking tool.
This helps debug API key issues and connection problems.

INPUT FILES:
    None (reads environment variables)

OUTPUT FILES:
    None (prints to console)

Usage:
    python test_api_connections.py

Version: 1.0 (2024-07-15)
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai():
    """Test OpenAI API connection."""
    print("\n🔍 Testing OpenAI API...")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test with latest model
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            print("✅ OpenAI API working!")
            return True
        else:
            print(f"❌ OpenAI API failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ OpenAI API error: {e}")
        return False

def test_anthropic():
    """Test Anthropic API connection."""
    print("\n🔍 Testing Anthropic API...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test with a simple message using the latest model name
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            print("✅ Anthropic API working!")
            return True
        else:
            print(f"❌ Anthropic API error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Anthropic API exception: {e}")
        return False

def test_kimi():
    """Test Kimi API connection via Groq."""
    print("\n🔍 Testing Kimi API...")
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test with Groq API endpoint for Kimi - OpenAI compatible
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "moonshotai/kimi-k2-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"🌐 Response status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Kimi API working!")
            return True
        else:
            print(f"❌ Kimi API failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_gemini():
    """Test Gemini API connection."""
    print("\n🔍 Testing Gemini API...")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test with correct Gemini API endpoint using URL parameter authentication
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{"text": "Hello"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 10
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"🌐 Response status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Gemini API working!")
            return True
        else:
            print(f"❌ Gemini API failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_grok():
    """Test Grok API connection."""
    print("\n🔍 Testing Grok API...")
    api_key = os.getenv("XAI_API_KEY")
    
    if not api_key:
        print("❌ XAI_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test with correct xAI Grok API endpoint
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "grok-4-0709",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"🌐 Response status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Grok API working!")
            return True
        else:
            print(f"❌ Grok API failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_deepseek():
    """Test DeepSeek API connection."""
    print("\n🔍 Testing DeepSeek API...")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test with correct DeepSeek API endpoint - OpenAI compatible
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"🌐 Response status: {response.status_code}")
        if response.status_code == 200:
            print("✅ DeepSeek API working!")
            return True
        else:
            print(f"❌ DeepSeek API failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def main():
    """Test all API connections."""
    print("🚀 Testing API Connections for All Providers")
    print("=" * 50)
    
    results = {}
    results["OpenAI"] = test_openai()
    results["Anthropic"] = test_anthropic()
    results["Kimi"] = test_kimi()
    results["Gemini"] = test_gemini()
    results["Grok"] = test_grok()
    results["DeepSeek"] = test_deepseek()
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS:")
    for provider, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {provider}: {'Working' if status else 'Failed'}")
    
    working_count = sum(results.values())
    total_count = len(results)
    print(f"\n🎯 Summary: {working_count}/{total_count} APIs working")

if __name__ == "__main__":
    main() 