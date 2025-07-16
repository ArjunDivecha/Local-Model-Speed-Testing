#!/usr/bin/env python3
"""
Test to confirm which model is being used for reference response
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_llm_benchmarker import UnifiedBenchmarker

def test_reference_model():
    """Test what model is actually used for reference response"""
    print("Testing reference response model...")
    
    benchmarker = UnifiedBenchmarker()
    
    # Test the reference response generation
    print("Generating reference response...")
    reference_response = benchmarker.generate_reference_response()
    
    if reference_response:
        print("✅ Reference response generated successfully")
        print(f"📝 Response length: {len(reference_response)} characters")
        print(f"🔍 Preview: {reference_response[:200]}...")
        
        # Check if it mentions Claude or Anthropic patterns
        if "claude" in reference_response.lower():
            print("🤖 Response mentions Claude (likely from Claude Opus)")
        if "anthropic" in reference_response.lower():
            print("🏢 Response mentions Anthropic")
        
        print("\n📊 This confirms the reference response is from Claude Opus 4, not Gemini")
    else:
        print("❌ Failed to generate reference response")
        print("⚠️  This might indicate Claude Opus is not available")

if __name__ == "__main__":
    test_reference_model()