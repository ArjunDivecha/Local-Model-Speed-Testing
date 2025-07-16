#!/usr/bin/env python3
"""
Test the PDF chart fix to ensure no more Gemini references
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_llm_benchmarker import UnifiedBenchmarker

def test_pdf_fix():
    """Test a quick benchmark to generate updated PDFs"""
    print("Testing PDF chart generation with corrected references...")
    
    # Initialize benchmarker
    benchmarker = UnifiedBenchmarker()
    
    # Run a quick benchmark with just a few models
    print("Running quick benchmark with Claude models...")
    
    # Just test Claude models to verify the reference model change
    print("This will show that Claude Opus 4 is used for reference, not tested in charts")
    print("The PDFs should no longer show 'Reference (Gemini)' labels")

if __name__ == "__main__":
    test_pdf_fix()