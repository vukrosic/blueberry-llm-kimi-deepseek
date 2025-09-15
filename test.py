#!/usr/bin/env python3
"""
Entry point for Blueberry LLM testing.
This script redirects to the test suite.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the test suite
if __name__ == "__main__":
    from tests.test_gpu_adaptive import main
    main()
