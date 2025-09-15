#!/usr/bin/env python3
"""
Entry point for Blueberry LLM training.
This script redirects to the core training functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main training script
if __name__ == "__main__":
    from core.train_auto import main
    main()
