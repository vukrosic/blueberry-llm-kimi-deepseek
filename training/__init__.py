"""
Training utilities for blueberry-llm.

This module provides training loops, evaluation functions, and utilities
for training large language models with GPU-adaptive optimizations.
"""

from .trainer import train_model
from .evaluation import evaluate_model, compute_perplexity

__all__ = [
    'train_model',
    'evaluate_model', 
    'compute_perplexity',
]
