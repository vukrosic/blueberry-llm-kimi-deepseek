"""
Model architectures and components for blueberry-llm.

This module provides modular neural network components that can be
composed to create different LLM architectures.
"""

from .layers import T4Linear, Rotary
from .components import MultiHeadAttention, Expert, TopKRouter, MixtureOfExperts, MoETransformerBlock
from .t4_llm import T4MoEMinimalLLM, create_model

__all__ = [
    # Basic layers
    'T4Linear',
    'Rotary',
    
    # Complex components
    'MultiHeadAttention',
    'Expert',
    'TopKRouter', 
    'MixtureOfExperts',
    'MoETransformerBlock',
    
    # Full models
    'T4MoEMinimalLLM',
    'create_model',
]
