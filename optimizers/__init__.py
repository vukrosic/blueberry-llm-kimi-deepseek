"""
Optimizers for blueberry-llm training.

This module provides advanced optimizers like Muon that are optimized
for training large language models.
"""

from .muon import Muon, zeropower_via_newtonschulz5
from .factory import setup_optimizers, get_lr_scheduler

__all__ = [
    'Muon',
    'zeropower_via_newtonschulz5',
    'setup_optimizers',
    'get_lr_scheduler',
]
