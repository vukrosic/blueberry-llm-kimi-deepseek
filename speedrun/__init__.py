"""
T4 Speedrun Challenge Module

This module provides the T4 speedrun challenge functionality,
including configuration, timing, validation, and leaderboard management.
"""

from .config import (
    T4SpeedrunConfig,
    get_t4_speedrun_config,
    create_custom_t4_config,
    get_memory_optimized_config,
    get_performance_optimized_config,
    get_balanced_config
)

from .speedrun import run_speedrun, SpeedrunTimer, SpeedrunValidator, SpeedrunResults
from .leaderboard import SpeedrunLeaderboard

__version__ = "1.0.0"
__author__ = "Blueberry LLM Team"

__all__ = [
    "T4SpeedrunConfig",
    "get_t4_speedrun_config",
    "create_custom_t4_config",
    "get_memory_optimized_config",
    "get_performance_optimized_config",
    "get_balanced_config",
    "run_speedrun",
    "SpeedrunTimer",
    "SpeedrunValidator",
    "SpeedrunResults",
    "SpeedrunLeaderboard",
]
