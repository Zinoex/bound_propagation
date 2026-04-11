"""
Relaxation module for computing linear approximations of operations.

This module provides the infrastructure for computing linear relaxations
of operations given concrete interval bounds. Relaxations are shared between
forward and backward LBP propagation methods.
"""

# Import submodules to trigger auto-registration of strategies
from . import bilinear, elementwise
from .base import (
    RelaxationRegistry,
    RelaxationStrategy,
    register_relaxation_strategy,
)
from .linear_relaxation import LinearRelaxation

__all__ = [
    "LinearRelaxation",
    "RelaxationStrategy",
    "RelaxationRegistry",
    "register_relaxation_strategy",
    "elementwise",
    "bilinear",
]
