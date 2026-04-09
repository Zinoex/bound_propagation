"""
Relaxation module for computing linear approximations of operations.

This module provides the infrastructure for computing linear relaxations
of operations given concrete interval bounds. Relaxations are shared between
forward and backward LBP propagation methods.
"""

from bound_propagation.relaxations.base import (
    RelaxationRegistry,
    RelaxationStrategy,
    register_relaxation_strategy,
)
from bound_propagation.relaxations.linear_relaxation import LinearRelaxation

# Import submodules to trigger auto-registration of strategies
from bound_propagation.relaxations import bilinear, elementwise

__all__ = [
    "LinearRelaxation",
    "RelaxationStrategy",
    "RelaxationRegistry",
    "register_relaxation_strategy",
    "elementwise",
    "bilinear",
]
