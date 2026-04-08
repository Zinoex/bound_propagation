"""
Bound representation types for bound propagation analysis.

This module provides different representations of bounds that can be propagated
through computation graphs:

- AbstractBounds: Base interface for all bound types
- IntervalBounds: Simple interval arithmetic bounds [lower, upper]
- LinearBounds: Affine bounds with linear relaxations (for CROWN-style methods)
"""

from .abstract_bounds import AbstractBounds
from .interval_bounds import IntervalBounds
from .linear_bounds import LinearBounds

__all__ = [
    "AbstractBounds",
    "IntervalBounds",
    "LinearBounds",
]
