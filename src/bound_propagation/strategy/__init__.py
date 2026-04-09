"""
Bounding strategy framework for bound propagation.

This module provides the strategy pattern implementation for computing bounds
through different propagation methods (IBP, forward, backward, CROWN, etc.).

Key components:
- BoundingStrategy: Abstract base class for all strategies
- StrategyConfig: Configuration for strategy execution
- StrategyRegistry: Global registry mapping (operation, method) -> strategy
- BoundPropagator: Orchestrates bound computation through a graph
"""

from .config import StrategyConfig  # noqa: I001
from .propagator import BoundPropagator
from .registry import (
    StrategyRegistry,
    get_global_registry,
    get_strategy,
    register_fallback,
    register_strategy,
)
from .strategy import BoundingStrategy

from . import crown  # noqa: F401
from . import ibp  # noqa: F401

__all__ = [
    "BoundingStrategy",
    "StrategyConfig",
    "StrategyRegistry",
    "BoundPropagator",
    "get_global_registry",
    "register_strategy",
    "register_fallback",
    "get_strategy",
]
