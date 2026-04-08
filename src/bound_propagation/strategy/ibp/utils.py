"""Utility functions for IBP strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...bounds import AbstractBounds

from ...bounds import IntervalBounds


def verify_interval_bounds(input_bounds: list[AbstractBounds]) -> None:
    """
    Verify all inputs are IntervalBounds.
    
    Args:
        input_bounds: List of bounds to verify
        
    Raises:
        ValueError: If any bound is not an IntervalBounds instance
    """
    for i, bounds in enumerate(input_bounds):
        if not isinstance(bounds, IntervalBounds):
            raise ValueError(
                f"IBP requires IntervalBounds, got {type(bounds).__name__} for input {i}"
            )
