"""Custom exceptions raised by bound_propagation."""

from __future__ import annotations


class DimensionMismatchError(ValueError):
    """Raised when tensor shapes are incompatible (broadcast or reduction)."""
