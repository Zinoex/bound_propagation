"""
Interval bounds representation.

Simple lower and upper bound tensors for interval arithmetic.
"""

from __future__ import annotations

import torch
from plum import dispatch

from ..regions import AbstractRegion, HyperRectangle
from .abstract_bounds import AbstractBounds


class IntervalBounds(AbstractBounds):
    """
    Interval bounds using simple lower and upper bound tensors.

    This is the simplest form of bounds - just [lower, upper] intervals
    for each element. Propagation uses interval arithmetic rules.

    Attributes:
        region: Input region defining the domain
        lower: Lower bound tensor
        upper: Upper bound tensor
    """

    def __init__(self, region: AbstractRegion, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """
        Initialize interval bounds.

        Args:
            region: Input region (e.g., HyperRectangle)
            lower: Lower bound tensor
            upper: Upper bound tensor

        Raises:
            ValueError: If shapes don't match or bounds are invalid
        """
        super().__init__(region)

        if lower.shape != upper.shape:
            raise ValueError(f"Lower and upper bounds must have same shape: {lower.shape} vs {upper.shape}")

        if lower.device != upper.device:
            raise ValueError(f"Lower and upper bounds must be on same device: {lower.device} vs {upper.device}")

        # Check that lower <= upper (allow some numerical tolerance)
        if not torch.all(lower <= upper + 1e-6):
            violations = torch.sum(lower > upper + 1e-6).item()
            raise ValueError(f"Lower bound must be <= upper bound (found {violations} violations)")

        self._lower = lower
        self._upper = upper

    @property
    def lower(self) -> torch.Tensor:
        """Get lower bound tensor."""
        return self._lower

    @property
    def upper(self) -> torch.Tensor:
        """Get upper bound tensor."""
        return self._upper

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of bounded tensor."""
        return tuple(self._lower.shape)

    @property
    def device(self) -> torch.device:
        """Get device of bounds."""
        return self._lower.device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of bounds."""
        return self._lower.dtype

    def to(self, device: str | torch.device) -> IntervalBounds:
        """
        Move bounds to a device.

        Args:
            device: Target device

        Returns:
            New bounds on target device
        """
        return IntervalBounds(
            region=self.region.to(device),
            lower=self._lower.to(device),
            upper=self._upper.to(device),
        )

    def clone(self) -> IntervalBounds:
        """
        Create a deep copy of these bounds.

        Returns:
            Cloned interval bounds
        """
        return IntervalBounds(
            region=self.region,  # Regions are immutable, so no need to clone
            lower=self._lower.clone(),
            upper=self._upper.clone(),
        )

    @property
    def width(self) -> torch.Tensor:
        """
        Get interval width (upper - lower).

        Returns:
            Tensor of interval widths
        """
        return self._upper - self._lower

    @property
    def center(self) -> torch.Tensor:
        """
        Get interval center (lower + upper) / 2.

        Returns:
            Tensor of interval centers
        """
        return (self._lower + self._upper) / 2

    def contains(self, value: torch.Tensor, tolerance: float = 1e-6) -> torch.Tensor:
        """
        Check if value is contained in bounds.

        Args:
            value: Tensor to check
            tolerance: Numerical tolerance for boundary checks

        Returns:
            Boolean tensor indicating containment for each element
        """
        return (self._lower - tolerance <= value) & (value <= self._upper + tolerance)

    def intersection(self, other: IntervalBounds) -> IntervalBounds:
        """
        Compute intersection with another interval.

        Args:
            other: Other interval bounds

        Returns:
            Interval bounds representing the intersection

        Raises:
            ValueError: If shapes don't match
        """
        if self.shape != other.shape:
            raise ValueError(f"Cannot intersect intervals with different shapes: {self.shape} vs {other.shape}")

        new_lower = torch.maximum(self._lower, other._lower)
        new_upper = torch.minimum(self._upper, other._upper)

        return IntervalBounds(region=self.region, lower=new_lower, upper=new_upper)

    def union(self, other: IntervalBounds) -> IntervalBounds:
        """
        Compute union (hull) with another interval.

        Args:
            other: Other interval bounds

        Returns:
            Interval bounds representing the smallest enclosing interval

        Raises:
            ValueError: If shapes don't match
        """
        if self.shape != other.shape:
            raise ValueError(f"Cannot union intervals with different shapes: {self.shape} vs {other.shape}")

        new_lower = torch.minimum(self._lower, other._lower)
        new_upper = torch.maximum(self._upper, other._upper)

        return IntervalBounds(region=self.region, lower=new_lower, upper=new_upper)

    @staticmethod
    def from_tensor(tensor: torch.Tensor, epsilon: float = 0.0) -> IntervalBounds:
        """
        Create interval bounds from a concrete tensor value.

        Args:
            tensor: Concrete tensor value
            epsilon: Optional perturbation radius (creates [tensor - epsilon, tensor + epsilon])

        Returns:
            Interval bounds around the tensor
        """
        from ..regions import HyperRectangle

        if epsilon == 0.0:
            region = HyperRectangle(lower=tensor.clone(), upper=tensor.clone())
            return IntervalBounds(region=region, lower=tensor.clone(), upper=tensor.clone())
        else:
            region = HyperRectangle(lower=tensor - epsilon, upper=tensor + epsilon)
            return IntervalBounds(
                region=region,
                lower=tensor - epsilon,
                upper=tensor + epsilon,
            )

    @staticmethod
    def unbounded(shape: tuple[int, ...], device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> IntervalBounds:
        """
        Create unbounded interval (-inf, +inf).

        Args:
            shape: Shape of the interval
            device: Device to create bounds on
            dtype: Data type for bounds

        Returns:
            Unbounded interval bounds
        """
        from ..regions import HyperRectangle

        lower = torch.full(shape, float("-inf"), device=device, dtype=dtype)
        upper = torch.full(shape, float("inf"), device=device, dtype=dtype)
        region = HyperRectangle(lower=lower, upper=upper)
        return IntervalBounds(
            region=region,
            lower=lower,
            upper=upper,
        )


@dispatch
def concretize(region: HyperRectangle, bounds: IntervalBounds) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concretize interval bounds given a hyperrectangle region.

    For interval bounds, simply returns the bounds as they are already concrete.

    Args:
        region: The hyperrectangle input region
        bounds: The interval bounds to concretize
    Returns:
        Tuple of (lower, upper) concrete bounds
    """
    return bounds.lower, bounds.upper
