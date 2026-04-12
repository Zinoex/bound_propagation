from __future__ import annotations

import torch
from plum import dispatch

from .abstract_bounds import AbstractBounds


class IntervalBounds(AbstractBounds):
    """
    Interval bounds using simple lower and upper bound tensors.

    This is the simplest form of bounds - just [lower, upper] intervals
    for each element. Propagation uses interval arithmetic rules.

    Attributes:
        lower: Lower bound tensor
        upper: Upper bound tensor
    """

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """
        Initialize interval bounds.

        Args:
            region: Input region (e.g., HyperRectangle)
            lower: Lower bound tensor
            upper: Upper bound tensor

        Raises:
            ValueError: If shapes don't match or bounds are invalid
        """
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
            lower=self._lower.to(device),
            upper=self._upper.to(device),
        )

    def __getitem__(self, item) -> IntervalBounds:
        """
        Slice/index the bounds.

        Args:
            item: Slice/index specification (e.g., for batch slicing)
        Returns:
            New bounds corresponding to the slice/index
        """
        return IntervalBounds(
            lower=self._lower[item],
            upper=self._upper[item],
        )

    def concretize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concretize interval bounds to get lower and upper tensors.

        For interval bounds, this simply returns the lower and upper tensors.

        Returns:
            Tuple of (lower, upper) tensors
        """
        return self._lower, self._upper

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

    @staticmethod
    @dispatch
    def unbounded_like(x: torch.Tensor) -> IntervalBounds:
        """
        Create unbounded interval bounds ([-inf, inf]).

        Args:
            x: Tensor to match shape, device, and dtype

        Returns:
            Unbounded IntervalBounds
        """
        lower = torch.full_like(x, float("-inf"))
        upper = torch.full_like(x, float("inf"))
        return IntervalBounds(lower, upper)

    @staticmethod
    @dispatch
    def unbounded_like(x: IntervalBounds) -> IntervalBounds:  # noqa: F811
        """
        Create unbounded interval bounds ([-inf, inf]) matching another IntervalBounds.

        Args:
            x: IntervalBounds to match shape, device, and dtype

        Returns:
            Unbounded IntervalBounds
        """
        lower = torch.full_like(x.lower, float("-inf"))
        upper = torch.full_like(x.upper, float("inf"))
        return IntervalBounds(lower, upper)

    def clone(self) -> IntervalBounds:
        """
        Create a copy of these bounds.

        Returns:
            New IntervalBounds with cloned tensors
        """
        return IntervalBounds(
            lower=self._lower.clone(),
            upper=self._upper.clone(),
        )
