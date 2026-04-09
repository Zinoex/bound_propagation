"""
HyperRectangle input region.

Represents a box-constrained input domain [lower, upper].
"""

from __future__ import annotations

import torch

from .abstract import AbstractRegion


class HyperRectangle(AbstractRegion):
    """
    Hyperrectangle input region.

    Represents box constraints on inputs: lower[i] <= x[i] <= upper[i] for each i.
    This is the most common type of input region for bound propagation.

    Attributes:
        lower: Lower bounds for each input dimension
        upper: Upper bounds for each input dimension
    """

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """
        Initialize hyperrectangle region.

        Args:
            lower: Lower bound tensor
            upper: Upper bound tensor

        Raises:
            ValueError: If shapes don't match or bounds are invalid
        """
        if lower.shape != upper.shape:
            raise ValueError(f"Lower and upper bounds must have same shape: {lower.shape} vs {upper.shape}")

        if lower.device != upper.device:
            raise ValueError(f"Lower and upper bounds must be on same device: {lower.device} vs {upper.device}")

        if not torch.all(lower <= upper + 1e-6):
            violations = torch.sum(lower > upper + 1e-6).item()
            raise ValueError(f"Lower bound must be <= upper bound (found {violations} violations)")

        self.lower = lower
        self.upper = upper

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of the region."""
        return tuple(self.lower.shape)

    @property
    def device(self) -> torch.device:
        """Get device of the region."""
        return self.lower.device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of the region."""
        return self.lower.dtype

    def to(self, device: str | torch.device) -> HyperRectangle:
        """Move region to a device."""
        return HyperRectangle(
            lower=self.lower.to(device),
            upper=self.upper.to(device),
        )

    def __getitem__(self, item) -> HyperRectangle:
        """Slice the hyperrectangle."""
        return HyperRectangle(
            lower=self.lower[item],
            upper=self.upper[item],
        )

    def maximize(self, direction: torch.Tensor) -> torch.Tensor:
        """
        Get the point in the hyperrectangle that maximizes the linear function defined by direction.

        Args:
            direction: Coefficients of the linear function to maximize
        Returns:
            sup_{x in hyperrectangle} direction^T x
        """
        return torch.where(direction >= 0, self.upper, self.lower) * direction

    def minimize(self, direction: torch.Tensor) -> torch.Tensor:
        """
        Get the point in the hyperrectangle that minimizes the linear function defined by direction.

        Args:
            direction: Coefficients of the linear function to minimize
        Returns:
            inf_{x in hyperrectangle} direction^T x
        """
        return torch.where(direction >= 0, self.lower, self.upper) * direction

    @property
    def width(self) -> torch.Tensor:
        """Get width of the hyperrectangle (upper - lower)."""
        return self.upper - self.lower

    @property
    def center(self) -> torch.Tensor:
        """Get center of the hyperrectangle ((lower + upper) / 2)."""
        return (self.lower + self.upper) / 2

    @staticmethod
    def from_eps(center: torch.Tensor, epsilon: float) -> HyperRectangle:
        """
        Create hyperrectangle from center and epsilon perturbation.

        Args:
            center: Center point
            epsilon: Perturbation radius (creates [center - eps, center + eps])

        Returns:
            Hyperrectangle region
        """
        return HyperRectangle(
            lower=center - epsilon,
            upper=center + epsilon,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"HyperRectangle(shape={self.shape}, lower={self.lower.min():.3f}..{self.lower.max():.3f}, upper={self.upper.min():.3f}..{self.upper.max():.3f})"
