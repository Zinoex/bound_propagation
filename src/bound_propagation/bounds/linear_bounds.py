"""
Linear bounds representation with affine relaxations.

Linear bounds represent regions as affine functions of input variables,
enabling tighter bounds through linear relaxations (used in CROWN methods).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from plum import dispatch

from ..regions import AbstractRegion, HyperRectangle
from .abstract_bounds import AbstractBounds

if TYPE_CHECKING:
    from .interval_bounds import IntervalBounds


class LinearBounds(AbstractBounds):
    """
    Linear bounds using affine relaxations.

    Represents bounds as affine functions: lower = W_l @ x + b_l, upper = W_u @ x + b_u
    This allows for tighter bounds through linear relaxations of non-linear operations.

    Used in CROWN-style bound propagation methods.

    Attributes:
        region: Input region defining the domain
        linear_lower: Linear coefficients for lower bound (W_l)
        bias_lower: Bias term for lower bound (b_l)
        linear_upper: Linear coefficients for upper bound (W_u)
        bias_upper: Bias term for upper bound (b_u)
    """

    def __init__(
        self,
        region: AbstractRegion,
        linear_lower: torch.Tensor | None,
        bias_lower: torch.Tensor,
        linear_upper: torch.Tensor | None,
        bias_upper: torch.Tensor,
    ) -> None:
        """
        Initialize linear bounds.

        Args:
            region: Input region (e.g., HyperRectangle)
            linear_lower: Linear coefficients for lower bound (can be None for constant bounds)
            bias_lower: Bias for lower bound
            linear_upper: Linear coefficients for upper bound (can be None for constant bounds)
            bias_upper: Bias for upper bound
        """
        super().__init__(region)

        self.linear_lower = linear_lower
        self.bias_lower = bias_lower
        self.linear_upper = linear_upper
        self.bias_upper = bias_upper

    @property
    def lower(self) -> torch.Tensor:
        """
        Get concrete lower bound.

        Note: For linear bounds, this requires evaluation at concrete input bounds.
        This property returns the bias term only. Use concretize() for full bounds.
        """
        return self.bias_lower

    @property
    def upper(self) -> torch.Tensor:
        """
        Get concrete upper bound.

        Note: For linear bounds, this requires evaluation at concrete input bounds.
        This property returns the bias term only. Use concretize() for full bounds.
        """
        return self.bias_upper

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of bounded tensor."""
        return tuple(self.bias_lower.shape)

    @property
    def device(self) -> torch.device:
        """Get device of bounds."""
        return self.bias_lower.device

    def to(self, device: str | torch.device) -> LinearBounds:
        """Move bounds to a device."""
        return LinearBounds(
            region=self.region.to(device),
            linear_lower=self.linear_lower.to(device) if self.linear_lower is not None else None,
            bias_lower=self.bias_lower.to(device),
            linear_upper=self.linear_upper.to(device) if self.linear_upper is not None else None,
            bias_upper=self.bias_upper.to(device),
        )

    def clone(self) -> LinearBounds:
        """Create a deep copy."""
        return LinearBounds(
            region=self.region,  # Regions are immutable
            linear_lower=self.linear_lower.clone() if self.linear_lower is not None else None,
            bias_lower=self.bias_lower.clone(),
            linear_upper=self.linear_upper.clone() if self.linear_upper is not None else None,
            bias_upper=self.bias_upper.clone(),
        )

    def forward_compose(self, bounds: LinearBounds) -> LinearBounds:
        ...

    def backward_compose(self, bounds: LinearBounds) -> LinearBounds:
        ...

    @staticmethod
    def from_interval_bounds(bounds: IntervalBounds) -> LinearBounds:
        """
        Create linear bounds from interval bounds.

        Creates constant linear bounds: lower = 0*x + bounds.lower, upper = 0*x + bounds.upper

        Args:
            bounds: Interval bounds to convert

        Returns:
            Linear bounds with zero coefficients
        """
        # Linear bounds with no dependence on inputs (constant bounds)
        return LinearBounds(
            region=bounds.region,
            linear_lower=None,  # No linear dependence
            bias_lower=bounds.lower,
            linear_upper=None,  # No linear dependence
            bias_upper=bounds.upper,
        )




@dispatch
def concretize(region: HyperRectangle, bounds: LinearBounds) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: F811
    """
    Concretize linear bounds given a hyperrectangle region.

    For linear bounds, evaluates the affine functions at the box extremes:
    - Lower bound: minimize W_l @ x + b_l over x in [lower, upper]
    - Upper bound: maximize W_u @ x + b_u over x in [lower, upper]

    For each weight coefficient:
    - Use region.lower if coefficient > 0 (for minimization)
    - Use region.upper if coefficient < 0 (for minimization)
    - Vice versa for maximization

    Args:
        region: The hyperrectangle input region
        bounds: The linear bounds to concretize

    Returns:
        Tuple of (lower, upper) concrete bounds
    """
    # Flatten the hyperrectangle bounds for easier computation
    input_lower = region.lower.flatten()
    input_upper = region.upper.flatten()

    # Lower bound computation: minimize W_l @ x + b_l
    lower_result = bounds.bias_lower.clone()
    if bounds.linear_lower is not None:
        positive_mask = bounds.linear_lower > 0
        contributions = torch.where(
            positive_mask,
            bounds.linear_lower * input_lower,
            bounds.linear_lower * input_upper,
        )
        lower_result = lower_result + contributions.sum(dim=-1)

    # Upper bound computation: maximize W_u @ x + b_u
    upper_result = bounds.bias_upper.clone()
    if bounds.linear_upper is not None:
        positive_mask = bounds.linear_upper > 0
        contributions = torch.where(
            positive_mask,
            bounds.linear_upper * input_upper,
            bounds.linear_upper * input_lower,
        )
        upper_result = upper_result + contributions.sum(dim=-1)

    return lower_result, upper_result
