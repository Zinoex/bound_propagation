    from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from ..strategy import BoundingStrategy
from .utils import verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class ForwardCrownReluStrategy(BoundingStrategy):
    """
    Forward CROWN strategy for RELU operation.

    For ReLU y = max(0, x), we use adaptive linear relaxations:
    - If x >= 0 (active): y = x (identity)
    - If x <= 0 (inactive): y = 0 (zero)
    - If x crosses zero: linear relaxation
      - Lower bound: y >= 0 (always valid)
      - Upper bound: y <= (u/(u-l)) * (x - l) where [l, u] are concrete bounds
    """

    @property
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        return "forward"

    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> AbstractBounds:
        """
        Compute forward CROWN bounds for ReLU.

        Args:
            node: The RELU node
            input_bounds: List with one LinearBounds for the input
            config: Strategy configuration

        Returns:
            LinearBounds for the output
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"RELU requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize to get interval bounds for determining ReLU behavior
        lower, upper = bounds.concretize()

        # Determine ReLU cases element-wise
        active = lower >= 0  # Always positive - ReLU is identity
        inactive = upper <= 0  # Always negative - ReLU is zero
        crossing = ~active & ~inactive  # Can be positive or negative

        # Initialize output bounds
        output_shape = lower.shape
        device = lower.device
        dtype = lower.dtype

        # For lower bound:
        # - Active: keep original
        # - Inactive:zero
        # - Crossing: zero (tightest linear lower bound)
        if bounds.linear_lower is not None:
            linear_lower = torch.where(
                active.unsqueeze(-1),
                bounds.linear_lower,
                torch.zeros_like(bounds.linear_lower),
            )
        else:
            linear_lower = None

        bias_lower = torch.where(
            active,
            bounds.bias_lower,
            torch.zeros_like(bounds.bias_lower),
        )

        # For upper bound:
        # - Active: keep original
        # - Inactive: zero
        # - Crossing: linear relaxation y <= slope * (x - l)
        #   where slope = u / (u - l), giving y <= slope * x - slope * l
        if crossing.any():
            # Compute slope for crossing neurons
            slope = torch.zeros_like(upper)
            denominator = upper - lower
            valid_denom = torch.abs(denominator) > 1e-8
            slope_value = upper / torch.where(valid_denom, denominator, torch.ones_like(denominator))
            slope = torch.where(crossing & valid_denom, slope_value, torch.zeros_like(slope))

            # Upper bound linear: slope * W_u
            if bounds.linear_upper is not None:
                linear_upper = torch.where(
                    active.unsqueeze(-1),
                    bounds.linear_upper,
                    torch.where(
                        crossing.unsqueeze(-1),
                        slope.unsqueeze(-1) * bounds.linear_upper,
                        torch.zeros_like(bounds.linear_upper),
                    ),
                )
            elif bounds.linear_lower is not None:
                # Use lower if upper not available
                linear_upper = torch.where(
                    active.unsqueeze(-1),
                    bounds.linear_lower,
                    torch.where(
                        crossing.unsqueeze(-1),
                        slope.unsqueeze(-1) * bounds.linear_lower,
                        torch.zeros_like(bounds.linear_lower),
                    ),
                )
            else:
                linear_upper = None

            # Upper bound bias: slope * b_u - slope * l
            bias_crossing = slope * bounds.bias_upper - slope * lower
            bias_upper = torch.where(
                active,
                bounds.bias_upper,
                torch.where(
                    crossing,
                    bias_crossing,
                    torch.zeros_like(bounds.bias_upper),
                ),
            )
        else:
            # No crossing case
            if bounds.linear_upper is not None:
                linear_upper = torch.where(
                    active.unsqueeze(-1),
                    bounds.linear_upper,
                    torch.zeros_like(bounds.linear_upper),
                )
            elif bounds.linear_lower is not None:
                linear_upper = torch.where(
                    active.unsqueeze(-1),
                    bounds.linear_lower,
                    torch.zeros_like(bounds.linear_lower),
                )
            else:
                linear_upper = None

            bias_upper = torch.where(
                active,
                bounds.bias_upper,
                torch.zeros_like(bounds.bias_upper),
            )

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
