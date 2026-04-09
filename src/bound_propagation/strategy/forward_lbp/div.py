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


class ForwardLBPDivStrategy(BoundingStrategy):
    """
    Forward LBP strategy for DIV operation.

    For division z = x / y:
    - If y is constant: W_l^z = W_l^x / y, b_l^z = b_l^x / y (handle sign)
    - If y varies: concretize to intervals
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
        Compute forward LBP bounds for division.

        Args:
            node: The DIV node
            input_bounds: List of two LinearBounds for the operands
            config: Strategy configuration

        Returns:
            LinearBounds for the quotient
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 2:
            raise ValueError(f"DIV requires exactly 2 inputs, got {len(input_bounds)}")

        bounds_a: LinearBounds = input_bounds[0]
        bounds_b: LinearBounds = input_bounds[1]

        # Check if divisor is constant
        b_is_constant = bounds_b.linear_lower is None and bounds_b.linear_upper is None

        if b_is_constant:
            # Divide by constant
            divisor = bounds_b.bias_lower
            return self._divide_by_constant(bounds_a, divisor)
        else:
            # Divisor varies - concretize
            lower_a, upper_a = bounds_a.concretize()
            lower_b, upper_b = bounds_b.concretize()

            # Check for division by zero
            if torch.any((lower_b <= 0) & (upper_b >= 0)):
                # Division by zero possible - use safe bounds
                lower = torch.full_like(lower_a, float("-inf"))
                upper = torch.full_like(upper_a, float("inf"))
            else:
                # Compute interval division
                quotients = [
                    lower_a / lower_b,
                    lower_a / upper_b,
                    upper_a / lower_b,
                    upper_a / upper_b,
                ]
                lower = torch.min(torch.stack(quotients), dim=0)[0]
                upper = torch.max(torch.stack(quotients), dim=0)[0]

            return LinearBounds(
                region=bounds_a.region,
                linear_lower=None,
                bias_lower=lower,
                linear_upper=None,
                bias_upper=upper,
            )

    def _divide_by_constant(self, bounds: LinearBounds, divisor: torch.Tensor) -> LinearBounds:
        """
        Divide linear bounds by a constant.

        Args:
            bounds: LinearBounds to divide
            divisor: Constant divisor

        Returns:
            Scaled LinearBounds
        """
        # Similar to multiplication, but with division
        positive_mask = divisor > 0

        # Handle linear coefficients
        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            linear_lower_pos = bounds.linear_lower / divisor.unsqueeze(-1)
            linear_lower_neg = bounds.linear_upper / divisor.unsqueeze(-1)
            linear_lower = torch.where(positive_mask.unsqueeze(-1), linear_lower_pos, linear_lower_neg)

            linear_upper_pos = bounds.linear_upper / divisor.unsqueeze(-1)
            linear_upper_neg = bounds.linear_lower / divisor.unsqueeze(-1)
            linear_upper = torch.where(positive_mask.unsqueeze(-1), linear_upper_pos, linear_upper_neg)
        elif bounds.linear_lower is not None:
            linear_lower = bounds.linear_lower / divisor.unsqueeze(-1)
            linear_upper = bounds.linear_lower / divisor.unsqueeze(-1)
        elif bounds.linear_upper is not None:
            linear_lower = bounds.linear_upper / divisor.unsqueeze(-1)
            linear_upper = bounds.linear_upper / divisor.unsqueeze(-1)
        else:
            linear_lower = None
            linear_upper = None

        # Handle bias terms
        bias_lower_pos = bounds.bias_lower / divisor
        bias_lower_neg = bounds.bias_upper / divisor
        bias_lower = torch.where(positive_mask, bias_lower_pos, bias_lower_neg)

        bias_upper_pos = bounds.bias_upper / divisor
        bias_upper_neg = bounds.bias_lower / divisor
        bias_upper = torch.where(positive_mask, bias_upper_pos, bias_upper_neg)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
