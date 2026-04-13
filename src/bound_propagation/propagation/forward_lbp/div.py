from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPDivStrategy(ForwardLBPStrategy):
    """
    Forward LBP strategy for DIV operation when both inputs are abstract.

    For division z = x / y where both have linear dependencies:
    - Concretize to intervals and compute interval division
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"DIV requires exactly 2 inputs, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds) or not isinstance(input_bounds[1], LinearBounds):
            raise TypeError("ForwardLBPDivStrategy requires both inputs to be LinearBounds")

        bounds_a: LinearBounds = input_bounds[0]
        bounds_b: LinearBounds = input_bounds[1]

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


class ForwardLBPDivConstant(ForwardLBPStrategy):
    """Forward LBP strategy for DIV when the second input (divisor) is constant: x / c."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"DIV requires exactly 2 inputs, got {len(input_bounds)}")

        x = input_bounds[0]
        c = input_bounds[1]

        if not isinstance(x, LinearBounds) or isinstance(c, LinearBounds):
            raise TypeError(
                "ForwardLBPDivConstant requires the first input to be LinearBounds "
                "and the second input to be torch.Tensor or Number"
            )

        return self._divide_by_constant(x, c)

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


class ForwardLBPConstantDiv(ForwardLBPStrategy):
    """Forward LBP strategy for DIV when the first input (dividend) is constant: c / x."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"DIV requires exactly 2 inputs, got {len(input_bounds)}")

        c = input_bounds[0]
        x = input_bounds[1]

        if not isinstance(x, LinearBounds) or isinstance(c, LinearBounds):
            raise TypeError(
                "ForwardLBPConstantDiv requires the second input to be LinearBounds "
                "and the first input to be torch.Tensor or Number"
            )

        # c / x: concretize x and compute interval division
        lower_x, upper_x = x.concretize()

        # Check for division by zero
        if torch.any((lower_x <= 0) & (upper_x >= 0)):
            lower = torch.full_like(lower_x, float("-inf"))
            upper = torch.full_like(upper_x, float("inf"))
        else:
            quotients = [c / lower_x, c / upper_x]
            lower = torch.min(torch.stack(quotients), dim=0)[0]
            upper = torch.max(torch.stack(quotients), dim=0)[0]

        return LinearBounds(
            region=x.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
