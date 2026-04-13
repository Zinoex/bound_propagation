from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPMul(ForwardLBPStrategy):
    """
    Forward LBP strategy for MUL operation when both inputs are abstract.

    For multiplication z = x * y where both have linear dependencies:
    - Concretize to intervals and compute interval product
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"mul requires exactly 2 inputs, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds) or not isinstance(input_bounds[1], LinearBounds):
            raise TypeError("ForwardLBPMul requires both inputs to be LinearBounds")

        bounds_a = input_bounds[0]
        bounds_b = input_bounds[1]

        # Both have linear dependencies - concretize and convert back
        lower_a, upper_a = bounds_a.concretize()
        lower_b, upper_b = bounds_b.concretize()

        # Compute interval product
        products = [
            lower_a * lower_b,
            lower_a * upper_b,
            upper_a * lower_b,
            upper_a * upper_b,
        ]
        lower = torch.stack(products).min(dim=0)[0]
        upper = torch.stack(products).max(dim=0)[0]

        # Return as constant LinearBounds
        return LinearBounds(
            region=bounds_a.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )


class ForwardLBPMulWithConstant(ForwardLBPStrategy):
    """
    Forward LBP strategy for MUL when at least one input is constant.

    For multiplication z = x * c:
    - If c >= 0: W_l^z = c * W_l^x, b_l^z = c * b_l^x
    - If c < 0: bounds flip (lower becomes upper, upper becomes lower)
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"mul requires exactly 2 inputs, got {len(input_bounds)}")

        left = input_bounds[0]
        right = input_bounds[1]

        if isinstance(left, LinearBounds):
            bounds, constant = left, right
        elif isinstance(right, LinearBounds):
            bounds, constant = right, left
        else:
            raise TypeError(
                f"ForwardLBPMulWithConstant requires one input to be LinearBounds and the other to be "
                f"torch.Tensor or Number, got {type(left)} and {type(right)}"
            )

        return self._multiply_by_constant(bounds, constant)

    def _multiply_by_constant(self, bounds: LinearBounds, constant: torch.Tensor) -> LinearBounds:
        """
        Multiply linear bounds by a constant.

        Args:
            bounds: LinearBounds to multiply
            constant: Constant multiplier

        Returns:
            Scaled LinearBounds
        """
        # When multiplying by positive constant, bounds stay same order
        # When multiplying by negative constant, bounds flip
        positive_mask = constant >= 0

        # For positive values: lower *= c, upper *= c
        # For negative values: lower = c * old_upper, upper = c * old_lower
        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            linear_lower_pos = constant.unsqueeze(-1) * bounds.linear_lower
            linear_lower_neg = constant.unsqueeze(-1) * bounds.linear_upper
            linear_lower = torch.where(positive_mask.unsqueeze(-1), linear_lower_pos, linear_lower_neg)

            linear_upper_pos = constant.unsqueeze(-1) * bounds.linear_upper
            linear_upper_neg = constant.unsqueeze(-1) * bounds.linear_lower
            linear_upper = torch.where(positive_mask.unsqueeze(-1), linear_upper_pos, linear_upper_neg)
        elif bounds.linear_lower is not None:
            linear_lower = constant.unsqueeze(-1) * bounds.linear_lower
            linear_upper = constant.unsqueeze(-1) * bounds.linear_lower
        elif bounds.linear_upper is not None:
            linear_lower = constant.unsqueeze(-1) * bounds.linear_upper
            linear_upper = constant.unsqueeze(-1) * bounds.linear_upper
        else:
            linear_lower = None
            linear_upper = None

        bias_lower_pos = constant * bounds.bias_lower
        bias_lower_neg = constant * bounds.bias_upper
        bias_lower = torch.where(positive_mask, bias_lower_pos, bias_lower_neg)

        bias_upper_pos = constant * bounds.bias_upper
        bias_upper_neg = constant * bounds.bias_lower
        bias_upper = torch.where(positive_mask, bias_upper_pos, bias_upper_neg)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
