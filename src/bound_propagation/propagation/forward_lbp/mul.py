from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPMulStrategy(ForwardLBPStrategy):
    """
    Forward LBP strategy for MUL operation.

    For multiplication z = x * y:
    - If y is constant: W_l^z = y * W_l^x, b_l^z = y * b_l^x (handle sign)
    - If both vary: concretize to intervals(?)
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"MUL requires exactly 2 inputs, got {len(input_bounds)}")

        bounds_a: LinearBounds = input_bounds[0]
        bounds_b: LinearBounds = input_bounds[1]

        # Check if one operand is constant (no linear dependency)
        a_is_constant = bounds_a.linear_lower is None and bounds_a.linear_upper is None
        b_is_constant = bounds_b.linear_lower is None and bounds_b.linear_upper is None

        if b_is_constant:
            # y is constant: z = x * c
            return self._multiply_by_constant(bounds_a, bounds_b.bias_lower)
        elif a_is_constant:
            # x is constant: z = c * y
            return self._multiply_by_constant(bounds_b, bounds_a.bias_lower)
        else:
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
            lower = torch.min(torch.stack(products), dim=0)[0]
            upper = torch.max(torch.stack(products), dim=0)[0]

            # Return as constant LinearBounds
            return LinearBounds(
                region=bounds_a.region,
                linear_lower=None,
                bias_lower=lower,
                linear_upper=None,
                bias_upper=upper,
            )

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
