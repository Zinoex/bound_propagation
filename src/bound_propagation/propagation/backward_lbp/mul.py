from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from ..strategy import ForwardBoundingStrategy
from .utils import verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class BackwardLBPMulStrategy(ForwardBoundingStrategy):
    """
    Backward LBP strategy for MUL operation.

    For multiplication z = x * y, in backward mode the bounds are computed
    the same way as forward mode since multiplication is a linear/bilinear operation.
    """

    @property
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        return "backward"

    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> AbstractBounds:
        """
        Compute backward LBP bounds for multiplication.

        Args:
            node: The MUL node
            input_bounds: List of two LinearBounds for the operands
            config: Strategy configuration

        Returns:
            LinearBounds for the product
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 2:
            raise ValueError(f"MUL requires exactly 2 inputs, got {len(input_bounds)}")

        bounds_a: LinearBounds = input_bounds[0]
        bounds_b: LinearBounds = input_bounds[1]

        # Check if one operand is constant (no linear dependency)
        a_is_constant = not bounds_a.linear_lowers and not bounds_a.linear_uppers
        b_is_constant = not bounds_b.linear_lowers and not bounds_b.linear_uppers

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
                regions=[],
                linear_lower=[],
                bias_lower=lower,
                linear_upper=[],
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
        linear_lower = [
            torch.where(
                positive_mask.unsqueeze(-1),
                constant.unsqueeze(-1) * lower_linear,
                constant.unsqueeze(-1) * upper_linear,
            )
            for lower_linear, upper_linear in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
        ]
        linear_upper = [
            torch.where(
                positive_mask.unsqueeze(-1),
                constant.unsqueeze(-1) * upper_linear,
                constant.unsqueeze(-1) * lower_linear,
            )
            for lower_linear, upper_linear in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
        ]

        bias_lower_pos = constant * bounds.bias_lower
        bias_lower_neg = constant * bounds.bias_upper
        bias_lower = torch.where(positive_mask, bias_lower_pos, bias_lower_neg)

        bias_upper_pos = constant * bounds.bias_upper
        bias_upper_neg = constant * bounds.bias_lower
        bias_upper = torch.where(positive_mask, bias_upper_pos, bias_upper_neg)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
