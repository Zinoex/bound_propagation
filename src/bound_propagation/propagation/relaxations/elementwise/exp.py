"""
Exponential relaxation strategy.

Computes linear relaxations for the exponential function.
"""

import torch

from ....bounds import IntervalBounds
from ....ir import Node, OperationType
from ..base import (
    RelaxationStrategy,
    register_relaxation_strategy,
)
from ..linear_relaxation import LinearRelaxation


@register_relaxation_strategy
class ExpRelaxationStrategy(RelaxationStrategy):
    """
    Relaxation strategy for exponential function.

    Exp is a convex function: exp(x).
    This implementation provides a basic relaxation using tangent and
    secant lines.
    """

    @property
    def supported_op_type(self) -> OperationType:
        return OperationType.EXP

    def relax(
        self,
        node: Node,
        interval_inputs: list[IntervalBounds],
    ) -> LinearRelaxation:
        """
        Compute linear relaxation for exp.

        For a convex function:
        - Lower bound: secant line (under-approximates)
        - Upper bound: tangent at upper point (over-approximates)

        Args:
            node: The exp operation node.
            interval_inputs: List containing single IntervalBounds for the input.

        Returns:
            LinearRelaxation with diagonal coefficients (element-wise slopes).

        Raises:
            ValueError: If number of inputs is not 1.
        """
        if len(interval_inputs) != 1:
            raise ValueError(
                f"Exp expects 1 input, got {len(interval_inputs)}"
            )

        input_bounds = interval_inputs[0]
        lower, upper = input_bounds.concretize()

        # Compute exp at bounds
        lower_act = torch.exp(lower)
        upper_act = torch.exp(upper)

        zero_width = torch.isclose(lower, upper)

        # Secant line slope
        slope = torch.where(
            zero_width,
            torch.zeros_like(lower),
            (upper_act - lower_act) / (upper - lower)
        )

        # Lower bound: secant line (for convex function)
        alpha_lower = torch.where(zero_width, torch.zeros_like(lower), slope)
        beta_lower = torch.where(
            zero_width,
            lower_act,
            lower_act - slope * lower
        )

        # Upper bound: tangent at upper point (for convex function)
        upper_deriv = upper_act  # derivative of exp(x) is exp(x)
        alpha_upper = torch.where(zero_width, torch.zeros_like(lower), upper_deriv)
        beta_upper = torch.where(
            zero_width,
            upper_act,
            upper_act - upper_deriv * upper
        )

        # Create diagonal relaxation
        return LinearRelaxation.create_diagonal(
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            beta_lower=beta_lower,
            beta_upper=beta_upper,
        )
