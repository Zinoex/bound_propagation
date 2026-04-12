"""
Logarithm relaxation strategy.

Computes linear relaxations for the natural logarithm function.
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
class LogRelaxationStrategy(RelaxationStrategy):
    """
    Relaxation strategy for natural logarithm function.

    Log is a concave function defined for positive inputs: log(x).
    This implementation provides a basic relaxation. More sophisticated
    strategies can be added later.
    """

    @property
    def supported_op_type(self) -> OperationType:
        return OperationType.LOG

    def relax(
        self,
        node: Node,
        interval_inputs: list[IntervalBounds],
    ) -> LinearRelaxation:
        """
        Compute linear relaxation for log.

        For now, this uses a simple secant line approximation for both
        upper and lower bounds. A tighter relaxation would use tangent
        lines for the lower bound.

        Args:
            node: The log operation node.
            interval_inputs: List containing single IntervalBounds for the input.

        Returns:
            LinearRelaxation with diagonal coefficients (element-wise slopes).

        Raises:
            ValueError: If number of inputs is not 1 or if inputs are non-positive.
        """
        if len(interval_inputs) != 1:
            raise ValueError(f"Log expects 1 input, got {len(interval_inputs)}")

        input_bounds = interval_inputs[0]
        lower, upper = input_bounds.concretize()

        # Check that all inputs are positive
        if torch.any(lower <= 0):
            raise ValueError(f"Log relaxation requires positive lower bounds. Got minimum lower bound: {lower.min().item()}")

        # Compute log at bounds
        lower_act = torch.log(lower)
        upper_act = torch.log(upper)

        # Use secant line for both bounds (simple approach)
        # For a concave function, secant provides lower bound
        # Tangent at lower point provides upper bound
        zero_width = torch.isclose(lower, upper)

        slope = torch.where(zero_width, torch.zeros_like(lower), (upper_act - lower_act) / (upper - lower))

        # For exact points
        alpha_lower = torch.where(zero_width, torch.zeros_like(lower), slope)
        beta_lower = torch.where(zero_width, lower_act, lower_act - slope * lower)

        # Upper bound: tangent at lower point (concave function)
        lower_deriv = 1.0 / lower
        alpha_upper = torch.where(zero_width, torch.zeros_like(lower), lower_deriv)
        beta_upper = torch.where(zero_width, upper_act, lower_act - lower_deriv * lower)

        # Create diagonal relaxation
        return LinearRelaxation.create_diagonal(
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            beta_lower=beta_lower,
            beta_upper=beta_upper,
        )
