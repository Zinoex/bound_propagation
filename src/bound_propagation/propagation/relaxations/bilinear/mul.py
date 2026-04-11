"""
Multiplication relaxation strategy.

Computes linear relaxations for element-wise multiplication.
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
class MulRelaxationStrategy(RelaxationStrategy):
    """
    Relaxation strategy for element-wise multiplication.

    For z = x * y, we need to compute linear bounds:
        z_lower >= a1_l * x + a2_l * y + b_l
        z_upper <= a1_u * x + a2_u * y + b_u

    This is a bilinear operation, so we use interval arithmetic to
    compute the four corners and derive the linear relaxation.
    """

    @property
    def supported_op_type(self) -> OperationType:
        return OperationType.MUL

    def relax(
        self,
        node: Node,
        interval_inputs: list[IntervalBounds],
    ) -> LinearRelaxation:
        """
        Compute linear relaxation for element-wise multiplication.

        For z = x * y where x ∈ [x_l, x_u] and y ∈ [y_l, y_u]:
        We compute the min/max over the four corners:
            (x_l, y_l), (x_l, y_u), (x_u, y_l), (x_u, y_u)

        Then derive linear coefficients that bound these corners.

        Args:
            node: The multiplication operation node.
            interval_inputs: List containing two IntervalBounds (for x and y).

        Returns:
            LinearRelaxation with coefficients for both inputs.

        Raises:
            ValueError: If number of inputs is not 2.
        """
        if len(interval_inputs) != 2:
            raise ValueError(
                f"Mul expects 2 inputs, got {len(interval_inputs)}"
            )

        x_bounds = interval_inputs[0]
        y_bounds = interval_inputs[1]

        x_l, x_u = x_bounds.concretize()
        y_l, y_u = y_bounds.concretize()

        # Compute the four corners
        corner_ll = x_l * y_l
        corner_lu = x_l * y_u
        corner_ul = x_u * y_l
        corner_uu = x_u * y_u

        # Find min and max over corners
        z_l = torch.min(torch.min(corner_ll, corner_lu), torch.min(corner_ul, corner_uu))
        z_u = torch.max(torch.max(corner_ll, corner_lu), torch.max(corner_ul, corner_uu))

        # For a simple relaxation, we use interval arithmetic bounds
        # and express them as linear functions with coefficients
        # z ~ a1*x + a2*y + b

        # Strategy: use the middle values for coefficients
        x_mid = (x_l + x_u) / 2
        y_mid = (y_l + y_u) / 2

        # Lower bound: z >= y_mid * x + x_mid * y - x_mid * y_mid + adjustment
        # This is the bilinear relaxation at the midpoint
        # We adjust the bias to ensure soundness
        alpha1_lower = y_mid
        alpha2_lower = x_mid
        beta_lower = z_l - (alpha1_lower * x_mid + alpha2_lower * y_mid)

        # Upper bound: similar approach
        alpha1_upper = y_mid
        alpha2_upper = x_mid
        beta_upper = z_u - (alpha1_upper * x_mid + alpha2_upper * y_mid)

        # Create multi-input relaxation
        return LinearRelaxation(
            coeffs_lower=[alpha1_lower, alpha2_lower],
            coeffs_upper=[alpha1_upper, alpha2_upper],
            bias_lower=beta_lower,
            bias_upper=beta_upper,
            input_shapes=[x_l.shape, y_l.shape],
            output_shape=z_l.shape,
        )
