"""
Division relaxation strategy.

Computes linear relaxations for element-wise division.
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
class DivRelaxationStrategy(RelaxationStrategy):
    """
    Relaxation strategy for element-wise division.

    For z = x / y, we need to ensure y doesn't contain zero and compute
    linear bounds similar to multiplication.
    """

    @property
    def supported_op_type(self) -> OperationType:
        return OperationType.DIV

    def relax(
        self,
        node: Node,
        interval_inputs: list[IntervalBounds],
    ) -> LinearRelaxation:
        """
        Compute linear relaxation for element-wise division.

        For z = x / y where x ∈ [x_l, x_u] and y ∈ [y_l, y_u]:
        We require that 0 ∉ [y_l, y_u] (no division by zero).

        Then we compute bounds using interval arithmetic on the corners.

        Args:
            node: The division operation node.
            interval_inputs: List containing two IntervalBounds (for x and y).

        Returns:
            LinearRelaxation with coefficients for both inputs.

        Raises:
            ValueError: If number of inputs is not 2 or if divisor contains zero.
        """
        if len(interval_inputs) != 2:
            raise ValueError(
                f"Div expects 2 inputs, got {len(interval_inputs)}"
            )

        x_bounds = interval_inputs[0]
        y_bounds = interval_inputs[1]

        x_l, x_u = x_bounds.concretize()
        y_l, y_u = y_bounds.concretize()

        # Check for division by zero
        if torch.any((y_l <= 0) & (y_u >= 0)):
            raise ValueError(
                "Division relaxation requires divisor bounds that don't contain zero"
            )

        # Compute the four corners
        corner_ll = x_l / y_l
        corner_lu = x_l / y_u
        corner_ul = x_u / y_l
        corner_uu = x_u / y_u

        # Find min and max over corners (accounting for sign flips when y is negative)
        z_l = torch.min(torch.min(corner_ll, corner_lu), torch.min(corner_ul, corner_uu))
        z_u = torch.max(torch.max(corner_ll, corner_lu), torch.max(corner_ul, corner_uu))

        # Use midpoint linearization strategy
        x_mid = (x_l + x_u) / 2
        y_mid = (y_l + y_u) / 2

        # For division: z = x / y ≈ (1/y_mid) * x - (x_mid/y_mid^2) * y + x_mid/y_mid
        # This is the first-order Taylor approximation

        # Coefficients for x
        alpha1_mid = 1.0 / y_mid
        # Coefficients for y (negative because derivative wrt y is -x/y^2)
        alpha2_mid = -(x_mid / (y_mid * y_mid))

        # Adjust biases to ensure soundness
        beta_lower = z_l - (alpha1_mid * x_mid + alpha2_mid * y_mid)
        beta_upper = z_u - (alpha1_mid * x_mid + alpha2_mid * y_mid)

        # Create multi-input relaxation
        return LinearRelaxation(
            coeffs_lower=[alpha1_mid, alpha2_mid],
            coeffs_upper=[alpha1_mid, alpha2_mid],
            bias_lower=beta_lower,
            bias_upper=beta_upper,
            input_shapes=[x_l.shape, y_l.shape],
            output_shape=z_l.shape,
        )
