from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..linear_relaxations.sigmoid import compute_sigmoid_alpha_beta
from .base import ForwardLinearBoundingStrategy
from .utils import apply_linear_relaxation

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPSigmoidStrategy(ForwardLinearBoundingStrategy):
    """
    Forward LBP strategy for SIGMOID operation.

    Uses adaptive linear relaxations based on the input bounds regime.
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"SIGMOID requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize to get interval bounds for determining relaxation
        lower, upper = bounds.concretize()

        # Compute alpha/beta parameters for sigmoid relaxation
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(
            lower, upper
        )

        # Apply the linear relaxation to the bounds
        return apply_linear_relaxation(
            bounds, alpha_lower, beta_lower, alpha_upper, beta_upper
        )
