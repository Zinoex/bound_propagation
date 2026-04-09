from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..linear_relaxations.relu import compute_relu_alpha_beta
from .base import ForwardLBPStrategy
from .utils import apply_linear_relaxation

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPReluStrategy(ForwardLBPStrategy):
    """
    Forward LBP strategy for RELU operation.

    For ReLU y = max(0, x), we use adaptive linear relaxations:
    - If x >= 0 (active): y = x (identity)
    - If x <= 0 (inactive): y = 0 (zero)
    - If x crosses zero: linear relaxation
      - Lower bound: y >= 0 (always valid)
      - Upper bound: y <= (u/(u-l)) * (x - l) where [l, u] are concrete bounds
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"RELU requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize to get interval bounds for determining ReLU behavior
        lower, upper = bounds.concretize()

        # Compute alpha/beta parameters for ReLU relaxation
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_relu_alpha_beta(
            lower, upper, adaptive=False
        )

        # Apply the linear relaxation to the bounds
        return apply_linear_relaxation(
            bounds, alpha_lower, beta_lower, alpha_upper, beta_upper
        )
