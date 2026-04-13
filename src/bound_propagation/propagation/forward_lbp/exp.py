from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..linear_relaxations.exp import compute_exp_alpha_beta
from .base import ForwardLBPStrategy
from .utils import apply_linear_relaxation

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPExp(ForwardLBPStrategy):
    """
    Forward LBP strategy for EXP operation.

    Exponential is monotonic, so we apply it to concretized bounds.
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"exp requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPExpStrategy requires input to be LinearBounds")

        bounds = input_bounds[0]

        # Concretize to get interval bounds for determining relaxation
        lower, upper = bounds.concretize()

        # Compute alpha/beta parameters for cos relaxation
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_exp_alpha_beta(lower, upper)

        # Apply the linear relaxation to the bounds
        return apply_linear_relaxation(bounds, alpha_lower, beta_lower, alpha_upper, beta_upper)
