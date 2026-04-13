from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from ..linear_relaxations.cos import compute_cos_alpha_beta
from .base import ForwardLBPStrategy
from .utils import apply_linear_relaxation

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPCosStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for COS operation using linear relaxation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"COS requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPCosStrategy requires input to be LinearBounds")

        bounds: LinearBounds = input_bounds[0]

        # Concretize to get interval bounds for determining relaxation
        lower, upper = bounds.concretize()

        # Compute alpha/beta parameters for cos relaxation
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_cos_alpha_beta(lower, upper)

        # Apply the linear relaxation to the bounds
        return apply_linear_relaxation(bounds, alpha_lower, beta_lower, alpha_upper, beta_upper)
