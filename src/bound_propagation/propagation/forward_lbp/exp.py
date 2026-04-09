from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLinearBoundingStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPExpStrategy(ForwardLinearBoundingStrategy):
    """
    Forward LBP strategy for EXP operation.

    Exponential is monotonic, so we apply it to concretized bounds.
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"EXP requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize and apply exp (monotonic function)
        lower, upper = bounds.concretize()
        lower_out = torch.exp(lower)
        upper_out = torch.exp(upper)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower_out,
            linear_upper=None,
            bias_upper=upper_out,
        )
