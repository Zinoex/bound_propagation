from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLinearBoundingStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPLogStrategy(ForwardLinearBoundingStrategy):
    """
    Forward LBP strategy for LOG operation.

    Logarithm is monotonic for positive inputs.
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"LOG requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize and apply log (monotonic function)
        lower, upper = bounds.concretize()

        # Ensure positive inputs
        lower = torch.clamp(lower, min=1e-8)
        upper = torch.clamp(upper, min=1e-8)

        lower_out = torch.log(lower)
        upper_out = torch.log(upper)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower_out,
            linear_upper=None,
            bias_upper=upper_out,
        )
