from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPLogStrategy(ForwardLBPStrategy):
    """
    Forward LBP strategy for LOG operation.

    Logarithm is monotonic for positive inputs.
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"log requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPLogStrategy requires input to be LinearBounds")

        bounds = input_bounds[0]

        # Concretize and apply log (monotonic function)
        lower, upper = bounds.concretize()

        lower_out = lower.log()
        upper_out = upper.log()

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower_out,
            linear_upper=None,
            bias_upper=upper_out,
        )
