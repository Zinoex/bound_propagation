from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

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

        # Concretize and apply exp (monotonic function)
        lower, upper = bounds.concretize()
        lower_out = lower.exp()
        upper_out = upper.exp()

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower_out,
            linear_upper=None,
            bias_upper=upper_out,
        )
