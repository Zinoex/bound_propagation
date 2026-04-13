from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPSelectStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for SELECT operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"SELECT requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPSelectStrategy requires input to be LinearBounds")

        bounds: LinearBounds = input_bounds[0]
        dim = node.attributes.get("dim", 0)
        index = node.attributes.get("index", 0)

        # Concretize and apply select
        lower, upper = bounds.concretize()

        lower = torch.select(lower, dim, index)
        upper = torch.select(upper, dim, index)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
