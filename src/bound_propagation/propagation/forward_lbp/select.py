from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPSelect(ForwardLBPStrategy):
    """Forward LBP strategy for SELECT operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"select requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPSelect requires input to be LinearBounds")

        bounds = input_bounds[0]
        dim = node.attributes.get("dim", 0)
        index = node.attributes.get("index", 0)

        # Concretize and apply select
        lower, upper = bounds.concretize()

        lower = lower.select(dim, index)
        upper = upper.select(dim, index)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
