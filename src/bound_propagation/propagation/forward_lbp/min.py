from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPMinStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for MIN reduction operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"MIN requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPMinStrategy requires input to be LinearBounds")

        bounds: LinearBounds = input_bounds[0]

        dim = node.attributes.get("dim")
        keep_dim = node.attributes.get("keepdim", False)

        # Concretize and apply min
        lower, upper = bounds.concretize()

        if dim is not None:
            lower = torch.min(lower, dim=dim, keepdim=keep_dim).values
            upper = torch.min(upper, dim=dim, keepdim=keep_dim).values
        else:
            lower = torch.min(lower)
            upper = torch.min(upper)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
