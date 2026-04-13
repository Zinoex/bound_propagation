from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPConcatStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for CONCAT operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) < 1:
            raise ValueError(f"CONCAT requires at least 1 input, got {len(input_bounds)}")

        # Check all inputs are LinearBounds
        for i, inp in enumerate(input_bounds):
            if not isinstance(inp, LinearBounds):
                raise TypeError(
                    f"ForwardLBPConcatStrategy requires all inputs to be LinearBounds, but input {i} is {type(inp)}"
                )

        bounds_list: list[LinearBounds] = input_bounds  # type: ignore
        dim = node.attributes.get("dim", 0)

        # Concretize all inputs and concatenate
        lowers = [b.concretize()[0] for b in bounds_list]
        uppers = [b.concretize()[1] for b in bounds_list]

        lower = torch.cat(lowers, dim=dim)
        upper = torch.cat(uppers, dim=dim)

        return LinearBounds(
            region=bounds_list[0].region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
