from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPStack(ForwardIBPStrategy):
    """
    IBP strategy for STACK operation:
    stack(I_1, I_2, ...; dim) = [stack(a_1, a_2, ...; dim), stack(b_1, b_2, ...; dim)],
    where I_j = [a_j, b_j].
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:

        # All must be IntervalBounds
        for i, b in enumerate(input_bounds):
            if not isinstance(b, IntervalBounds):
                raise TypeError(f"IBPStack requires all inputs to be IntervalBounds, but input {i} is {type(b)}")

        bounds = cast(list[IntervalBounds], input_bounds)

        shape = bounds[0].lower.shape
        for i, b in enumerate(bounds):
            if b.lower.shape != shape:
                raise ValueError(
                    "All inputs to stack must have the same shape , "
                    f"but input {i} has shape {b.lower.shape} and expected shape {shape}"
                )

        dim = node.attributes.get("dim", 0)

        lower = torch.stack([b.lower for b in bounds], dim=dim)
        upper = torch.stack([b.upper for b in bounds], dim=dim)

        return IntervalBounds(lower, upper)
