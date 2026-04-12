from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPCat(ForwardIBPStrategy):
    """
    IBP strategy for CAT operation:
    cat(I_1, I_2, ...; dim) = [cat(a_1, a_2, ...; dim), cat(b_1, b_2, ...; dim)],
    where I_j = [a_j, b_j].
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        dim = node.attributes.get("dim", 0)

        # All must be IntervalBounds
        for i, b in enumerate(input_bounds):
            if not isinstance(b, IntervalBounds):
                raise TypeError(f"IBPCat requires all inputs to be IntervalBounds, but input {i} is {type(b)}")

        bounds = cast(list[IntervalBounds], input_bounds)

        shape = bounds[0].lower.shape
        for i, b in enumerate(bounds):
            # All must have the same shape except in the concatenation dimension
            if b.lower.shape[0:dim] != shape[0:dim] or b.lower.shape[dim + 1 :] != shape[dim + 1 :]:
                raise ValueError(
                    "All inputs to cat must have the same shape except in the concatenation dimension, "
                    f"but input {i} has shape {b.lower.shape} and expected shape {shape}"
                )

        lower = torch.cat([b.lower for b in bounds], dim=dim)
        upper = torch.cat([b.upper for b in bounds], dim=dim)

        return IntervalBounds(lower, upper)
