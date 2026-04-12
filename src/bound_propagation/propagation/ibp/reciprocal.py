from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPReciprocal(ForwardIBPStrategy):
    """IBP strategy for reciprocal."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"reciprocal requires 1 input, got {len(input_bounds)}")

        x = input_bounds[0]

        if not isinstance(x, IntervalBounds):
            raise TypeError(
                f"IBPReciprocal requires the input to be IntervalBounds, got {type(x)}"
            )

        # For c / [a, b],
        # - if [a, b] contains 0, return [-inf, inf]
        # - else, return [c/b, c/a]
        unbounded_mask = (x.lower <= 0) & (x.upper >= 0)

        lower = 1 / x.upper
        lower = torch.where(unbounded_mask, float("-inf"), lower)

        upper = 1 / x.lower
        upper = torch.where(unbounded_mask, float("inf"), upper)

        return IntervalBounds(lower, upper)
