from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPExp(ForwardIBPStrategy):
    """IBP strategy for EXP operation: exp([a, b]) = [exp(a), exp(b)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"exp requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPExp requires input to be IntervalBounds")

        # Exp is monotonic
        lower = torch.exp(x_bounds.lower)
        upper = torch.exp(x_bounds.upper)

        return IntervalBounds(lower, upper)
