from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPCos(ForwardIBPStrategy):
    """IBP strategy for COS operation: cos([a, b]) = [cos(a), cos(b)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"cos requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPCos requires input to be IntervalBounds")

        # Analyze regimes based on the period of cosine
        # Peak occurs at 2k*π for integer k (where cos = 1)
        # Trough occurs at (2k+1)*π for integer k (where cos = -1)
        #
        # An interval [a, b] contains a peak if: floor(b / 2π) >= ceil(a / 2π)
        # An interval [a, b] contains a trough if: floor((b - π) / 2π) >= ceil((a - π) / 2π)

        two_pi = 2 * torch.pi
        pi = torch.pi

        # Check for peak: does [a, b] contain any 2k*π?
        # This is true when floor(b / 2π) >= ceil(a / 2π)
        includes_peak = torch.floor(x_bounds.upper / two_pi) >= torch.ceil(x_bounds.lower / two_pi)

        # Check for trough: does [a, b] contain any (2k+1)*π?
        # Shift by π and check: floor((b - π) / 2π) >= ceil((a - π) / 2π)
        includes_trough = torch.floor((x_bounds.upper - pi) / two_pi) >= torch.ceil(
            (x_bounds.lower - pi) / two_pi
        )

        cos_lower = torch.cos(x_bounds.lower)
        cos_upper = torch.cos(x_bounds.upper)

        lower = torch.where(
            includes_trough,
            torch.tensor(-1.0, device=x_bounds.lower.device, dtype=x_bounds.lower.dtype),
            torch.min(cos_lower, cos_upper),
        )
        upper = torch.where(
            includes_peak,
            torch.tensor(1.0, device=x_bounds.upper.device, dtype=x_bounds.upper.dtype),
            torch.max(cos_lower, cos_upper),
        )

        return IntervalBounds(lower, upper)
