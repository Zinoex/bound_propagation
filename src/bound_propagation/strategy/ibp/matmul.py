from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from ..strategy import ForwardBoundingStrategy
from .utils import verify_interval_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class IBPMatmulStrategy(ForwardBoundingStrategy):
    """IBP strategy for MATMUL operation: A @ B."""

    @property
    def method_name(self) -> str:
        return "ibp"

    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> AbstractBounds:
        verify_interval_bounds(input_bounds)

        if len(input_bounds) != 2:
            raise ValueError(f"MATMUL requires 2 inputs, got {len(input_bounds)}")

        a_bounds: IntervalBounds = input_bounds[0]  # ty:ignore[invalid-assignment]
        b_bounds: IntervalBounds = input_bounds[1]  # ty:ignore[invalid-assignment]

        # For matrix multiplication A @ B, we need to handle each element separately
        # This is complex for general case, so for now we'll handle simple cases

        # Check if this is actually a weight multiplication (one input is constant)
        if torch.all(a_bounds.lower == a_bounds.upper):
            # A is constant
            weight = a_bounds.lower
            return self._matmul_with_constant(weight, b_bounds, left=True)
        elif torch.all(b_bounds.lower == b_bounds.upper):
            # B is constant
            weight = b_bounds.lower
            return self._matmul_with_constant(weight, a_bounds, left=False)
        else:
            # Both are intervals - use general but loose bound
            raise NotImplementedError("MATMUL with two interval inputs not yet implemented. Use MATMUL only with constant weights.")

    def _matmul_with_constant(
        self,
        weight: torch.Tensor,
        bounds: IntervalBounds,
        left: bool,
    ) -> IntervalBounds:
        """
        Compute bounds for matmul with constant weight.

        Args:
            weight: Constant weight matrix
            bounds: Interval bounds for variable input
            left: If True, weight @ bounds, else bounds @ weight
        """
        lower_out = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
        upper_out = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)

        # Similar to linear layer
        for i in range(weight.shape[0]):
            w_row = weight[i]
            pos_mask = w_row >= 0
            neg_mask = w_row < 0

            lower_out[i] = torch.sum(w_row[pos_mask] * bounds.lower[pos_mask]) + torch.sum(w_row[neg_mask] * bounds.upper[neg_mask])
            upper_out[i] = torch.sum(w_row[pos_mask] * bounds.upper[pos_mask]) + torch.sum(w_row[neg_mask] * bounds.lower[neg_mask])

        return IntervalBounds(bounds.region, lower_out, upper_out)
