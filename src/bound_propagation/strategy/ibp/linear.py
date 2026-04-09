from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import IntervalBoundingStrategy
from .utils import verify_interval_bounds

if TYPE_CHECKING:
    from ...ir import Node


class IBPLinearStrategy(IntervalBoundingStrategy):
    """IBP strategy for LINEAR operation: y = x @ W^T + b."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:
        verify_interval_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"LINEAR requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]

        # Get weight and bias from node attributes
        weight = node.attributes.get("weight")
        bias = node.attributes.get("bias")

        if weight is None:
            raise ValueError("LINEAR node missing weight attribute")

        # Compute bounds for W^T @ x
        # For each output: lower_i = sum_j (W_ij * x_j) where we choose x_j based on sign of W_ij
        lower_out = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
        upper_out = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)

        # For each row of weight
        for i in range(weight.shape[0]):
            w_row = weight[i]  # Shape: (in_features,)

            # Positive weights: multiply by input bounds directly
            # Negative weights: swap bounds
            pos_mask = w_row >= 0
            neg_mask = w_row < 0

            # Lower: pos*lower + neg*upper
            lower_out[i] = torch.sum(w_row[pos_mask] * x_bounds.lower[pos_mask]) + torch.sum(w_row[neg_mask] * x_bounds.upper[neg_mask])

            # Upper: pos*upper + neg*lower
            upper_out[i] = torch.sum(w_row[pos_mask] * x_bounds.upper[pos_mask]) + torch.sum(w_row[neg_mask] * x_bounds.lower[neg_mask])

        # Add bias if present
        if bias is not None:
            lower_out = lower_out + bias
            upper_out = upper_out + bias

        return IntervalBounds(x_bounds.region, lower_out, upper_out)
