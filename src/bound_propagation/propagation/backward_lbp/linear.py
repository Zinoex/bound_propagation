from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import BackwardLBPBoundingStrategy

if TYPE_CHECKING:
    from ...ir import Node


class BackwardLBPLinearStrategy(BackwardLBPBoundingStrategy):
    """
    Backward LBP strategy for LINEAR operation.

    For linear layer y = x @ W^T + b, in backward mode the computation
    is the same as forward mode since it's an exact linear operation.
    """
    def propagate_backwards(
        self,
        node: Node,
        output_bounds: LinearBounds,
    ) -> list[LinearBounds]:
        bounds: LinearBounds = input_bounds[0]

        # Get weight and bias from node attributes
        weight = node.attributes.get("weight")
        bias = node.attributes.get("bias")

        if weight is None:
            raise ValueError(f"LINEAR node {node.id} missing 'weight' attribute")

        # Linear transformation: y = x @ W^T + b
        # Lower bound: W @ W_l^x @ x0 + W @ b_l^x + b
        # Upper bound: W @ W_u^x @ x0 + W @ b_u^x + b

        # Handle positive and negative weights separately for tight bounds
        weight_pos = torch.clamp(weight, min=0)
        weight_neg = torch.clamp(weight, max=0)

        # Lower bound computation
        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            # W_l^y = weight_pos @ W_l^x + weight_neg @ W_u^x
            linear_lower = weight_pos @ bounds.linear_lower + weight_neg @ bounds.linear_upper
        elif bounds.linear_lower is not None:
            linear_lower = weight_pos @ bounds.linear_lower + weight_neg @ bounds.linear_lower
        elif bounds.linear_upper is not None:
            linear_lower = weight_pos @ bounds.linear_upper + weight_neg @ bounds.linear_upper
        else:
            linear_lower = None

        # b_l^y = weight_pos @ b_l^x + weight_neg @ b_u^x + b
        bias_lower = weight_pos @ bounds.bias_lower + weight_neg @ bounds.bias_upper
        if bias is not None:
            bias_lower = bias_lower + bias

        # Upper bound computation
        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            # W_u^y = weight_pos @ W_u^x + weight_neg @ W_l^x
            linear_upper = weight_pos @ bounds.linear_upper + weight_neg @ bounds.linear_lower
        elif bounds.linear_upper is not None:
            linear_upper = weight_pos @ bounds.linear_upper + weight_neg @ bounds.linear_upper
        elif bounds.linear_lower is not None:
            linear_upper = weight_pos @ bounds.linear_lower + weight_neg @ bounds.linear_lower
        else:
            linear_upper = None

        # b_u^y = weight_pos @ b_u^x + weight_neg @ b_l^x + b
        bias_upper = weight_pos @ bounds.bias_upper + weight_neg @ bounds.bias_lower
        if bias is not None:
            bias_upper = bias_upper + bias

        return [LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )]
