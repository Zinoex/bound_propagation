from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from ..strategy import BoundingStrategy
from .utils import verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class ForwardCrownLinearStrategy(BoundingStrategy):
    """
    Forward CROWN strategy for LINEAR operation.

    For linear layer y = x @ W^T + b:
    - If x has bounds: W_l^y @ x0 + b_l^y = W @ W_l^x @ x0 + W @ b_l^x + b
    - This is exact (no approximation needed for linear operations)
    """

    @property
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        return "forward"

    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> AbstractBounds:
        """
        Compute forward CROWN bounds for linear layer.

        Args:
            node: The LINEAR node
            input_bounds: List with one LinearBounds for the input
            config: Strategy configuration

        Returns:
            LinearBounds for the output
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"LINEAR requires exactly 1 input, got {len(input_bounds)}")

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

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
