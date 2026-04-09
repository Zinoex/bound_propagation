"""
ReLU relaxation strategy.

Computes linear relaxations for the ReLU activation function.
"""

import torch

from bound_propagation.bounds.interval_bounds import IntervalBounds
from bound_propagation.ir.node import Node
from bound_propagation.ir.operations import OperationType
from bound_propagation.relaxations.base import (
    RelaxationStrategy,
    register_relaxation_strategy,
)
from bound_propagation.relaxations.linear_relaxation import LinearRelaxation


def compute_relu_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
    adaptive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for ReLU linear relaxation.
    
    For ReLU, the relaxation divides the input space into three regimes:
    1. Negative regime (upper <= 0): output is always 0
    2. Positive regime (lower >= 0): output is identity (y = x)
    3. Crossing regime (lower < 0 < upper): linear approximation
    
    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        adaptive: Whether to use adaptive ReLU relaxation (chooses slope
                 based on which bound is tighter)
    
    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
        where y_lower >= alpha_lower * x + beta_lower
              y_upper <= alpha_upper * x + beta_upper
    """
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Determine regimes
    zero_width = torch.isclose(lower, upper)
    negative = (~zero_width) & (upper <= 0)
    positive = (~zero_width) & (lower >= 0)
    crossing = (~zero_width) & (lower < 0) & (upper > 0)

    # Zero-width: use the value itself
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = torch.relu(lower[zero_width])
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = torch.relu(upper[zero_width])

    # Negative regime: output is always 0
    alpha_lower[negative] = 0
    beta_lower[negative] = 0
    alpha_upper[negative] = 0
    beta_upper[negative] = 0

    # Positive regime: output is identity
    alpha_lower[positive] = 1
    beta_lower[positive] = 0
    alpha_upper[positive] = 1
    beta_upper[positive] = 0

    # Crossing regime: use linear relaxation
    if crossing.any():
        l_cross = lower[crossing]
        u_cross = upper[crossing]

        z = u_cross / (u_cross - l_cross)

        if adaptive:
            # Adaptive: choose slope based on which bound is tighter
            a = (u_cross >= torch.abs(l_cross)).to(lower.dtype)
        else:
            a = z

        alpha_lower[crossing] = a
        beta_lower[crossing] = 0
        alpha_upper[crossing] = z
        beta_upper[crossing] = -l_cross * z

    return alpha_lower, beta_lower, alpha_upper, beta_upper


@register_relaxation_strategy
class ReluRelaxationStrategy(RelaxationStrategy):
    """
    Relaxation strategy for ReLU activation function.
    
    ReLU is a piecewise linear function: relu(x) = max(0, x)
    The relaxation uses different linear approximations depending on
    whether the input interval is negative, positive, or crossing zero.
    """
    
    def __init__(self, adaptive: bool = False):
        """
        Initialize ReLU relaxation strategy.
        
        Args:
            adaptive: If True, use adaptive slope selection in crossing regime.
        """
        self.adaptive = adaptive
    
    @property
    def supported_op_type(self) -> OperationType:
        return OperationType.RELU
    
    def relax(
        self,
        node: Node,
        interval_inputs: list[IntervalBounds],
    ) -> LinearRelaxation:
        """
        Compute linear relaxation for ReLU.
        
        Args:
            node: The ReLU operation node.
            interval_inputs: List containing single IntervalBounds for the input.
        
        Returns:
            LinearRelaxation with diagonal coefficients (element-wise slopes).
        
        Raises:
            ValueError: If number of inputs is not 1.
        """
        if len(interval_inputs) != 1:
            raise ValueError(
                f"ReLU expects 1 input, got {len(interval_inputs)}"
            )
        
        input_bounds = interval_inputs[0]
        lower, upper = input_bounds.concretize()
        
        # Compute alpha/beta parameters
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_relu_alpha_beta(
            lower, upper, adaptive=self.adaptive
        )
        
        # Create diagonal relaxation
        return LinearRelaxation.create_diagonal(
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            beta_lower=beta_lower,
            beta_upper=beta_upper,
        )
