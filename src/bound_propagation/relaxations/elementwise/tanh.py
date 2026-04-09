"""
Tanh relaxation strategy.

Computes linear relaxations for the tanh activation function.
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


def compute_tanh_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for tanh linear relaxation.
    
    Tanh is a smooth S-shaped function symmetric around the origin: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    The relaxation uses different strategies depending on the regime:
    - Negative regime: secant line for upper, tangent at midpoint for lower
    - Positive regime: tangent at midpoint for upper, secant line for lower
    - Crossing regime: tangent lines for bounds
    
    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
    
    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
    """
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Determine regimes
    zero_width = torch.isclose(lower, upper)

    # Compute tanh and derivative
    lower_act = torch.tanh(lower)
    upper_act = torch.tanh(upper)

    def tanh_derivative(x):
        t = torch.tanh(x)
        return 1 - t * t

    lower_prime = tanh_derivative(lower)
    upper_prime = tanh_derivative(upper)

    # Midpoint for tangent line
    d = (lower + upper) * 0.5
    d_act = torch.tanh(d)
    d_prime = tanh_derivative(d)

    # Slope of secant line
    slope = torch.where(
        zero_width,
        torch.zeros_like(lower),
        (upper_act - lower_act) / (upper - lower)
    )

    # Zero-width case
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = lower_act[zero_width]
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width
    non_zero = ~zero_width

    negative = non_zero & (upper <= 0)
    positive = non_zero & (lower >= 0)
    crossing = non_zero & (lower < 0) & (upper > 0)

    # Negative regime
    if negative.any():
        alpha_upper[negative] = slope[negative]
        beta_upper[negative] = upper_act[negative] - slope[negative] * upper[negative]

        alpha_lower[negative] = d_prime[negative]
        beta_lower[negative] = d_act[negative] - d_prime[negative] * d[negative]

    # Positive regime
    if positive.any():
        alpha_upper[positive] = d_prime[positive]
        beta_upper[positive] = d_act[positive] - d_prime[positive] * d[positive]

        alpha_lower[positive] = slope[positive]
        beta_lower[positive] = lower_act[positive] - slope[positive] * lower[positive]

    # Crossing regime
    if crossing.any():
        alpha_upper[crossing] = lower_prime[crossing]
        beta_upper[crossing] = lower_act[crossing] - lower_prime[crossing] * lower[crossing]

        alpha_lower[crossing] = upper_prime[crossing]
        beta_lower[crossing] = lower_act[crossing] - upper_prime[crossing] * lower[crossing]

    return alpha_lower, beta_lower, alpha_upper, beta_upper


@register_relaxation_strategy
class TanhRelaxationStrategy(RelaxationStrategy):
    """
    Relaxation strategy for tanh activation function.
    
    Tanh is a smooth, monotonic function symmetric around the origin
    that maps inputs to (-1, 1). The relaxation uses tangent and secant
    lines depending on the regime.
    """
    
    @property
    def supported_op_type(self) -> OperationType:
        return OperationType.TANH
    
    def relax(
        self,
        node: Node,
        interval_inputs: list[IntervalBounds],
    ) -> LinearRelaxation:
        """
        Compute linear relaxation for tanh.
        
        Args:
            node: The tanh operation node.
            interval_inputs: List containing single IntervalBounds for the input.
        
        Returns:
            LinearRelaxation with diagonal coefficients (element-wise slopes).
        
        Raises:
            ValueError: If number of inputs is not 1.
        """
        if len(interval_inputs) != 1:
            raise ValueError(
                f"Tanh expects 1 input, got {len(interval_inputs)}"
            )
        
        input_bounds = interval_inputs[0]
        lower, upper = input_bounds.concretize()
        
        # Compute alpha/beta parameters
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_tanh_alpha_beta(
            lower, upper
        )
        
        # Create diagonal relaxation
        return LinearRelaxation.create_diagonal(
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            beta_lower=beta_lower,
            beta_upper=beta_upper,
        )
