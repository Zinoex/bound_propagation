"""
Utility functions for backward LBP strategies.

Helper functions for working with linear bounds in backward LBP-style propagation.
"""

from __future__ import annotations

import torch

from ...bounds import LinearBounds


def verify_linear_bounds(input_bounds: list) -> None:
    """
    Verify that input bounds are LinearBounds.

    Args:
        input_bounds: List of bounds to verify

    Raises:
        TypeError: If any bounds are not LinearBounds
    """
    for i, bounds in enumerate(input_bounds):
        if not isinstance(bounds, LinearBounds):
            raise TypeError(
                f"Backward LBP strategies require LinearBounds, "
                f"but input {i} has type {type(bounds).__name__}"
            )


def create_element_wise_relaxation_bounds(
    region,
    alpha_lower: torch.Tensor,
    beta_lower: torch.Tensor,
    alpha_upper: torch.Tensor,
    beta_upper: torch.Tensor,
) -> LinearBounds:
    """
    Create LinearBounds representing an element-wise linear relaxation.

    For element-wise nonlinear functions, the relaxation is:
    y_i = alpha_i * x_i + beta_i

    This creates bounds with diagonal weight matrices where the diagonal
    contains the alpha values.

    Args:
        region: Input region
        alpha_lower: Slope for lower bound relaxation (shape: output_shape)
        beta_lower: Bias for lower bound relaxation (shape: output_shape)
        alpha_upper: Slope for upper bound relaxation (shape: output_shape)
        beta_upper: Bias for upper bound relaxation (shape: output_shape)

    Returns:
        LinearBounds with diagonal structure representing the relaxation
    """
    # Flatten to create diagonal matrices
    output_size = alpha_lower.numel()

    # Create diagonal matrices
    linear_lower = torch.diag(alpha_lower.flatten())
    linear_upper = torch.diag(alpha_upper.flatten())

    return LinearBounds(
        region=region,
        linear_lower=linear_lower,
        bias_lower=beta_lower,
        linear_upper=linear_upper,
        bias_upper=beta_upper,
    )


def apply_linear_relaxation_backward(
    bounds: LinearBounds,
    alpha_lower: torch.Tensor,
    beta_lower: torch.Tensor,
    alpha_upper: torch.Tensor,
    beta_upper: torch.Tensor,
) -> LinearBounds:
    """
    Apply element-wise linear relaxation to bounds using backward composition.

    For unary nonlinear functions in backward mode, we compute the inverse relaxation.
    If y = f(x) and we have bounds on y, we compute bounds on x.

    Args:
        bounds: Output LinearBounds (bounds on y)
        alpha_lower: Slope for lower bound relaxation (shape: input_shape)
        beta_lower: Bias for lower bound relaxation (shape: input_shape)
        alpha_upper: Slope for upper bound relaxation (shape: input_shape)
        beta_upper: Bias for upper bound relaxation (shape: input_shape)

    Returns:
        LinearBounds with relaxation applied via backward composition
    """
    # Create the relaxation as LinearBounds with diagonal structure
    relaxation = create_element_wise_relaxation_bounds(
        bounds.region,
        alpha_lower,
        beta_lower,
        alpha_upper,
        beta_upper,
    )

    # Use backward composition to apply the relaxation
    # bounds.backward_compose(relaxation) computes bounds o relaxation
    return bounds.backward_compose(relaxation)


def compute_relu_alpha_beta(
    lower: torch.Tensor, upper: torch.Tensor, adaptive: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for ReLU linear relaxation.

    ReLU has three regimes:
    - Negative (upper <= 0): ReLU(x) = 0
    - Positive (lower >= 0): ReLU(x) = x  
    - Crossing: 0 in [lower, upper]

    Args:
        lower: Lower bounds on input
        upper: Upper bounds on input
        adaptive: Whether to use adaptive relaxation (currently ignored)

    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
    """
    # Detect regimes
    zero_width = torch.abs(upper - lower) < 1e-8
    negative = upper <= 0
    positive = lower >= 0
    crossing = ~(negative | positive | zero_width)

    # Initialize alpha/beta tensors
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(upper)
    beta_upper = torch.zeros_like(upper)

    # Negative regime: y = 0 (alpha=0, beta=0)
    # (already initialized to zero)

    # Positive regime: y = x (alpha=1, beta=0)
    alpha_lower = torch.where(positive, torch.ones_like(alpha_lower), alpha_lower)
    alpha_upper = torch.where(positive, torch.ones_like(alpha_upper), alpha_upper)

    # Crossing regime:
    # Lower bound: y >= 0 (alpha=0, beta=0)
    # Upper bound: y <= slope * (x - l) where slope = u / (u - l)
    if crossing.any():
        slope = torch.zeros_like(upper)
        denominator = upper - lower
        valid_denom = torch.abs(denominator) > 1e-8
        slope_value = upper / torch.where(
            valid_denom, denominator, torch.ones_like(denominator)
        )
        slope = torch.where(crossing & valid_denom, slope_value, torch.zeros_like(slope))

        alpha_upper = torch.where(crossing, slope, alpha_upper)
        beta_upper = torch.where(crossing, -slope * lower, beta_upper)

    return alpha_lower, beta_lower, alpha_upper, beta_upper


def compute_sigmoid_alpha_beta(
    lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for sigmoid linear relaxation.

    Uses adaptive tangent and secant lines based on input bounds.

    Args:
        lower: Lower bounds on input
        upper: Upper bounds on input

    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
    """
    # Compute sigmoid values
    sigmoid_lower = torch.sigmoid(lower)
    sigmoid_upper = torch.sigmoid(upper)

    # Detect regimes
    zero_width = torch.abs(upper - lower) < 1e-8
    negative = upper <= 0
    positive = lower >= 0
    crossing = ~(negative | positive | zero_width)

    # Initialize to zero
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(upper)
    beta_upper = torch.zeros_like(upper)

    # For negative regime: use tangent line at lower bound for lower, secant for upper
    if negative.any():
        # Derivative at lower: sigmoid(l) * (1 - sigmoid(l))
        d_lower = sigmoid_lower * (1 - sigmoid_lower)
        alpha_lower = torch.where(negative, d_lower, alpha_lower)
        beta_lower = torch.where(negative, sigmoid_lower - d_lower * lower, beta_lower)

        # Secant line
        slope = (sigmoid_upper - sigmoid_lower) / torch.clamp(upper - lower, min=1e-8)
        alpha_upper = torch.where(negative, slope, alpha_upper)
        beta_upper = torch.where(negative, sigmoid_lower - slope * lower, beta_upper)

    # For positive regime: use secant line for lower, tangent at upper for upper
    if positive.any():
        # Secant line
        slope = (sigmoid_upper - sigmoid_lower) / torch.clamp(upper - lower, min=1e-8)
        alpha_lower = torch.where(positive, slope, alpha_lower)
        beta_lower = torch.where(positive, sigmoid_lower - slope * lower, beta_lower)

        # Derivative at upper: sigmoid(u) * (1 - sigmoid(u))
        d_upper = sigmoid_upper * (1 - sigmoid_upper)
        alpha_upper = torch.where(positive, d_upper, alpha_upper)
        beta_upper = torch.where(positive, sigmoid_upper - d_upper * upper, beta_upper)

    # For crossing regime: use tangent lines at bounds
    if crossing.any():
        # Midpoint for tangent
        mid = (lower + upper) / 2
        sigmoid_mid = torch.sigmoid(mid)
        d_mid = sigmoid_mid * (1 - sigmoid_mid)

        alpha_lower = torch.where(crossing, d_mid, alpha_lower)
        beta_lower = torch.where(crossing, sigmoid_mid - d_mid * mid, beta_lower)

        alpha_upper = torch.where(crossing, d_mid, alpha_upper)
        beta_upper = torch.where(crossing, sigmoid_mid - d_mid * mid, beta_upper)

    return alpha_lower, beta_lower, alpha_upper, beta_upper


def compute_tanh_alpha_beta(
    lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for tanh linear relaxation.

    Similar to sigmoid but for tanh function.

    Args:
        lower: Lower bounds on input
        upper: Upper bounds on input

    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
    """
    # Compute tanh values
    tanh_lower = torch.tanh(lower)
    tanh_upper = torch.tanh(upper)

    # Detect regimes
    zero_width = torch.abs(upper - lower) < 1e-8
    negative = upper <= 0
    positive = lower >= 0
    crossing = ~(negative | positive | zero_width)

    # Initialize to zero
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(upper)
    beta_upper = torch.zeros_like(upper)

    # For negative regime: use tangent line at lower bound for lower, secant for upper
    if negative.any():
        # Derivative at lower: 1 - tanh(l)^2
        d_lower = 1 - tanh_lower**2
        alpha_lower = torch.where(negative, d_lower, alpha_lower)
        beta_lower = torch.where(negative, tanh_lower - d_lower * lower, beta_lower)

        # Secant line
        slope = (tanh_upper - tanh_lower) / torch.clamp(upper - lower, min=1e-8)
        alpha_upper = torch.where(negative, slope, alpha_upper)
        beta_upper = torch.where(negative, tanh_lower - slope * lower, beta_upper)

    # For positive regime: use secant line for lower, tangent at upper for upper
    if positive.any():
        # Secant line
        slope = (tanh_upper - tanh_lower) / torch.clamp(upper - lower, min=1e-8)
        alpha_lower = torch.where(positive, slope, alpha_lower)
        beta_lower = torch.where(positive, tanh_lower - slope * lower, beta_lower)

        # Derivative at upper: 1 - tanh(u)^2
        d_upper = 1 - tanh_upper**2
        alpha_upper = torch.where(positive, d_upper, alpha_upper)
        beta_upper = torch.where(positive, tanh_upper - d_upper * upper, beta_upper)

    # For crossing regime: use tangent lines at midpoint
    if crossing.any():
        mid = (lower + upper) / 2
        tanh_mid = torch.tanh(mid)
        d_mid = 1 - tanh_mid**2

        alpha_lower = torch.where(crossing, d_mid, alpha_lower)
        beta_lower = torch.where(crossing, tanh_mid - d_mid * mid, beta_lower)

        alpha_upper = torch.where(crossing, d_mid, alpha_upper)
        beta_upper = torch.where(crossing, tanh_mid - d_mid * mid, beta_upper)

    return alpha_lower, beta_lower, alpha_upper, beta_upper
