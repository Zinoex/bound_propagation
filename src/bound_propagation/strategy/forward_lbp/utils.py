"""
Utility functions for forward LBP strategies.

Helper functions for working with linear bounds in LBP-style propagation.
"""

from __future__ import annotations

import torch

from ...bounds import IntervalBounds, LinearBounds


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
                f"LBP strategies require LinearBounds, "
                f"but input {i} has type {type(bounds).__name__}"
            )


def linearize_bounds(bounds: IntervalBounds | LinearBounds) -> LinearBounds:
    """
    Convert bounds to LinearBounds if needed.

    If bounds are already LinearBounds, returns as-is.
    If bounds are IntervalBounds, creates LinearBounds with zero coefficients (constants).

    Args:
        bounds: Bounds to linearize

    Returns:
        LinearBounds representation
    """
    if isinstance(bounds, LinearBounds):
        return bounds

    if isinstance(bounds, IntervalBounds):
        # Convert IntervalBounds to LinearBounds with zero coefficients
        # This represents constant bounds: lower = 0 @ x + lower, upper = 0 @ x + upper
        return LinearBounds(
            region=bounds.region,
            linear_lower=None,  # No linear dependency
            bias_lower=bounds.lower,
            linear_upper=None,  # No linear dependency
            bias_upper=bounds.upper,
        )

    raise TypeError(f"Cannot linearize bounds of type {type(bounds).__name__}")


def create_identity_bounds(region, shape: tuple[int, ...]) -> LinearBounds:
    """
    Create identity linear bounds (output = input).

    Used for input nodes in forward-mode LBP.

    Args:
        region: Input region
        shape: Shape of the output

    Returns:
        LinearBounds with identity mapping
    """
    # Flatten input dimension
    input_size = region.lower.numel()
    output_size = torch.Size(shape).numel()

    # Create identity matrix if input and output sizes match
    if input_size == output_size:
        identity = torch.eye(output_size, dtype=region.dtype, device=region.device)
        bias = torch.zeros(output_size, dtype=region.dtype, device=region.device)

        return LinearBounds(
            region=region,
            linear_lower=identity,
            bias_lower=bias,
            linear_upper=identity,
            bias_upper=bias,
        )
    else:
        # If sizes don't match, we can't create identity mapping
        # Fall back to IntervalBounds converted to LinearBounds

        lower, upper = region.lower, region.upper
        return LinearBounds(
            region=region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
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


def apply_linear_relaxation(
    bounds: LinearBounds,
    alpha_lower: torch.Tensor,
    beta_lower: torch.Tensor,
    alpha_upper: torch.Tensor,
    beta_upper: torch.Tensor,
) -> LinearBounds:
    """
    Apply element-wise linear relaxation to bounds using forward composition.

    For unary nonlinear functions according to Table 6 in the LBP paper:
    - W_out = diag(alpha) @ W_in
    - b_out = diag(alpha) @ b_in + beta

    Args:
        bounds: Input LinearBounds
        alpha_lower: Slope for lower bound relaxation (shape: output_shape)
        beta_lower: Bias for lower bound relaxation (shape: output_shape)
        alpha_upper: Slope for upper bound relaxation (shape: output_shape)
        beta_upper: Bias for upper bound relaxation (shape: output_shape)

    Returns:
        LinearBounds with relaxation applied
    """
    # Create the relaxation as LinearBounds with diagonal structure
    relaxation = create_element_wise_relaxation_bounds(
        bounds.region,
        alpha_lower,
        beta_lower,
        alpha_upper,
        beta_upper,
    )

    # Use forward composition to apply the relaxation
    return bounds.forward_compose(relaxation)


def compute_relu_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
    adaptive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for ReLU linear relaxation.

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        adaptive: Whether to use adaptive ReLU relaxation

    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
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


def compute_sigmoid_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for sigmoid linear relaxation.

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

    # Compute sigmoid and derivative
    lower_act = torch.sigmoid(lower)
    upper_act = torch.sigmoid(upper)

    def sigmoid_derivative(x):
        s = torch.sigmoid(x)
        return s * (1 - s)

    lower_prime = sigmoid_derivative(lower)
    upper_prime = sigmoid_derivative(upper)

    # Midpoint for tangent line
    d = (lower + upper) * 0.5
    d_act = torch.sigmoid(d)
    d_prime = sigmoid_derivative(d)

    # Slope of secant line
    slope = torch.where(
        zero_width,
        torch.zeros_like(lower),
        (upper_act - lower_act) / (upper - lower)
    )

    # Zero-width case: use the value itself
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = lower_act[zero_width]
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width cases
    non_zero = ~zero_width

    # Determine negative/positive regimes
    negative = non_zero & (upper <= 0)
    positive = non_zero & (lower >= 0)
    crossing = non_zero & (lower < 0) & (upper > 0)

    # Negative regime
    if negative.any():
        # Upper: secant line between lower and upper
        alpha_upper[negative] = slope[negative]
        beta_upper[negative] = upper_act[negative] - slope[negative] * upper[negative]

        # Lower: tangent line at midpoint
        alpha_lower[negative] = d_prime[negative]
        beta_lower[negative] = d_act[negative] - d_prime[negative] * d[negative]

    # Positive regime
    if positive.any():
        # Upper: tangent at midpoint
        alpha_upper[positive] = d_prime[positive]
        beta_upper[positive] = d_act[positive] - d_prime[positive] * d[positive]

        # Lower: secant line
        alpha_lower[positive] = slope[positive]
        beta_lower[positive] = lower_act[positive] - slope[positive] * lower[positive]

    # Crossing regime (contains both negative and positive)
    if crossing.any():
        # Upper: minimum of two tangent lines at lower and upper
        # Use tangent at lower since sigmoid is concave in crossing region typically
        alpha_upper[crossing] = lower_prime[crossing]
        beta_upper[crossing] = lower_act[crossing] - lower_prime[crossing] * lower[crossing]

        # Lower: secant line
        alpha_lower[crossing] = slope[crossing]
        beta_lower[crossing] = lower_act[crossing] - slope[crossing] * lower[crossing]

    return alpha_lower, beta_lower, alpha_upper, beta_upper


def compute_tanh_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for tanh linear relaxation.

    Tanh has similar structure to sigmoid but is symmetric around origin.

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

        alpha_lower[crossing] = slope[crossing]
        beta_lower[crossing] = lower_act[crossing] - slope[crossing] * lower[crossing]

    return alpha_lower, beta_lower, alpha_upper, beta_upper
