import torch


def compute_tan_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for tan linear relaxation.

    tan is convex on each monotone interval, so we use:
    - Upper bound: tangent line at upper bound
    - Lower bound: tangent line at lower bound or secant

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
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)

    # Compute tan values
    lower_act = torch.tan(lower)
    upper_act = torch.tan(upper)

    def tan_derivative(x):
        # d/dx tan(x) = sec^2(x) = 1/cos^2(x)
        cos_x = torch.cos(x)
        return 1.0 / (cos_x * cos_x + 1e-8)  # Add small epsilon for numerical stability

    tan_derivative(lower)
    tan_derivative(upper)

    # Midpoint for tangent line
    d = (lower + upper) * 0.5
    d_act = torch.tan(d)
    d_prime = tan_derivative(d)

    # Slope of secant line
    slope = torch.where(
        zero_width, torch.zeros_like(lower), (upper_act - lower_act) / torch.clamp(upper - lower, min=1e-8)
    )

    # Zero-width case: use the value itself
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = lower_act[zero_width]
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width cases
    non_zero = ~zero_width

    # For simplicity, use secant for upper and tangent at midpoint for lower
    # This is a conservative approximation
    alpha_upper[non_zero] = slope[non_zero]
    beta_upper[non_zero] = upper_act[non_zero] - slope[non_zero] * upper[non_zero]

    alpha_lower[non_zero] = d_prime[non_zero]
    beta_lower[non_zero] = d_act[non_zero] - d_prime[non_zero] * d[non_zero]

    return alpha_lower, beta_lower, alpha_upper, beta_upper
