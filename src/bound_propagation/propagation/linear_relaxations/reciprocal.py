import torch


def compute_reciprocal_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for reciprocal (1/x) linear relaxation.

    reciprocal is convex for x > 0 and convex for x < 0, so:
    - When interval is all positive or all negative: use secant for lower, tangent for upper
    - When interval crosses zero: handle specially (may need to use safe bounds)

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
    crosses_zero = (lower < 0) & (upper > 0)
    all_positive = lower > 0
    all_negative = upper < 0

    # Compute reciprocal values (avoid division by zero)
    eps = 1e-8
    lower_safe = torch.where(torch.abs(lower) < eps, eps * torch.sign(lower + eps), lower)
    upper_safe = torch.where(torch.abs(upper) < eps, eps * torch.sign(upper + eps), upper)

    lower_act = 1.0 / lower_safe
    upper_act = 1.0 / upper_safe

    def reciprocal_derivative(x):
        # d/dx (1/x) = -1/x^2
        x_safe = torch.where(torch.abs(x) < eps, eps * torch.sign(x + eps), x)
        return -1.0 / (x_safe * x_safe)

    reciprocal_derivative(lower_safe)
    reciprocal_derivative(upper_safe)

    # Midpoint for tangent line
    d = (lower + upper) * 0.5
    d_safe = torch.where(torch.abs(d) < eps, eps * torch.sign(d + eps), d)
    d_act = 1.0 / d_safe
    d_prime = reciprocal_derivative(d_safe)

    # Slope of secant line
    slope = torch.where(
        zero_width, torch.zeros_like(lower), (upper_act - lower_act) / torch.clamp(upper - lower, min=eps)
    )

    # Zero-width case: use the value itself
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = lower_act[zero_width]
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = upper_act[zero_width]

    # Crosses zero: use safe infinite bounds
    alpha_lower[crosses_zero] = 0
    beta_lower[crosses_zero] = float("-inf")
    alpha_upper[crosses_zero] = 0
    beta_upper[crosses_zero] = float("inf")

    # All positive or all negative: 1/x is convex
    valid = all_positive | all_negative
    if valid.any():
        # Upper bound: tangent at midpoint
        alpha_upper[valid] = d_prime[valid]
        beta_upper[valid] = d_act[valid] - d_prime[valid] * d_safe[valid]

        # Lower bound: secant line
        alpha_lower[valid] = slope[valid]
        beta_lower[valid] = upper_act[valid] - slope[valid] * upper_safe[valid]

    return alpha_lower, beta_lower, alpha_upper, beta_upper
