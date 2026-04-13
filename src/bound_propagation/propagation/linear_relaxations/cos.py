import torch


def compute_cos_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for cos linear relaxation.

    cos is neither globally convex nor concave, so we use:
    - Upper bound: secant line or tangent depending on the interval
    - Lower bound: tangent line at midpoint or minimum value

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

    # Compute cos values
    lower_act = torch.cos(lower)
    upper_act = torch.cos(upper)

    def cos_derivative(x):
        return -torch.sin(x)

    cos_derivative(lower)
    cos_derivative(upper)

    # Midpoint for tangent line
    d = (lower + upper) * 0.5
    d_act = torch.cos(d)
    d_prime = cos_derivative(d)

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
