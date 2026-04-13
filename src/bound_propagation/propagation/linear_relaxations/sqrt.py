import torch


def compute_sqrt_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for sqrt linear relaxation.

    sqrt is concave, so:
    - Upper bound: secant line connecting (lower, sqrt(lower)) and (upper, sqrt(upper))
    - Lower bound: tangent line at a suitable point (midpoint)

    Args:
        lower: Lower bounds of pre-activation (must be >= 0)
        upper: Upper bounds of pre-activation (must be >= 0)
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
    """
    if torch.any(lower < 0):
        raise ValueError("sqrt requires non-negative lower bounds")

    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Determine regimes
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)

    # Compute sqrt values
    lower_act = torch.sqrt(lower)
    upper_act = torch.sqrt(upper)

    def sqrt_derivative(x):
        # d/dx sqrt(x) = 1/(2*sqrt(x))
        # Handle zero case
        return torch.where(x > 0, 1.0 / (2.0 * torch.sqrt(x)), torch.zeros_like(x))

    # Slope of secant line
    slope = torch.where(
        zero_width,
        torch.zeros_like(lower),
        (upper_act - lower_act) / torch.clamp(upper - lower, min=1e-8),
    )

    # Zero-width case: use the value itself
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = lower_act[zero_width]
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width cases
    non_zero = ~zero_width

    # sqrt is concave everywhere
    # For concave functions:
    # - Secant line lies BELOW the curve (use for lower bound)
    # - Tangent line lies ABOVE the curve (use for upper bound)

    # Lower bound: secant line
    alpha_lower[non_zero] = slope[non_zero]
    beta_lower[non_zero] = lower_act[non_zero] - slope[non_zero] * lower[non_zero]

    # Upper bound: tangent at upper point (steepest tangent for tightest upper bound)
    upper_prime = sqrt_derivative(upper)
    alpha_upper[non_zero] = upper_prime[non_zero]
    beta_upper[non_zero] = upper_act[non_zero] - upper_prime[non_zero] * upper[non_zero]

    return alpha_lower, beta_lower, alpha_upper, beta_upper
