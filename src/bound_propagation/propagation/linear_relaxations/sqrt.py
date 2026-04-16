import torch

from .base import ElementwiseLinearRelaxation


def compute_sqrt_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseLinearRelaxation:
    """
    Compute alpha/beta parameters for sqrt linear relaxation.

    sqrt is concave, so:
    - Lower bound: secant line connecting (lower, sqrt(lower)) and (upper, sqrt(upper))
    - Upper bound: tangent line at midpoint

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        ElementwiseLinearRelaxation encapsulating the relaxation
    """
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
        return torch.where(x > 0, 1.0 / (2.0 * torch.sqrt(x)), 0)

    # Midpoint for tangent line
    midpoint = (lower + upper) * 0.5
    midpoint_act = torch.sqrt(midpoint)
    midpoint_prime = sqrt_derivative(midpoint)

    # Slope of secant line
    slope = (upper_act - lower_act) / (upper - lower)

    # Zero-width case: use the value itself
    beta_lower[zero_width] = lower_act[zero_width]
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width cases
    non_zero = ~zero_width

    # sqrt is concave everywhere (for x > 0):
    # - Lower bound: secant line
    # - Upper bound: tangent line at midpoint

    # Lower bound: secant line
    alpha_lower[non_zero] = slope[non_zero]
    beta_lower[non_zero] = lower_act[non_zero] - slope[non_zero] * lower[non_zero]

    # Upper bound: tangent at midpoint
    alpha_upper[non_zero] = midpoint_prime[non_zero]
    beta_upper[non_zero] = midpoint_act[non_zero] - midpoint_prime[non_zero] * midpoint[non_zero]

    # Invalid regime: sqrt is undefined for negative inputs
    invalid = lower < 0
    alpha_lower[invalid] = float("nan")
    beta_lower[invalid] = float("nan")
    alpha_upper[invalid] = float("nan")
    beta_upper[invalid] = float("nan")

    return ElementwiseLinearRelaxation(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
