import torch

from .base import ElementwiseLinearRelaxation


def compute_abs_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseLinearRelaxation:
    """
    Compute alpha/beta parameters for abs linear relaxation.

    abs(x) is piecewise linear:
    - For x >= 0: abs(x) = x
    - For x < 0: abs(x) = -x

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
    all_positive = (lower >= 0) & ~zero_width
    all_negative = (upper <= 0) & ~zero_width
    crosses_zero = (lower < 0) & (upper > 0) & ~zero_width

    # Zero-width case: use the value itself
    lower_act = torch.abs(lower[zero_width])
    upper_act = torch.abs(upper[zero_width])
    beta_lower[zero_width] = torch.min(lower_act, upper_act)
    beta_upper[zero_width] = torch.max(lower_act, upper_act)

    # All positive: abs(x) = x
    alpha_lower[all_positive] = 1
    alpha_upper[all_positive] = 1

    # All negative: abs(x) = -x
    alpha_lower[all_negative] = -1
    alpha_upper[all_negative] = -1

    # Crosses zero
    # Upper bound: line connecting (lower, abs(lower)) and (upper, abs(upper))
    lower, upper = lower[crosses_zero], upper[crosses_zero]

    lower_act = torch.abs(lower)
    upper_act = torch.abs(upper)

    slope = (upper_act - lower_act) / (upper - lower)

    alpha_upper[crosses_zero] = slope
    beta_upper[crosses_zero] = upper_act - slope * upper

    # For lower bound, use upper bound slope but zero intercept (line through the origin)
    alpha_lower[crosses_zero] = slope

    return ElementwiseLinearRelaxation(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
