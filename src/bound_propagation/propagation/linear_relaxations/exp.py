import torch

from .base import ElementwiseLinearRelaxation


def compute_exp_relaxation(
    lower: torch.Tensor, upper: torch.Tensor, zero_threshold: float = 1e-8
) -> ElementwiseLinearRelaxation:
    """
    Compute alpha/beta parameters for exp linear relaxation.

    exp(x) is convex, so we can use the tangent line at the midpoint for the lower bound relaxation,
    and the secant line between (lower, exp(lower)) and (upper, exp(upper)) for the upper bound relaxation.

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        ElementwiseLinearRelaxation encapsulating the relaxation
    """
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)
    midpoint = (lower + upper) / 2

    exp_lower = torch.exp(lower)
    exp_upper = torch.exp(upper)
    exp_mid = torch.exp(midpoint)

    alpha_lower = torch.where(zero_width, 0, exp_mid)
    beta_lower = torch.where(zero_width, exp_lower, exp_mid - alpha_lower * midpoint)

    slope = (exp_upper - exp_lower) / (upper - lower)

    alpha_upper = torch.where(zero_width, 0, slope)
    beta_upper = torch.where(zero_width, exp_upper, exp_lower - slope * lower)

    return ElementwiseLinearRelaxation(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
