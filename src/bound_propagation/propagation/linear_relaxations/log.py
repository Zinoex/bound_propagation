import torch


def compute_log_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for log linear relaxation.

    log(x) is concave, so the upper bound is the tangent line at the lower bound,
    and the lower bound is the secant line connecting (lower, log(lower)) and (upper, log(upper)).

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation (should be > 0 for log to be defined)
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
    """
    log_lower = torch.log(lower)
    log_upper = torch.log(upper)

    zero_width = torch.isclose(lower, upper, atol=zero_threshold)
    slope = (log_upper - log_lower) / (upper - lower)

    alpha_lower = torch.where(zero_width, 0, slope)
    beta_lower = torch.where(zero_width, log_lower, log_lower - slope * lower)

    midpoint = (lower + upper) / 2

    alpha_upper = torch.where(zero_width, 0, 1 / midpoint)
    beta_upper = torch.where(zero_width, log_upper, torch.log(midpoint) - alpha_upper * midpoint)

    return alpha_lower, beta_lower, alpha_upper, beta_upper
