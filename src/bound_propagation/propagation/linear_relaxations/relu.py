import torch


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
