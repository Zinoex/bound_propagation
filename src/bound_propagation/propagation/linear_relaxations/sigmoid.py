import torch


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
