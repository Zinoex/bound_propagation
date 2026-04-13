import torch


def compute_sigmoid_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for sigmoid linear relaxation.

    Sigmoid has an inflection point at x=0 (convex for x<0, concave for x>0).

    For intervals that cross zero:
    - Upper bound: Use secant if sigmoid'(upper) >= secant_slope, else tangent at upper
    - Lower bound: Use secant if sigmoid'(lower) >= secant_slope, else tangent at lower

    This ensures sound bounds while preferring simpler secant when valid.

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
    """
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Determine regimes
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)

    # Compute sigmoid and derivative
    lower_act = torch.sigmoid(lower)
    upper_act = torch.sigmoid(upper)

    def sigmoid_derivative(x):
        s = torch.sigmoid(x)
        return s * (1 - s)

    # Midpoint for tangent line
    d = (lower + upper) * 0.5
    d_act = torch.sigmoid(d)
    d_prime = sigmoid_derivative(d)

    # Slope of secant line
    slope = torch.where(zero_width, torch.zeros_like(lower), (upper_act - lower_act) / (upper - lower))

    # Zero-width case: use the value itself
    beta_lower[zero_width] = lower_act[zero_width]
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width cases
    non_zero = ~zero_width

    # Determine negative/positive regimes
    negative = non_zero & (upper <= 0)
    positive = non_zero & (lower >= 0)
    crossing = non_zero & (lower < 0) & (upper > 0)

    # Negative regime
    # Upper: secant line between lower and upper
    alpha_upper[negative] = slope[negative]
    beta_upper[negative] = upper_act[negative] - slope[negative] * upper[negative]

    # Lower: tangent line at midpoint
    alpha_lower[negative] = d_prime[negative]
    beta_lower[negative] = d_act[negative] - d_prime[negative] * d[negative]

    # Positive regime
    # Upper: tangent at midpoint
    alpha_upper[positive] = d_prime[positive]
    beta_upper[positive] = d_act[positive] - d_prime[positive] * d[positive]

    # Lower: secant line
    alpha_lower[positive] = slope[positive]
    beta_lower[positive] = lower_act[positive] - slope[positive] * lower[positive]

    # Crossing regime (contains both negative and positive)
    lower_prime = sigmoid_derivative(lower)
    upper_prime = sigmoid_derivative(upper)

    # Upper bound strategy:
    # Check if sigmoid'(upper) >= secant slope
    # If yes, secant is a valid upper bound (simpler)
    # Otherwise, use tangent at upper
    use_secant_upper = upper_prime[crossing] >= slope[crossing]

    # For secant case
    alpha_upper[crossing] = torch.where(use_secant_upper, slope[crossing], upper_prime[crossing])
    beta_upper[crossing] = torch.where(
        use_secant_upper,
        upper_act[crossing] - slope[crossing] * upper[crossing],
        upper_act[crossing] - upper_prime[crossing] * upper[crossing],
    )

    # Lower bound strategy:
    # Check if sigmoid'(lower) >= secant slope
    # If yes, secant is a valid lower bound
    # Otherwise, use tangent at lower
    use_secant_lower = lower_prime[crossing] >= slope[crossing]

    alpha_lower[crossing] = torch.where(use_secant_lower, slope[crossing], lower_prime[crossing])
    beta_lower[crossing] = torch.where(
        use_secant_lower,
        lower_act[crossing] - slope[crossing] * lower[crossing],
        lower_act[crossing] - lower_prime[crossing] * lower[crossing],
    )

    return alpha_lower, beta_lower, alpha_upper, beta_upper
