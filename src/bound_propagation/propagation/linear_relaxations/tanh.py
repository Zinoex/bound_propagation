import torch

from .base import ElementwiseLinearRelaxation


def compute_tanh_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseLinearRelaxation:
    """
    Compute alpha/beta parameters for tanh linear relaxation.

    Tanh has similar structure to sigmoid: inflection point at x=0 (convex for x<0, concave for x>0).

    For intervals that cross zero:
    - Upper bound: Use secant if tanh'(upper) >= secant_slope, else tangent at upper
    - Lower bound: Use secant if tanh'(lower) >= secant_slope, else tangent at lower

    This ensures sound bounds while preferring simpler secant when valid.

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

    # Compute tanh and derivative
    lower_act = torch.tanh(lower)
    upper_act = torch.tanh(upper)

    def tanh_derivative(x):
        t = torch.tanh(x)
        return 1 - t * t

    lower_prime = tanh_derivative(lower)
    upper_prime = tanh_derivative(upper)

    # Midpoint for tangent line
    d = (lower + upper) * 0.5
    d_act = torch.tanh(d)
    d_prime = tanh_derivative(d)

    # Slope of secant line
    slope = torch.where(zero_width, torch.zeros_like(lower), (upper_act - lower_act) / (upper - lower))

    # Zero-width case
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = lower_act[zero_width]
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width
    non_zero = ~zero_width

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
        # Upper bound strategy:
        # Check if tanh'(upper) >= secant slope
        # If yes, secant is a valid upper bound (simpler)
        # Otherwise, use tangent at upper
        use_secant_upper = upper_prime[crossing] >= slope[crossing]

        alpha_upper[crossing] = torch.where(use_secant_upper, slope[crossing], upper_prime[crossing])
        beta_upper[crossing] = torch.where(
            use_secant_upper,
            upper_act[crossing] - slope[crossing] * upper[crossing],
            upper_act[crossing] - upper_prime[crossing] * upper[crossing],
        )

        # Lower bound strategy:
        # Check if tanh'(lower) >= secant slope
        # If yes, secant is a valid lower bound
        # Otherwise, use tangent at lower
        use_secant_lower = lower_prime[crossing] >= slope[crossing]

        alpha_lower[crossing] = torch.where(use_secant_lower, slope[crossing], lower_prime[crossing])
        beta_lower[crossing] = torch.where(
            use_secant_lower,
            lower_act[crossing] - slope[crossing] * lower[crossing],
            lower_act[crossing] - lower_prime[crossing] * lower[crossing],
        )

    return ElementwiseLinearRelaxation(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
