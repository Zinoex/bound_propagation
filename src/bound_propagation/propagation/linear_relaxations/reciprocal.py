import torch

from .base import ElementwiseLinearRelaxation


def compute_reciprocal_relaxation(
    lower: torch.Tensor, upper: torch.Tensor, zero_threshold: float = 1e-8
) -> ElementwiseLinearRelaxation:
    """
    Compute alpha/beta parameters for reciprocal (1/x) linear relaxation.

    reciprocal is convex for x > 0 and convex for x < 0, so:
    - When interval is all positive or all negative: use secant for lower, tangent for upper
    - When interval crosses zero: handle specially (may need to use safe bounds)

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
    crosses_zero = (lower < 0) & (upper > 0)
    all_positive = lower > 0
    all_negative = upper < 0

    # Compute reciprocal values (avoid division by zero)
    eps = 1e-8
    lower_safe = torch.where(torch.abs(lower) < eps, eps * torch.sign(lower + eps), lower)
    upper_safe = torch.where(torch.abs(upper) < eps, eps * torch.sign(upper + eps), upper)

    lower_act = 1.0 / lower_safe
    upper_act = 1.0 / upper_safe

    def reciprocal_derivative(x):
        # d/dx (1/x) = -1/x^2
        x_safe = torch.where(torch.abs(x) < eps, eps * torch.sign(x + eps), x)
        return -1.0 / (x_safe * x_safe)

    reciprocal_derivative(lower_safe)
    reciprocal_derivative(upper_safe)

    # Midpoint for tangent line
    d = (lower + upper) * 0.5
    d_safe = torch.where(torch.abs(d) < eps, eps * torch.sign(d + eps), d)
    d_act = 1.0 / d_safe
    d_prime = reciprocal_derivative(d_safe)

    # Slope of secant line
    slope = torch.where(
        zero_width, torch.zeros_like(lower), (upper_act - lower_act) / torch.clamp(upper - lower, min=eps)
    )

    # Case 1: Zero-width intervals
    # alphas = 0, beta_lower = 1/upper, beta_upper = 1/lower
    alpha_lower[zero_width] = 0.0
    alpha_upper[zero_width] = 0.0
    beta_lower[zero_width] = upper_act[zero_width]
    beta_upper[zero_width] = lower_act[zero_width]

    # Case 4: Crosses zero
    # alphas = 0, beta_lower = -inf, beta_upper = inf
    alpha_lower[crosses_zero] = 0.0
    alpha_upper[crosses_zero] = 0.0
    beta_lower[crosses_zero] = float("-inf")
    beta_upper[crosses_zero] = float("inf")

    # Case 2: All positive (x > 0)
    # Upper bound: secant line
    # Lower bound: tangent at midpoint
    alpha_upper[all_positive] = slope[all_positive]
    beta_upper[all_positive] = upper_act[all_positive] - slope[all_positive] * upper_safe[all_positive]
    alpha_lower[all_positive] = d_prime[all_positive]
    beta_lower[all_positive] = d_act[all_positive] - d_prime[all_positive] * d_safe[all_positive]

    # Case 3: All negative (x < 0)
    # Upper bound: tangent at midpoint
    # Lower bound: secant line
    alpha_upper[all_negative] = d_prime[all_negative]
    beta_upper[all_negative] = d_act[all_negative] - d_prime[all_negative] * d_safe[all_negative]
    alpha_lower[all_negative] = slope[all_negative]
    beta_lower[all_negative] = upper_act[all_negative] - slope[all_negative] * upper_safe[all_negative]

    return ElementwiseLinearRelaxation(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
