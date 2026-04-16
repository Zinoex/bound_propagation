import math

import torch

from .base import ElementwiseLinearRelaxation


def compute_tan_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseLinearRelaxation:
    """
    Compute alpha/beta parameters for tan linear relaxation.

    tan has asymptotes at x = π/2 + nπ and inflection points at x = nπ.
    - In (-π/2, 0): tan is concave (tan'' < 0)
    - In (0, π/2): tan is convex (tan'' > 0)

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        zero_threshold: Threshold for zero-width intervals

    Returns:
        ElementwiseLinearRelaxation encapsulating the relaxation
    """
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Determine regimes
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)

    # Check for asymptote crossings: tan has asymptotes at x = π/2 + nπ
    # This happens when cos(x) = 0, which is at x = ±π/2, ±3π/2, ±5π/2, ...
    # We detect this by checking if the interval contains any x where cos(x) ≈ 0
    # More precisely, check if there exists n such that π/2 + nπ ∈ [lower, upper]

    # Find the asymptote positions in the range [lower, upper]
    # Asymptotes are at (2n+1)π/2 for integer n
    # For each interval, check if it contains an asymptote
    half_pi = math.pi / 2.0

    # Compute which asymptote index (n) the lower and upper bounds are near
    # lower_n = floor((lower - π/2) / π) gives the index of the asymptote at or below lower
    # If lower_n != upper_n, we cross an asymptote
    lower_asymptote_idx = torch.floor((lower - half_pi) / math.pi)
    upper_asymptote_idx = torch.floor((upper - half_pi) / math.pi)

    crosses_asymptote = lower_asymptote_idx != upper_asymptote_idx

    # For intervals crossing asymptotes, set infinite bounds
    beta_lower[crosses_asymptote] = float("-inf")
    beta_upper[crosses_asymptote] = float("inf")
    alpha_lower[crosses_asymptote] = 0.0
    alpha_upper[crosses_asymptote] = 0.0

    # Compute tan values for non-asymptote-crossing intervals
    valid = ~crosses_asymptote & ~zero_width
    lower_act = torch.tan(lower)
    upper_act = torch.tan(upper)

    def tan_derivative(x):
        # d/dx tan(x) = sec^2(x) = 1/cos^2(x)
        cos_x = torch.cos(x)
        return 1.0 / (cos_x * cos_x + 1e-8)

    lower_prime = tan_derivative(lower)
    upper_prime = tan_derivative(upper)

    # Slope of secant line
    slope = (upper_act - lower_act) / torch.clamp(upper - lower, min=1e-8)

    # Zero-width case: use the value itself; if it also crosses an asymptote, we already set to inf above
    beta_lower[zero_width & ~crosses_asymptote] = lower_act[zero_width & ~crosses_asymptote]
    beta_upper[zero_width & ~crosses_asymptote] = upper_act[zero_width & ~crosses_asymptote]

    # Determine convexity: tan is convex when tan(x) > 0, concave when tan(x) < 0
    # Equivalently: convex when x ∈ (nπ, nπ + π/2), concave when x ∈ (nπ - π/2, nπ)
    # We can check this by looking at sin(x): tan is convex when sin(lower) and sin(upper) > 0
    # Actually simpler: tan is convex when tan(midpoint) > 0, concave when < 0
    midpoint = (lower + upper) * 0.5
    midpoint_tan = torch.tan(midpoint)

    # But we need to handle crossing zero (inflection point) specially
    crosses_zero = (lower < 0) & (upper > 0) & valid

    # Convex regime: tan > 0 (e.g., x ∈ (0, π/2))
    # Use secant for upper bound, tangent for lower bound
    convex = (midpoint_tan > zero_threshold) & valid & ~crosses_zero

    # Concave regime: tan < 0 (e.g., x ∈ (-π/2, 0))
    # Use tangent for upper bound, secant for lower bound
    concave = (midpoint_tan < -zero_threshold) & valid & ~crosses_zero

    # Handle convex regime
    if convex.any():
        alpha_upper[convex] = slope[convex]
        beta_upper[convex] = upper_act[convex] - slope[convex] * upper[convex]

        alpha_lower[convex] = lower_prime[convex]
        beta_lower[convex] = lower_act[convex] - lower_prime[convex] * lower[convex]

    # Handle concave regime
    if concave.any():
        alpha_upper[concave] = upper_prime[concave]
        beta_upper[concave] = upper_act[concave] - upper_prime[concave] * upper[concave]

        alpha_lower[concave] = slope[concave]
        beta_lower[concave] = lower_act[concave] - slope[concave] * lower[concave]

    # Handle crossing zero (inflection point)
    # tan is concave left, convex right (opposite of sigmoid!)
    # For this type of S-curve, neither secant nor endpoint tangents work perfectly.
    # Solution: Use tangent slope at inflection point (x=0, slope=1) but adjust
    # intercepts to ensure the lines pass through or beyond the endpoints.
    if crosses_zero.any():
        # tan'(0) = 1
        inflection_slope = torch.ones_like(lower[crosses_zero])

        # For lower bound: y = 1*x + β_lower
        # Need: 1*lower + β_lower <= tan(lower) AND 1*upper + β_lower <= tan(upper)
        # => β_lower <= tan(lower) - lower AND β_lower <= tan(upper) - upper
        beta_lower_from_lower = lower_act[crosses_zero] - inflection_slope * lower[crosses_zero]
        beta_lower_from_upper = upper_act[crosses_zero] - inflection_slope * upper[crosses_zero]
        beta_lower_val = torch.minimum(beta_lower_from_lower, beta_lower_from_upper)

        # For upper bound: y = 1*x + β_upper
        # Need: 1*lower + β_upper >= tan(lower) AND 1*upper + β_upper >= tan(upper)
        # => β_upper >= tan(lower) - lower AND β_upper >= tan(upper) - upper
        beta_upper_from_lower = lower_act[crosses_zero] - inflection_slope * lower[crosses_zero]
        beta_upper_from_upper = upper_act[crosses_zero] - inflection_slope * upper[crosses_zero]
        beta_upper_val = torch.maximum(beta_upper_from_lower, beta_upper_from_upper)

        alpha_lower[crosses_zero] = inflection_slope
        beta_lower[crosses_zero] = beta_lower_val

        alpha_upper[crosses_zero] = inflection_slope
        beta_upper[crosses_zero] = beta_upper_val

    return ElementwiseLinearRelaxation(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
