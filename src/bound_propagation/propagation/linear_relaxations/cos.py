import math

import torch


def compute_cos_alpha_beta(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha/beta parameters for cos linear relaxation.

    cos is neither globally convex nor concave. Strategy:
    - Always use secant line connecting endpoints as one bound
    - Use tangent line at appropriate endpoint for the other bound
    - Determine which bound gets secant based on convexity of the interval

    For convex regions: tangent is lower bound, secant is upper bound
    For concave regions: secant is lower bound, tangent is upper bound

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        zero_threshold: Threshold for considering an interval as zero-width

    Returns:
        Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
    """
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Handle zero-width case
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)

    lower_act, upper_act = torch.cos(lower[zero_width]), torch.cos(upper[zero_width])
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = torch.min(lower_act, upper_act)
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = torch.max(lower_act, upper_act)

    non_zero = ~zero_width
    if not torch.any(non_zero):
        return alpha_lower, beta_lower, alpha_upper, beta_upper

    # Work with non-zero width intervals
    lower_nz = lower[non_zero]
    upper_nz = upper[non_zero]

    # Compute cos values and derivatives at endpoints
    cos_lower = torch.cos(lower_nz)
    cos_upper = torch.cos(upper_nz)
    dcos_lower = -torch.sin(lower_nz)  # derivative of cos
    dcos_upper = -torch.sin(upper_nz)

    # Secant line connecting endpoints
    secant_slope = (cos_upper - cos_lower) / (upper_nz - lower_nz)
    secant_beta = cos_upper - secant_slope * upper_nz

    # Tangent lines at endpoints
    tangent_lower_slope = dcos_lower
    tangent_lower_beta = cos_lower - dcos_lower * lower_nz
    tangent_upper_slope = dcos_upper
    tangent_upper_beta = cos_upper - dcos_upper * upper_nz

    # Check if interval contains critical points (extrema)
    # Maxima occur at x = 2*k*π for integer k (cos = 1)
    # Minima occur at x = (2*k+1)*π for integer k (cos = -1)

    k_max_lower = torch.ceil(lower_nz / (2 * math.pi))
    k_max_upper = torch.floor(upper_nz / (2 * math.pi))
    has_maximum = k_max_lower <= k_max_upper

    k_min_lower = torch.ceil((lower_nz - math.pi) / (2 * math.pi))
    k_min_upper = torch.floor((upper_nz - math.pi) / (2 * math.pi))
    has_minimum = k_min_lower <= k_min_upper

    # Special case: if interval contains both maximum and minimum, use constant bounds
    # This ensures we capture the full range [-1, 1] or appropriate subset
    has_both_extrema = has_maximum & has_minimum

    # Initialize alpha and beta
    alpha_nz_lower = torch.zeros_like(secant_slope)
    beta_nz_lower = torch.zeros_like(secant_beta)
    alpha_nz_upper = torch.zeros_like(secant_slope)
    beta_nz_upper = torch.zeros_like(secant_beta)

    # For intervals with both extrema, use tight constant bounds
    alpha_nz_lower = torch.where(has_both_extrema, 0, alpha_nz_lower)
    beta_nz_lower = torch.where(has_both_extrema, -1, beta_nz_lower)
    alpha_nz_upper = torch.where(has_both_extrema, 0, alpha_nz_upper)
    beta_nz_upper = torch.where(has_both_extrema, 1, beta_nz_upper)

    # Determine if secant is upper or lower bound by checking midpoint
    mid = (lower_nz + upper_nz) * 0.5
    cos_mid = torch.cos(mid)
    secant_at_mid = secant_slope * mid + secant_beta

    # If cos(mid) > secant(mid), function is above secant (concave), so secant is lower bound
    # If cos(mid) < secant(mid), function is below secant (convex), so secant is upper bound
    # Use a small threshold for numerical stability
    secant_is_lower = cos_mid > secant_at_mid + 1e-7

    # For tangent line, choose the endpoint based on:
    # - If secant is lower bound, we need upper bound from tangent
    # - If secant is upper bound, we need lower bound from tangent
    # - Choose the tangent that gives the tighter (less conservative) bound

    # Initialize with secant for appropriate bound (only for intervals without both extrema)
    init_mask = ~has_both_extrema
    alpha_nz_lower = torch.where(init_mask & secant_is_lower, secant_slope, alpha_nz_lower)
    beta_nz_lower = torch.where(init_mask & secant_is_lower, secant_beta, beta_nz_lower)
    alpha_nz_upper = torch.where(init_mask & ~secant_is_lower, secant_slope, alpha_nz_upper)
    beta_nz_upper = torch.where(init_mask & ~secant_is_lower, secant_beta, beta_nz_upper)

    # For the other bound, choose the better tangent
    # When secant is lower bound, we need upper bound from tangent
    # Choose tangent that gives tighter upper bound (lower line above the function)
    when_secant_is_lower = secant_is_lower

    # Simple heuristic: use tangent at the endpoint with smaller |derivative| for upper bound (flatter)
    # and tangent at endpoint with larger |derivative| for lower bound (steeper)
    use_lower_endpoint = torch.abs(dcos_lower) <= torch.abs(dcos_upper)

    # When secant is lower, we need upper from tangent
    # When secant is upper, we need lower from tangent
    tangent_slope = torch.where(use_lower_endpoint, tangent_lower_slope, tangent_upper_slope)
    tangent_beta = torch.where(use_lower_endpoint, tangent_lower_beta, tangent_upper_beta)

    # Special case: if interval contains an extremum, we need to be more careful
    # For maximum: tangents from both sides should both be valid upper bounds (below peak)
    # For minimum: tangents from both sides should both be valid lower bounds (above trough)

    # When has_maximum and secant_is_lower, use tangent from endpoint closer to maximum
    # When has_minimum and not secant_is_lower, use tangent from endpoint closer to minimum

    # Find which endpoint is closer to the maximum
    # Maximum is at some 2kπ in [lower_nz, upper_nz]
    max_point = 2 * math.pi * k_max_lower
    closer_to_max_is_lower = torch.abs(lower_nz - max_point) <= torch.abs(upper_nz - max_point)

    # When secant is lower bound and we have maximum, use tangent at point closer to maximum for upper
    use_lower_for_max = has_maximum & ~has_both_extrema & when_secant_is_lower & closer_to_max_is_lower
    use_upper_for_max = has_maximum & ~has_both_extrema & when_secant_is_lower & ~closer_to_max_is_lower

    tangent_slope = torch.where(use_lower_for_max, tangent_lower_slope, tangent_slope)
    tangent_beta = torch.where(use_lower_for_max, tangent_lower_beta, tangent_beta)
    tangent_slope = torch.where(use_upper_for_max, tangent_upper_slope, tangent_slope)
    tangent_beta = torch.where(use_upper_for_max, tangent_upper_beta, tangent_beta)

    # Find which endpoint is closer to the minimum
    min_point = math.pi + 2 * math.pi * k_min_lower
    closer_to_min_is_lower = torch.abs(lower_nz - min_point) <= torch.abs(upper_nz - min_point)

    # When secant is upper bound and we have minimum, use tangent at point closer to minimum for lower
    use_lower_for_min = has_minimum & ~has_both_extrema & ~when_secant_is_lower & closer_to_min_is_lower
    use_upper_for_min = has_minimum & ~has_both_extrema & ~when_secant_is_lower & ~closer_to_min_is_lower

    tangent_slope = torch.where(use_lower_for_min, tangent_lower_slope, tangent_slope)
    tangent_beta = torch.where(use_lower_for_min, tangent_lower_beta, tangent_beta)
    tangent_slope = torch.where(use_upper_for_min, tangent_upper_slope, tangent_slope)
    tangent_beta = torch.where(use_upper_for_min, tangent_upper_beta, tangent_beta)

    # Set the tangent bound (only for intervals without both extrema)
    update_mask = ~has_both_extrema
    alpha_nz_lower = torch.where(update_mask & ~when_secant_is_lower, tangent_slope, alpha_nz_lower)
    beta_nz_lower = torch.where(update_mask & ~when_secant_is_lower, tangent_beta, beta_nz_lower)
    alpha_nz_upper = torch.where(update_mask & when_secant_is_lower, tangent_slope, alpha_nz_upper)
    beta_nz_upper = torch.where(update_mask & when_secant_is_lower, tangent_beta, beta_nz_upper)

    # For intervals with single extremum, ensure the extremum value is included
    # When has maximum (and not minimum), upper bound should be >= 1
    only_max = has_maximum & ~has_minimum & ~has_both_extrema
    # Check if current upper bound captures the maximum
    # Evaluate upper bound at the maximum point
    max_point = 2 * math.pi * k_max_lower
    upper_at_max = alpha_nz_upper * max_point + beta_nz_upper
    max_not_captured = only_max & (upper_at_max < 0.99)

    # Fall back to constant bounds for these intervals
    alpha_nz_lower = torch.where(max_not_captured, 0, alpha_nz_lower)
    beta_nz_lower = torch.where(max_not_captured, torch.minimum(cos_lower, cos_upper), beta_nz_lower)
    alpha_nz_upper = torch.where(max_not_captured, 0, alpha_nz_upper)
    beta_nz_upper = torch.where(max_not_captured, torch.ones_like(beta_nz_upper), beta_nz_upper)

    # When has minimum (and not maximum), lower bound should be <= -1
    # For intervals with minimum, tangent lines often extend below -1, so use constant bounds
    only_min = has_minimum & ~has_maximum & ~has_both_extrema

    # For simplicity and correctness, use constant bounds when crossing minimum
    # A more sophisticated approach could use piecewise linear approximation
    alpha_nz_lower = torch.where(only_min, 0, alpha_nz_lower)
    beta_nz_lower = torch.where(only_min, torch.full_like(beta_nz_lower, -1.0), beta_nz_lower)
    alpha_nz_upper = torch.where(only_min, 0, alpha_nz_upper)
    beta_nz_upper = torch.where(only_min, torch.maximum(cos_lower, cos_upper), beta_nz_upper)

    # Assign back to output tensors
    alpha_lower[non_zero] = alpha_nz_lower
    beta_lower[non_zero] = beta_nz_lower
    alpha_upper[non_zero] = alpha_nz_upper
    beta_upper[non_zero] = beta_nz_upper

    return alpha_lower, beta_lower, alpha_upper, beta_upper
