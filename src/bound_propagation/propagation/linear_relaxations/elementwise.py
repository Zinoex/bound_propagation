from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final, overload

import torch


@final
@dataclass
class ElementwiseParams:
    """
    Element-wise linear relaxation for unary operations y = f(x).

    Stores four element-wise tensors (same shape as x / y):
        y_lower >= alpha_lower * x + beta_lower
        y_upper <= alpha_upper * x + beta_upper

    The abstract dimension convention for LinearBounds linear terms is
    (*batch_dims, *output_dims, *input_dims). alpha and beta live in
    (*batch_dims, *output_dims).

    Attributes:
        alpha_lower: Element-wise slopes for the lower bound.
        beta_lower:  Element-wise biases for the lower bound.
        alpha_upper: Element-wise slopes for the upper bound.
        beta_upper:  Element-wise biases for the upper bound.
    """

    alpha_lower: torch.Tensor
    beta_lower: torch.Tensor
    alpha_upper: torch.Tensor
    beta_upper: torch.Tensor


def compute_abs_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
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
        ElementwiseParams encapsulating the relaxation parameters
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

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


@overload
def compute_clamp_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_val: float | None,
    max_val: float | None,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams: ...


@overload
def compute_clamp_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_val: torch.Tensor | None,
    max_val: torch.Tensor | None,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams: ...


def compute_clamp_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_val: float | torch.Tensor | None = None,
    max_val: float | torch.Tensor | None = None,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for clamp linear relaxation.

    clamp(x, min, max) = min(max(x, min), max)

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        min_val: Minimum clamp value (default: -inf)
        max_val: Maximum clamp value (default: +inf)
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        ElementwiseParams encapsulating the relaxation parameters
    """

    # TODO: assert the overload inputs

    if max_val is None:
        lower_clamped = torch.clamp(lower, min=min_val)
        upper_clamped = torch.clamp(upper, min=min_val)
        max_val = float("inf")
    elif min_val is None:
        lower_clamped = torch.clamp(lower, max=max_val)
        upper_clamped = torch.clamp(upper, max=max_val)
        min_val = float("-inf")
    elif isinstance(min_val, torch.Tensor) and not isinstance(max_val, torch.Tensor):
        raise ValueError("If min_val is a tensor, max_val must be None or a tensor")
    elif isinstance(max_val, torch.Tensor) and not isinstance(min_val, torch.Tensor):
        raise ValueError("If max_val is a tensor, min_val must be None or a tensor")
    else:
        lower_clamped = torch.clamp(lower, min=min_val, max=max_val)  # ty:ignore[no-matching-overload]
        upper_clamped = torch.clamp(upper, min=min_val, max=max_val)  # ty:ignore[no-matching-overload]

    assert min_val is not None and max_val is not None

    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Determine regimes
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)
    not_zero_width = ~zero_width
    below_min = (upper <= min_val) & not_zero_width
    above_max = (lower >= max_val) & not_zero_width
    in_range = (lower >= min_val) & (upper <= max_val) & not_zero_width
    crosses_min = (lower < min_val) & (upper > min_val) & (upper <= max_val) & not_zero_width
    crosses_max = (lower >= min_val) & (lower < max_val) & (upper > max_val) & not_zero_width
    crosses_both = (lower < min_val) & (upper > max_val) & not_zero_width

    # Zero-width case: use beta_lower = clamp(lower) and beta_upper = clamp(upper)
    beta_lower[zero_width] = lower_clamped[zero_width]
    beta_upper[zero_width] = upper_clamped[zero_width]

    # Below min: constant at min
    beta_lower[below_min] = min_val[below_min] if isinstance(min_val, torch.Tensor) else min_val
    beta_upper[below_min] = min_val[below_min] if isinstance(min_val, torch.Tensor) else min_val

    # Above max: constant at max
    beta_lower[above_max] = max_val[above_max] if isinstance(max_val, torch.Tensor) else max_val
    beta_upper[above_max] = max_val[above_max] if isinstance(max_val, torch.Tensor) else max_val

    # In range: identity
    alpha_lower[in_range] = 1
    alpha_upper[in_range] = 1

    # Crosses min:
    # Function has corner at (min_val, min_val), can't be tightly bounded by single line
    # Lower bound: horizontal at min_val (sound, conservative)
    # Upper bound: horizontal at max(clamp(lower), clamp(upper)) (sound, conservative)
    lower_clamped_min, upper_clamped_min = lower_clamped[crosses_min], upper_clamped[crosses_min]

    alpha_lower[crosses_min] = 0
    beta_lower[crosses_min] = min_val[crosses_min] if isinstance(min_val, torch.Tensor) else min_val
    alpha_upper[crosses_min] = 0
    beta_upper[crosses_min] = torch.maximum(lower_clamped_min, upper_clamped_min)

    # Crosses max:
    # Function has corner at (max_val, max_val), can't be tightly bounded by single line
    # Lower bound: horizontal at min(clamp(lower), clamp(upper)) (sound, conservative)
    # Upper bound: horizontal at max_val (sound, conservative)
    lower_clamped_max, upper_clamped_max = lower_clamped[crosses_max], upper_clamped[crosses_max]

    alpha_lower[crosses_max] = 0
    beta_lower[crosses_max] = torch.minimum(lower_clamped_max, upper_clamped_max)
    alpha_upper[crosses_max] = 0
    beta_upper[crosses_max] = max_val[crosses_max] if isinstance(max_val, torch.Tensor) else max_val

    # Crosses both:
    # Function has corners at both (min_val, min_val) and (max_val, max_val)
    # Lower bound: horizontal at min_val (conservative)
    # Upper bound: horizontal at max_val (conservative)
    alpha_lower[crosses_both] = 0
    beta_lower[crosses_both] = min_val[crosses_both] if isinstance(min_val, torch.Tensor) else min_val
    alpha_upper[crosses_both] = 0
    beta_upper[crosses_both] = max_val[crosses_both] if isinstance(max_val, torch.Tensor) else max_val

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_cos_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
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
        ElementwiseParams encapsulating the relaxation parameters
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
        return ElementwiseParams(
            alpha_lower=alpha_lower,
            beta_lower=beta_lower,
            alpha_upper=alpha_upper,
            beta_upper=beta_upper,
        )

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

    # Check if interval contains inflection points (where convexity changes)
    # cos has inflection points at x = π/2 + k*π for integer k (where cos'' = 0)
    # Intervals crossing inflection points need more conservative bounds
    k_inflection_lower = torch.ceil((lower_nz - math.pi / 2) / math.pi)
    k_inflection_upper = torch.floor((upper_nz - math.pi / 2) / math.pi)
    has_inflection = k_inflection_lower <= k_inflection_upper

    # For intervals crossing inflection points WITHOUT extrema, use constant bounds
    # Secant isn't sound when crossing inflection points (can be above AND below curve)
    crosses_inflection_only = has_inflection & ~has_maximum & ~has_minimum & ~has_both_extrema
    alpha_nz_lower = torch.where(crosses_inflection_only, 0, alpha_nz_lower)
    beta_nz_lower = torch.where(crosses_inflection_only, torch.minimum(cos_lower, cos_upper), beta_nz_lower)
    alpha_nz_upper = torch.where(crosses_inflection_only, 0, alpha_nz_upper)
    beta_nz_upper = torch.where(crosses_inflection_only, torch.maximum(cos_lower, cos_upper), beta_nz_upper)

    # Assign back to output tensors
    alpha_lower[non_zero] = alpha_nz_lower
    beta_lower[non_zero] = beta_nz_lower
    alpha_upper[non_zero] = alpha_nz_upper
    beta_upper[non_zero] = beta_nz_upper

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_exp_relaxation(lower: torch.Tensor, upper: torch.Tensor, zero_threshold: float = 1e-8) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for exp linear relaxation.

    exp(x) is convex, so we can use the tangent line at the midpoint for the lower bound relaxation,
    and the secant line between (lower, exp(lower)) and (upper, exp(upper)) for the upper bound relaxation.

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        ElementwiseParams encapsulating the relaxation parameters
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

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_log_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for log linear relaxation.

    log(x) is concave, so the upper bound is the tangent line at the lower bound,
    and the lower bound is the secant line connecting (lower, log(lower)) and (upper, log(upper)).

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation (should be > 0 for log to be defined)
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        ElementwiseParams encapsulating the relaxation parameters
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

    # Invalid regime: log is undefined for non-positive inputs, so we can set those bounds to nan
    invalid = lower <= 0
    alpha_lower[invalid] = float("nan")
    beta_lower[invalid] = float("nan")
    alpha_upper[invalid] = float("nan")
    beta_upper[invalid] = float("nan")

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_reciprocal_relaxation(
    lower: torch.Tensor, upper: torch.Tensor, zero_threshold: float = 1e-8
) -> ElementwiseParams:
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
        ElementwiseParams encapsulating the relaxation parameters
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

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_constant_div_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    constant: object,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
    """Compute linear-relaxation parameters for constant-over-bounds division ``constant / x``.

    The function ``f(x) = c / x`` is a scaled reciprocal.

    - For ``c > 0``, lower/upper relaxations keep the same orientation as reciprocal.
    - For ``c < 0``, multiplying by a negative constant flips inequalities, so lower/upper are swapped.
    - For intervals crossing zero, returns ``(-inf, inf)`` bounds element-wise.

    Args:
        lower: Lower bounds of denominator ``x``.
        upper: Upper bounds of denominator ``x``.
        constant: Numerator constant (scalar or tensor broadcastable to ``lower``).
        zero_threshold: Threshold used by reciprocal relaxation for zero-width handling.

    Returns:
        ElementwiseParams encapsulating the relaxation parameters.
    """
    if not isinstance(lower, torch.Tensor):
        raise TypeError(f"lower must be a torch.Tensor, got {type(lower)!r}")
    if not isinstance(upper, torch.Tensor):
        raise TypeError(f"upper must be a torch.Tensor, got {type(upper)!r}")
    if lower.shape != upper.shape:
        raise ValueError(f"lower and upper must have the same shape, got {lower.shape} and {upper.shape}")
    if zero_threshold < 0:
        raise ValueError(f"zero_threshold must be non-negative, got {zero_threshold}")

    constant_tensor = torch.as_tensor(constant, dtype=lower.dtype, device=lower.device)
    try:
        constant_tensor = torch.broadcast_to(constant_tensor, lower.shape)
    except RuntimeError as error:
        constant_shape = tuple(constant_tensor.shape)
        raise ValueError(
            f"constant must be broadcastable to denominator bounds shape {lower.shape}, "
            f"got constant shape {constant_shape}"
        ) from error

    recip = compute_reciprocal_relaxation(lower, upper, zero_threshold=zero_threshold)

    positive_constant = constant_tensor > 0
    zero_constant = constant_tensor == 0
    crosses_zero = (lower < 0) & (upper > 0)

    # Multiplication by a negative constant flips lower/upper inequalities.
    alpha_lower = constant_tensor * torch.where(positive_constant, recip.alpha_lower, recip.alpha_upper)
    beta_lower = constant_tensor * torch.where(positive_constant, recip.beta_lower, recip.beta_upper)
    alpha_upper = constant_tensor * torch.where(positive_constant, recip.alpha_upper, recip.alpha_lower)
    beta_upper = constant_tensor * torch.where(positive_constant, recip.beta_upper, recip.beta_lower)

    # For intervals crossing zero, the function contains asymptotes, so return infinite bounds.
    alpha_lower[crosses_zero] = 0.0
    beta_lower[crosses_zero] = float("-inf")
    alpha_upper[crosses_zero] = 0.0
    beta_upper[crosses_zero] = float("inf")

    # For zero constant, the function is constant zero, so alpha=0, beta=0.
    # This must take precedence over zero-crossing handling.
    alpha_lower[zero_constant] = 0.0
    beta_lower[zero_constant] = 0.0
    alpha_upper[zero_constant] = 0.0
    beta_upper[zero_constant] = 0.0

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_relu_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    adaptive: bool = False,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for ReLU linear relaxation.

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        adaptive: Whether to use adaptive ReLU relaxation
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        ElementwiseParams encapsulating the relaxation parameters
    """
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Determine regimes
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)
    # negative = (~zero_width) & (upper <= 0)
    positive = (~zero_width) & (lower >= 0)
    crossing = (~zero_width) & (lower < 0) & (upper > 0)

    # Zero-width: use the value itself
    beta_lower[zero_width] = torch.relu(lower[zero_width])
    beta_upper[zero_width] = torch.relu(upper[zero_width])

    # Negative regime: output is always 0

    # Positive regime: output is identity
    alpha_lower[positive] = 1
    alpha_upper[positive] = 1

    # Crossing regime: use linear relaxation
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

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_sigmoid_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
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
        ElementwiseParams encapsulating the relaxation parameters
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

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_sin_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for sin linear relaxation.

    sin is neither globally convex nor concave. Strategy:
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
        ElementwiseParams encapsulating the relaxation parameters
    """
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Handle zero-width case
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)

    lower_act, upper_act = torch.sin(lower[zero_width]), torch.sin(upper[zero_width])
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = torch.min(lower_act, upper_act)
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = torch.max(lower_act, upper_act)

    non_zero = ~zero_width
    if not torch.any(non_zero):
        return ElementwiseParams(
            alpha_lower=alpha_lower,
            beta_lower=beta_lower,
            alpha_upper=alpha_upper,
            beta_upper=beta_upper,
        )

    # Work with non-zero width intervals
    lower_nz = lower[non_zero]
    upper_nz = upper[non_zero]

    # Compute sin values and derivatives at endpoints
    sin_lower = torch.sin(lower_nz)
    sin_upper = torch.sin(upper_nz)
    dsin_lower = torch.cos(lower_nz)  # derivative of sin
    dsin_upper = torch.cos(upper_nz)

    # Secant line connecting endpoints
    secant_slope = (sin_upper - sin_lower) / (upper_nz - lower_nz)
    secant_beta = sin_upper - secant_slope * upper_nz

    # Tangent lines at endpoints
    tangent_lower_slope = dsin_lower
    tangent_lower_beta = sin_lower - dsin_lower * lower_nz
    tangent_upper_slope = dsin_upper
    tangent_upper_beta = sin_upper - dsin_upper * upper_nz

    # Check if interval contains critical points (extrema)
    # Maxima occur at x = π/2 + 2*k*π for integer k (sin = 1)
    # Minima occur at x = 3π/2 + 2*k*π for integer k (sin = -1)

    k_max_lower = torch.ceil((lower_nz - math.pi / 2) / (2 * math.pi))
    k_max_upper = torch.floor((upper_nz - math.pi / 2) / (2 * math.pi))
    has_maximum = k_max_lower <= k_max_upper

    k_min_lower = torch.ceil((lower_nz - 3 * math.pi / 2) / (2 * math.pi))
    k_min_upper = torch.floor((upper_nz - 3 * math.pi / 2) / (2 * math.pi))
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
    sin_mid = torch.sin(mid)
    secant_at_mid = secant_slope * mid + secant_beta

    # If sin(mid) > secant(mid), function is above secant (concave), so secant is lower bound
    # If sin(mid) < secant(mid), function is below secant (convex), so secant is upper bound
    # Use a small threshold for numerical stability
    secant_is_lower = sin_mid > secant_at_mid + 1e-7

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
    use_lower_endpoint = torch.abs(dsin_lower) <= torch.abs(dsin_upper)

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
    # Maximum is at some π/2 + 2kπ in [lower_nz, upper_nz]
    max_point = math.pi / 2 + 2 * math.pi * k_max_lower
    closer_to_max_is_lower = torch.abs(lower_nz - max_point) <= torch.abs(upper_nz - max_point)

    # When secant is lower bound and we have maximum, use tangent at point closer to maximum for upper
    use_lower_for_max = has_maximum & ~has_both_extrema & when_secant_is_lower & closer_to_max_is_lower
    use_upper_for_max = has_maximum & ~has_both_extrema & when_secant_is_lower & ~closer_to_max_is_lower

    tangent_slope = torch.where(use_lower_for_max, tangent_lower_slope, tangent_slope)
    tangent_beta = torch.where(use_lower_for_max, tangent_lower_beta, tangent_beta)
    tangent_slope = torch.where(use_upper_for_max, tangent_upper_slope, tangent_slope)
    tangent_beta = torch.where(use_upper_for_max, tangent_upper_beta, tangent_beta)

    # Find which endpoint is closer to the minimum
    min_point = 3 * math.pi / 2 + 2 * math.pi * k_min_lower
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
    max_point = math.pi / 2 + 2 * math.pi * k_max_lower
    upper_at_max = alpha_nz_upper * max_point + beta_nz_upper
    max_not_captured = only_max & (upper_at_max < 0.99)

    # Fall back to constant bounds for these intervals
    alpha_nz_lower = torch.where(max_not_captured, 0, alpha_nz_lower)
    beta_nz_lower = torch.where(max_not_captured, torch.minimum(sin_lower, sin_upper), beta_nz_lower)
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
    beta_nz_upper = torch.where(only_min, torch.maximum(sin_lower, sin_upper), beta_nz_upper)

    # Check if interval contains inflection points (where convexity changes)
    # sin has inflection points at x = k*π for integer k (where sin'' = 0)
    # Intervals crossing inflection points need more conservative bounds
    k_inflection_lower = torch.ceil(lower_nz / math.pi)
    k_inflection_upper = torch.floor(upper_nz / math.pi)
    has_inflection = k_inflection_lower <= k_inflection_upper

    # For intervals crossing inflection points WITHOUT extrema, use constant bounds
    # Secant isn't sound when crossing inflection points (can be above AND below curve)
    crosses_inflection_only = has_inflection & ~has_maximum & ~has_minimum & ~has_both_extrema
    alpha_nz_lower = torch.where(crosses_inflection_only, 0, alpha_nz_lower)
    beta_nz_lower = torch.where(crosses_inflection_only, torch.minimum(sin_lower, sin_upper), beta_nz_lower)
    alpha_nz_upper = torch.where(crosses_inflection_only, 0, alpha_nz_upper)
    beta_nz_upper = torch.where(crosses_inflection_only, torch.maximum(sin_lower, sin_upper), beta_nz_upper)

    # Assign back to output tensors
    alpha_lower[non_zero] = alpha_nz_lower
    beta_lower[non_zero] = beta_nz_lower
    alpha_upper[non_zero] = alpha_nz_upper
    beta_upper[non_zero] = beta_nz_upper

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_sqrt_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for sqrt linear relaxation.

    sqrt is concave, so:
    - Lower bound: secant line connecting (lower, sqrt(lower)) and (upper, sqrt(upper))
    - Upper bound: tangent line at midpoint

    Args:
        lower: Lower bounds of pre-activation
        upper: Upper bounds of pre-activation
        zero_threshold: Threshold to treat bounds as zero-width

    Returns:
        ElementwiseParams encapsulating the relaxation parameters
    """
    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    # Determine regimes
    zero_width = torch.isclose(lower, upper, atol=zero_threshold)

    # Compute sqrt values
    lower_act = torch.sqrt(lower)
    upper_act = torch.sqrt(upper)

    def sqrt_derivative(x):
        # d/dx sqrt(x) = 1/(2*sqrt(x))
        # Handle zero case
        return torch.where(x > 0, 1.0 / (2.0 * torch.sqrt(x)), 0)

    # Midpoint for tangent line
    midpoint = (lower + upper) * 0.5
    midpoint_act = torch.sqrt(midpoint)
    midpoint_prime = sqrt_derivative(midpoint)

    # Slope of secant line
    slope = (upper_act - lower_act) / (upper - lower)

    # Zero-width case: use the value itself
    beta_lower[zero_width] = lower_act[zero_width]
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width cases
    non_zero = ~zero_width

    # sqrt is concave everywhere (for x > 0):
    # - Lower bound: secant line
    # - Upper bound: tangent line at midpoint

    # Lower bound: secant line
    alpha_lower[non_zero] = slope[non_zero]
    beta_lower[non_zero] = lower_act[non_zero] - slope[non_zero] * lower[non_zero]

    # Upper bound: tangent at midpoint
    alpha_upper[non_zero] = midpoint_prime[non_zero]
    beta_upper[non_zero] = midpoint_act[non_zero] - midpoint_prime[non_zero] * midpoint[non_zero]

    # Invalid regime: sqrt is undefined for negative inputs
    invalid = lower < 0
    alpha_lower[invalid] = float("nan")
    beta_lower[invalid] = float("nan")
    alpha_upper[invalid] = float("nan")
    beta_upper[invalid] = float("nan")

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_tan_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
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

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_tanh_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
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

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
