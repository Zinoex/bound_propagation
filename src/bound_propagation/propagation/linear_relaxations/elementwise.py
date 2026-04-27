"""Element-wise linear relaxations ``y = f(x)`` for unary activations.

Each ``compute_*_relaxation`` produces an :class:`ElementwiseParams` of slopes
and biases ``(α_lower, β_lower, α_upper, β_upper)`` such that, on the input
interval ``[l, u]``,

.. math::

    α_L \\, x + β_L \\;\\le\\; f(x) \\;\\le\\; α_U \\, x + β_U
    \\quad \\forall x \\in [l, u].

The relaxation is method-agnostic: IBP concretizes immediately, forward LBP
composes the slopes into running ``LinearBounds``, backward LBP attaches
them to the A-matrix recurrence in :mod:`backward_lbp`.

Citation table
--------------
- ReLU (relu): Zhang et al. 2018 "CROWN", Eq. 16. Lower slope ∈ [0, 1] is
  optimizable per α-CROWN; upper is always the secant.
- Sigmoid / tanh: auto-LiRPA §3.2 (Xu et al. 2020), tangent-line construction
  with a free tangent point per side on the inflection-crossing regime.
- Exp / log / sqrt / pow / reciprocal: standard convex/concave envelope
  reasoning — secant on the convex/concave side, tangent on the other; the
  tangent-point fraction is a free α-knob.
- Abs / clamp / ReLU crossing: piecewise-linear envelopes parameterized by
  the in-regime slope or corner-crossing slopes.
- Sin / cos / tan: regime detection (critical points, asymptotes for tan)
  followed by per-regime envelope; tangent-point fraction is a free α-knob
  on subregimes without critical points / asymptotes.

α-knob conventions
------------------
Every free knob is parameterized as ``α ∈ [0, 1]`` and the resolver
(:mod:`alpha_resolvers`) maps the fraction to the underlying geometric
quantity (slope, tangent point, eta). This keeps the optimizer's projection
trivial (clip to ``[0, 1]``); ops are responsible for the geometric mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final, overload

import torch

from ...bounds import IntervalBounds


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
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_abs_lower: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for abs linear relaxation.

    abs(x) is piecewise linear:
    - For x >= 0: abs(x) = x
    - For x < 0: abs(x) = -x

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_abs_lower : torch.Tensor | None
        Optional alpha-CROWN override for the lower-bound slope in the
        crossing (``l < 0 < u``) regime. Must be unit-interval fractions
        broadcastable to the shape of ``bounds.lower``. Each fraction
        ``alpha`` maps to a through-origin lower-bound slope
        ``m = 2 * alpha - 1 in [-1, 1]``. Values outside the crossing
        regime are ignored. Soundness: for ``l < 0 < u`` and any
        ``m in [-1, 1]``, the line ``y = m * x`` satisfies ``m*x <= |x|``
        pointwise (``m <= 1`` ensures soundness on ``x > 0``; ``m >= -1``
        ensures soundness on ``x < 0``). Note that the upper bound (secant)
        is already the tightest linear upper bound of a convex function
        and therefore has no optimization degree of freedom.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Determine regimes
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)
    all_positive = (bounds.lower >= 0) & ~zero_width
    all_negative = (bounds.upper <= 0) & ~zero_width
    crosses_zero = (bounds.lower < 0) & (bounds.upper > 0) & ~zero_width

    # Zero-width case: use the value itself
    lower_act = torch.abs(bounds.lower[zero_width])
    upper_act = torch.abs(bounds.upper[zero_width])
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
    lower, upper = bounds.lower[crosses_zero], bounds.upper[crosses_zero]

    lower_act = torch.abs(lower)
    upper_act = torch.abs(upper)

    slope = (upper_act - lower_act) / (upper - lower)

    alpha_upper[crosses_zero] = slope
    beta_upper[crosses_zero] = upper_act - slope * upper

    # Lower bound: through-origin line with slope in [-1, 1].
    if alpha_abs_lower is not None:
        # Map the [0, 1] fraction to a slope in [-1, 1].
        alpha_lower[crosses_zero] = 2.0 * alpha_abs_lower[crosses_zero] - 1.0
    else:
        alpha_lower[crosses_zero] = slope

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


@overload
def compute_clamp_relaxation(
    bounds: IntervalBounds,
    min_val: float | None,
    max_val: float | None,
    zero_threshold: float = 1e-8,
    *,
    alpha_clamp_crosses_min_lower: torch.Tensor | None = None,
    alpha_clamp_crosses_max_upper: torch.Tensor | None = None,
) -> ElementwiseParams: ...


@overload
def compute_clamp_relaxation(
    bounds: IntervalBounds,
    min_val: torch.Tensor | None,
    max_val: torch.Tensor | None,
    zero_threshold: float = 1e-8,
    *,
    alpha_clamp_crosses_min_lower: torch.Tensor | None = None,
    alpha_clamp_crosses_max_upper: torch.Tensor | None = None,
) -> ElementwiseParams: ...


def compute_clamp_relaxation(
    bounds: IntervalBounds,
    min_val: float | torch.Tensor | None = None,
    max_val: float | torch.Tensor | None = None,
    zero_threshold: float = 1e-8,
    *,
    alpha_clamp_crosses_min_lower: torch.Tensor | None = None,
    alpha_clamp_crosses_max_upper: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for clamp linear relaxation.

    clamp(x, min, max) = min(max(x, min), max)

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds of the pre-activation.
    min_val : float | torch.Tensor | None
        Minimum clamp value (``-inf`` if ``None``).
    max_val : float | torch.Tensor | None
        Maximum clamp value (``+inf`` if ``None``).
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_clamp_crosses_min_lower : torch.Tensor | None
        Optional alpha-CROWN override for the lower-bound slope on elements
        where the interval crosses ``min_val`` (but not ``max_val``). The
        fraction ``alpha in [0, 1]`` maps to slope ``a = alpha``, with the
        line written as ``y = a * (x - min_val) + min_val``. Default
        fraction ``0`` reproduces the current horizontal lower bound.
        Soundness: for ``l < min_val < u <= max_val`` and any
        ``a in [0, 1]``, ``y <= clamp(x)`` pointwise on ``[l, u]``.
    alpha_clamp_crosses_max_upper : torch.Tensor | None
        Optional alpha-CROWN override for the upper-bound slope on elements
        where the interval crosses ``max_val`` (but not ``min_val``). The
        fraction ``alpha in [0, 1]`` maps to slope ``a = alpha``, with the
        line written as ``y = a * (x - max_val) + max_val``. Default
        fraction ``0`` reproduces the current horizontal upper bound.
        Soundness: symmetric to the crosses_min lower-bound case.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """

    if max_val is None:
        lower_clamped = torch.clamp(bounds.lower, min=min_val)
        upper_clamped = torch.clamp(bounds.upper, min=min_val)
        max_val = float("inf")
    elif min_val is None:
        lower_clamped = torch.clamp(bounds.lower, max=max_val)
        upper_clamped = torch.clamp(bounds.upper, max=max_val)
        min_val = float("-inf")
    elif isinstance(min_val, torch.Tensor) and not isinstance(max_val, torch.Tensor):
        raise ValueError("If min_val is a tensor, max_val must be None or a tensor")
    elif isinstance(max_val, torch.Tensor) and not isinstance(min_val, torch.Tensor):
        raise ValueError("If max_val is a tensor, min_val must be None or a tensor")
    else:
        lower_clamped = torch.clamp(bounds.lower, min=min_val, max=max_val)  # ty:ignore[no-matching-overload]
        upper_clamped = torch.clamp(bounds.upper, min=min_val, max=max_val)  # ty:ignore[no-matching-overload]

    assert min_val is not None and max_val is not None

    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Determine regimes
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)
    not_zero_width = ~zero_width
    below_min = (bounds.upper <= min_val) & not_zero_width
    above_max = (bounds.lower >= max_val) & not_zero_width
    in_range = (bounds.lower >= min_val) & (bounds.upper <= max_val) & not_zero_width
    crosses_min = (bounds.lower < min_val) & (bounds.upper > min_val) & (bounds.upper <= max_val) & not_zero_width
    crosses_max = (bounds.lower >= min_val) & (bounds.lower < max_val) & (bounds.upper > max_val) & not_zero_width
    crosses_both = (bounds.lower < min_val) & (bounds.upper > max_val) & not_zero_width

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
    # Function has corner at (min_val, min_val), can't be tightly bounded by single line.
    # Lower bound: sound family y = a*(x - min_val) + min_val with slope a in [0, 1];
    # default a = 0 (horizontal at min_val). The upper bound stays at the current
    # horizontal default, which has no meaningful [0, 1] slope knob.
    lower_clamped_min, upper_clamped_min = lower_clamped[crosses_min], upper_clamped[crosses_min]

    min_val_crosses_min = min_val[crosses_min] if isinstance(min_val, torch.Tensor) else min_val
    if alpha_clamp_crosses_min_lower is not None:
        slope_lower_cm = alpha_clamp_crosses_min_lower[crosses_min]
        alpha_lower[crosses_min] = slope_lower_cm
        # y = a*(x - min_val) + min_val  =>  alpha = a, beta = (1 - a) * min_val.
        beta_lower[crosses_min] = (1.0 - slope_lower_cm) * min_val_crosses_min
    else:
        alpha_lower[crosses_min] = 0
        beta_lower[crosses_min] = min_val_crosses_min
    alpha_upper[crosses_min] = 0
    beta_upper[crosses_min] = torch.maximum(lower_clamped_min, upper_clamped_min)

    # Crosses max:
    # Function has corner at (max_val, max_val), can't be tightly bounded by single line.
    # Upper bound: sound family y = a*(x - max_val) + max_val with slope a in [0, 1];
    # default a = 0 (horizontal at max_val). The lower bound stays at its current default.
    lower_clamped_max, upper_clamped_max = lower_clamped[crosses_max], upper_clamped[crosses_max]

    alpha_lower[crosses_max] = 0
    beta_lower[crosses_max] = torch.minimum(lower_clamped_max, upper_clamped_max)

    max_val_crosses_max = max_val[crosses_max] if isinstance(max_val, torch.Tensor) else max_val
    if alpha_clamp_crosses_max_upper is not None:
        slope_upper_cmx = alpha_clamp_crosses_max_upper[crosses_max]
        alpha_upper[crosses_max] = slope_upper_cmx
        # y = a*(x - max_val) + max_val  =>  alpha = a, beta = (1 - a) * max_val.
        beta_upper[crosses_max] = (1.0 - slope_upper_cmx) * max_val_crosses_max
    else:
        alpha_upper[crosses_max] = 0
        beta_upper[crosses_max] = max_val_crosses_max

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
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_cos_tangent_frac: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for cos linear relaxation.

    cos is neither globally convex nor concave. Strategy:
    - Always use secant line connecting endpoints as one bound
    - Use tangent line at appropriate endpoint for the other bound
    - Determine which bound gets secant based on convexity of the interval

    For convex regions: tangent is lower bound, secant is upper bound
    For concave regions: secant is lower bound, tangent is upper bound

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_cos_tangent_frac : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point fraction, only
        active on the safe subregime (no extremum, no inflection). See
        :func:`compute_sin_relaxation` for the soundness argument. Ignored
        everywhere else.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Handle zero-width case
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)

    lower_act, upper_act = torch.cos(bounds.lower[zero_width]), torch.cos(bounds.upper[zero_width])
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = torch.min(lower_act, upper_act)
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = torch.max(lower_act, upper_act)

    non_zero = ~zero_width

    # Work with non-zero width intervals
    lower_nz = bounds.lower[non_zero]
    upper_nz = bounds.upper[non_zero]

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
    beta_nz_upper = torch.where(max_not_captured, 1, beta_nz_upper)

    # When has minimum (and not maximum), lower bound should be <= -1
    # For intervals with minimum, tangent lines often extend below -1, so use constant bounds
    only_min = has_minimum & ~has_maximum & ~has_both_extrema

    # For simplicity and correctness, use constant bounds when crossing minimum
    # A more sophisticated approach could use piecewise linear approximation
    alpha_nz_lower = torch.where(only_min, 0, alpha_nz_lower)
    beta_nz_lower = torch.where(only_min, -1, beta_nz_lower)
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

    # Alpha-CROWN override on the safe (no-extremum, no-inflection) subregime.
    if alpha_cos_tangent_frac is not None:
        alpha_nz = alpha_cos_tangent_frac[non_zero]
        d_opt = lower_nz + alpha_nz * (upper_nz - lower_nz)
        tangent_opt_slope = -torch.sin(d_opt)
        tangent_opt_beta = torch.cos(d_opt) - tangent_opt_slope * d_opt
        safe = ~has_maximum & ~has_minimum & ~has_inflection & ~has_both_extrema
        use_opt_upper = safe & when_secant_is_lower
        alpha_nz_upper = torch.where(use_opt_upper, tangent_opt_slope, alpha_nz_upper)
        beta_nz_upper = torch.where(use_opt_upper, tangent_opt_beta, beta_nz_upper)
        use_opt_lower = safe & ~when_secant_is_lower
        alpha_nz_lower = torch.where(use_opt_lower, tangent_opt_slope, alpha_nz_lower)
        beta_nz_lower = torch.where(use_opt_lower, tangent_opt_beta, beta_nz_lower)

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


def compute_exp_relaxation(
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_exp_tangent_lower: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for exp linear relaxation.

    exp(x) is globally convex, so the lower bound is a tangent at a free
    point ``d`` and the upper bound is the secant (already tightest).

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds of the pre-activation.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_exp_tangent_lower : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point fraction used
        as the lower bound. Maps ``alpha in [0, 1]`` to
        ``d = l + alpha * (u - l)``. Default ``0.5`` reproduces the current
        midpoint tangent. Soundness: ``exp`` is strictly convex everywhere,
        so the tangent at any ``d`` lies on or below the function
        pointwise. Thus every ``alpha in [0, 1]`` produces a sound lower
        bound on ``[l, u]``.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)
    width = bounds.upper - bounds.lower
    if alpha_exp_tangent_lower is not None:
        d_tangent = bounds.lower + alpha_exp_tangent_lower * width
    else:
        d_tangent = (bounds.lower + bounds.upper) / 2

    exp_lower = torch.exp(bounds.lower)
    exp_upper = torch.exp(bounds.upper)
    exp_d = torch.exp(d_tangent)

    alpha_lower = torch.where(zero_width, 0, exp_d)
    beta_lower = torch.where(zero_width, exp_lower, exp_d - alpha_lower * d_tangent)

    slope = (exp_upper - exp_lower) / (bounds.upper - bounds.lower)

    alpha_upper = torch.where(zero_width, 0, slope)
    beta_upper = torch.where(zero_width, exp_upper, exp_lower - slope * bounds.lower)

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_log_relaxation(
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_log_tangent_upper: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for log linear relaxation.

    log(x) is strictly concave on its domain (``x > 0``), so the upper bound
    is a tangent at a free point ``d`` and the lower bound is the secant.

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds of the pre-activation (must satisfy
        ``lower > 0`` for a valid relaxation).
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_log_tangent_upper : torch.Tensor | None
        Optional alpha-CROWN override for the upper-bound tangent-point
        fraction. Maps ``alpha in [0, 1]`` to ``d = l + alpha * (u - l)``.
        Default ``0.5`` reproduces the current midpoint tangent. Soundness:
        ``log`` is strictly concave, so the tangent at any ``d`` lies on
        or above the function pointwise. Every ``alpha in [0, 1]`` is
        sound on the valid domain.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    log_lower = torch.log(bounds.lower)
    log_upper = torch.log(bounds.upper)

    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)
    slope = (log_upper - log_lower) / (bounds.upper - bounds.lower)

    alpha_lower = torch.where(zero_width, 0, slope)
    beta_lower = torch.where(zero_width, log_lower, log_lower - slope * bounds.lower)

    width = bounds.upper - bounds.lower
    if alpha_log_tangent_upper is not None:
        d_tangent = bounds.lower + alpha_log_tangent_upper * width
    else:
        d_tangent = (bounds.lower + bounds.upper) / 2

    alpha_upper = torch.where(zero_width, 0, 1 / d_tangent)
    beta_upper = torch.where(zero_width, log_upper, torch.log(d_tangent) - alpha_upper * d_tangent)

    # Invalid regime: log is undefined for non-positive inputs, so we can set those bounds to nan
    invalid = bounds.lower <= 0
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
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_reciprocal_tangent_lower: torch.Tensor | None = None,
    alpha_reciprocal_tangent_upper: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for reciprocal (1/x) linear relaxation.

    reciprocal is strictly convex on ``x > 0`` and strictly concave on
    ``x < 0``. On each sign-homogeneous branch the tangent at a free
    point ``d`` provides a sound one-sided bound.

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_reciprocal_tangent_lower : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point fraction used
        as the lower bound on the all-positive branch (``l > 0``). Maps
        ``alpha in [0, 1]`` to ``d = l + alpha * (u - l)``. Default
        ``0.5`` reproduces the midpoint tangent. Soundness: tangents of a
        strictly convex function lie below it pointwise. Ignored in the
        all-negative, crossing, or zero-width regimes.
    alpha_reciprocal_tangent_upper : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point fraction used
        as the upper bound on the all-negative branch (``u < 0``). Default
        ``0.5``. Soundness is symmetric (tangents of a concave function
        lie above it). Ignored in other regimes.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Determine regimes
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)
    crosses_zero = (bounds.lower < 0) & (bounds.upper > 0)
    all_positive = bounds.lower > 0
    all_negative = bounds.upper < 0

    # Compute reciprocal values (avoid division by zero)
    eps = 1e-8
    lower_safe = torch.where(torch.abs(bounds.lower) < eps, eps * torch.sign(bounds.lower + eps), bounds.lower)
    upper_safe = torch.where(torch.abs(bounds.upper) < eps, eps * torch.sign(bounds.upper + eps), bounds.upper)

    lower_act = 1.0 / lower_safe
    upper_act = 1.0 / upper_safe

    def reciprocal_derivative(x):
        # d/dx (1/x) = -1/x^2
        x_safe = torch.where(torch.abs(x) < eps, eps * torch.sign(x + eps), x)
        return -1.0 / (x_safe * x_safe)

    # Separate tangent points for the all-positive (lower bound) and
    # all-negative (upper bound) branches so each can be optimized
    # independently. Midpoint default reproduces the prior behavior.
    width = bounds.upper - bounds.lower
    if alpha_reciprocal_tangent_lower is not None:
        d_lower = bounds.lower + alpha_reciprocal_tangent_lower * width
    else:
        d_lower = (bounds.lower + bounds.upper) * 0.5
    if alpha_reciprocal_tangent_upper is not None:
        d_upper = bounds.lower + alpha_reciprocal_tangent_upper * width
    else:
        d_upper = (bounds.lower + bounds.upper) * 0.5

    d_lower_safe = torch.where(torch.abs(d_lower) < eps, eps * torch.sign(d_lower + eps), d_lower)
    d_upper_safe = torch.where(torch.abs(d_upper) < eps, eps * torch.sign(d_upper + eps), d_upper)
    d_lower_act = 1.0 / d_lower_safe
    d_upper_act = 1.0 / d_upper_safe
    d_lower_prime = reciprocal_derivative(d_lower_safe)
    d_upper_prime = reciprocal_derivative(d_upper_safe)

    # Slope of secant line
    slope = torch.where(
        zero_width,
        torch.zeros_like(bounds.lower),
        (upper_act - lower_act) / torch.clamp(bounds.upper - bounds.lower, min=eps),
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

    # Case 2: All positive (x > 0): convex branch.
    # Upper bound: secant line. Lower bound: tangent at optimizable point.
    alpha_upper[all_positive] = slope[all_positive]
    beta_upper[all_positive] = upper_act[all_positive] - slope[all_positive] * upper_safe[all_positive]
    alpha_lower[all_positive] = d_lower_prime[all_positive]
    beta_lower[all_positive] = d_lower_act[all_positive] - d_lower_prime[all_positive] * d_lower_safe[all_positive]

    # Case 3: All negative (x < 0): concave branch.
    # Upper bound: tangent at optimizable point. Lower bound: secant line.
    alpha_upper[all_negative] = d_upper_prime[all_negative]
    beta_upper[all_negative] = d_upper_act[all_negative] - d_upper_prime[all_negative] * d_upper_safe[all_negative]
    alpha_lower[all_negative] = slope[all_negative]
    beta_lower[all_negative] = upper_act[all_negative] - slope[all_negative] * upper_safe[all_negative]

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_constant_div_relaxation(
    bounds: IntervalBounds,
    constant: torch.Tensor | torch.types.Number,
    zero_threshold: float = 1e-8,
) -> ElementwiseParams:
    """Compute linear-relaxation parameters for constant-over-bounds division ``constant / x``.

    The function ``f(x) = c / x`` is a scaled reciprocal.

    - For ``c > 0``, lower/upper relaxations keep the same orientation as reciprocal.
    - For ``c < 0``, multiplying by a negative constant flips inequalities, so lower/upper are swapped.
    - For intervals crossing zero, returns ``(-inf, inf)`` bounds element-wise.

    Args:
        bounds: IntervalBounds object containing lower and upper bounds of the denominator.
        constant: Numerator constant (scalar or tensor broadcastable to ``lower``).
        zero_threshold: Threshold used by reciprocal relaxation for zero-width handling.

    Returns:
        ElementwiseParams encapsulating the relaxation parameters.
    """
    if zero_threshold < 0:
        raise ValueError(f"zero_threshold must be non-negative, got {zero_threshold}")

    constant_tensor = torch.as_tensor(constant, dtype=bounds.lower.dtype, device=bounds.lower.device)
    try:
        constant_tensor = torch.broadcast_to(constant_tensor, bounds.lower.shape)
    except RuntimeError as error:
        constant_shape = tuple(constant_tensor.shape)
        raise ValueError(
            f"constant must be broadcastable to denominator bounds shape {bounds.lower.shape}, "
            f"got constant shape {constant_shape}"
        ) from error

    recip = compute_reciprocal_relaxation(bounds, zero_threshold=zero_threshold)

    positive_constant = constant_tensor > 0
    zero_constant = constant_tensor == 0
    crosses_zero = (bounds.lower < 0) & (bounds.upper > 0)

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
    bounds: IntervalBounds,
    adaptive: bool = False,
    zero_threshold: float = 1e-8,
    *,
    alpha_relu_lower: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for ReLU linear relaxation.

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds of the pre-activation.
    adaptive : bool
        When ``True`` and ``alpha_relu_lower`` is ``None``, pick the crossing
        slope adaptively (``1`` if ``|u| >= |l|`` else ``0``); otherwise the
        crossing slope defaults to ``z = u / (u - l)``.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_relu_lower : torch.Tensor | None
        Optional alpha-CROWN override for the lower-bound slope in the
        crossing regime. Must be a tensor of unit-interval fractions
        broadcastable to the shape of ``bounds.lower``. Each fraction
        ``alpha`` maps to crossing-regime lower slope ``a = alpha`` directly
        (since the valid slope range is ``[0, 1]``). Values in non-crossing
        elements are ignored. Soundness: for any ``l < 0 < u`` and any
        ``a in [0, 1]``, the line ``y = a * x`` satisfies ``a*x <= ReLU(x)``
        pointwise on ``[l, u]``.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Determine regimes
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)
    # negative = (~zero_width) & (upper <= 0)
    positive = (~zero_width) & (bounds.lower >= 0)
    crossing = (~zero_width) & (bounds.lower < 0) & (bounds.upper > 0)

    # Zero-width: use the value itself
    beta_lower[zero_width] = torch.relu(bounds.lower[zero_width])
    beta_upper[zero_width] = torch.relu(bounds.upper[zero_width])

    # Negative regime: output is always 0

    # Positive regime: output is identity
    alpha_lower[positive] = 1
    alpha_upper[positive] = 1

    # Crossing regime: use linear relaxation
    l_cross = bounds.lower[crossing]
    u_cross = bounds.upper[crossing]

    z = u_cross / (u_cross - l_cross)

    if alpha_relu_lower is not None:
        # Alpha-CROWN override: the lower slope is simply the fraction in [0, 1].
        a = alpha_relu_lower[crossing]
    elif adaptive:
        # Adaptive: choose slope based on which bound is tighter.
        a = (u_cross >= torch.abs(l_cross)).to(bounds.lower.dtype)
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


def compute_pow_relaxation(
    bounds: IntervalBounds,
    power: int,
    zero_threshold: float = 1e-8,
    *,
    alpha_pow_tangent: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for the linear relaxation of ``y = x ** power``.

    Currently supports ``power == 2`` only. ``y = x²`` is convex everywhere,
    so the standard relaxation is:

    - **Upper bound (chord)** through ``(l, l²)`` and ``(u, u²)``:
      ``y_upper = (l + u) * x - l*u``.
    - **Lower bound (tangent)** at ``t ∈ [l, u]``:
      ``y_lower = 2*t * x - t²``. Default ``t = (l + u) / 2``.

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds of the input.
    power : int
        Integer exponent. Must equal ``2``; other powers raise
        ``NotImplementedError``.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_pow_tangent : torch.Tensor | None
        Optional alpha-CROWN override for the tangent point. Each fraction
        ``alpha ∈ [0, 1]`` maps to ``t = l + alpha * (u - l)``. The default
        ``alpha = 1/2`` gives the centered tangent.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    if power != 2:
        raise NotImplementedError(f"compute_pow_relaxation currently only supports power=2, got power={power}")

    lower = bounds.lower
    upper = bounds.upper

    alpha_lower = torch.zeros_like(lower)
    beta_lower = torch.zeros_like(lower)
    alpha_upper = torch.zeros_like(lower)
    beta_upper = torch.zeros_like(lower)

    zero_width = torch.isclose(lower, upper, atol=zero_threshold)
    nontrivial = ~zero_width

    # Zero-width: both bounds equal x².
    sq_lower = lower[zero_width] ** 2
    beta_lower[zero_width] = sq_lower
    beta_upper[zero_width] = sq_lower

    # Upper bound: chord through endpoints.
    l_nt = lower[nontrivial]
    u_nt = upper[nontrivial]
    alpha_upper[nontrivial] = u_nt + l_nt
    beta_upper[nontrivial] = -l_nt * u_nt

    # Lower bound: tangent at t.
    if alpha_pow_tangent is not None:
        alpha_clamped = alpha_pow_tangent.clamp(0.0, 1.0)
        t = l_nt + alpha_clamped[nontrivial] * (u_nt - l_nt)
    else:
        t = (l_nt + u_nt) / 2
    alpha_lower[nontrivial] = 2 * t
    beta_lower[nontrivial] = -(t**2)

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_sigmoid_relaxation(
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_sigmoid_tangent_lower: torch.Tensor | None = None,
    alpha_sigmoid_tangent_upper: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for sigmoid linear relaxation.

    Sigmoid has an inflection point at x=0 (convex for x<0, concave for x>0).

    For intervals that cross zero:
    - Upper bound: Use secant if sigmoid'(upper) >= secant_slope, else tangent at upper
    - Lower bound: Use secant if sigmoid'(lower) >= secant_slope, else tangent at lower

    This ensures sound bounds while preferring simpler secant when valid.

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds of the pre-activation.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_sigmoid_tangent_lower : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point *fraction* used
        when sigmoid is convex (``u <= 0``, "negative-only" regime) and the
        tangent forms the lower bound. Maps ``alpha in [0, 1]`` to tangent
        abscissa ``d = l + alpha * (u - l) in [l, u]``. Default fraction
        ``0.5`` reproduces the current midpoint tangent. Soundness: on any
        strictly-convex subinterval, the tangent at any ``d in [l, u]``
        satisfies ``tangent(x) <= sigmoid(x)`` pointwise on ``[l, u]``.
        This override is ignored in non-negative-only regimes.
    alpha_sigmoid_tangent_upper : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point fraction used
        when sigmoid is concave (``l >= 0``, "positive-only" regime) and the
        tangent forms the upper bound. Same mapping and default as above.
        Soundness is symmetric: on concave subintervals, every tangent lies
        on or above the function. This override is ignored outside the
        positive-only regime (including crossing intervals, where the
        tangent logic is regime-dependent and not every ``d`` is safe).

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Determine regimes
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)

    # Compute sigmoid and derivative
    lower_act = torch.sigmoid(bounds.lower)
    upper_act = torch.sigmoid(bounds.upper)

    def sigmoid_derivative(x):
        s = torch.sigmoid(x)
        return s * (1 - s)

    # Optimizable tangent points (default: midpoint, reproducing prior behavior).
    width = bounds.upper - bounds.lower
    if alpha_sigmoid_tangent_lower is not None:
        d_lower = bounds.lower + alpha_sigmoid_tangent_lower * width
    else:
        d_lower = (bounds.lower + bounds.upper) * 0.5
    if alpha_sigmoid_tangent_upper is not None:
        d_upper = bounds.lower + alpha_sigmoid_tangent_upper * width
    else:
        d_upper = (bounds.lower + bounds.upper) * 0.5

    d_lower_act = torch.sigmoid(d_lower)
    d_lower_prime = sigmoid_derivative(d_lower)
    d_upper_act = torch.sigmoid(d_upper)
    d_upper_prime = sigmoid_derivative(d_upper)

    # Slope of secant line
    slope = torch.where(zero_width, 0, (upper_act - lower_act) / (bounds.upper - bounds.lower))

    # Zero-width case: use the value itself
    beta_lower[zero_width] = lower_act[zero_width]
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width cases
    non_zero = ~zero_width

    # Determine negative/positive regimes
    negative = non_zero & (bounds.upper <= 0)
    positive = non_zero & (bounds.lower >= 0)
    crossing = non_zero & (bounds.lower < 0) & (bounds.upper > 0)

    # Negative regime
    # Upper: secant line between lower and upper
    alpha_upper[negative] = slope[negative]
    beta_upper[negative] = upper_act[negative] - slope[negative] * bounds.upper[negative]

    # Lower: tangent line at (optimizable) tangent point.
    alpha_lower[negative] = d_lower_prime[negative]
    beta_lower[negative] = d_lower_act[negative] - d_lower_prime[negative] * d_lower[negative]

    # Positive regime
    # Upper: tangent at (optimizable) tangent point.
    alpha_upper[positive] = d_upper_prime[positive]
    beta_upper[positive] = d_upper_act[positive] - d_upper_prime[positive] * d_upper[positive]

    # Lower: secant line
    alpha_lower[positive] = slope[positive]
    beta_lower[positive] = lower_act[positive] - slope[positive] * bounds.lower[positive]

    # Crossing regime (contains both negative and positive)
    lower_prime = sigmoid_derivative(bounds.lower)
    upper_prime = sigmoid_derivative(bounds.upper)

    # Upper bound strategy:
    # Check if sigmoid'(upper) >= secant slope
    # If yes, secant is a valid upper bound (simpler)
    # Otherwise, use tangent at upper
    use_secant_upper = upper_prime[crossing] >= slope[crossing]

    # For secant case
    alpha_upper[crossing] = torch.where(use_secant_upper, slope[crossing], upper_prime[crossing])
    beta_upper[crossing] = torch.where(
        use_secant_upper,
        upper_act[crossing] - slope[crossing] * bounds.upper[crossing],
        upper_act[crossing] - upper_prime[crossing] * bounds.upper[crossing],
    )

    # Lower bound strategy:
    # Check if sigmoid'(lower) >= secant slope
    # If yes, secant is a valid lower bound
    # Otherwise, use tangent at lower
    use_secant_lower = lower_prime[crossing] >= slope[crossing]

    alpha_lower[crossing] = torch.where(use_secant_lower, slope[crossing], lower_prime[crossing])
    beta_lower[crossing] = torch.where(
        use_secant_lower,
        lower_act[crossing] - slope[crossing] * bounds.lower[crossing],
        lower_act[crossing] - lower_prime[crossing] * bounds.lower[crossing],
    )

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )


def compute_sin_relaxation(
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_sin_tangent_frac: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for sin linear relaxation.

    sin is neither globally convex nor concave. Strategy:
    - Always use secant line connecting endpoints as one bound
    - Use tangent line at appropriate endpoint for the other bound
    - Determine which bound gets secant based on convexity of the interval

    For convex regions: tangent is lower bound, secant is upper bound
    For concave regions: secant is lower bound, tangent is upper bound

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_sin_tangent_frac : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point fraction.
        Active only in the "safe" subregime where the interval contains
        neither a maximum, a minimum, nor an inflection point — i.e. a
        subinterval on which sin is strictly monotone and either strictly
        convex or strictly concave. In that regime, the tangent at any
        ``d = l + alpha * (u - l)`` is globally sound (below sin on convex
        subintervals, above sin on concave subintervals). Outside the safe
        regime the override is silently ignored and the analytical defaults
        (extrema-aware constant/secant fallbacks) are used.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Handle zero-width case
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)

    lower_act, upper_act = torch.sin(bounds.lower[zero_width]), torch.sin(bounds.upper[zero_width])
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = torch.min(lower_act, upper_act)
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = torch.max(lower_act, upper_act)

    non_zero = ~zero_width

    # Work with non-zero width intervals
    lower_nz = bounds.lower[non_zero]
    upper_nz = bounds.upper[non_zero]

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

    # Alpha-CROWN override: in the safe subregime (strictly monotone + single-convex/concave),
    # replace the endpoint-tangent heuristic with a tangent at an optimizable point.
    if alpha_sin_tangent_frac is not None:
        alpha_nz = alpha_sin_tangent_frac[non_zero]
        d_opt = lower_nz + alpha_nz * (upper_nz - lower_nz)
        tangent_opt_slope = torch.cos(d_opt)
        tangent_opt_beta = torch.sin(d_opt) - tangent_opt_slope * d_opt
        safe = ~has_maximum & ~has_minimum & ~has_inflection & ~has_both_extrema
        # Concave subinterval: secant is lower, tangent is upper.
        use_opt_upper = safe & when_secant_is_lower
        alpha_nz_upper = torch.where(use_opt_upper, tangent_opt_slope, alpha_nz_upper)
        beta_nz_upper = torch.where(use_opt_upper, tangent_opt_beta, beta_nz_upper)
        # Convex subinterval: secant is upper, tangent is lower.
        use_opt_lower = safe & ~when_secant_is_lower
        alpha_nz_lower = torch.where(use_opt_lower, tangent_opt_slope, alpha_nz_lower)
        beta_nz_lower = torch.where(use_opt_lower, tangent_opt_beta, beta_nz_lower)

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
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_sqrt_tangent_upper: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for sqrt linear relaxation.

    sqrt is strictly concave on ``x > 0``, so the upper bound is a tangent
    at a free point ``d`` and the lower bound is the secant.

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds (requires ``lower >= 0`` for validity).
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_sqrt_tangent_upper : torch.Tensor | None
        Optional alpha-CROWN override for the upper-bound tangent-point
        fraction. Maps ``alpha in [0, 1]`` to ``d = l + alpha * (u - l)``.
        Default ``0.5`` reproduces the current midpoint tangent. Soundness:
        tangents of a strictly concave function lie on or above the
        function pointwise; every ``alpha in [0, 1]`` is sound on the
        strictly-positive part of the interval. Elements where
        ``bounds.lower < 0`` remain NaN (undefined domain).

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Determine regimes
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)

    # Compute sqrt values
    lower_act = torch.sqrt(bounds.lower)
    upper_act = torch.sqrt(bounds.upper)

    def sqrt_derivative(x):
        # d/dx sqrt(x) = 1/(2*sqrt(x))
        # Handle zero case
        return torch.where(x > 0, 1.0 / (2.0 * torch.sqrt(x)), 0)

    width = bounds.upper - bounds.lower
    if alpha_sqrt_tangent_upper is not None:
        d_tangent = bounds.lower + alpha_sqrt_tangent_upper * width
    else:
        d_tangent = (bounds.lower + bounds.upper) * 0.5
    d_act = torch.sqrt(d_tangent)
    d_prime = sqrt_derivative(d_tangent)

    # Slope of secant line
    slope = (upper_act - lower_act) / (bounds.upper - bounds.lower)

    # Zero-width case: use the value itself
    beta_lower[zero_width] = lower_act[zero_width]
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width cases
    non_zero = ~zero_width

    # sqrt is concave everywhere (for x > 0):
    # - Lower bound: secant line
    # - Upper bound: tangent line at optimizable point

    # Lower bound: secant line
    alpha_lower[non_zero] = slope[non_zero]
    beta_lower[non_zero] = lower_act[non_zero] - slope[non_zero] * bounds.lower[non_zero]

    # Upper bound: tangent at optimizable point
    alpha_upper[non_zero] = d_prime[non_zero]
    beta_upper[non_zero] = d_act[non_zero] - d_prime[non_zero] * d_tangent[non_zero]

    # Invalid regime: sqrt is undefined for negative inputs
    invalid = bounds.lower < 0
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
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_tan_tangent_frac: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for tan linear relaxation.

    tan has asymptotes at x = π/2 + nπ and inflection points at x = nπ.
    - In (-π/2, 0): tan is concave (tan'' < 0)
    - In (0, π/2): tan is convex (tan'' > 0)

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_tan_tangent_frac : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point fraction. Only
        active in the safe subregimes: a strictly-convex half-branch
        (``[kπ, kπ + π/2)``) contributes the lower-bound tangent, and a
        strictly-concave half-branch (``(kπ - π/2, kπ]``) contributes the
        upper-bound tangent. Each is optimized independently per element.
        Ignored on asymptote-crossing, inflection-crossing (``crosses_zero``
        within a branch), and zero-width elements.

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Determine regimes
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)

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
    lower_asymptote_idx = torch.floor((bounds.lower - half_pi) / math.pi)
    upper_asymptote_idx = torch.floor((bounds.upper - half_pi) / math.pi)

    crosses_asymptote = lower_asymptote_idx != upper_asymptote_idx

    # For intervals crossing asymptotes, set infinite bounds
    beta_lower[crosses_asymptote] = float("-inf")
    beta_upper[crosses_asymptote] = float("inf")
    alpha_lower[crosses_asymptote] = 0.0
    alpha_upper[crosses_asymptote] = 0.0

    # Compute tan values for non-asymptote-crossing intervals
    valid = ~crosses_asymptote & ~zero_width
    lower_act = torch.tan(bounds.lower)
    upper_act = torch.tan(bounds.upper)

    def tan_derivative(x):
        # d/dx tan(x) = sec^2(x) = 1/cos^2(x)
        cos_x = torch.cos(x)
        return 1.0 / (cos_x * cos_x + 1e-8)

    lower_prime = tan_derivative(bounds.lower)
    upper_prime = tan_derivative(bounds.upper)

    # Slope of secant line
    slope = (upper_act - lower_act) / torch.clamp(bounds.upper - bounds.lower, min=1e-8)

    # Zero-width case: use the value itself; if it also crosses an asymptote, we already set to inf above
    beta_lower[zero_width & ~crosses_asymptote] = lower_act[zero_width & ~crosses_asymptote]
    beta_upper[zero_width & ~crosses_asymptote] = upper_act[zero_width & ~crosses_asymptote]

    # Determine convexity: tan is convex when tan(x) > 0, concave when tan(x) < 0
    # Equivalently: convex when x ∈ (nπ, nπ + π/2), concave when x ∈ (nπ - π/2, nπ)
    # We can check this by looking at sin(x): tan is convex when sin(bounds.lower) and sin(bounds.upper) > 0
    # Actually simpler: tan is convex when tan(midpoint) > 0, concave when < 0
    midpoint = (bounds.lower + bounds.upper) * 0.5
    midpoint_tan = torch.tan(midpoint)

    # But we need to handle crossing zero (inflection point) specially
    crosses_zero = (bounds.lower < 0) & (bounds.upper > 0) & valid

    # Optimizable tangent point (default: lower endpoint for convex, upper endpoint for concave).
    width = bounds.upper - bounds.lower
    if alpha_tan_tangent_frac is not None:
        d_opt = bounds.lower + alpha_tan_tangent_frac * width
        d_opt_act = torch.tan(d_opt)
        d_opt_prime = tan_derivative(d_opt)
    else:
        d_opt = bounds.lower  # unused; defaults kept in branches below
        d_opt_act = lower_act
        d_opt_prime = lower_prime

    # Convex regime: tan > 0 (e.g., x ∈ (0, π/2))
    # Use secant for upper bound, tangent for lower bound (at any d in [l, u]).
    convex = (midpoint_tan > zero_threshold) & valid & ~crosses_zero

    alpha_upper[convex] = slope[convex]
    beta_upper[convex] = upper_act[convex] - slope[convex] * bounds.upper[convex]

    if alpha_tan_tangent_frac is not None:
        alpha_lower[convex] = d_opt_prime[convex]
        beta_lower[convex] = d_opt_act[convex] - d_opt_prime[convex] * d_opt[convex]
    else:
        alpha_lower[convex] = lower_prime[convex]
        beta_lower[convex] = lower_act[convex] - lower_prime[convex] * bounds.lower[convex]

    # Concave regime: tan < 0 (e.g., x ∈ (-π/2, 0))
    # Use tangent for upper bound (at any d in [l, u]), secant for lower bound.
    concave = (midpoint_tan < -zero_threshold) & valid & ~crosses_zero

    if alpha_tan_tangent_frac is not None:
        alpha_upper[concave] = d_opt_prime[concave]
        beta_upper[concave] = d_opt_act[concave] - d_opt_prime[concave] * d_opt[concave]
    else:
        alpha_upper[concave] = upper_prime[concave]
        beta_upper[concave] = upper_act[concave] - upper_prime[concave] * bounds.upper[concave]

    alpha_lower[concave] = slope[concave]
    beta_lower[concave] = lower_act[concave] - slope[concave] * bounds.lower[concave]

    # Handle crossing zero (inflection point)
    # tan is concave left, convex right (opposite of sigmoid!)
    # For this type of S-curve, neither secant nor endpoint tangents work perfectly.
    # Solution: Use tangent slope at inflection point (x=0, slope=1) but adjust
    # intercepts to ensure the lines pass through or beyond the endpoints.

    # tan'(0) = 1
    inflection_slope = 1.0

    # For lower bound: y = 1*x + β_lower
    # Need: 1*lower + β_lower <= tan(lower) AND 1*upper + β_lower <= tan(upper)
    # => β_lower <= tan(lower) - lower AND β_lower <= tan(upper) - upper
    beta_lower_from_lower = lower_act[crosses_zero] - inflection_slope * bounds.lower[crosses_zero]
    beta_lower_from_upper = upper_act[crosses_zero] - inflection_slope * bounds.upper[crosses_zero]
    beta_lower_val = torch.minimum(beta_lower_from_lower, beta_lower_from_upper)

    # For upper bound: y = 1*x + β_upper
    # Need: 1*lower + β_upper >= tan(lower) AND 1*upper + β_upper >= tan(upper)
    # => β_upper >= tan(lower) - lower AND β_upper >= tan(upper) - upper
    beta_upper_from_lower = lower_act[crosses_zero] - inflection_slope * bounds.lower[crosses_zero]
    beta_upper_from_upper = upper_act[crosses_zero] - inflection_slope * bounds.upper[crosses_zero]
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
    bounds: IntervalBounds,
    zero_threshold: float = 1e-8,
    *,
    alpha_tanh_tangent_lower: torch.Tensor | None = None,
    alpha_tanh_tangent_upper: torch.Tensor | None = None,
) -> ElementwiseParams:
    """
    Compute alpha/beta parameters for tanh linear relaxation.

    Tanh has similar structure to sigmoid: inflection point at x=0 (convex for x<0, concave for x>0).

    For intervals that cross zero:
    - Upper bound: Use secant if tanh'(upper) >= secant_slope, else tangent at upper
    - Lower bound: Use secant if tanh'(lower) >= secant_slope, else tangent at lower

    This ensures sound bounds while preferring simpler secant when valid.

    Parameters
    ----------
    bounds : IntervalBounds
        Lower and upper bounds of the pre-activation.
    zero_threshold : float
        Threshold below which an interval is treated as zero-width.
    alpha_tanh_tangent_lower : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point fraction in the
        negative-only (``u <= 0``, strictly convex) regime, where the
        tangent forms the lower bound. Maps ``alpha in [0, 1]`` to
        ``d = l + alpha * (u - l)``. Default ``0.5`` reproduces midpoint
        tangent. Soundness: every tangent of a strictly convex function
        lies on or below it. Ignored outside the negative-only regime.
    alpha_tanh_tangent_upper : torch.Tensor | None
        Optional alpha-CROWN override for the tangent-point fraction in
        the positive-only (``l >= 0``, strictly concave) regime, where the
        tangent forms the upper bound. Same mapping and soundness argument.
        Ignored outside the positive-only regime (including crossing).

    Returns
    -------
    ElementwiseParams
        The relaxation parameters.
    """
    alpha_lower = torch.zeros_like(bounds.lower)
    beta_lower = torch.zeros_like(bounds.lower)
    alpha_upper = torch.zeros_like(bounds.lower)
    beta_upper = torch.zeros_like(bounds.lower)

    # Determine regimes
    zero_width = torch.isclose(bounds.lower, bounds.upper, atol=zero_threshold)

    # Compute tanh and derivative
    lower_act = torch.tanh(bounds.lower)
    upper_act = torch.tanh(bounds.upper)

    def tanh_derivative(x):
        t = torch.tanh(x)
        return 1 - t * t

    lower_prime = tanh_derivative(bounds.lower)
    upper_prime = tanh_derivative(bounds.upper)

    # Optimizable tangent points (default: midpoint reproduces prior behavior).
    width = bounds.upper - bounds.lower
    if alpha_tanh_tangent_lower is not None:
        d_lower = bounds.lower + alpha_tanh_tangent_lower * width
    else:
        d_lower = (bounds.lower + bounds.upper) * 0.5
    if alpha_tanh_tangent_upper is not None:
        d_upper = bounds.lower + alpha_tanh_tangent_upper * width
    else:
        d_upper = (bounds.lower + bounds.upper) * 0.5

    d_lower_act = torch.tanh(d_lower)
    d_lower_prime = tanh_derivative(d_lower)
    d_upper_act = torch.tanh(d_upper)
    d_upper_prime = tanh_derivative(d_upper)

    # Slope of secant line
    slope = torch.where(
        zero_width, torch.zeros_like(bounds.lower), (upper_act - lower_act) / (bounds.upper - bounds.lower)
    )

    # Zero-width case
    alpha_lower[zero_width] = 0
    beta_lower[zero_width] = lower_act[zero_width]
    alpha_upper[zero_width] = 0
    beta_upper[zero_width] = upper_act[zero_width]

    # Non-zero width
    non_zero = ~zero_width

    negative = non_zero & (bounds.upper <= 0)
    positive = non_zero & (bounds.lower >= 0)
    crossing = non_zero & (bounds.lower < 0) & (bounds.upper > 0)

    # Negative regime
    if negative.any():
        # Upper: secant line between lower and upper
        alpha_upper[negative] = slope[negative]
        beta_upper[negative] = upper_act[negative] - slope[negative] * bounds.upper[negative]

        # Lower: tangent line at (optimizable) tangent point.
        alpha_lower[negative] = d_lower_prime[negative]
        beta_lower[negative] = d_lower_act[negative] - d_lower_prime[negative] * d_lower[negative]

    # Positive regime
    if positive.any():
        # Upper: tangent at (optimizable) tangent point.
        alpha_upper[positive] = d_upper_prime[positive]
        beta_upper[positive] = d_upper_act[positive] - d_upper_prime[positive] * d_upper[positive]

        # Lower: secant line
        alpha_lower[positive] = slope[positive]
        beta_lower[positive] = lower_act[positive] - slope[positive] * bounds.lower[positive]

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
            upper_act[crossing] - slope[crossing] * bounds.upper[crossing],
            upper_act[crossing] - upper_prime[crossing] * bounds.upper[crossing],
        )

        # Lower bound strategy:
        # Check if tanh'(lower) >= secant slope
        # If yes, secant is a valid lower bound
        # Otherwise, use tangent at lower
        use_secant_lower = lower_prime[crossing] >= slope[crossing]

        alpha_lower[crossing] = torch.where(use_secant_lower, slope[crossing], lower_prime[crossing])
        beta_lower[crossing] = torch.where(
            use_secant_lower,
            lower_act[crossing] - slope[crossing] * bounds.lower[crossing],
            lower_act[crossing] - lower_prime[crossing] * bounds.lower[crossing],
        )

    return ElementwiseParams(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
