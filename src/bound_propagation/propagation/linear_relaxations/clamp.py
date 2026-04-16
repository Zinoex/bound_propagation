from typing import overload

import torch

from .base import ElementwiseLinearRelaxation


@overload
def compute_clamp_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_val: float | None,
    max_val: float | None,
    zero_threshold: float = 1e-8,
) -> ElementwiseLinearRelaxation: ...


@overload
def compute_clamp_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_val: torch.Tensor | None,
    max_val: torch.Tensor | None,
    zero_threshold: float = 1e-8,
) -> ElementwiseLinearRelaxation: ...


def compute_clamp_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_val: float | torch.Tensor | None = None,
    max_val: float | torch.Tensor | None = None,
    zero_threshold: float = 1e-8,
) -> ElementwiseLinearRelaxation:
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
        ElementwiseLinearRelaxation encapsulating the relaxation
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

    return ElementwiseLinearRelaxation(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
