from __future__ import annotations

import torch
from plum import dispatch

from ..bounds import IntervalBounds, LinearBounds
from ..regions import HyperRectangle


@dispatch
def concretize(region: HyperRectangle, bounds: IntervalBounds) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concretize interval bounds given a hyperrectangle region.

    For interval bounds, simply returns the bounds as they are already concrete.

    Args:
        region: The hyperrectangle input region
        bounds: The interval bounds to concretize
    Returns:
        Tuple of (lower, upper) concrete bounds
    """
    return bounds.lower, bounds.upper



@dispatch
def concretize(region: HyperRectangle, bounds: LinearBounds) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: F811
    """
    Concretize linear bounds given a hyperrectangle region.

    For linear bounds, evaluates the affine functions at the box extremes:
    - Lower bound: minimize W_l @ x + b_l over x in [lower, upper]
    - Upper bound: maximize W_u @ x + b_u over x in [lower, upper]

    For each weight coefficient:
    - Use region.lower if coefficient > 0 (for minimization)
    - Use region.upper if coefficient < 0 (for minimization)
    - Vice versa for maximization

    Args:
        region: The hyperrectangle input region
        bounds: The linear bounds to concretize

    Returns:
        Tuple of (lower, upper) concrete bounds
    """
    # Flatten the hyperrectangle bounds for easier computation
    input_lower = region.lower.flatten()
    input_upper = region.upper.flatten()

    # Lower bound computation: minimize W_l @ x + b_l
    lower_result = bounds.bias_lower.clone()
    if bounds.linear_lower is not None:
        positive_mask = bounds.linear_lower > 0
        contributions = torch.where(
            positive_mask,
            bounds.linear_lower * input_lower,
            bounds.linear_lower * input_upper,
        )
        lower_result = lower_result + contributions.sum(dim=-1)

    # Upper bound computation: maximize W_u @ x + b_u
    upper_result = bounds.bias_upper.clone()
    if bounds.linear_upper is not None:
        positive_mask = bounds.linear_upper > 0
        contributions = torch.where(
            positive_mask,
            bounds.linear_upper * input_upper,
            bounds.linear_upper * input_lower,
        )
        upper_result = upper_result + contributions.sum(dim=-1)

    return lower_result, upper_result
