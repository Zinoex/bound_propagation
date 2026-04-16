from __future__ import annotations

import torch

from .base import PairedLinearRelaxation


def compute_mul_relaxation(
    lower_a: torch.Tensor,
    upper_a: torch.Tensor,
    lower_b: torch.Tensor,
    upper_b: torch.Tensor,
    eta_lower: torch.Tensor | torch.types.Number = 0.5,
    eta_upper: torch.Tensor | torch.types.Number = 0.5,
) -> PairedLinearRelaxation:
    """
    Compute a linear relaxation for the element-wise multiplication of two variables with given bounds.

    The relaxation is based on the McCormick envelopes, which provide valid linear bounds for bilinear terms.
    The `eta_lower` and `eta_upper` parameters control the convex combination of the two McCormick bounds.

    Arguments:
        lower_a: Lower bound of variable a.
        upper_a: Upper bound of variable a.
        lower_b: Lower bound of variable b.
        upper_b: Upper bound of variable b.
        eta_lower: Convex combination parameter for the lower bound (default 0.5).
        eta_upper: Convex combination parameter for the upper bound (default 0.5).
    Returns:
        A PairedLinearRelaxation object representing the linear relaxation of z = a * b.
    """

    alpha1_lower = lower_b * eta_lower + upper_b * (1 - eta_lower)  # coeff of a
    alpha2_lower = lower_a * eta_lower + upper_a * (1 - eta_lower)  # coeff of b
    bias_lower = -lower_a * lower_b * eta_lower - upper_a * upper_b * (1 - eta_lower)

    alpha1_upper = upper_b * eta_upper + lower_b * (1 - eta_upper)  # coeff of a
    alpha2_upper = lower_a * eta_upper + upper_a * (1 - eta_upper)  # coeff of b
    bias_upper = -lower_a * upper_b * eta_upper - upper_a * lower_b * (1 - eta_upper)

    relaxation = PairedLinearRelaxation(
        coeffs_lower=[alpha1_lower, alpha2_lower],
        coeffs_upper=[alpha1_upper, alpha2_upper],
        bias_lower=bias_lower,
        bias_upper=bias_upper,
    )
    return relaxation
