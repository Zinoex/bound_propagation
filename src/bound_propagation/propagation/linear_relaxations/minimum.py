from __future__ import annotations

import torch

from .base import PairedLinearRelaxation


def compute_minimum_relaxation(
    lower_a: torch.Tensor,
    upper_a: torch.Tensor,
    lower_b: torch.Tensor,
    upper_b: torch.Tensor,
    eta_lower: torch.Tensor | torch.types.Number = 0.5,
    eta_upper: torch.Tensor | torch.types.Number = 0.5,
) -> PairedLinearRelaxation:
    """Compute a linear relaxation for element-wise min(a, b).

    Three regimes (element-wise):
      - a dominates (upper_a <= lower_b): min = a
      - b dominates (upper_b <= lower_a): min = b
      - crossing: linear relaxation

    Upper bound (valid because min(a,b) <= a and min(a,b) <= b):
      z <= eta * a + (1-eta) * b

    Lower bound (convex underestimator of the concave min function):
      z >= lambda * a + (1-lambda) * b - delta
      delta = max((1-lambda)*(upper_b - lower_a), lambda*(upper_a - lower_b))

    Parameters
    ----------
    lower_a, upper_a : torch.Tensor
        Concrete bounds on the first input.
    lower_b, upper_b : torch.Tensor
        Concrete bounds on the second input.
    eta_lower : torch.Tensor or scalar
        Interpolation parameter for the lower bound plane tilt.
    eta_upper : torch.Tensor or scalar
        Interpolation parameter for the upper bound (0 = use b, 1 = use a).
    """
    a_dominates = upper_a <= lower_b
    b_dominates = upper_b <= lower_a
    crossing = ~a_dominates & ~b_dominates

    ones = torch.ones_like(lower_a)
    zeros = torch.zeros_like(lower_a)

    # Lower bound coefficients
    eta_l_val = torch.as_tensor(eta_lower, dtype=lower_a.dtype, device=lower_a.device).expand_as(lower_a)
    eta_l = torch.where(a_dominates, ones, torch.where(b_dominates, zeros, eta_l_val))
    alpha1_lower = eta_l
    alpha2_lower = 1 - eta_l
    # delta = max((1-lambda)*(ub - la), lambda*(ua - lb))
    delta_opt1 = (1 - eta_l) * (upper_b - lower_a).clamp(min=0)
    delta_opt2 = eta_l * (upper_a - lower_b).clamp(min=0)
    bias_lower = torch.where(crossing, -torch.maximum(delta_opt1, delta_opt2), zeros)

    # Upper bound coefficients
    eta_u_val = torch.as_tensor(eta_upper, dtype=lower_a.dtype, device=lower_a.device).expand_as(lower_a)
    eta_u = torch.where(a_dominates, ones, torch.where(b_dominates, zeros, eta_u_val))
    alpha1_upper = eta_u
    alpha2_upper = 1 - eta_u
    bias_upper = zeros

    return PairedLinearRelaxation(
        coeffs_lower=[alpha1_lower, alpha2_lower],
        coeffs_upper=[alpha1_upper, alpha2_upper],
        bias_lower=bias_lower,
        bias_upper=bias_upper,
    )
