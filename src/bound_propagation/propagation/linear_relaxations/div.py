from __future__ import annotations

import torch

from .base import PairedLinearRelaxation
from .reciprocal import compute_reciprocal_relaxation


def compute_div_relaxation(
    lower_a: torch.Tensor,
    upper_a: torch.Tensor,
    lower_b: torch.Tensor,
    upper_b: torch.Tensor,
    zero_threshold: float = 1e-8,
    eta_lower: torch.Tensor | torch.types.Number = 0.5,
    eta_upper: torch.Tensor | torch.types.Number = 0.5,
) -> PairedLinearRelaxation:
    """
    Compute element-wise linear relaxation for z = a / b given bounds on a and b.

    Decomposes as z = a * (1/b):
      1. Compute a linear relaxation of w = 1/b as a function of b.
      2. Compute concrete bounds on w (1/x is decreasing, so [1/ub, 1/lb]).
      3. Apply a McCormick relaxation for z = a * w using concrete bounds on a and w.
      4. Substitute the linear relaxation of w back to express z linearly in a and b.

    Elements where b's domain crosses zero are set to [-inf, +inf] (undefined).
    Those elements are temporarily replaced with dummy bounds [1, 2] to prevent
    NaN from poisoning the computation for the remaining elements.

    Args:
        lower_a: Lower bounds of numerator
        upper_a: Upper bounds of numerator
        lower_b: Lower bounds of denominator
        upper_b: Upper bounds of denominator
        zero_threshold: Passed through to compute_reciprocal_relaxation
        eta_lower: The convex combination parameter for the McCormick lower bound
        eta_upper: The convex combination parameter for the McCormick upper bound

    Returns:
        PairedLinearRelaxation encapsulating z = a / b with inputs ordered [a, b].
    """
    crosses_zero = (lower_b <= 0) & (upper_b >= 0)

    # Replace zero-crossing elements with safe dummy bounds [1, 2] to avoid NaN
    safe_lower_b = torch.where(crosses_zero, torch.ones_like(lower_b), lower_b)
    safe_upper_b = torch.where(crosses_zero, 2 * torch.ones_like(upper_b), upper_b)

    # Step 1: Linear relaxation for w = 1/b
    # alpha_lower/upper and beta_lower/upper satisfy:
    #   alpha_lower * b + beta_lower <= w <= alpha_upper * b + beta_upper
    recip = compute_reciprocal_relaxation(safe_lower_b, safe_upper_b, zero_threshold)

    # Step 2: Concrete bounds on w = 1/b
    # 1/x is strictly decreasing on each sign, so lower_w = 1/ub and upper_w = 1/lb.
    # safe_lower_b and safe_upper_b are guaranteed non-zero (either original non-crossing
    # values or the dummy [1, 2]), so no additional eps guard is needed.
    lower_w = 1.0 / safe_upper_b
    upper_w = 1.0 / safe_lower_b

    # Step 3: McCormick relaxation for z = a * w
    la, ua = lower_a, upper_a
    lw, uw = lower_w, upper_w

    # Lower bound candidates (combined via convex combination with parameter eta_lower):
    #   Candidate 1: z >= lw * a + la * w - la * lw
    #   Candidate 2: z >= uw * a + ua * w - ua * uw

    coeff_a_lower = lw * eta_lower + uw * (1 - eta_lower)
    coeff_w_lower = la * eta_lower + ua * (1 - eta_lower)
    bias_lower_mcc = -la * lw * eta_lower - ua * uw * (1 - eta_lower)

    # Upper bound candidates (combined via convex combination with parameter eta_upper):
    #   Candidate 1: z <= lw * a + ua * w - ua * lw
    #   Candidate 2: z <= uw * a + la * w - la * uw

    coeff_a_upper = lw * eta_upper + uw * (1 - eta_upper)
    coeff_w_upper = ua * eta_upper + la * (1 - eta_upper)
    bias_upper_mcc = -ua * lw * eta_upper - la * uw * (1 - eta_upper)

    # Step 4: Substitute w with its linear relaxation in terms of b.
    # For the lower bound z >= coeff_a_lower * a + coeff_w_lower * w + bias_lower_mcc:
    #   coeff_w_lower >= 0 -> use lower relaxation of w (alpha_lower * b + beta_lower)
    #   coeff_w_lower <  0 -> use upper relaxation of w (alpha_upper * b + beta_upper)
    cwl_pos = coeff_w_lower.clamp(min=0)
    cwl_neg = coeff_w_lower.clamp(max=0)
    coeff_b_lower = cwl_pos * recip.alpha_lower + cwl_neg * recip.alpha_upper
    bias_lower = bias_lower_mcc + cwl_pos * recip.beta_lower + cwl_neg * recip.beta_upper

    # For the upper bound z <= coeff_a_upper * a + coeff_w_upper * w + bias_upper_mcc:
    #   coeff_w_upper >= 0 -> use upper relaxation of w (alpha_upper * b + beta_upper)
    #   coeff_w_upper <  0 -> use lower relaxation of w (alpha_lower * b + beta_lower)
    cwu_pos = coeff_w_upper.clamp(min=0)
    cwu_neg = coeff_w_upper.clamp(max=0)
    coeff_b_upper = cwu_pos * recip.alpha_upper + cwu_neg * recip.alpha_lower
    bias_upper = bias_upper_mcc + cwu_pos * recip.beta_upper + cwu_neg * recip.beta_lower

    # Override zero-crossing elements: output is [-inf, +inf]
    coeff_a_lower = torch.where(crosses_zero, 0, coeff_a_lower)
    coeff_b_lower = torch.where(crosses_zero, 0, coeff_b_lower)
    bias_lower = torch.where(crosses_zero, float("-inf"), bias_lower)

    coeff_a_upper = torch.where(crosses_zero, 0, coeff_a_upper)
    coeff_b_upper = torch.where(crosses_zero, 0, coeff_b_upper)
    bias_upper = torch.where(crosses_zero, float("inf"), bias_upper)

    return PairedLinearRelaxation(
        coeffs_lower=[coeff_a_lower, coeff_b_lower],
        coeffs_upper=[coeff_a_upper, coeff_b_upper],
        bias_lower=bias_lower,
        bias_upper=bias_upper,
    )
