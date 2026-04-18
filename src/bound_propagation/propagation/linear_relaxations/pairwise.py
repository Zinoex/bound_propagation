from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from .elementwise import compute_reciprocal_relaxation


@final
@dataclass
class PairedParams:
    """
    Parameters for pairwise linear relaxations (e.g., multiplication, division, max, min).
    For a binary operation z = f(a, b), the linear relaxation has the form:
        lower_bound: z >= alpha_lower_a * a + alpha_lower_b * b + bias_lower
        upper_bound: z <= alpha_upper_a * a + alpha_upper_b * b + bias_upper

    The abstract dimension convention for these LinearBounds linear terms are
    (*batch_dims, *output_dims, *input_dims) since they are pairwise.
    Therefore, alpha and beta live in (*batch_dims, *output_dims).

    Attributes:
        alpha_lower_a: Coefficient for a in the lower bound.
        alpha_upper_a: Coefficient for a in the upper bound.
        alpha_lower_b: Coefficient for b in the lower bound.
        alpha_upper_b: Coefficient for b in the upper bound.
        bias_lower: Bias term in the lower bound.
        bias_upper: Bias term in the upper bound.
    """

    alpha_lower_a: torch.Tensor
    alpha_upper_a: torch.Tensor
    alpha_lower_b: torch.Tensor
    alpha_upper_b: torch.Tensor
    bias_lower: torch.Tensor
    bias_upper: torch.Tensor


def compute_mul_relaxation(
    lower_a: torch.Tensor,
    upper_a: torch.Tensor,
    lower_b: torch.Tensor,
    upper_b: torch.Tensor,
    eta_lower: torch.Tensor | torch.types.Number = 0.5,
    eta_upper: torch.Tensor | torch.types.Number = 0.5,
) -> PairedParams:
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
        A PairedParams object representing the linear relaxation of z = a * b.
    """

    alpha1_lower = lower_b * eta_lower + upper_b * (1 - eta_lower)  # coeff of a
    alpha2_lower = lower_a * eta_lower + upper_a * (1 - eta_lower)  # coeff of b
    bias_lower = -lower_a * lower_b * eta_lower - upper_a * upper_b * (1 - eta_lower)

    alpha1_upper = upper_b * eta_upper + lower_b * (1 - eta_upper)  # coeff of a
    alpha2_upper = lower_a * eta_upper + upper_a * (1 - eta_upper)  # coeff of b
    bias_upper = -lower_a * upper_b * eta_upper - upper_a * lower_b * (1 - eta_upper)

    relaxation = PairedParams(
        alpha_lower_a=alpha1_lower,
        alpha_upper_a=alpha1_upper,
        alpha_lower_b=alpha2_lower,
        alpha_upper_b=alpha2_upper,
        bias_lower=bias_lower,
        bias_upper=bias_upper,
    )
    return relaxation


def compute_div_relaxation(
    lower_a: torch.Tensor,
    upper_a: torch.Tensor,
    lower_b: torch.Tensor,
    upper_b: torch.Tensor,
    zero_threshold: float = 1e-8,
    eta_lower: torch.Tensor | torch.types.Number = 0.5,
    eta_upper: torch.Tensor | torch.types.Number = 0.5,
) -> PairedParams:
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
        PairedParams encapsulating z = a / b with inputs ordered [a, b].
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

    return PairedParams(
        alpha_lower_a=coeff_a_lower,
        alpha_upper_a=coeff_a_upper,
        alpha_lower_b=coeff_b_lower,
        alpha_upper_b=coeff_b_upper,
        bias_lower=bias_lower,
        bias_upper=bias_upper,
    )


def compute_maximum_relaxation(
    lower_a: torch.Tensor,
    upper_a: torch.Tensor,
    lower_b: torch.Tensor,
    upper_b: torch.Tensor,
    eta_lower: torch.Tensor | torch.types.Number = 0.5,
    eta_upper: torch.Tensor | torch.types.Number = 0.5,
) -> PairedParams:
    """Compute a linear relaxation for element-wise max(a, b).

    Three regimes (element-wise):
      - a dominates (lower_a >= upper_b): max = a
      - b dominates (lower_b >= upper_a): max = b
      - crossing: linear relaxation

    Lower bound (valid because max(a,b) >= a and max(a,b) >= b):
      z >= eta * a + (1-eta) * b

    Upper bound (concave overestimator of the convex max function):
      z <= lambda * a + (1-lambda) * b + delta
      delta = max((1-lambda)*(upper_a - lower_b), lambda*(upper_b - lower_a))

    Parameters
    ----------
    lower_a, upper_a : torch.Tensor
        Concrete bounds on the first input.
    lower_b, upper_b : torch.Tensor
        Concrete bounds on the second input.
    eta_lower : torch.Tensor or scalar
        Interpolation parameter for the lower bound (0 = use b, 1 = use a).
    eta_upper : torch.Tensor or scalar
        Interpolation parameter for the upper bound plane tilt.
    """
    a_dominates = lower_a >= upper_b
    b_dominates = lower_b >= upper_a
    crossing = ~a_dominates & ~b_dominates

    ones = torch.ones_like(lower_a)
    zeros = torch.zeros_like(lower_a)

    # Lower bound coefficients
    eta_l_val = torch.as_tensor(eta_lower, dtype=lower_a.dtype, device=lower_a.device).expand_as(lower_a)
    eta_l = torch.where(a_dominates, ones, torch.where(b_dominates, zeros, eta_l_val))
    alpha1_lower = eta_l
    alpha2_lower = 1 - eta_l
    bias_lower = zeros

    # Upper bound coefficients
    eta_u_val = torch.as_tensor(eta_upper, dtype=lower_a.dtype, device=lower_a.device).expand_as(lower_a)
    eta_u = torch.where(a_dominates, ones, torch.where(b_dominates, zeros, eta_u_val))
    alpha1_upper = eta_u
    alpha2_upper = 1 - eta_u
    # delta = max((1-lambda)*(ua - lb), lambda*(ub - la))
    delta_opt1 = (1 - eta_u) * (upper_a - lower_b).clamp(min=0)
    delta_opt2 = eta_u * (upper_b - lower_a).clamp(min=0)
    bias_upper = torch.where(crossing, torch.maximum(delta_opt1, delta_opt2), zeros)

    return PairedParams(
        alpha_lower_a=alpha1_lower,
        alpha_upper_a=alpha1_upper,
        alpha_lower_b=alpha2_lower,
        alpha_upper_b=alpha2_upper,
        bias_lower=bias_lower,
        bias_upper=bias_upper,
    )


def compute_minimum_relaxation(
    lower_a: torch.Tensor,
    upper_a: torch.Tensor,
    lower_b: torch.Tensor,
    upper_b: torch.Tensor,
    eta_lower: torch.Tensor | torch.types.Number = 0.5,
    eta_upper: torch.Tensor | torch.types.Number = 0.5,
) -> PairedParams:
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

    return PairedParams(
        alpha_lower_a=alpha1_lower,
        alpha_upper_a=alpha1_upper,
        alpha_lower_b=alpha2_lower,
        alpha_upper_b=alpha2_upper,
        bias_lower=bias_lower,
        bias_upper=bias_upper,
    )
