from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from ...bounds import LinearBounds
from ...regions import SimpleRegion
from .base import AbstractLinearRelaxation, SymbolicLinearRelaxation
from .elementwise import compute_reciprocal_relaxation


@final
@dataclass
class PairedLinearRelaxation(AbstractLinearRelaxation):
    """
    Paired linear relaxation for binary operations z = f(x1, x2).

    Represents:
        z_lower >= alpha1_lower * x1 + alpha2_lower * x2 + beta_lower
        z_upper <= alpha1_upper * x1 + alpha2_upper * x2 + beta_upper

    The abstract dimension convention for linear terms is
    (*batch_dims, *output_dims, *input_dims).  Each element of coeffs_lower /
    coeffs_upper lives in (*batch_dims, *output_dims); the corresponding input's
    trailing axes are broadcast in forward.

    Attributes:
        coeffs_lower: List of element-wise coefficient tensors, one per input,
                      giving the lower-bound relaxation slopes.
        coeffs_upper: Same structure for the upper-bound relaxation slopes.
        bias_lower:   Element-wise bias for the lower bound.
        bias_upper:   Element-wise bias for the upper bound.
    """

    coeffs_lower: list[torch.Tensor]
    coeffs_upper: list[torch.Tensor]
    bias_lower: torch.Tensor
    bias_upper: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.coeffs_lower) != len(self.coeffs_upper):
            raise ValueError(
                f"coeffs_lower and coeffs_upper must have the same length, "
                f"got {len(self.coeffs_lower)} vs {len(self.coeffs_upper)}"
            )
        if len(self.coeffs_lower) != 2:
            raise ValueError("PairedLinearRelaxation requires exactly 2 input coefficients")

    # ------------------------------------------------------------------
    # Forward composition
    # ------------------------------------------------------------------

    def forward(self, input_bounds: list[LinearBounds]) -> LinearBounds:
        """
        Compose z = sum_i(alpha_i * x_i) + beta with linear bounds on each x_i.

        Linear terms have shape (*batch_dims, *output_dims, *input_dims).
        Contributions from all inputs are merged by input_id so that shared regions
        are accumulated correctly.
        """
        if len(input_bounds) != 2:
            raise ValueError(f"PairedLinearRelaxation expects 2 input bounds, got {len(input_bounds)}")

        def broadcast(alpha: torch.Tensor, linear: torch.Tensor) -> torch.Tensor:
            # alpha: (*batch_dims, *output_dims)
            # linear: (*batch_dims, *output_dims, *input_dims)
            extra = linear.ndim - alpha.ndim
            return alpha.reshape(alpha.shape + (1,) * extra)

        # Merge linear contributions by input_id (handles shared regions)
        merged_lower: dict[int, tuple[SimpleRegion, torch.Tensor]] = {}
        merged_upper: dict[int, tuple[SimpleRegion, torch.Tensor]] = {}
        ordered_ids: list[int] = []

        bias_lower = self.bias_lower.clone()
        bias_upper = self.bias_upper.clone()

        for alpha_lower, alpha_upper, bounds in zip(self.coeffs_lower, self.coeffs_upper, input_bounds, strict=True):
            al_pos = alpha_lower.clamp(min=0)
            al_neg = alpha_lower.clamp(max=0)
            au_pos = alpha_upper.clamp(min=0)
            au_neg = alpha_upper.clamp(max=0)

            # Bias contribution from this input
            bias_lower = bias_lower + al_pos * bounds.bias_lower + al_neg * bounds.bias_upper
            bias_upper = bias_upper + au_pos * bounds.bias_upper + au_neg * bounds.bias_lower

            # Linear contributions
            for iid, region, wl, wu in zip(
                bounds.input_ids, bounds.regions, bounds.linear_lowers, bounds.linear_uppers, strict=True
            ):
                contrib_lower = broadcast(al_pos, wl) * wl + broadcast(al_neg, wu) * wu
                contrib_upper = broadcast(au_pos, wu) * wu + broadcast(au_neg, wl) * wl

                if iid in merged_lower:
                    merged_lower[iid] = (merged_lower[iid][0], merged_lower[iid][1] + contrib_lower)
                    merged_upper[iid] = (merged_upper[iid][0], merged_upper[iid][1] + contrib_upper)
                else:
                    ordered_ids.append(iid)
                    merged_lower[iid] = (region, contrib_lower)
                    merged_upper[iid] = (region, contrib_upper)

        regions = [merged_lower[iid][0] for iid in ordered_ids]
        linear_lower = [merged_lower[iid][1] for iid in ordered_ids]
        linear_upper = [merged_upper[iid][1] for iid in ordered_ids]

        return LinearBounds(
            regions=regions,
            linear_lower=linear_lower or None,
            bias_lower=bias_lower,
            linear_upper=linear_upper or None,
            bias_upper=bias_upper,
            input_ids=ordered_ids or None,
        )

    def symbolic_forward(self, inputs: list[SymbolicLinearRelaxation]) -> SymbolicLinearRelaxation:
        if len(inputs) != 2:
            raise ValueError(f"PairedLinearRelaxation expects exactly 2 inputs, got {len(inputs)}")
        return SymbolicPairedLinearRelaxation(concrete_relaxation=self, input_left=inputs[0], input_right=inputs[1])


@final
@dataclass
class SymbolicPairedLinearRelaxation(SymbolicLinearRelaxation):
    concrete_relaxation: PairedLinearRelaxation

    input_left: SymbolicLinearRelaxation
    input_right: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        r = self.concrete_relaxation
        node_ndim = r.coeffs_lower[0].ndim - batch_ndim
        bounded_ndim = A_lower.ndim - r.coeffs_lower[0].ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            """Broadcast ``(*batch, *node)`` → ``(*batch, *bounded_out, *node)``."""
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        A_l_pos = A_lower.clamp(min=0)
        A_l_neg = A_lower.clamp(max=0)
        A_u_pos = A_upper.clamp(min=0)
        A_u_neg = A_upper.clamp(max=0)

        # Left input: sign decomposition on coeffs[0]
        new_A_lower_left = A_l_pos * bc(r.coeffs_lower[0]) + A_l_neg * bc(r.coeffs_upper[0])
        new_A_upper_left = A_u_pos * bc(r.coeffs_upper[0]) + A_u_neg * bc(r.coeffs_lower[0])
        bounds_left = self.input_left.backward(new_A_lower_left, new_A_upper_left, batch_ndim)

        # Right input: sign decomposition on coeffs[1]
        new_A_lower_right = A_l_pos * bc(r.coeffs_lower[1]) + A_l_neg * bc(r.coeffs_upper[1])
        new_A_upper_right = A_u_pos * bc(r.coeffs_upper[1]) + A_u_neg * bc(r.coeffs_lower[1])
        bounds_right = self.input_right.backward(new_A_lower_right, new_A_upper_right, batch_ndim)

        # Bias contribution: sum over the trailing node dimensions.
        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        delta_bias_lower = A_l_pos * bc(r.bias_lower) + A_l_neg * bc(r.bias_upper)
        delta_bias_upper = A_u_pos * bc(r.bias_upper) + A_u_neg * bc(r.bias_lower)
        if sum_dims:
            delta_bias_lower = delta_bias_lower.sum(dim=sum_dims)
            delta_bias_upper = delta_bias_upper.sum(dim=sum_dims)

        bias_lower = bounds_left.bias_lower + bounds_right.bias_lower + delta_bias_lower
        bias_upper = bounds_left.bias_upper + bounds_right.bias_upper + delta_bias_upper

        # Merge linear contributions by input_id (handles shared regions between left and right)
        merged: dict[int, tuple[SimpleRegion, torch.Tensor, torch.Tensor]] = {}
        ordered_ids: list[int] = []

        for bounds in [bounds_left, bounds_right]:
            for iid, region, wl, wu in zip(
                bounds.input_ids, bounds.regions, bounds.linear_lowers, bounds.linear_uppers, strict=True
            ):
                if iid in merged:
                    merged[iid] = (merged[iid][0], merged[iid][1] + wl, merged[iid][2] + wu)
                else:
                    ordered_ids.append(iid)
                    merged[iid] = (region, wl, wu)

        regions = [merged[iid][0] for iid in ordered_ids]
        linear_lower = [merged[iid][1] for iid in ordered_ids]
        linear_upper = [merged[iid][2] for iid in ordered_ids]

        return LinearBounds(
            regions=regions,
            linear_lower=linear_lower or None,
            bias_lower=bias_lower,
            linear_upper=linear_upper or None,
            bias_upper=bias_upper,
            input_ids=ordered_ids or None,
            validate=False,
        )


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


def compute_maximum_relaxation(
    lower_a: torch.Tensor,
    upper_a: torch.Tensor,
    lower_b: torch.Tensor,
    upper_b: torch.Tensor,
    eta_lower: torch.Tensor | torch.types.Number = 0.5,
    eta_upper: torch.Tensor | torch.types.Number = 0.5,
) -> PairedLinearRelaxation:
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

    return PairedLinearRelaxation(
        coeffs_lower=[alpha1_lower, alpha2_lower],
        coeffs_upper=[alpha1_upper, alpha2_upper],
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
