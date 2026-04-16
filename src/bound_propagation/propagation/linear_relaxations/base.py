"""
AbstractLinearRelaxation: Type hierarchy for linear approximations of operations.

Provides:
  AbstractLinearRelaxation – base class guaranteeing forward_compose and backward_compose.
  ElementwiseLinearRelaxation – for unary element-wise operations: y ≥ alpha*x + beta.
  PairedLinearRelaxation – for binary operations: z ≥ alpha1*x + alpha2*y + beta.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import torch

from ...bounds import LinearBounds
from ...regions import SimpleRegion


class AbstractLinearRelaxation(ABC):
    """
    Abstract base for linear relaxations of operations.

    Subtypes must implement forward_compose to compose the relaxation with
    incoming linear bounds.
    """

    @abstractmethod
    def forward_compose(self, input_bounds: list[LinearBounds]) -> LinearBounds:
        """
        Compose this relaxation with incoming linear bounds (forward direction).

        Given linear bounds on the inputs (each expressed as an affine function of
        some root variable x0), substitute them into the relaxation to obtain linear
        bounds on the output, also expressed as an affine function of x0.

        Args:
            input_bounds: One LinearBounds per input of the relaxed operation.

        Returns:
            LinearBounds representing the output bounds in terms of x0.
        """


@final
@dataclass
class ElementwiseLinearRelaxation(AbstractLinearRelaxation):
    """
    Element-wise linear relaxation for unary operations y = f(x).

    Stores four element-wise tensors (same shape as x / y):
        y_lower >= alpha_lower * x + beta_lower
        y_upper <= alpha_upper * x + beta_upper

    The abstract dimension convention for LinearBounds linear terms is
    (*batch_dims, *output_dims, *input_dims).  alpha and beta live in
    (*batch_dims, *output_dims); forward_compose appends the input trailing
    axes via broadcasting.

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

    # ------------------------------------------------------------------
    # Forward composition
    # ------------------------------------------------------------------

    def forward_compose(self, input_bounds: list[LinearBounds]) -> LinearBounds:
        """
        Compose: y = alpha * x + beta  composed with  x = W @ x0 + b  →  y = W_new @ x0 + b_new.

        Linear terms have shape (*batch_dims, *output_dims, *input_dims).
        alpha/beta have shape (*batch_dims, *output_dims); trailing input axes are
        broadcast by appending ones.

        Handles signed alpha via positive/negative clamping so the result is always
        a valid lower/upper bound.
        """
        if len(input_bounds) != 1:
            raise ValueError(f"ElementwiseLinearRelaxation expects 1 input bound, got {len(input_bounds)}")
        bounds = input_bounds[0]

        al_pos = self.alpha_lower.clamp(min=0)
        al_neg = self.alpha_lower.clamp(max=0)
        au_pos = self.alpha_upper.clamp(min=0)
        au_neg = self.alpha_upper.clamp(max=0)

        def broadcast(alpha: torch.Tensor, linear: torch.Tensor) -> torch.Tensor:
            # alpha: (*batch_dims, *output_dims)
            # linear: (*batch_dims, *output_dims, *input_dims)
            # Append one dimension per input axis so broadcasting is correct.
            extra = linear.ndim - alpha.ndim
            return alpha.reshape(alpha.shape + (1,) * extra)

        # Lower bound: alpha_lower_pos * W_lower  +  alpha_lower_neg * W_upper
        linear_lower = [
            broadcast(al_pos, wl) * wl + broadcast(al_neg, wu) * wu
            for wl, wu in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
        ]
        bias_lower = al_pos * bounds.bias_lower + al_neg * bounds.bias_upper + self.beta_lower

        # Upper bound: alpha_upper_pos * W_upper  +  alpha_upper_neg * W_lower
        linear_upper = [
            broadcast(au_pos, wu) * wu + broadcast(au_neg, wl) * wl
            for wl, wu in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
        ]
        bias_upper = au_pos * bounds.bias_upper + au_neg * bounds.bias_lower + self.beta_upper

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower or None,
            bias_lower=bias_lower,
            linear_upper=linear_upper or None,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids or None,
        )


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
    trailing axes are broadcast in forward_compose.

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

    def forward_compose(self, input_bounds: list[LinearBounds]) -> LinearBounds:
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
