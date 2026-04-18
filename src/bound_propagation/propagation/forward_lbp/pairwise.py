from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from ...regions import SimpleRegion
from ..linear_relaxations.elementwise import compute_constant_div_relaxation
from ..linear_relaxations.pairwise import (
    PairedParams,
    compute_div_relaxation,
    compute_maximum_relaxation,
    compute_minimum_relaxation,
    compute_mul_relaxation,
)
from .base import ForwardLBPStrategy
from .elementwise import ElementwiseForwardRelaxation

if TYPE_CHECKING:
    from ..context import PropagationContext


@final
@dataclass
class PairedForwardRelaxation:
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

    params: PairedParams

    # ------------------------------------------------------------------
    # Forward composition
    # ------------------------------------------------------------------

    def forward(self, input_bounds_a: LinearBounds, input_bounds_b: LinearBounds) -> LinearBounds:
        """
        Compose z = sum_i(alpha_i * x_i) + beta with linear bounds on each x_i.

        Linear terms have shape (*batch_dims, *output_dims, *input_dims).
        Contributions from all inputs are merged by input_id so that shared regions
        are accumulated correctly.
        """

        def broadcast(alpha: torch.Tensor, linear: torch.Tensor) -> torch.Tensor:
            # alpha: (*batch_dims, *output_dims)
            # linear: (*batch_dims, *output_dims, *input_dims)
            extra = linear.ndim - alpha.ndim
            return alpha.reshape(alpha.shape + (1,) * extra)

        # Merge linear contributions by input_id (handles shared regions)
        merged_lower: dict[int, tuple[SimpleRegion, torch.Tensor]] = {}
        merged_upper: dict[int, tuple[SimpleRegion, torch.Tensor]] = {}
        ordered_ids: list[int] = []

        bias_lower = self.params.bias_lower.clone()
        bias_upper = self.params.bias_upper.clone()

        # input_bounds_a
        al_pos = self.params.alpha_lower_a.clamp(min=0)
        al_neg = self.params.alpha_lower_a.clamp(max=0)
        au_pos = self.params.alpha_upper_a.clamp(min=0)
        au_neg = self.params.alpha_upper_a.clamp(max=0)

        # Bias contribution from input a
        bias_lower = bias_lower + al_pos * input_bounds_a.bias_lower + al_neg * input_bounds_a.bias_upper
        bias_upper = bias_upper + au_pos * input_bounds_a.bias_upper + au_neg * input_bounds_a.bias_lower

        # Linear contributions from input a
        for iid, region, wl, wu in zip(
            input_bounds_a.input_ids,
            input_bounds_a.regions,
            input_bounds_a.linear_lowers,
            input_bounds_a.linear_uppers,
            strict=True,
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

        # input_bounds_b
        al_pos = self.params.alpha_lower_b.clamp(min=0)
        al_neg = self.params.alpha_lower_b.clamp(max=0)
        au_pos = self.params.alpha_upper_b.clamp(min=0)
        au_neg = self.params.alpha_upper_b.clamp(max=0)

        # Bias contribution from input b
        bias_lower = bias_lower + al_pos * input_bounds_b.bias_lower + al_neg * input_bounds_b.bias_upper
        bias_upper = bias_upper + au_pos * input_bounds_b.bias_upper + au_neg * input_bounds_b.bias_lower

        # Linear contributions from input b
        for iid, region, wl, wu in zip(
            input_bounds_b.input_ids,
            input_bounds_b.regions,
            input_bounds_b.linear_lowers,
            input_bounds_b.linear_uppers,
            strict=True,
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


class ForwardLBPDiv(ForwardLBPStrategy):
    """Forward LBP strategy for division (abstract/abstract, abstract/constant, constant/abstract)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, _kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._div_bounds(left, right)

        if isinstance(left, LinearBounds) and not isinstance(right, LinearBounds):
            return self._divide_by_constant(left, right)

        if isinstance(right, LinearBounds) and not isinstance(left, LinearBounds):
            return self._constant_div(left, right)

        raise TypeError(f"ForwardLBPDiv requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _div_bounds(self, a: LinearBounds, b: LinearBounds) -> LinearBounds:
        """Bound z = a / b for abstract numerator and denominator.

        Decomposes as z = a * (1/b):
          1. Linearise 1/b via a reciprocal relaxation applied to b's concrete bounds.
          2. Compose that relaxation with b's LinearBounds to get LinearBounds for w = 1/b.
          3. Apply a McCormick relaxation for z = a * w.

        When an element of b's domain crosses zero the reciprocal is undefined, so
        the output for that element is set to [-∞, +∞].  To prevent NaN from
        propagating through the McCormick computation (which arises from ±∞ in
        the reciprocal when zero is crossed), crossing elements are temporarily
        replaced with safe dummy values [1, 2] before computing the relaxation;
        their outputs are then overridden unconditionally to [-∞, +∞].
        """
        bounds_a = a.concretize()
        bounds_b = b.concretize()

        params = compute_div_relaxation(bounds_a, bounds_b)
        relaxation = PairedForwardRelaxation(params)
        return relaxation.forward(a, b)

    def _divide_by_constant(self, bounds: LinearBounds, divisor: torch.Tensor | torch.types.Number) -> LinearBounds:
        divisor = torch.as_tensor(divisor, dtype=bounds.bias_lower.dtype, device=bounds.bias_lower.device).expand_as(
            bounds.bias_lower
        )
        positive_mask = divisor > 0

        linear_lower = [
            torch.where(mask, lower_linear / scale, upper_linear / scale)
            for lower_linear, upper_linear in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
            for scale, mask in [
                (
                    divisor.reshape(
                        *divisor.shape,
                        *([1] * (lower_linear.ndim - divisor.ndim)),
                    ),
                    positive_mask.reshape(
                        *positive_mask.shape,
                        *([1] * (lower_linear.ndim - positive_mask.ndim)),
                    ),
                )
            ]
        ]
        linear_upper = [
            torch.where(mask, upper_linear / scale, lower_linear / scale)
            for lower_linear, upper_linear in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
            for scale, mask in [
                (
                    divisor.reshape(
                        *divisor.shape,
                        *([1] * (lower_linear.ndim - divisor.ndim)),
                    ),
                    positive_mask.reshape(
                        *positive_mask.shape,
                        *([1] * (lower_linear.ndim - positive_mask.ndim)),
                    ),
                )
            ]
        ]

        bias_lower_pos = bounds.bias_lower / divisor
        bias_lower_neg = bounds.bias_upper / divisor
        bias_lower = torch.where(positive_mask, bias_lower_pos, bias_lower_neg)

        bias_upper_pos = bounds.bias_upper / divisor
        bias_upper_neg = bounds.bias_lower / divisor
        bias_upper = torch.where(positive_mask, bias_upper_pos, bias_upper_neg)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )

    def _constant_div(self, constant: torch.Tensor | torch.types.Number, bounds: LinearBounds) -> LinearBounds:
        concrete_bounds = bounds.concretize()
        params = compute_constant_div_relaxation(concrete_bounds, constant)
        relaxation = ElementwiseForwardRelaxation(params)
        return relaxation.forward(bounds)


class ForwardLBPMul(ForwardLBPStrategy):
    """Forward LBP strategy for multiplication (abstract*abstract or abstract*constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._mul_bounds(left, right)

        if isinstance(left, LinearBounds):
            return self._multiply_by_constant(left, right)

        if isinstance(right, LinearBounds):
            return self._multiply_by_constant(right, left)

        raise TypeError(f"ForwardLBPMul requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _mul_bounds(self, a: LinearBounds, b: LinearBounds) -> LinearBounds:
        """McCormick relaxation for element-wise z = a * b.

        Uses PairedForwardRelaxation to preserve linear structure from both inputs.
        At each element, we choose element-wise between the two McCormick lower bounds
        and the two McCormick upper bounds by evaluating tightness at the midpoint.

        McCormick lower bounds (both valid):
          z ≥ lb * a + la * b - la*lb
          z ≥ ub * a + ua * b - ua*ub

        McCormick upper bounds (both valid):
          z ≤ ub * a + la * b - la*ub
          z ≤ lb * a + ua * b - ua*lb
        """
        bounds_a = a.concretize()
        bounds_b = b.concretize()

        params = compute_mul_relaxation(bounds_a, bounds_b)
        relaxation = PairedForwardRelaxation(params)
        return relaxation.forward(a, b)

    def _multiply_by_constant(self, bounds: LinearBounds, constant: torch.Tensor | torch.types.Number) -> LinearBounds:
        constant_tensor = torch.as_tensor(
            constant, dtype=bounds.bias_lower.dtype, device=bounds.bias_lower.device
        ).expand_as(bounds.bias_lower)
        positive_mask = constant_tensor >= 0

        linear_lower = [
            torch.where(mask, scale * lower_linear, scale * upper_linear)
            for lower_linear, upper_linear in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
            for scale, mask in [
                (
                    constant_tensor.reshape(
                        *constant_tensor.shape,
                        *([1] * (lower_linear.ndim - constant_tensor.ndim)),
                    ),
                    positive_mask.reshape(
                        *positive_mask.shape,
                        *([1] * (lower_linear.ndim - positive_mask.ndim)),
                    ),
                )
            ]
        ]
        linear_upper = [
            torch.where(mask, scale * upper_linear, scale * lower_linear)
            for lower_linear, upper_linear in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
            for scale, mask in [
                (
                    constant_tensor.reshape(
                        *constant_tensor.shape,
                        *([1] * (lower_linear.ndim - constant_tensor.ndim)),
                    ),
                    positive_mask.reshape(
                        *positive_mask.shape,
                        *([1] * (lower_linear.ndim - positive_mask.ndim)),
                    ),
                )
            ]
        ]

        bias_lower_pos = constant_tensor * bounds.bias_lower
        bias_lower_neg = constant_tensor * bounds.bias_upper
        bias_lower = torch.where(positive_mask, bias_lower_pos, bias_lower_neg)

        bias_upper_pos = constant_tensor * bounds.bias_upper
        bias_upper_neg = constant_tensor * bounds.bias_lower
        bias_upper = torch.where(positive_mask, bias_upper_pos, bias_upper_neg)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )


class ForwardLBPMaximum(ForwardLBPStrategy):
    """Forward LBP strategy for element-wise maximum (abstract+abstract or abstract+constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._max_bounds(left, right)

        if isinstance(left, LinearBounds):
            return self._max_bounds(left, self._constant_to_bounds(right, left))

        if isinstance(right, LinearBounds):
            return self._max_bounds(self._constant_to_bounds(left, right), right)

        raise TypeError(f"ForwardLBPMaximum requires at least one LinearBounds, got {type(left)} and {type(right)}")

    @staticmethod
    def _constant_to_bounds(constant: torch.Tensor | torch.types.Number, reference: LinearBounds) -> LinearBounds:
        constant_tensor = torch.as_tensor(
            constant, dtype=reference.bias_lower.dtype, device=reference.bias_lower.device
        )
        constant_tensor = constant_tensor.expand_as(reference.bias_lower)
        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=constant_tensor,
            linear_upper=[],
            bias_upper=constant_tensor,
        )

    @staticmethod
    def _max_bounds(a: LinearBounds, b: LinearBounds) -> LinearBounds:
        bounds_a = a.concretize()
        bounds_b = b.concretize()
        params = compute_maximum_relaxation(bounds_a, bounds_b)
        relaxation = PairedForwardRelaxation(params)
        return relaxation.forward(a, b)


class ForwardLBPMinimum(ForwardLBPStrategy):
    """Forward LBP strategy for element-wise minimum (abstract+abstract or abstract+constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._min_bounds(left, right)

        if isinstance(left, LinearBounds):
            return self._min_bounds(left, self._constant_to_bounds(right, left))

        if isinstance(right, LinearBounds):
            return self._min_bounds(self._constant_to_bounds(left, right), right)

        raise TypeError(f"ForwardLBPMinimum requires at least one LinearBounds, got {type(left)} and {type(right)}")

    @staticmethod
    def _constant_to_bounds(constant: torch.Tensor | torch.types.Number, reference: LinearBounds) -> LinearBounds:
        constant_tensor = torch.as_tensor(
            constant, dtype=reference.bias_lower.dtype, device=reference.bias_lower.device
        )
        constant_tensor = constant_tensor.expand_as(reference.bias_lower)
        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=constant_tensor,
            linear_upper=[],
            bias_upper=constant_tensor,
        )

    @staticmethod
    def _min_bounds(a: LinearBounds, b: LinearBounds) -> LinearBounds:
        bounds_a = a.concretize()
        bounds_b = b.concretize()
        params = compute_minimum_relaxation(bounds_a, bounds_b)
        relaxation = PairedForwardRelaxation(params)
        return relaxation.forward(a, b)
