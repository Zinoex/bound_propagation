from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from ...regions import SimpleRegion
from ..linear_relaxations.alpha_resolvers import (
    resolve_div_etas,
    resolve_max_etas,
    resolve_min_etas,
    resolve_mul_etas,
)
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

        Linear terms have shape ``(*output_bias_shape, *input_axes)``. When the
        pairwise op broadcasts (the input bias prefixes don't match the output
        bias prefix), the per-input linear contributions are aligned by:

        * appending ``len(input_axes)`` trailing size-1 dims to alpha so it
          slots in before the input axes, and
        * prepending leading size-1 dims to ``wl`` / ``wu`` so its bias prefix
          broadcasts against the (larger) output bias prefix.

        Contributions are merged by input_id so shared regions accumulate.
        """
        output_ndim = self.params.bias_lower.ndim

        def contribute(
            alpha: torch.Tensor,
            linear: torch.Tensor,
            input_bias_ndim: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n_input_axes = linear.ndim - input_bias_ndim
            alpha_padded = alpha.reshape(tuple(alpha.shape) + (1,) * n_input_axes)
            linear_padded = linear.reshape((1,) * (output_ndim - input_bias_ndim) + tuple(linear.shape))
            return alpha_padded, linear_padded

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
        input_bias_ndim_a = input_bounds_a.bias_lower.ndim
        for iid, region, wl, wu in zip(
            input_bounds_a.input_ids,
            input_bounds_a.regions,
            input_bounds_a.linear_lowers,
            input_bounds_a.linear_uppers,
            strict=True,
        ):
            al_pos_b, wl_b = contribute(al_pos, wl, input_bias_ndim_a)
            al_neg_b, wu_b = contribute(al_neg, wu, input_bias_ndim_a)
            au_pos_b, _ = contribute(au_pos, wu, input_bias_ndim_a)
            au_neg_b, _ = contribute(au_neg, wl, input_bias_ndim_a)
            contrib_lower = al_pos_b * wl_b + al_neg_b * wu_b
            contrib_upper = au_pos_b * wu_b + au_neg_b * wl_b

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
        input_bias_ndim_b = input_bounds_b.bias_lower.ndim
        for iid, region, wl, wu in zip(
            input_bounds_b.input_ids,
            input_bounds_b.regions,
            input_bounds_b.linear_lowers,
            input_bounds_b.linear_uppers,
            strict=True,
        ):
            al_pos_b, wl_b = contribute(al_pos, wl, input_bias_ndim_b)
            al_neg_b, wu_b = contribute(al_neg, wu, input_bias_ndim_b)
            au_pos_b, _ = contribute(au_pos, wu, input_bias_ndim_b)
            au_neg_b, _ = contribute(au_neg, wl, input_bias_ndim_b)
            contrib_lower = al_pos_b * wl_b + al_neg_b * wu_b
            contrib_upper = au_pos_b * wu_b + au_neg_b * wl_b

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
            return self._div_bounds(left, right, node, ctx)

        if isinstance(left, LinearBounds) and not isinstance(right, LinearBounds):
            return self._divide_by_constant(left, right)

        if isinstance(right, LinearBounds) and not isinstance(left, LinearBounds):
            return self._constant_div(left, right)

        raise TypeError(f"ForwardLBPDiv requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _div_bounds(self, a: LinearBounds, b: LinearBounds, node: fx.Node, ctx: PropagationContext) -> LinearBounds:
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

        eta_lo, eta_up = resolve_div_etas(ctx.alpha_provider, node, bounds_a)
        params = compute_div_relaxation(
            bounds_a,
            bounds_b,
            eta_lower=eta_lo if eta_lo is not None else 0.5,
            eta_upper=eta_up if eta_up is not None else 0.5,
        )
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
            return self._mul_bounds(left, right, node, ctx)

        if isinstance(left, LinearBounds):
            return self._multiply_by_constant(left, right)

        if isinstance(right, LinearBounds):
            return self._multiply_by_constant(right, left)

        raise TypeError(f"ForwardLBPMul requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _mul_bounds(self, a: LinearBounds, b: LinearBounds, node: fx.Node, ctx: PropagationContext) -> LinearBounds:
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

        eta_lo, eta_up = resolve_mul_etas(ctx.alpha_provider, node, bounds_a)
        params = compute_mul_relaxation(
            bounds_a,
            bounds_b,
            eta_lower=eta_lo if eta_lo is not None else 0.5,
            eta_upper=eta_up if eta_up is not None else 0.5,
        )
        relaxation = PairedForwardRelaxation(params)
        return relaxation.forward(a, b)

    def _multiply_by_constant(self, bounds: LinearBounds, constant: torch.Tensor | torch.types.Number) -> LinearBounds:
        constant_tensor = torch.as_tensor(
            constant, dtype=bounds.bias_lower.dtype, device=bounds.bias_lower.device
        ).expand_as(bounds.bias_lower)
        positive_mask = constant_tensor >= 0

        # Building the broadcast shape as a single tuple keeps the call
        # well-formed when both shapes are empty (0-D input).
        def _broadcast(t: torch.Tensor, target_ndim: int) -> torch.Tensor:
            return t.reshape(tuple(t.shape) + (1,) * (target_ndim - t.ndim))

        linear_lower = [
            torch.where(
                _broadcast(positive_mask, lower_linear.ndim),
                _broadcast(constant_tensor, lower_linear.ndim) * lower_linear,
                _broadcast(constant_tensor, lower_linear.ndim) * upper_linear,
            )
            for lower_linear, upper_linear in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
        ]
        linear_upper = [
            torch.where(
                _broadcast(positive_mask, lower_linear.ndim),
                _broadcast(constant_tensor, lower_linear.ndim) * upper_linear,
                _broadcast(constant_tensor, lower_linear.ndim) * lower_linear,
            )
            for lower_linear, upper_linear in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
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
            return self._max_bounds(left, right, node, ctx)

        if isinstance(left, LinearBounds):
            return self._max_bounds(left, self._constant_to_bounds(right, left), node, ctx)

        if isinstance(right, LinearBounds):
            return self._max_bounds(self._constant_to_bounds(left, right), right, node, ctx)

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
    def _max_bounds(a: LinearBounds, b: LinearBounds, node: fx.Node, ctx: PropagationContext) -> LinearBounds:
        bounds_a = a.concretize()
        bounds_b = b.concretize()
        eta_lo, eta_up = resolve_max_etas(ctx.alpha_provider, node, bounds_a)
        params = compute_maximum_relaxation(
            bounds_a,
            bounds_b,
            eta_lower=eta_lo if eta_lo is not None else 0.5,
            eta_upper=eta_up if eta_up is not None else 0.5,
        )
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
            return self._min_bounds(left, right, node, ctx)

        if isinstance(left, LinearBounds):
            return self._min_bounds(left, self._constant_to_bounds(right, left), node, ctx)

        if isinstance(right, LinearBounds):
            return self._min_bounds(self._constant_to_bounds(left, right), right, node, ctx)

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
    def _min_bounds(a: LinearBounds, b: LinearBounds, node: fx.Node, ctx: PropagationContext) -> LinearBounds:
        bounds_a = a.concretize()
        bounds_b = b.concretize()
        eta_lo, eta_up = resolve_min_etas(ctx.alpha_provider, node, bounds_a)
        params = compute_minimum_relaxation(
            bounds_a,
            bounds_b,
            eta_lower=eta_lo if eta_lo is not None else 0.5,
            eta_upper=eta_up if eta_up is not None else 0.5,
        )
        relaxation = PairedForwardRelaxation(params)
        return relaxation.forward(a, b)
