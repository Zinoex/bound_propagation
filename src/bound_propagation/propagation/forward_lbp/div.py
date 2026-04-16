from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from ..linear_relaxations.constant_div import compute_constant_div_relaxation
from ..linear_relaxations.div import compute_div_relaxation
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


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
        lower_a, upper_a = a.concretize()
        lower_b, upper_b = b.concretize()

        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)

        return relaxation.forward_compose([a, b])

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
        lower_x, upper_x = bounds.concretize()
        relaxation = compute_constant_div_relaxation(lower_x, upper_x, constant)
        return relaxation.forward_compose([bounds])
