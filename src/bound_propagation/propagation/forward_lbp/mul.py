from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from ..linear_relaxations.mul import compute_mul_relaxation
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


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

        Uses PairedLinearRelaxation to preserve linear structure from both inputs.
        At each element, we choose element-wise between the two McCormick lower bounds
        and the two McCormick upper bounds by evaluating tightness at the midpoint.

        McCormick lower bounds (both valid):
          z ≥ lb * a + la * b - la*lb
          z ≥ ub * a + ua * b - ua*ub

        McCormick upper bounds (both valid):
          z ≤ ub * a + la * b - la*ub
          z ≤ lb * a + ua * b - ua*lb
        """
        la, ua = a.concretize()
        lb, ub = b.concretize()

        relaxation = compute_mul_relaxation(la, ua, lb, ub)
        return relaxation.forward_compose([a, b])

    def _multiply_by_constant(self, bounds: LinearBounds, constant: object) -> LinearBounds:
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
