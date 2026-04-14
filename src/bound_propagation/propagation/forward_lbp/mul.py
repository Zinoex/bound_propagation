from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
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
        lower_a, upper_a = a.concretize()
        lower_b, upper_b = b.concretize()

        products = [
            lower_a * lower_b,
            lower_a * upper_b,
            upper_a * lower_b,
            upper_a * upper_b,
        ]
        lower = torch.stack(products).min(dim=0)[0]
        upper = torch.stack(products).max(dim=0)[0]

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=lower,
            linear_upper=[],
            bias_upper=upper,
        )

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
