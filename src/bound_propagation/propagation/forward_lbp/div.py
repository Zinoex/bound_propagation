from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
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
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._div_bounds(left, right)

        if isinstance(left, LinearBounds) and not isinstance(right, LinearBounds):
            return self._divide_by_constant(left, right)

        if isinstance(right, LinearBounds) and not isinstance(left, LinearBounds):
            return self._constant_div(left, right)

        raise TypeError(f"ForwardLBPDiv requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _div_bounds(self, a: LinearBounds, b: LinearBounds) -> LinearBounds:
        lower_a, upper_a = a.concretize()
        lower_b, upper_b = b.concretize()

        if torch.any((lower_b <= 0) & (upper_b >= 0)):
            lower = torch.full_like(lower_a, float("-inf"))
            upper = torch.full_like(upper_a, float("inf"))
        else:
            quotients = [
                lower_a / lower_b,
                lower_a / upper_b,
                upper_a / lower_b,
                upper_a / upper_b,
            ]
            lower = torch.min(torch.stack(quotients), dim=0)[0]
            upper = torch.max(torch.stack(quotients), dim=0)[0]

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=lower,
            linear_upper=[],
            bias_upper=upper,
        )

    def _divide_by_constant(self, bounds: LinearBounds, divisor: object) -> LinearBounds:
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

    def _constant_div(self, constant: object, bounds: LinearBounds) -> LinearBounds:
        lower_x, upper_x = bounds.concretize()
        constant_tensor = torch.as_tensor(constant, dtype=lower_x.dtype, device=lower_x.device)

        if torch.any((lower_x <= 0) & (upper_x >= 0)):
            lower = torch.full_like(lower_x, float("-inf"))
            upper = torch.full_like(upper_x, float("inf"))
        else:
            quotients = [constant_tensor / lower_x, constant_tensor / upper_x]
            lower = torch.min(torch.stack(quotients), dim=0)[0]
            upper = torch.max(torch.stack(quotients), dim=0)[0]

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=lower,
            linear_upper=[],
            bias_upper=upper,
        )
