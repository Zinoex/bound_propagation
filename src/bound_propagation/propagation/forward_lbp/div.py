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
        # TODO: implement a more precise method that tracks linear terms instead of just concretizing and re-abstracting
        lower_a, upper_a = a.concretize()
        lower_b, upper_b = b.concretize()

        crosses_zero = (lower_b <= 0) & (upper_b >= 0)
        safe_lower_b = torch.where(crosses_zero, 1, lower_b)
        safe_upper_b = torch.where(crosses_zero, 1, upper_b)

        quotients = [
            lower_a / safe_lower_b,
            lower_a / safe_upper_b,
            upper_a / safe_lower_b,
            upper_a / safe_upper_b,
        ]
        finite_lower = torch.min(torch.stack(quotients), dim=0)[0]
        finite_upper = torch.max(torch.stack(quotients), dim=0)[0]

        lower = torch.where(crosses_zero, float("-inf"), finite_lower)
        upper = torch.where(crosses_zero, float("inf"), finite_upper)

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
        # TODO: implement a more precise method that tracks linear terms instead of just concretizing and re-abstracting
        lower_x, upper_x = bounds.concretize()
        constant_tensor = torch.as_tensor(constant, dtype=lower_x.dtype, device=lower_x.device)

        crosses_zero = (lower_x <= 0) & (upper_x >= 0)
        safe_lower_x = torch.where(crosses_zero, 1, lower_x)
        safe_upper_x = torch.where(crosses_zero, 1, upper_x)

        quotients = [constant_tensor / safe_lower_x, constant_tensor / safe_upper_x]
        finite_lower = torch.min(torch.stack(quotients), dim=0)[0]
        finite_upper = torch.max(torch.stack(quotients), dim=0)[0]

        lower = torch.where(crosses_zero, float("-inf"), finite_lower)
        upper = torch.where(crosses_zero, float("inf"), finite_upper)

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=lower,
            linear_upper=[],
            bias_upper=upper,
        )
