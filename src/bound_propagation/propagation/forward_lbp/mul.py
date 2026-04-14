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
            region=a.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )

    def _multiply_by_constant(self, bounds: LinearBounds, constant: object) -> LinearBounds:
        positive_mask = constant >= 0

        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            linear_lower_pos = constant.unsqueeze(-1) * bounds.linear_lower
            linear_lower_neg = constant.unsqueeze(-1) * bounds.linear_upper
            linear_lower = torch.where(positive_mask.unsqueeze(-1), linear_lower_pos, linear_lower_neg)

            linear_upper_pos = constant.unsqueeze(-1) * bounds.linear_upper
            linear_upper_neg = constant.unsqueeze(-1) * bounds.linear_lower
            linear_upper = torch.where(positive_mask.unsqueeze(-1), linear_upper_pos, linear_upper_neg)
        elif bounds.linear_lower is not None:
            linear_lower = constant.unsqueeze(-1) * bounds.linear_lower
            linear_upper = constant.unsqueeze(-1) * bounds.linear_lower
        elif bounds.linear_upper is not None:
            linear_lower = constant.unsqueeze(-1) * bounds.linear_upper
            linear_upper = constant.unsqueeze(-1) * bounds.linear_upper
        else:
            linear_lower = None
            linear_upper = None

        bias_lower_pos = constant * bounds.bias_lower
        bias_lower_neg = constant * bounds.bias_upper
        bias_lower = torch.where(positive_mask, bias_lower_pos, bias_lower_neg)

        bias_upper_pos = constant * bounds.bias_upper
        bias_upper_neg = constant * bounds.bias_lower
        bias_upper = torch.where(positive_mask, bias_upper_pos, bias_upper_neg)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
