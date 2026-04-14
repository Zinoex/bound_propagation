from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPAdd(ForwardLBPStrategy):
    """Forward LBP strategy for addition (abstract+abstract or abstract+constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._add_bounds(left, right)

        if isinstance(left, LinearBounds):
            return self._add_constant(left, right)

        if isinstance(right, LinearBounds):
            return self._add_constant(right, left)

        raise TypeError(f"ForwardLBPAdd requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _add_bounds(self, a: LinearBounds, b: LinearBounds) -> LinearBounds:
        if a.linear_lower is not None and b.linear_lower is not None:
            linear_lower = a.linear_lower + b.linear_lower
        elif a.linear_lower is not None:
            linear_lower = a.linear_lower
        elif b.linear_lower is not None:
            linear_lower = b.linear_lower
        else:
            linear_lower = None

        if a.linear_upper is not None and b.linear_upper is not None:
            linear_upper = a.linear_upper + b.linear_upper
        elif a.linear_upper is not None:
            linear_upper = a.linear_upper
        elif b.linear_upper is not None:
            linear_upper = b.linear_upper
        else:
            linear_upper = None

        return LinearBounds(
            region=a.region,
            linear_lower=linear_lower,
            bias_lower=a.bias_lower + b.bias_lower,
            linear_upper=linear_upper,
            bias_upper=a.bias_upper + b.bias_upper,
        )

    def _add_constant(self, bounds: LinearBounds, constant: object) -> LinearBounds:
        return LinearBounds(
            region=bounds.region,
            linear_lower=bounds.linear_lower,
            bias_lower=bounds.bias_lower + constant,
            linear_upper=bounds.linear_upper,
            bias_upper=bounds.bias_upper + constant,
        )
