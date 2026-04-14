from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPSub(ForwardLBPStrategy):
    """Forward LBP strategy for subtraction (abstract-abstract or abstract-constant or constant-abstract)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._sub_bounds(left, right)

        if isinstance(left, LinearBounds):
            # x - c
            return LinearBounds(
                region=left.region,
                linear_lower=left.linear_lower,
                bias_lower=left.bias_lower - right,
                linear_upper=left.linear_upper,
                bias_upper=left.bias_upper - right,
            )

        if isinstance(right, LinearBounds):
            # c - x: flip signs and bounds
            return LinearBounds(
                region=right.region,
                linear_lower=-right.linear_upper if right.linear_upper is not None else None,
                bias_lower=left - right.bias_upper,
                linear_upper=-right.linear_lower if right.linear_lower is not None else None,
                bias_upper=left - right.bias_lower,
            )

        raise TypeError(f"ForwardLBPSub requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _sub_bounds(self, a: LinearBounds, b: LinearBounds) -> LinearBounds:
        if a.linear_lower is not None and b.linear_upper is not None:
            linear_lower = a.linear_lower - b.linear_upper
        elif a.linear_lower is not None:
            linear_lower = a.linear_lower
        elif b.linear_upper is not None:
            linear_lower = -b.linear_upper
        else:
            linear_lower = None

        if a.linear_upper is not None and b.linear_lower is not None:
            linear_upper = a.linear_upper - b.linear_lower
        elif a.linear_upper is not None:
            linear_upper = a.linear_upper
        elif b.linear_lower is not None:
            linear_upper = -b.linear_lower
        else:
            linear_upper = None

        return LinearBounds(
            region=a.region,
            linear_lower=linear_lower,
            bias_lower=a.bias_lower - b.bias_upper,
            linear_upper=linear_upper,
            bias_upper=a.bias_upper - b.bias_lower,
        )
