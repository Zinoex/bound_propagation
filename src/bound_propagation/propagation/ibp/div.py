from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPDiv(ForwardIBPStrategy):
    """IBP strategy for division (all combinations of abstract/constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, IntervalBounds) and isinstance(right, IntervalBounds):
            return self._interval_div_interval(left, right)

        if isinstance(left, IntervalBounds):
            return self._interval_div_constant(left, right)

        if isinstance(right, IntervalBounds):
            return self._constant_div_interval(left, right)

        raise TypeError(f"IBPDiv requires at least one IntervalBounds, got {type(left)} and {type(right)}")

    @staticmethod
    def _interval_div_interval(x: IntervalBounds, y: IntervalBounds) -> IntervalBounds:
        # TODO: Fix because this is not sound.
        if torch.any((y.lower <= 0) & (y.upper >= 0)):
            return IntervalBounds.unbounded_like(x)

        ll = x.lower / y.lower
        lu = x.lower / y.upper
        ul = x.upper / y.lower
        uu = x.upper / y.upper

        lower = torch.min(torch.min(ll, lu), torch.min(ul, uu))
        upper = torch.max(torch.max(ll, lu), torch.max(ul, uu))
        return IntervalBounds(lower, upper)

    @staticmethod
    def _interval_div_constant(interval: IntervalBounds, c: torch.Tensor | torch.types.Number) -> IntervalBounds:
        if isinstance(c, torch.Tensor):
            lower = torch.where(c >= 0, interval.lower / c, interval.upper / c)
            lower = torch.where(c == 0, float("-inf"), lower)
            upper = torch.where(c >= 0, interval.upper / c, interval.lower / c)
            upper = torch.where(c == 0, float("inf"), upper)
            return IntervalBounds(lower, upper)

        if c == 0:
            return IntervalBounds.unbounded_like(interval)
        if c > 0:
            return IntervalBounds(interval.lower / c, interval.upper / c)
        return IntervalBounds(interval.upper / c, interval.lower / c)

    @staticmethod
    def _constant_div_interval(c: torch.Tensor | torch.types.Number, interval: IntervalBounds) -> IntervalBounds:
        unbounded_mask = (interval.lower <= 0) & (interval.upper >= 0)

        if isinstance(c, torch.Tensor):
            lower = torch.where(c >= 0, c / interval.upper, c / interval.lower)
            lower = torch.where(unbounded_mask, float("-inf"), lower)
            upper = torch.where(c >= 0, c / interval.lower, c / interval.upper)
            upper = torch.where(unbounded_mask, float("inf"), upper)
            return IntervalBounds(lower, upper)

        if c == 0:
            zero = torch.zeros_like(interval.lower)
            return IntervalBounds(zero, zero)

        lower = c / interval.upper
        lower = torch.where(unbounded_mask, float("-inf"), lower)
        upper = c / interval.lower
        upper = torch.where(unbounded_mask, float("inf"), upper)
        return IntervalBounds(lower, upper)
