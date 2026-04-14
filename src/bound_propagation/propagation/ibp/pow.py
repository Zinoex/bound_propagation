from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPPow(ForwardIBPStrategy):
    """IBP strategy for integer power."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPPow requires input to be IntervalBounds")

        power = args[1] if len(args) > 1 else kwargs.get("exponent", 1)

        if isinstance(power, torch.Tensor) and torch.is_floating_point(power):
            raise TypeError("IBPPow requires power to be an integer tensor")

        lower, upper = x_bounds.lower, x_bounds.upper

        if isinstance(power, torch.Tensor):
            return self._tensor_power(lower, upper, power)

        return self._scalar_power(lower, upper, power)

    @staticmethod
    def _scalar_power(lower: torch.Tensor, upper: torch.Tensor, power: int) -> IntervalBounds:
        if power == 0:
            ones = torch.ones_like(lower)
            return IntervalBounds(ones, ones)

        if power % 2 == 0:
            lower_act = torch.pow(lower, power)
            upper_act = torch.pow(upper, power)
            lower_out = torch.where((lower < 0) & (upper > 0), 0.0, torch.min(lower_act, upper_act))
            upper_out = torch.max(lower_act, upper_act)
            return IntervalBounds(lower_out, upper_out)

        return IntervalBounds(torch.pow(lower, power), torch.pow(upper, power))

    @staticmethod
    def _tensor_power(lower: torch.Tensor, upper: torch.Tensor, power: torch.Tensor) -> IntervalBounds:
        if torch.all(power == 0):
            ones = torch.ones_like(lower)
            return IntervalBounds(ones, ones)

        lower_act = torch.pow(lower, power)
        upper_act = torch.pow(upper, power)

        lower_out = torch.zeros_like(lower)
        upper_out = torch.zeros_like(upper)

        power_expanded = power.expand_as(lower)
        zero_power = power_expanded == 0
        even = ((power_expanded % 2) == 0) & (~zero_power)
        odd = (~even) & (~zero_power)
        crossing = (lower < 0) & (upper > 0)

        # power == 0
        lower_out[zero_power] = 1.0
        upper_out[zero_power] = 1.0

        # Even powers
        crossing_even = crossing & even
        not_crossing_even = (~crossing) & even
        lower_out[crossing_even] = 0.0
        lower_out[not_crossing_even] = torch.min(lower_act[not_crossing_even], upper_act[not_crossing_even])
        upper_out[even] = torch.max(lower_act[even], upper_act[even])

        # Odd powers
        lower_out[odd] = lower_act[odd]
        upper_out[odd] = upper_act[odd]

        return IntervalBounds(lower_out, upper_out)
