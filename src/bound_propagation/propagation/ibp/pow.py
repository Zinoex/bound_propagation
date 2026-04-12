from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPPow(ForwardIBPStrategy):
    """IBP strategy for power activation: pow([a, b], n)."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"pow requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPPow requires input to be IntervalBounds")

        power = node.attributes.get("power", 1)

        if isinstance(power, torch.Tensor):# and dtype is integer type:
            if torch.is_floating_point(power):
                raise TypeError("IBPPow requires power to be an integer tensor")


        lower, upper = x_bounds.lower, x_bounds.upper

        if isinstance(power, torch.Tensor):
            lower_act, upper_act = torch.pow(lower, power), torch.pow(upper, power)

            lower_out, upper_out = torch.zeros_like(lower), torch.zeros_like(upper)

            # expand power to match input shape for even/odd checks
            power_expanded = power.expand_as(lower)

            # Even powers
            even = (power_expanded % 2) == 0
            crossing = (lower < 0) & (upper > 0)

            crossing, not_crossing = (crossing & even), ((~crossing) & even)

            lower_out[crossing] = 0.0
            lower_out[not_crossing] = torch.min(lower_act[not_crossing], upper_act[not_crossing])

            upper_out[even] = torch.max(lower_act[even], upper_act[even])

            # Odd powers
            lower_out[~even] = lower_act[~even]
            upper_out[~even] = upper_act[~even]

            return IntervalBounds(lower_out, upper_out)
        else:
            if power % 2 == 0:
                # Even power: [a, b] → [min(a^n, b^n), max(a^n, b^n)] with special handling for crossing zero
                lower_act, upper_act = torch.pow(lower, power), torch.pow(upper, power)

                lower_out = torch.where((lower < 0) & (upper > 0), 0.0, torch.min(lower_act, upper_act))
                upper_out = torch.max(lower_act, upper_act)

                return IntervalBounds(lower_out, upper_out)
            else:
                # Odd power: [a, b] → [a^n, b^n]
                return IntervalBounds(torch.pow(lower, power), torch.pow(upper, power))
