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

        if isinstance(power, torch.Tensor):
            if torch.is_floating_point(power):
                raise TypeError("IBPPow requires power to be an integer tensor")

        lower, upper = x_bounds.lower, x_bounds.upper

        # Special case: x^0 = 1 for all x
        if isinstance(power, torch.Tensor):
            # Check if all powers are 0
            if torch.all(power == 0):
                ones = torch.ones_like(lower)
                return IntervalBounds(ones, ones)

            # Check if some powers are 0 (mixed case)
            if torch.any(power == 0):
                lower_act, upper_act = torch.pow(lower, power), torch.pow(upper, power)

                lower_out, upper_out = torch.zeros_like(lower), torch.zeros_like(upper)

                # expand power to match input shape for even/odd checks
                power_expanded = power.expand_as(lower)

                # Handle power == 0 case
                zero_power = power_expanded == 0
                lower_out[zero_power] = 1.0
                upper_out[zero_power] = 1.0

                # Even powers (excluding 0)
                even = ((power_expanded % 2) == 0) & (~zero_power)
                crossing = (lower < 0) & (upper > 0)

                crossing_even, not_crossing_even = (crossing & even), ((~crossing) & even)

                lower_out[crossing_even] = 0.0
                lower_out[not_crossing_even] = torch.min(lower_act[not_crossing_even], upper_act[not_crossing_even])

                upper_out[even] = torch.max(lower_act[even], upper_act[even])

                # Odd powers
                odd = (~even) & (~zero_power)
                lower_out[odd] = lower_act[odd]
                upper_out[odd] = upper_act[odd]

                return IntervalBounds(lower_out, upper_out)

            # No zeros, proceed with normal tensor logic
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
            # Scalar power case
            if power == 0:
                # x^0 = 1 for all x
                ones = torch.ones_like(lower)
                return IntervalBounds(ones, ones)

            if power % 2 == 0:
                # Even power: [a, b] → [min(a^n, b^n), max(a^n, b^n)] with special handling for crossing zero
                lower_act, upper_act = torch.pow(lower, power), torch.pow(upper, power)

                lower_out = torch.where((lower < 0) & (upper > 0), 0.0, torch.min(lower_act, upper_act))
                upper_out = torch.max(lower_act, upper_act)

                return IntervalBounds(lower_out, upper_out)
            else:
                # Odd power: [a, b] → [a^n, b^n]
                return IntervalBounds(torch.pow(lower, power), torch.pow(upper, power))
