from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPDiv(ForwardIBPStrategy):
    """IBP strategy for DIV operation: [a, b] / [c, d]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"div requires 2 inputs, got {len(input_bounds)}")

        x_bounds = input_bounds[0]
        y_bounds = input_bounds[1]

        if not isinstance(x_bounds, IntervalBounds) or not isinstance(y_bounds, IntervalBounds):
            raise TypeError("IBPDiv requires both inputs to be IntervalBounds")

        # TODO: Fix because this is not sound.

        # Check if divisor can be zero
        if torch.any((y_bounds.lower <= 0) & (y_bounds.upper >= 0)):
            # Division by interval containing zero - return unbounded
            return IntervalBounds.unbounded_like(x_bounds)

        # Compute all four quotients
        ll = x_bounds.lower / y_bounds.lower
        lu = x_bounds.lower / y_bounds.upper
        ul = x_bounds.upper / y_bounds.lower
        uu = x_bounds.upper / y_bounds.upper

        # Take min and max
        lower = torch.min(torch.min(ll, lu), torch.min(ul, uu))
        upper = torch.max(torch.max(ll, lu), torch.max(ul, uu))

        return IntervalBounds(lower, upper)


class IBPDivConstant(ForwardIBPStrategy):
    """IBP strategy for DIV when divisor is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"div requires 2 inputs, got {len(input_bounds)}")

        x = input_bounds[0]
        y = input_bounds[1]

        if isinstance(x, IntervalBounds):
            interval = x
            c = y
        elif isinstance(y, IntervalBounds):
            interval = y
            c = x
        else:
            raise TypeError(
                "IBPDivConstantStrategy requires the first input to be IntervalBounds and "
                f"the second input to be torch.Tensor or Number, got {type(x)} and {type(y)}"
            )

        if isinstance(c, torch.Tensor):
            lower = torch.where(
                c >= 0,
                interval.lower / c,
                interval.upper / c,
            )
            lower = torch.where(c == 0, float("-inf"), lower)

            upper = torch.where(
                c >= 0,
                interval.upper / c,
                interval.lower / c,
            )
            upper = torch.where(c == 0, float("inf"), upper)
            return IntervalBounds(lower, upper)
        elif isinstance(c, torch.types.Number):
            if c == 0:
                return IntervalBounds.unbounded_like(interval)

            lower = interval.lower / c
            upper = interval.upper / c

            if c < 0:
                lower, upper = upper, lower

            return IntervalBounds(lower, upper)
        else:
            raise TypeError(f"Constant input must be torch.Tensor or Number, got {type(c)}")


class IBPConstantDiv(ForwardIBPStrategy):
    """IBP strategy for DIV when dividend is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"div requires 2 inputs, got {len(input_bounds)}")

        x = input_bounds[0]
        y = input_bounds[1]

        if isinstance(x, (torch.Tensor, torch.types.Number)) and isinstance(y, IntervalBounds):
            c = x
            interval = y
        elif isinstance(y, (torch.Tensor, torch.types.Number)) and isinstance(x, IntervalBounds):
            c = y
            interval = x
        else:
            raise TypeError(
                "IBPConstantDiv requires the first input to be torch.Tensor or Number and "
                f"the second input to be IntervalBounds, got {type(x)} and {type(y)}"
            )

        # For c / [a, b], if
        # - [a, b] contains 0, return [-inf, inf]
        # - c > 0, return [c/b, c/a]
        # - c < 0, return [c/a, c/b]
        unbounded_mask = (interval.lower <= 0) & (interval.upper >= 0)

        if isinstance(c, torch.Tensor):
            lower = torch.where(
                c >= 0,
                c / interval.upper,
                c / interval.lower,
            )
            lower = torch.where(unbounded_mask, float("-inf"), lower)

            upper = torch.where(
                c >= 0,
                c / interval.lower,
                c / interval.upper,
            )
            upper = torch.where(unbounded_mask, float("inf"), upper)

            return IntervalBounds(lower, upper)
        elif isinstance(c, torch.types.Number):
            if c == 0:
                zero = torch.zeros_like(interval.lower)
                return IntervalBounds(zero, zero)

            lower = c / interval.upper
            lower = torch.where(unbounded_mask, float("-inf"), lower)

            upper = c / interval.lower
            upper = torch.where(unbounded_mask, float("inf"), upper)

            return IntervalBounds(lower, upper)
        else:
            raise TypeError(f"Constant input must be torch.Tensor or Number, got {type(c)}")
