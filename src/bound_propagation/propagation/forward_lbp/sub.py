from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy
from .utils import combine_linear_terms

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
                regions=left.regions,
                linear_lower=left.linear_lowers,
                bias_lower=left.bias_lower - right,
                linear_upper=left.linear_uppers,
                bias_upper=left.bias_upper - right,
                input_ids=left.input_ids,
            )

        if isinstance(right, LinearBounds):
            # c - x: flip signs and bounds
            return LinearBounds(
                regions=right.regions,
                linear_lower=[-linear for linear in right.linear_uppers],
                bias_lower=left - right.bias_upper,
                linear_upper=[-linear for linear in right.linear_lowers],
                bias_upper=left - right.bias_lower,
                input_ids=right.input_ids,
            )

        raise TypeError(f"ForwardLBPSub requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _sub_bounds(self, a: LinearBounds, b: LinearBounds) -> LinearBounds:
        lower_regions, linear_lower, input_ids = combine_linear_terms([(a, "lower", 1.0), (b, "upper", -1.0)])
        upper_regions, linear_upper, upper_input_ids = combine_linear_terms([(a, "upper", 1.0), (b, "lower", -1.0)])

        if input_ids != upper_input_ids:
            raise ValueError(f"Lower and upper input IDs must match, got {input_ids} vs {upper_input_ids}")

        return LinearBounds(
            regions=lower_regions or upper_regions,
            linear_lower=linear_lower,
            bias_lower=a.bias_lower - b.bias_upper,
            linear_upper=linear_upper,
            bias_upper=a.bias_upper - b.bias_lower,
            input_ids=input_ids,
        )
