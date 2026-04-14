from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy
from .utils import combine_linear_terms

if TYPE_CHECKING:
    import torch
    import torch.fx as fx

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
        lower_regions, linear_lower, input_ids = combine_linear_terms([(a, "lower", 1.0), (b, "lower", 1.0)])
        upper_regions, linear_upper, upper_input_ids = combine_linear_terms([(a, "upper", 1.0), (b, "upper", 1.0)])

        if input_ids != upper_input_ids:
            raise ValueError(f"Lower and upper input IDs must match, got {input_ids} vs {upper_input_ids}")

        return LinearBounds(
            regions=lower_regions or upper_regions,
            linear_lower=linear_lower,
            bias_lower=a.bias_lower + b.bias_lower,
            linear_upper=linear_upper,
            bias_upper=a.bias_upper + b.bias_upper,
            input_ids=input_ids,
        )

    def _add_constant(self, bounds: LinearBounds, constant: torch.Tensor | torch.types.Number) -> LinearBounds:
        return LinearBounds(
            regions=bounds.regions,
            linear_lower=bounds.linear_lowers,
            bias_lower=bounds.bias_lower + constant,
            linear_upper=bounds.linear_uppers,
            bias_upper=bounds.bias_upper + constant,
            input_ids=bounds.input_ids,
        )
