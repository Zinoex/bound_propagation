from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy
from .utils import transform_linear_terms

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPView(ForwardLBPStrategy):
    """Forward LBP strategy for view."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPView requires input to be LinearBounds")

        if len(args) == 2 and isinstance(args[1], (tuple, list, torch.Size)):
            shape = tuple(args[1])
        else:
            shape = tuple(args[1:])

        output_ndim = bounds.bias_lower.ndim

        linear_lower = transform_linear_terms(
            bounds.linear_lowers,
            lambda linear: linear.view(*shape, *linear.shape[output_ndim:]),
        )
        linear_upper = transform_linear_terms(
            bounds.linear_uppers,
            lambda linear: linear.view(*shape, *linear.shape[output_ndim:]),
        )

        bias_lower = bounds.bias_lower.view(*shape)
        bias_upper = bounds.bias_upper.view(*shape)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
