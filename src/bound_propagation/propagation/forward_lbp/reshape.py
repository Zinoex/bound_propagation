from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy
from .utils import transform_linear_terms

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPReshape(ForwardLBPStrategy):
    """Forward LBP strategy for reshape."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPReshape requires input to be LinearBounds")

        if len(args) == 2 and isinstance(args[1], (tuple, list, torch.Size)):
            target_shape = tuple(args[1])
        else:
            target_shape = tuple(args[1:])

        bias_lower = bounds.bias_lower.reshape(target_shape)
        bias_upper = bounds.bias_upper.reshape(target_shape)
        linear_lower = transform_linear_terms(
            bounds.linear_lowers,
            lambda linear: linear.reshape(*target_shape, linear.shape[-1]),
        )
        linear_upper = transform_linear_terms(
            bounds.linear_uppers,
            lambda linear: linear.reshape(*target_shape, linear.shape[-1]),
        )

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
