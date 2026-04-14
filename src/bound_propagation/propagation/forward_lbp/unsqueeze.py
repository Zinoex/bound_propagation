from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy
from .utils import transform_linear_terms

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPUnsqueeze(ForwardLBPStrategy):
    """Forward LBP strategy for unsqueeze."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPUnsqueeze requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        output_ndim = bounds.bias_lower.ndim
        if dim < 0:
            dim += output_ndim + 1
        if dim < 0 or dim > output_ndim:
            raise ValueError(f"unsqueeze dim must be in [0, {output_ndim}], got {dim}")

        linear_lower = transform_linear_terms(bounds.linear_lowers, lambda linear: linear.unsqueeze(dim))
        linear_upper = transform_linear_terms(bounds.linear_uppers, lambda linear: linear.unsqueeze(dim))
        bias_lower = bounds.bias_lower.unsqueeze(dim)
        bias_upper = bounds.bias_upper.unsqueeze(dim)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
