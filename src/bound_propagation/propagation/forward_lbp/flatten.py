from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy
from .utils import transform_linear_terms

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPFlatten(ForwardLBPStrategy):
    """Forward LBP strategy for flatten."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPFlatten requires input to be LinearBounds")

        start_dim = args[1] if len(args) > 1 else kwargs.get("start_dim", 0)
        end_dim = args[2] if len(args) > 2 else kwargs.get("end_dim", -1)

        output_ndim = bounds.bias_lower.ndim
        if start_dim < 0:
            start_dim += output_ndim
        if end_dim < 0:
            end_dim += output_ndim

        if start_dim < 0 or start_dim >= output_ndim or end_dim < 0 or end_dim >= output_ndim:
            raise ValueError(
                f"flatten dims must be in [0, {output_ndim - 1}], got start_dim={start_dim}, end_dim={end_dim}"
            )

        if end_dim < start_dim:
            raise ValueError(f"flatten end_dim must be >= start_dim, got start_dim={start_dim}, end_dim={end_dim}")

        bias_lower = bounds.bias_lower.flatten(start_dim, end_dim)
        bias_upper = bounds.bias_upper.flatten(start_dim, end_dim)
        linear_lower = transform_linear_terms(
            bounds.linear_lowers,
            lambda linear: linear.flatten(start_dim, end_dim),
        )
        linear_upper = transform_linear_terms(
            bounds.linear_uppers,
            lambda linear: linear.flatten(start_dim, end_dim),
        )

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
