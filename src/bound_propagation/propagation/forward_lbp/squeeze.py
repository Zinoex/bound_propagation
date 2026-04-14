from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy
from .utils import transform_linear_terms

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPSqueeze(ForwardLBPStrategy):
    """Forward LBP strategy for squeeze."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPSqueeze requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        output_ndim = bounds.bias_lower.ndim

        if dim is not None:
            if dim < 0:
                dim += output_ndim
            if dim < 0 or dim >= output_ndim:
                raise ValueError(f"squeeze dim must be in [0, {output_ndim - 1}], got {dim}")

            linear_lower = transform_linear_terms(bounds.linear_lowers, lambda linear: linear.squeeze(dim))
            linear_upper = transform_linear_terms(bounds.linear_uppers, lambda linear: linear.squeeze(dim))
            bias_lower = bounds.bias_lower.squeeze(dim)
            bias_upper = bounds.bias_upper.squeeze(dim)
        else:
            target_shape = tuple(size for size in bounds.bias_lower.shape if size != 1)
            bias_lower = bounds.bias_lower.reshape(target_shape)
            bias_upper = bounds.bias_upper.reshape(target_shape)
            output_ndim = bounds.bias_lower.ndim

            linear_lower = transform_linear_terms(
                bounds.linear_lowers,
                lambda linear: linear.reshape(*target_shape, *linear.shape[output_ndim:]),
            )
            linear_upper = transform_linear_terms(
                bounds.linear_uppers,
                lambda linear: linear.reshape(*target_shape, *linear.shape[output_ndim:]),
            )

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
