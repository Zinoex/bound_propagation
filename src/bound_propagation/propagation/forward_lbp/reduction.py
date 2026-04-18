from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPSum(ForwardLBPStrategy):
    """Forward LBP strategy for sum reduction."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPSum requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        lower_bias = bounds.bias_lower.sum(dim, keepdim=keepdim)
        upper_bias = bounds.bias_upper.sum(dim, keepdim=keepdim)

        # Linear terms are only reduced on non-input dimensions (*batch_dims, *output_dims)
        linear_dim = tuple(range(bounds.shape)) if dim is None else dim

        linear_lowers = [linear.sum(linear_dim, keepdim=keepdim) for linear in bounds.linear_lowers]
        linear_uppers = [linear.sum(linear_dim, keepdim=keepdim) for linear in bounds.linear_uppers]

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lowers,
            bias_lower=lower_bias,
            linear_upper=linear_uppers,
            bias_upper=upper_bias,
            input_ids=bounds.input_ids,
        )


class ForwardLBPMean(ForwardLBPStrategy):
    """Forward LBP strategy for mean reduction."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPMean requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        lower_bias = bounds.bias_lower.mean(dim, keepdim=keepdim)
        upper_bias = bounds.bias_upper.mean(dim, keepdim=keepdim)

        # Linear terms are only reduced on non-input dimensions (*batch_dims, *output_dims)
        linear_dim = tuple(range(bounds.shape)) if dim is None else dim

        linear_lowers = [linear.mean(linear_dim, keepdim=keepdim) for linear in bounds.linear_lowers]
        linear_uppers = [linear.mean(linear_dim, keepdim=keepdim) for linear in bounds.linear_uppers]

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lowers,
            bias_lower=lower_bias,
            linear_upper=linear_uppers,
            bias_upper=upper_bias,
            input_ids=bounds.input_ids,
        )


class ForwardLBPMin(ForwardLBPStrategy):
    """Forward LBP strategy for min reduction."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPMin requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        lower, upper = bounds.concretize()

        # TODO: For min and max reductions, we need to compute a proper
        # linear relaxation unlike sum where the linear terms are just summed.
        # For now we just concretize the bounds and compute the min/max.

        if dim is not None:
            lower = lower.min(dim=dim, keepdim=keepdim).values
            upper = upper.min(dim=dim, keepdim=keepdim).values
        else:
            lower = lower.min()
            upper = upper.min()

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=lower,
            linear_upper=[],
            bias_upper=upper,
        )


class ForwardLBPMax(ForwardLBPStrategy):
    """Forward LBP strategy for max reduction."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPMax requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        lower, upper = bounds.concretize()

        if dim is not None:
            lower = lower.amax(dim=dim, keepdim=keepdim)
            upper = upper.amax(dim=dim, keepdim=keepdim)
        else:
            lower = lower.amax()
            upper = upper.amax()

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=lower,
            linear_upper=[],
            bias_upper=upper,
        )
