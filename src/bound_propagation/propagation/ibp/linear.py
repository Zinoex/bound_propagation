from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPLinear(ForwardIBPStrategy):
    """IBP strategy for linear: y = x @ W^T + b."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPLinear requires input to be IntervalBounds")

        if node.op == "call_module":
            if not isinstance(node.target, str):
                raise TypeError(f"Expected node.target to be str for call_module, got {type(node.target)}")

            module = ctx.get_module(node.target)
            weight: torch.Tensor = module.weight  # ty:ignore[invalid-assignment]
            bias: torch.Tensor | None = getattr(module, "bias", None)
        else:
            # F.linear(input, weight, bias=None)
            weight: torch.Tensor = args[1]
            bias: torch.Tensor | None = args[2] if len(args) > 2 else kwargs.get("bias")

        if weight.ndim not in (1, 2):
            raise ValueError(f"linear weight must be 1D or 2D, got shape {tuple(weight.shape)}")

        weight_pos = torch.clamp(weight, min=0)
        weight_neg = torch.clamp(weight, max=0)

        if weight.ndim == 2:
            lower = x_bounds.lower @ weight_pos.T + x_bounds.upper @ weight_neg.T
            upper = x_bounds.upper @ weight_pos.T + x_bounds.lower @ weight_neg.T
        else:
            # 1D weight: y = (x * w).sum(-1); the last feature dim is reduced.
            lower = (x_bounds.lower * weight_pos + x_bounds.upper * weight_neg).sum(-1)
            upper = (x_bounds.upper * weight_pos + x_bounds.lower * weight_neg).sum(-1)

        if bias is not None:
            lower = lower + bias
            upper = upper + bias

        return IntervalBounds(lower, upper, batch_ndim=x_bounds.batch_ndim)


class IBPAdd(ForwardIBPStrategy):
    """IBP strategy for addition (abstract+abstract or abstract+constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, IntervalBounds) and isinstance(right, IntervalBounds):
            return IntervalBounds(
                left.lower + right.lower,
                left.upper + right.upper,
                batch_ndim=max(left.batch_ndim, right.batch_ndim),
            )

        if isinstance(left, IntervalBounds):
            return IntervalBounds(left.lower + right, left.upper + right, batch_ndim=left.batch_ndim)

        if isinstance(right, IntervalBounds):
            return IntervalBounds(left + right.lower, left + right.upper, batch_ndim=right.batch_ndim)

        raise TypeError(f"IBPAdd requires at least one IntervalBounds, got {type(left)} and {type(right)}")


class IBPSub(ForwardIBPStrategy):
    """IBP strategy for subtraction (all combinations of abstract/constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, IntervalBounds) and isinstance(right, IntervalBounds):
            return IntervalBounds(
                left.lower - right.upper,
                left.upper - right.lower,
                batch_ndim=max(left.batch_ndim, right.batch_ndim),
            )

        if isinstance(left, IntervalBounds):
            # abstract - constant
            return IntervalBounds(left.lower - right, left.upper - right, batch_ndim=left.batch_ndim)

        if isinstance(right, IntervalBounds):
            # constant - abstract
            return IntervalBounds(left - right.upper, left - right.lower, batch_ndim=right.batch_ndim)

        raise TypeError(f"IBPSub requires at least one IntervalBounds, got {type(left)} and {type(right)}")


class IBPNeg(ForwardIBPStrategy):
    """IBP strategy for negation: -[a, b] = [-b, -a]."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPNeg requires input to be IntervalBounds")

        return IntervalBounds(-x_bounds.upper, -x_bounds.lower, batch_ndim=x_bounds.batch_ndim)
