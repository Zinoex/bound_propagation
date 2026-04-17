from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..linear_relaxations.elementwise import (
    compute_abs_relaxation,
    compute_clamp_relaxation,
    compute_cos_relaxation,
    compute_exp_relaxation,
    compute_log_relaxation,
    compute_reciprocal_relaxation,
    compute_relu_relaxation,
    compute_sigmoid_relaxation,
    compute_sin_relaxation,
    compute_sqrt_relaxation,
    compute_tan_relaxation,
    compute_tanh_relaxation,
)
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch.fx as fx

    from ..context import PropagationContext


class ForwardLBPAbs(ForwardLBPStrategy):
    """Forward LBP strategy for abs using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPAbs requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_abs_relaxation(lower, upper)
        return relaxation.forward([bounds])


class ForwardLBPClamp(ForwardLBPStrategy):
    """Forward LBP strategy for clamp using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPClamp requires input to be LinearBounds")

        min_val = args[1] if len(args) > 1 else kwargs.get("min")
        max_val = args[2] if len(args) > 2 else kwargs.get("max")

        lower, upper = bounds.concretize()
        relaxation = compute_clamp_relaxation(lower, upper, min_val, max_val)
        return relaxation.forward([bounds])


class ForwardLBPCos(ForwardLBPStrategy):
    """Forward LBP strategy for cos using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPCos requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_cos_relaxation(lower, upper)
        return relaxation.forward([bounds])


class ForwardLBPExp(ForwardLBPStrategy):
    """Forward LBP strategy for exp using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPExp requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_exp_relaxation(lower, upper)
        return relaxation.forward([bounds])


class ForwardLBPLog(ForwardLBPStrategy):
    """Forward LBP strategy for log using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPLog requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_log_relaxation(lower, upper)
        return relaxation.forward([bounds])


class ForwardLBPReciprocal(ForwardLBPStrategy):
    """Forward LBP strategy for reciprocal using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPReciprocal requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_reciprocal_relaxation(lower, upper)
        return relaxation.forward([bounds])


class ForwardLBPRelu(ForwardLBPStrategy):
    """Forward LBP strategy for ReLU using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPRelu requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_relu_relaxation(lower, upper, adaptive=False)
        return relaxation.forward([bounds])


class ForwardLBPSigmoid(ForwardLBPStrategy):
    """Forward LBP strategy for sigmoid using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPSigmoid requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_sigmoid_relaxation(lower, upper)
        return relaxation.forward([bounds])


class ForwardLBPSin(ForwardLBPStrategy):
    """Forward LBP strategy for sin using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPSin requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_sin_relaxation(lower, upper)
        return relaxation.forward([bounds])


class ForwardLBPSqrt(ForwardLBPStrategy):
    """Forward LBP strategy for sqrt using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPSqrt requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_sqrt_relaxation(lower, upper)
        return relaxation.forward([bounds])


class ForwardLBPTan(ForwardLBPStrategy):
    """Forward LBP strategy for tan using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPTan requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_tan_relaxation(lower, upper)
        return relaxation.forward([bounds])


class ForwardLBPTanh(ForwardLBPStrategy):
    """Forward LBP strategy for tanh using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPTanh requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_tanh_relaxation(lower, upper)
        return relaxation.forward([bounds])
