from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from ...bounds import LinearBounds
from ..linear_relaxations.elementwise import (
    ElementwiseParams,
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
    import torch
    import torch.fx as fx

    from ..context import PropagationContext


@final
@dataclass
class ElementwiseForwardRelaxation:
    """
    Element-wise linear relaxation for unary operations y = f(x).

    Stores four element-wise tensors (same shape as x / y):
        y_lower >= alpha_lower * x + beta_lower
        y_upper <= alpha_upper * x + beta_upper

    The abstract dimension convention for LinearBounds linear terms is
    (*batch_dims, *output_dims, *input_dims).  alpha and beta live in
    (*batch_dims, *output_dims); forward appends the input trailing
    axes via broadcasting.

    Attributes:
        alpha_lower: Element-wise slopes for the lower bound.
        beta_lower:  Element-wise biases for the lower bound.
        alpha_upper: Element-wise slopes for the upper bound.
        beta_upper:  Element-wise biases for the upper bound.
    """

    params: ElementwiseParams

    # ------------------------------------------------------------------
    # Forward composition
    # ------------------------------------------------------------------

    def forward(self, input_bounds: LinearBounds) -> LinearBounds:
        """
        Compose: y = alpha * x + beta  composed with  x = W @ x0 + b  →  y = W_new @ x0 + b_new.

        Linear terms have shape (*batch_dims, *output_dims, *input_dims).
        alpha/beta have shape (*batch_dims, *output_dims); trailing input axes are
        broadcast by appending ones.

        Handles signed alpha via positive/negative clamping so the result is always
        a valid lower/upper bound.
        """

        al_pos = self.params.alpha_lower.clamp(min=0)
        al_neg = self.params.alpha_lower.clamp(max=0)
        au_pos = self.params.alpha_upper.clamp(min=0)
        au_neg = self.params.alpha_upper.clamp(max=0)

        def broadcast(alpha: torch.Tensor, linear: torch.Tensor) -> torch.Tensor:
            # alpha: (*batch_dims, *output_dims)
            # linear: (*batch_dims, *output_dims, *input_dims)
            # Append one dimension per input axis so broadcasting is correct.
            extra = linear.ndim - alpha.ndim
            return alpha.reshape(alpha.shape + (1,) * extra)

        # Lower bound: alpha_lower_pos * W_lower  +  alpha_lower_neg * W_upper
        linear_lower = [
            broadcast(al_pos, wl) * wl + broadcast(al_neg, wu) * wu
            for wl, wu in zip(input_bounds.linear_lowers, input_bounds.linear_uppers, strict=True)
        ]
        bias_lower = al_pos * input_bounds.bias_lower + al_neg * input_bounds.bias_upper + self.params.beta_lower

        # Upper bound: alpha_upper_pos * W_upper  +  alpha_upper_neg * W_lower
        if input_bounds.linear_lowers is None and input_bounds.linear_uppers is None:
            linear_upper = None
        else:
            linear_upper = [
                broadcast(au_pos, wu) * wu + broadcast(au_neg, wl) * wl
                for wl, wu in zip(input_bounds.linear_lowers, input_bounds.linear_uppers, strict=True)
            ]
        bias_upper = au_pos * input_bounds.bias_upper + au_neg * input_bounds.bias_lower + self.params.beta_upper

        return LinearBounds(
            regions=input_bounds.regions,
            linear_lower=linear_lower or None,
            bias_lower=bias_lower,
            linear_upper=linear_upper or None,
            bias_upper=bias_upper,
            input_ids=input_bounds.input_ids or None,
        )


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

        concrete_bounds = bounds.concretize()
        params = compute_abs_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_clamp_relaxation(concrete_bounds, min_val, max_val)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_cos_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_exp_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_log_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_reciprocal_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_relu_relaxation(concrete_bounds, adaptive=False)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_sigmoid_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_sin_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_sqrt_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_tan_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)


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

        concrete_bounds = bounds.concretize()
        params = compute_tanh_relaxation(concrete_bounds)
        relaxation = ElementwiseForwardRelaxation(params=params)
        return relaxation.forward(bounds)
