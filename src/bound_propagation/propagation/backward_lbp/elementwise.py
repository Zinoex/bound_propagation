"""Backward LBP strategies for element-wise nonlinear operations.

These strategies concretize the input's symbolic subtree (via recursive CROWN)
to obtain interval bounds, compute the concrete relaxation, then wrap it in
a symbolic node via ``symbolic_forward``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.fx as fx
from beartype.typing import final

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
from .base import BackwardLBPStrategy, BackwardLinearRelaxation, concretize_symbolic

if TYPE_CHECKING:
    from ..context import PropagationContext


@final
@dataclass
class ElementwiseBackwardLinearRelaxation(BackwardLinearRelaxation):
    params: ElementwiseParams

    input: BackwardLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        r = self.concrete_relaxation
        node_ndim = r.alpha_lower.ndim - batch_ndim
        bounded_ndim = A_lower.ndim - r.alpha_lower.ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            """Broadcast ``(*batch, *node)`` → ``(*batch, *bounded_out, *node)``."""
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        A_l_pos = A_lower.clamp(min=0)
        A_l_neg = A_lower.clamp(max=0)
        A_u_pos = A_upper.clamp(min=0)
        A_u_neg = A_upper.clamp(max=0)

        # Sign decomposition: where A > 0 use same-side relaxation,
        # where A < 0 use opposite-side relaxation.
        new_A_lower = A_l_pos * bc(r.alpha_lower) + A_l_neg * bc(r.alpha_upper)
        new_A_upper = A_u_pos * bc(r.alpha_upper) + A_u_neg * bc(r.alpha_lower)

        bounds = self.input.backward(new_A_lower, new_A_upper, batch_ndim)

        # Bias contribution: sum over the trailing node dimensions.
        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        delta_bias_lower = A_l_pos * bc(r.beta_lower) + A_l_neg * bc(r.beta_upper)
        delta_bias_upper = A_u_pos * bc(r.beta_upper) + A_u_neg * bc(r.beta_lower)
        if sum_dims:
            delta_bias_lower = delta_bias_lower.sum(dim=sum_dims)
            delta_bias_upper = delta_bias_upper.sum(dim=sum_dims)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=bounds.linear_lowers,
            bias_lower=bounds.bias_lower + delta_bias_lower,
            linear_upper=bounds.linear_uppers,
            bias_upper=bounds.bias_upper + delta_bias_upper,
            input_ids=bounds.input_ids,
            validate=False,
        )


class _ElementwiseBackwardLBP(BackwardLBPStrategy):
    """Base for element-wise nonlinear backward LBP strategies."""

    def _get_input_sym_and_bounds(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> tuple[SymbolicLinearRelaxation, tuple]:
        args, _ = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(
                f"{self.__class__.__name__} requires input to be SymbolicLinearRelaxation, got {type(sym_input)}"
            )
        input_shape = node.args[0].meta["tensor_meta"]["shape"]
        input_dtype = node.args[0].meta["tensor_meta"]["dtype"]
        input_device = node.meta.get("device", "cpu")
        lower, upper = concretize_symbolic(sym_input, input_shape, input_dtype, input_device)
        return sym_input, (lower, upper)


class BackwardLBPRelu(_ElementwiseBackwardLBP):
    """Backward LBP strategy for ReLU."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_relu_relaxation(lower, upper, adaptive=False)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPSigmoid(_ElementwiseBackwardLBP):
    """Backward LBP strategy for sigmoid."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_sigmoid_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPTanh(_ElementwiseBackwardLBP):
    """Backward LBP strategy for tanh."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_tanh_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPExp(_ElementwiseBackwardLBP):
    """Backward LBP strategy for exp."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_exp_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPLog(_ElementwiseBackwardLBP):
    """Backward LBP strategy for log."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_log_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPSqrt(_ElementwiseBackwardLBP):
    """Backward LBP strategy for sqrt."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_sqrt_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPReciprocal(_ElementwiseBackwardLBP):
    """Backward LBP strategy for reciprocal."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_reciprocal_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPAbs(_ElementwiseBackwardLBP):
    """Backward LBP strategy for abs."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_abs_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPSin(_ElementwiseBackwardLBP):
    """Backward LBP strategy for sin."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_sin_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPCos(_ElementwiseBackwardLBP):
    """Backward LBP strategy for cos."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_cos_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPTan(_ElementwiseBackwardLBP):
    """Backward LBP strategy for tan."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        sym_input, (lower, upper) = self._get_input_sym_and_bounds(node, ctx)
        relaxation = compute_tan_relaxation(lower, upper)
        return relaxation.symbolic_forward([sym_input])


class BackwardLBPClamp(BackwardLBPStrategy):
    """Backward LBP strategy for clamp."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPClamp requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        min_val = args[1] if len(args) > 1 else kwargs.get("min")
        max_val = args[2] if len(args) > 2 else kwargs.get("max")

        input_shape = node.args[0].meta["tensor_meta"]["shape"]
        input_dtype = node.args[0].meta["tensor_meta"]["dtype"]
        input_device = node.meta.get("device", "cpu")
        lower, upper = concretize_symbolic(sym_input, input_shape, input_dtype, input_device)

        relaxation = compute_clamp_relaxation(lower, upper, min_val, max_val)
        return relaxation.symbolic_forward([sym_input])
