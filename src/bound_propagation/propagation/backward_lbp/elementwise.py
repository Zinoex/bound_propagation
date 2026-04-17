"""Backward LBP strategies for element-wise nonlinear operations.

These strategies concretize the input's symbolic subtree (via recursive CROWN)
to obtain interval bounds, compute the concrete relaxation, then wrap it in
a symbolic node via ``symbolic_forward``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ..linear_relaxations.base import SymbolicLinearRelaxation
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
from .base import BackwardLBPStrategy, concretize_symbolic

if TYPE_CHECKING:
    from ..context import PropagationContext


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
