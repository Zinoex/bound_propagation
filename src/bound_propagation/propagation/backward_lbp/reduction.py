"""Backward LBP strategies for reduction operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ..linear_relaxations.base import (
    SymbolicIntervalLeaf,
    SymbolicLinearRelaxation,
)
from ..linear_relaxations.reduction import SymbolicMean, SymbolicSum
from .base import BackwardLBPStrategy, concretize_symbolic

if TYPE_CHECKING:
    from ..context import PropagationContext


class BackwardLBPSum(BackwardLBPStrategy):
    """Backward LBP strategy for sum reduction."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPSum requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)
        source_shape = node.args[0].meta["tensor_meta"]["shape"]

        return SymbolicSum(dim=dim, keepdim=keepdim, source_shape=source_shape, input=sym_input)


class BackwardLBPMean(BackwardLBPStrategy):
    """Backward LBP strategy for mean reduction."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPMean requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)
        source_shape = node.args[0].meta["tensor_meta"]["shape"]

        return SymbolicMean(dim=dim, keepdim=keepdim, source_shape=source_shape, input=sym_input)


class BackwardLBPMax(BackwardLBPStrategy):
    """Backward LBP strategy for amax reduction.

    Since amax is nonlinear, this concretizes the input subtree and wraps the
    result as an interval leaf (breaks the symbolic chain).
    """

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPMax requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        input_shape = node.args[0].meta["tensor_meta"]["shape"]
        input_dtype = node.args[0].meta["tensor_meta"]["dtype"]
        input_device = node.meta.get("device", "cpu")
        lower, upper = concretize_symbolic(sym_input, input_shape, input_dtype, input_device)

        if dim is not None:
            lower = lower.amax(dim=dim, keepdim=keepdim)
            upper = upper.amax(dim=dim, keepdim=keepdim)
        else:
            lower = lower.amax()
            upper = upper.amax()

        return SymbolicIntervalLeaf(lower=lower, upper=upper)


class BackwardLBPMin(BackwardLBPStrategy):
    """Backward LBP strategy for amin reduction.

    Since amin is nonlinear, this concretizes the input subtree and wraps the
    result as an interval leaf (breaks the symbolic chain).
    """

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPMin requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        input_shape = node.args[0].meta["tensor_meta"]["shape"]
        input_dtype = node.args[0].meta["tensor_meta"]["dtype"]
        input_device = node.meta.get("device", "cpu")
        lower, upper = concretize_symbolic(sym_input, input_shape, input_dtype, input_device)

        if dim is not None:
            lower = lower.amin(dim=dim, keepdim=keepdim)
            upper = upper.amin(dim=dim, keepdim=keepdim)
        else:
            lower = lower.amin()
            upper = upper.amin()

        return SymbolicIntervalLeaf(lower=lower, upper=upper)
