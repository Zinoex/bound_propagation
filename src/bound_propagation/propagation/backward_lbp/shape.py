"""Backward LBP strategies for shape manipulation operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ..linear_relaxations.base import SymbolicLinearRelaxation
from ..linear_relaxations.shape import (
    SymbolicCatNode,
    SymbolicGetItem,
    SymbolicPermute,
    SymbolicReshape,
    SymbolicSelect,
    SymbolicSqueeze,
    SymbolicStackNode,
    SymbolicTranspose,
    SymbolicUnsqueeze,
)
from .base import BackwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class BackwardLBPReshape(BackwardLBPStrategy):
    """Backward LBP strategy for reshape / view / flatten."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, _ = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPReshape requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        source_shape = node.args[0].meta["tensor_meta"]["shape"]
        target_shape = node.meta["tensor_meta"]["shape"]

        return SymbolicReshape(source_shape=source_shape, target_shape=target_shape, input=sym_input)


class BackwardLBPUnsqueeze(BackwardLBPStrategy):
    """Backward LBP strategy for unsqueeze."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(
                f"BackwardLBPUnsqueeze requires input to be SymbolicLinearRelaxation, got {type(sym_input)}"
            )

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        output_ndim = len(node.meta["tensor_meta"]["shape"])

        # Normalize negative dim
        if dim < 0:
            dim += output_ndim

        return SymbolicUnsqueeze(dim=dim, output_ndim=output_ndim, input=sym_input)


class BackwardLBPSqueeze(BackwardLBPStrategy):
    """Backward LBP strategy for squeeze."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPSqueeze requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        input_ndim = len(node.args[0].meta["tensor_meta"]["shape"])

        if dim is not None:
            if dim < 0:
                dim += input_ndim
            return SymbolicSqueeze(dim=dim, input_ndim=input_ndim, input=sym_input)

        # squeeze(None) removes all size-1 dims -> use reshape
        source_shape = node.args[0].meta["tensor_meta"]["shape"]
        target_shape = node.meta["tensor_meta"]["shape"]
        return SymbolicReshape(source_shape=source_shape, target_shape=target_shape, input=sym_input)


class BackwardLBPTranspose(BackwardLBPStrategy):
    """Backward LBP strategy for transpose."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(
                f"BackwardLBPTranspose requires input to be SymbolicLinearRelaxation, got {type(sym_input)}"
            )

        dim0 = args[1] if len(args) > 1 else kwargs.get("dim0", 0)
        dim1 = args[2] if len(args) > 2 else kwargs.get("dim1", 1)
        output_ndim = len(node.meta["tensor_meta"]["shape"])

        if dim0 < 0:
            dim0 += output_ndim
        if dim1 < 0:
            dim1 += output_ndim

        return SymbolicTranspose(dim0=dim0, dim1=dim1, output_ndim=output_ndim, input=sym_input)


class BackwardLBPPermute(BackwardLBPStrategy):
    """Backward LBP strategy for permute."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPPermute requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        if len(args) == 2 and isinstance(args[1], (tuple, list)):
            dims = tuple(args[1])
        else:
            dims = tuple(args[1:])

        output_ndim = len(node.meta["tensor_meta"]["shape"])
        dims = tuple(d + output_ndim if d < 0 else d for d in dims)

        return SymbolicPermute(perm=dims, output_ndim=output_ndim, input=sym_input)


class BackwardLBPSelect(BackwardLBPStrategy):
    """Backward LBP strategy for select."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPSelect requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        index = args[2] if len(args) > 2 else kwargs.get("index", 0)
        source_shape = node.args[0].meta["tensor_meta"]["shape"]

        input_ndim = len(source_shape)
        if dim < 0:
            dim += input_ndim

        return SymbolicSelect(dim=dim, index=index, source_shape=source_shape, input=sym_input)


class BackwardLBPGetItem(BackwardLBPStrategy):
    """Backward LBP strategy for getitem (operator.getitem)."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, _ = ctx.resolve_args(node)
        sym_input = args[0]
        index = args[1]

        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPGetItem requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        source_shape = node.args[0].meta["tensor_meta"]["shape"]
        output_shape = node.meta["tensor_meta"]["shape"]

        return SymbolicGetItem(index=index, source_shape=source_shape, output_shape=output_shape, input=sym_input)


class BackwardLBPConcat(BackwardLBPStrategy):
    """Backward LBP strategy for torch.cat."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        inputs: list[SymbolicLinearRelaxation] = []
        split_sizes: list[int] = []

        for i, (t, raw_arg) in enumerate(zip(tensors, node.args[0], strict=True)):
            if not isinstance(t, SymbolicLinearRelaxation):
                raise TypeError(
                    f"BackwardLBPConcat requires all inputs to be SymbolicLinearRelaxation, but input {i} is {type(t)}"
                )
            inputs.append(t)
            split_sizes.append(raw_arg.meta["tensor_meta"]["shape"][dim])

        output_ndim = len(node.meta["tensor_meta"]["shape"])
        if dim < 0:
            dim += output_ndim

        return SymbolicCatNode(dim=dim, split_sizes=tuple(split_sizes), output_ndim=output_ndim, inputs=inputs)


class BackwardLBPStack(BackwardLBPStrategy):
    """Backward LBP strategy for torch.stack."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        inputs: list[SymbolicLinearRelaxation] = []
        for i, t in enumerate(tensors):
            if not isinstance(t, SymbolicLinearRelaxation):
                raise TypeError(
                    f"BackwardLBPStack requires all inputs to be SymbolicLinearRelaxation, but input {i} is {type(t)}"
                )
            inputs.append(t)

        output_ndim = len(node.meta["tensor_meta"]["shape"])
        if dim < 0:
            dim += output_ndim

        return SymbolicStackNode(dim=dim, output_ndim=output_ndim, inputs=inputs)
