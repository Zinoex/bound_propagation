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


@final
@dataclass
class SymbolicReshape(SymbolicLinearRelaxation):
    """Backward through reshape / flatten / view."""

    source_shape: tuple[int, ...]  # full input shape (with batch)
    target_shape: tuple[int, ...]  # full output shape (with batch)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        target_features = self.target_shape[batch_ndim:]
        source_features = self.source_shape[batch_ndim:]
        bounded_ndim = A_lower.ndim - batch_ndim - len(target_features)
        batch_shape = A_lower.shape[:batch_ndim]
        bounded_shape = A_lower.shape[batch_ndim : batch_ndim + bounded_ndim]

        new_shape = (*batch_shape, *bounded_shape, *source_features)
        new_A_lower = A_lower.reshape(new_shape)
        new_A_upper = A_upper.reshape(new_shape)

        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicUnsqueeze(SymbolicLinearRelaxation):
    """Backward through unsqueeze(dim)."""

    dim: int  # non-negative, absolute dim in the output tensor
    output_ndim: int  # total ndim of the output tensor (including batch)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        # The unsqueezed dim in A's node portion
        bounded_ndim = A_lower.ndim - self.output_ndim
        a_dim = batch_ndim + bounded_ndim + (self.dim - batch_ndim)
        new_A_lower = A_lower.squeeze(a_dim)
        new_A_upper = A_upper.squeeze(a_dim)
        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicSqueeze(SymbolicLinearRelaxation):
    """Backward through squeeze(dim).

    For squeeze without a dim (squeeze all size-1 dims), use SymbolicReshape instead.
    """

    dim: int  # non-negative, absolute dim in the input tensor that was squeezed
    input_ndim: int  # total ndim of the input tensor (including batch)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        # Inverse of squeeze is unsqueeze at the same position
        output_ndim = self.input_ndim - 1  # squeeze removes one dim
        bounded_ndim = A_lower.ndim - (output_ndim - batch_ndim) - batch_ndim
        a_dim = batch_ndim + bounded_ndim + (self.dim - batch_ndim)
        new_A_lower = A_lower.unsqueeze(a_dim)
        new_A_upper = A_upper.unsqueeze(a_dim)
        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicTranspose(SymbolicLinearRelaxation):
    """Backward through transpose(dim0, dim1)."""

    dim0: int  # non-negative, absolute dim in the output tensor
    dim1: int  # non-negative, absolute dim in the output tensor
    output_ndim: int  # total ndim of the output (including batch)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        # transpose is its own inverse
        bounded_ndim = A_lower.ndim - self.output_ndim
        a_dim0 = batch_ndim + bounded_ndim + (self.dim0 - batch_ndim)
        a_dim1 = batch_ndim + bounded_ndim + (self.dim1 - batch_ndim)
        new_A_lower = A_lower.transpose(a_dim0, a_dim1)
        new_A_upper = A_upper.transpose(a_dim0, a_dim1)
        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicPermute(SymbolicLinearRelaxation):
    """Backward through permute(dims)."""

    perm: tuple[int, ...]  # non-negative, absolute dims in the output tensor
    output_ndim: int  # total ndim of the output (including batch)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        bounded_ndim = A_lower.ndim - self.output_ndim
        node_perm = tuple(p - batch_ndim for p in self.perm)

        # Inverse permutation
        inv_perm = [0] * len(node_perm)
        for i, p in enumerate(node_perm):
            inv_perm[p] = i

        a_perm = (
            tuple(range(batch_ndim))
            + tuple(range(batch_ndim, batch_ndim + bounded_ndim))
            + tuple(batch_ndim + bounded_ndim + d for d in inv_perm)
        )
        new_A_lower = A_lower.permute(a_perm)
        new_A_upper = A_upper.permute(a_perm)
        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicSelect(SymbolicLinearRelaxation):
    """Backward through select(dim, index)."""

    dim: int  # non-negative, absolute dim in the input tensor
    index: int
    source_shape: tuple[int, ...]  # full input shape (including batch)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        source_features = self.source_shape[batch_ndim:]
        bounded_ndim = A_lower.ndim - batch_ndim - (len(source_features) - 1)
        a_dim = batch_ndim + bounded_ndim + (self.dim - batch_ndim)

        # Unsqueeze to restore the dimension, then pad with zeros
        A_lower_expanded = A_lower.unsqueeze(a_dim)
        A_upper_expanded = A_upper.unsqueeze(a_dim)

        full_shape = list(A_lower_expanded.shape)
        full_shape[a_dim] = source_features[self.dim - batch_ndim]

        new_A_lower = torch.zeros(full_shape, dtype=A_lower.dtype, device=A_lower.device)
        new_A_upper = torch.zeros(full_shape, dtype=A_upper.dtype, device=A_upper.device)

        new_A_lower.narrow(a_dim, self.index, 1).copy_(A_lower_expanded)
        new_A_upper.narrow(a_dim, self.index, 1).copy_(A_upper_expanded)

        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicGetItem(SymbolicLinearRelaxation):
    """Backward through getitem (indexing)."""

    index: object
    source_shape: tuple[int, ...]  # full input shape (including batch)
    output_shape: tuple[int, ...]  # full output shape (including batch)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        source_features = self.source_shape[batch_ndim:]
        output_features = self.output_shape[batch_ndim:]
        bounded_ndim = A_lower.ndim - batch_ndim - len(output_features)
        batch_shape = A_lower.shape[:batch_ndim]
        bounded_shape = A_lower.shape[batch_ndim : batch_ndim + bounded_ndim]

        full_shape = (*batch_shape, *bounded_shape, *source_features)
        new_A_lower = torch.zeros(full_shape, dtype=A_lower.dtype, device=A_lower.device)
        new_A_upper = torch.zeros(full_shape, dtype=A_upper.dtype, device=A_upper.device)

        prefix_slices = (slice(None),) * (batch_ndim + bounded_ndim)
        if isinstance(self.index, tuple):
            full_index = prefix_slices + self.index
        else:
            full_index = prefix_slices + (self.index,)

        new_A_lower[full_index] = A_lower
        new_A_upper[full_index] = A_upper

        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicCatNode(SymbolicLinearRelaxation):
    """Backward through cat(tensors, dim)."""

    dim: int  # non-negative, absolute dim in the output tensor
    split_sizes: tuple[int, ...]  # size of each input along the cat dim
    output_ndim: int  # total ndim of the output (including batch)
    inputs: list[SymbolicLinearRelaxation]

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        node_ndim = self.output_ndim - batch_ndim
        bounded_ndim = A_lower.ndim - batch_ndim - node_ndim
        a_dim = batch_ndim + bounded_ndim + (self.dim - batch_ndim)

        A_lowers = A_lower.split(list(self.split_sizes), dim=a_dim)
        A_uppers = A_upper.split(list(self.split_sizes), dim=a_dim)

        bounds_list = []
        for A_l, A_u, inp in zip(A_lowers, A_uppers, self.inputs, strict=True):
            bounds_list.append(inp.backward(A_l, A_u, batch_ndim))

        zero = torch.zeros_like(bounds_list[0].bias_lower)
        return _merge_backward_bounds(bounds_list, zero, zero)


@final
@dataclass
class SymbolicStackNode(SymbolicLinearRelaxation):
    """Backward through stack(tensors, dim)."""

    dim: int  # non-negative, absolute dim in the output tensor
    output_ndim: int  # total ndim of the output (including batch)
    inputs: list[SymbolicLinearRelaxation]

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        node_ndim = self.output_ndim - batch_ndim
        bounded_ndim = A_lower.ndim - batch_ndim - node_ndim
        a_dim = batch_ndim + bounded_ndim + (self.dim - batch_ndim)

        bounds_list = []
        for i, inp in enumerate(self.inputs):
            A_l = A_lower.select(a_dim, i)
            A_u = A_upper.select(a_dim, i)
            bounds_list.append(inp.backward(A_l, A_u, batch_ndim))

        zero = torch.zeros_like(bounds_list[0].bias_lower)
        return _merge_backward_bounds(bounds_list, zero, zero)
