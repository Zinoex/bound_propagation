from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from ...bounds import LinearBounds
from .base import SymbolicLinearRelaxation, _merge_backward_bounds


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
