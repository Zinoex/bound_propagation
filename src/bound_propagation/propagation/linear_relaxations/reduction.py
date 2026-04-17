from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from ...bounds import LinearBounds
from .base import SymbolicLinearRelaxation


@final
@dataclass
class SymbolicSum(SymbolicLinearRelaxation):
    """Backward through sum(dim, keepdim)."""

    dim: int | tuple[int, ...] | None
    keepdim: bool
    source_shape: tuple[int, ...]  # full input shape (including batch)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        source_features = self.source_shape[batch_ndim:]

        if self.dim is None:
            # Full reduction: A has no node dims, expand to source features
            new_A_lower = A_lower
            new_A_upper = A_upper
            for _ in source_features:
                new_A_lower = new_A_lower.unsqueeze(-1)
                new_A_upper = new_A_upper.unsqueeze(-1)
            new_A_lower = new_A_lower.expand(*A_lower.shape, *source_features)
            new_A_upper = new_A_upper.expand(*A_upper.shape, *source_features)
        else:
            dims = (self.dim,) if isinstance(self.dim, int) else self.dim
            norm_dims = tuple(d if d >= 0 else d + len(self.source_shape) for d in dims)
            node_dims = tuple(d - batch_ndim for d in norm_dims)

            if self.keepdim:
                node_ndim = len(source_features)
            else:
                node_ndim = len(source_features) - len(node_dims)
            bounded_ndim = A_lower.ndim - batch_ndim - node_ndim

            if not self.keepdim:
                new_A_lower = A_lower
                new_A_upper = A_upper
                for d in sorted(node_dims):
                    a_d = batch_ndim + bounded_ndim + d
                    new_A_lower = new_A_lower.unsqueeze(a_d)
                    new_A_upper = new_A_upper.unsqueeze(a_d)
            else:
                new_A_lower = A_lower
                new_A_upper = A_upper

            expand_shape = list(new_A_lower.shape)
            for d in node_dims:
                expand_shape[batch_ndim + bounded_ndim + d] = source_features[d]
            new_A_lower = new_A_lower.expand(expand_shape)
            new_A_upper = new_A_upper.expand(expand_shape)

        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicMean(SymbolicLinearRelaxation):
    """Backward through mean(dim, keepdim)."""

    dim: int | tuple[int, ...] | None
    keepdim: bool
    source_shape: tuple[int, ...]  # full input shape (including batch)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        # mean = sum / count
        if self.dim is None:
            count = 1
            for d in self.source_shape[batch_ndim:]:
                count *= d
        else:
            dims = (self.dim,) if isinstance(self.dim, int) else self.dim
            norm_dims = tuple(d if d >= 0 else d + len(self.source_shape) for d in dims)
            count = 1
            for d in norm_dims:
                count *= self.source_shape[d]

        sym_sum = SymbolicSum(
            dim=self.dim,
            keepdim=self.keepdim,
            source_shape=self.source_shape,
            input=self.input,
        )
        return sym_sum.backward(A_lower / count, A_upper / count, batch_ndim)
