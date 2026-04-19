"""Backward LBP strategies and relaxations for reduction operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.fx as fx
from beartype.typing import final

from .base import (
    BackwardContributions,
    BackwardLBPStrategy,
    BackwardRelaxation,
    IntermediateBoundsProvider,
    IntervalLeafRelaxation,
)

if TYPE_CHECKING:
    from .tape import BackwardTape


@final
@dataclass
class SumRelaxation(BackwardRelaxation):
    """Backward relaxation for sum reduction.

    Reverses the sum by expanding A-matrices back to the source shape
    and contributes zero bias.
    """

    dim: int | tuple[int, ...] | None
    keepdim: bool
    source_shape: tuple[int, ...]
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
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

        # Bias shape is everything except the source node dimensions
        source_node_ndim = len(source_features)
        bias_shape = new_A_lower.shape[: new_A_lower.ndim - source_node_ndim]
        zero = torch.zeros(bias_shape, dtype=A_lower.dtype, device=A_lower.device)

        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=zero,
            bias_upper=zero,
        )


@final
@dataclass
class MeanRelaxation(BackwardRelaxation):
    """Backward relaxation for mean reduction.

    Delegates to ``SumRelaxation`` after dividing the A-matrices by the
    number of elements being averaged.
    """

    dim: int | tuple[int, ...] | None
    keepdim: bool
    source_shape: tuple[int, ...]
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
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

        sum_relaxation = SumRelaxation(
            dim=self.dim,
            keepdim=self.keepdim,
            source_shape=self.source_shape,
            input_node=self.input_node,
        )
        return sum_relaxation.backward_through(A_lower / count, A_upper / count, batch_ndim)


class BackwardLBPSum(BackwardLBPStrategy):
    """Backward LBP strategy for sum reduction."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPSum requires input to be BackwardRelaxation, got {type(sym_input).__name__}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)
        source_shape = node.args[0].meta["tensor_meta"]["shape"]

        return SumRelaxation(dim=dim, keepdim=keepdim, source_shape=source_shape, input_node=node.args[0])


class BackwardLBPMean(BackwardLBPStrategy):
    """Backward LBP strategy for mean reduction."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPMean requires input to be BackwardRelaxation, got {type(sym_input).__name__}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)
        source_shape = node.args[0].meta["tensor_meta"]["shape"]

        return MeanRelaxation(dim=dim, keepdim=keepdim, source_shape=source_shape, input_node=node.args[0])


class BackwardLBPMax(BackwardLBPStrategy):
    """Backward LBP strategy for amax reduction.

    Since amax is nonlinear, this concretizes the input subtree and wraps
    the result as an interval leaf (breaks the symbolic chain).
    """

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPMax requires input to be BackwardRelaxation, got {type(sym_input).__name__}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        input_node: fx.Node = node.args[0]  # ty:ignore[invalid-assignment]
        input_bounds = bounds(input_node)
        lower, upper = input_bounds.lower, input_bounds.upper

        if dim is not None:
            lower = lower.amax(dim=dim, keepdim=keepdim)
            upper = upper.amax(dim=dim, keepdim=keepdim)
        else:
            lower = lower.amax()
            upper = upper.amax()

        return IntervalLeafRelaxation(lower=lower, upper=upper)


class BackwardLBPMin(BackwardLBPStrategy):
    """Backward LBP strategy for amin reduction.

    Since amin is nonlinear, this concretizes the input subtree and wraps
    the result as an interval leaf (breaks the symbolic chain).
    """

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPMin requires input to be BackwardRelaxation, got {type(sym_input).__name__}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        input_node: fx.Node = node.args[0]  # ty:ignore[invalid-assignment]
        input_bounds = bounds(input_node)
        lower, upper = input_bounds.lower, input_bounds.upper

        if dim is not None:
            lower = lower.amin(dim=dim, keepdim=keepdim)
            upper = upper.amin(dim=dim, keepdim=keepdim)
        else:
            lower = lower.amin()
            upper = upper.amin()

        return IntervalLeafRelaxation(lower=lower, upper=upper)
