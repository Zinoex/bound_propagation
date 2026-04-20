"""Backward LBP strategies for shape manipulation operations.

All shape operations are pure A-matrix dimension transforms with zero bias
contributions. No sign decomposition or concretization is needed.
"""

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
    accumulate_a_terms,
)

if TYPE_CHECKING:
    from .tape import BackwardTape


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _zero_bias(A: torch.Tensor, node_ndim: int) -> torch.Tensor:
    """Create zero bias tensor with shape ``(*batch, *bounded_out)``.

    Parameters
    ----------
    A : torch.Tensor
        An A-matrix whose leading dimensions encode batch and bounded-output
        shape, followed by ``node_ndim`` trailing node dimensions.
    node_ndim : int
        Number of trailing dimensions that belong to the node (not batch/bounded).

    Returns
    -------
    torch.Tensor
        Zero tensor with shape ``A.shape[:A.ndim - node_ndim]``.
    """
    bias_ndim = A.ndim - node_ndim
    return torch.zeros(A.shape[:bias_ndim], dtype=A.dtype, device=A.device)


# ---------------------------------------------------------------------------
# Relaxation dataclasses
# ---------------------------------------------------------------------------


@final
@dataclass
class ReshapeRelaxation(BackwardRelaxation):
    """Backward relaxation for reshape / flatten / view."""

    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        target_features = self.target_shape[batch_ndim:]
        source_features = self.source_shape[batch_ndim:]
        bounded_ndim = A_lower.ndim - batch_ndim - len(target_features)
        batch_shape = A_lower.shape[:batch_ndim]
        bounded_shape = A_lower.shape[batch_ndim : batch_ndim + bounded_ndim]

        new_shape = (*batch_shape, *bounded_shape, *source_features)
        new_A_lower = A_lower.reshape(new_shape)
        new_A_upper = A_upper.reshape(new_shape)

        node_ndim = len(target_features)
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=_zero_bias(A_lower, node_ndim),
            bias_upper=_zero_bias(A_upper, node_ndim),
        )


@final
@dataclass
class UnsqueezeRelaxation(BackwardRelaxation):
    """Backward relaxation for unsqueeze(dim)."""

    dim: int
    output_ndim: int
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        bounded_ndim = A_lower.ndim - self.output_ndim
        a_dim = batch_ndim + bounded_ndim + (self.dim - batch_ndim)
        new_A_lower = A_lower.squeeze(a_dim)
        new_A_upper = A_upper.squeeze(a_dim)

        node_ndim = self.output_ndim - batch_ndim
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=_zero_bias(A_lower, node_ndim),
            bias_upper=_zero_bias(A_upper, node_ndim),
        )


@final
@dataclass
class SqueezeRelaxation(BackwardRelaxation):
    """Backward relaxation for squeeze(dim).

    For squeeze without a dim (squeeze all size-1 dims), use
    ``ReshapeRelaxation`` instead.
    """

    dim: int
    input_ndim: int
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        output_ndim = self.input_ndim - 1
        bounded_ndim = A_lower.ndim - (output_ndim - batch_ndim) - batch_ndim
        a_dim = batch_ndim + bounded_ndim + (self.dim - batch_ndim)
        new_A_lower = A_lower.unsqueeze(a_dim)
        new_A_upper = A_upper.unsqueeze(a_dim)

        node_ndim = output_ndim - batch_ndim
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=_zero_bias(A_lower, node_ndim),
            bias_upper=_zero_bias(A_upper, node_ndim),
        )


@final
@dataclass
class TransposeRelaxation(BackwardRelaxation):
    """Backward relaxation for transpose(dim0, dim1)."""

    dim0: int
    dim1: int
    output_ndim: int
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        bounded_ndim = A_lower.ndim - self.output_ndim
        a_dim0 = batch_ndim + bounded_ndim + (self.dim0 - batch_ndim)
        a_dim1 = batch_ndim + bounded_ndim + (self.dim1 - batch_ndim)
        new_A_lower = A_lower.transpose(a_dim0, a_dim1)
        new_A_upper = A_upper.transpose(a_dim0, a_dim1)

        node_ndim = self.output_ndim - batch_ndim
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=_zero_bias(A_lower, node_ndim),
            bias_upper=_zero_bias(A_upper, node_ndim),
        )


@final
@dataclass
class PermuteRelaxation(BackwardRelaxation):
    """Backward relaxation for permute(dims)."""

    perm: tuple[int, ...]
    output_ndim: int
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
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

        node_ndim = self.output_ndim - batch_ndim
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=_zero_bias(A_lower, node_ndim),
            bias_upper=_zero_bias(A_upper, node_ndim),
        )


@final
@dataclass
class SelectRelaxation(BackwardRelaxation):
    """Backward relaxation for select(dim, index)."""

    dim: int
    index: int
    source_shape: tuple[int, ...]
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
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

        # select removes one dim, so output node_ndim = len(source_features) - 1
        node_ndim = len(source_features) - 1
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=_zero_bias(A_lower, node_ndim),
            bias_upper=_zero_bias(A_upper, node_ndim),
        )


@final
@dataclass
class GetItemRelaxation(BackwardRelaxation):
    """Backward relaxation for operator.getitem (indexing)."""

    index: object
    source_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
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

        node_ndim = len(output_features)
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=_zero_bias(A_lower, node_ndim),
            bias_upper=_zero_bias(A_upper, node_ndim),
        )


@final
@dataclass
class CatRelaxation(BackwardRelaxation):
    """Backward relaxation for torch.cat."""

    dim: int
    split_sizes: tuple[int, ...]
    output_ndim: int
    input_nodes: list[fx.Node]

    def predecessor_nodes(self) -> list[fx.Node]:
        return list(dict.fromkeys(self.input_nodes))

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        node_ndim = self.output_ndim - batch_ndim
        bounded_ndim = A_lower.ndim - batch_ndim - node_ndim
        a_dim = batch_ndim + bounded_ndim + (self.dim - batch_ndim)

        A_lowers = A_lower.split(list(self.split_sizes), dim=a_dim)
        A_uppers = A_upper.split(list(self.split_sizes), dim=a_dim)

        a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]] = {}
        for A_l, A_u, inp_node in zip(A_lowers, A_uppers, self.input_nodes, strict=True):
            accumulate_a_terms(a_terms, inp_node, A_l, A_u)

        return BackwardContributions(
            a_terms=a_terms,
            bias_lower=_zero_bias(A_lower, node_ndim),
            bias_upper=_zero_bias(A_upper, node_ndim),
        )


@final
@dataclass
class StackRelaxation(BackwardRelaxation):
    """Backward relaxation for torch.stack."""

    dim: int
    output_ndim: int
    input_nodes: list[fx.Node]

    def predecessor_nodes(self) -> list[fx.Node]:
        return list(dict.fromkeys(self.input_nodes))

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        node_ndim = self.output_ndim - batch_ndim
        bounded_ndim = A_lower.ndim - batch_ndim - node_ndim
        a_dim = batch_ndim + bounded_ndim + (self.dim - batch_ndim)

        a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, inp_node in enumerate(self.input_nodes):
            A_l = A_lower.select(a_dim, i)
            A_u = A_upper.select(a_dim, i)
            accumulate_a_terms(a_terms, inp_node, A_l, A_u)

        return BackwardContributions(
            a_terms=a_terms,
            bias_lower=_zero_bias(A_lower, node_ndim),
            bias_upper=_zero_bias(A_upper, node_ndim),
        )


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------


class BackwardLBPReshape(BackwardLBPStrategy):
    """Backward LBP strategy for reshape."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, _ = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPReshape requires input to be BackwardRelaxation, got {type(sym_input)}")

        source_shape = node.args[0].meta["tensor_meta"]["shape"]
        target_shape = node.meta["tensor_meta"]["shape"]

        return ReshapeRelaxation(
            source_shape=source_shape,
            target_shape=target_shape,
            input_node=node.args[0],
        )


class BackwardLBPFlatten(BackwardLBPStrategy):
    """Backward LBP strategy for flatten.

    Flatten is a pure dimension rearrangement, so the backward pass reuses
    :class:`ReshapeRelaxation` — only the source and target shapes (both already
    stored in ``tensor_meta``) are needed.
    """

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, _ = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPFlatten requires input to be BackwardRelaxation, got {type(sym_input)}")

        source_shape = node.args[0].meta["tensor_meta"]["shape"]
        target_shape = node.meta["tensor_meta"]["shape"]

        return ReshapeRelaxation(
            source_shape=source_shape,
            target_shape=target_shape,
            input_node=node.args[0],
        )


class BackwardLBPView(BackwardLBPStrategy):
    """Backward LBP strategy for view.

    Like reshape and flatten, view is a pure dimension rearrangement and reuses
    :class:`ReshapeRelaxation`.
    """

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, _ = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPView requires input to be BackwardRelaxation, got {type(sym_input)}")

        source_shape = node.args[0].meta["tensor_meta"]["shape"]
        target_shape = node.meta["tensor_meta"]["shape"]

        return ReshapeRelaxation(
            source_shape=source_shape,
            target_shape=target_shape,
            input_node=node.args[0],
        )


class BackwardLBPUnsqueeze(BackwardLBPStrategy):
    """Backward LBP strategy for unsqueeze."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPUnsqueeze requires input to be BackwardRelaxation, got {type(sym_input)}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        output_ndim = len(node.meta["tensor_meta"]["shape"])

        # Normalize negative dim
        if dim < 0:
            dim += output_ndim

        return UnsqueezeRelaxation(dim=dim, output_ndim=output_ndim, input_node=node.args[0])


class BackwardLBPSqueeze(BackwardLBPStrategy):
    """Backward LBP strategy for squeeze."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPSqueeze requires input to be BackwardRelaxation, got {type(sym_input)}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        input_ndim = len(node.args[0].meta["tensor_meta"]["shape"])

        if dim is not None:
            if dim < 0:
                dim += input_ndim
            return SqueezeRelaxation(dim=dim, input_ndim=input_ndim, input_node=node.args[0])

        # squeeze(None) removes all size-1 dims -> use reshape
        source_shape = node.args[0].meta["tensor_meta"]["shape"]
        target_shape = node.meta["tensor_meta"]["shape"]
        return ReshapeRelaxation(
            source_shape=source_shape,
            target_shape=target_shape,
            input_node=node.args[0],
        )


class BackwardLBPTranspose(BackwardLBPStrategy):
    """Backward LBP strategy for transpose."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPTranspose requires input to be BackwardRelaxation, got {type(sym_input)}")

        dim0 = args[1] if len(args) > 1 else kwargs.get("dim0", 0)
        dim1 = args[2] if len(args) > 2 else kwargs.get("dim1", 1)
        output_ndim = len(node.meta["tensor_meta"]["shape"])

        if dim0 < 0:
            dim0 += output_ndim
        if dim1 < 0:
            dim1 += output_ndim

        return TransposeRelaxation(
            dim0=dim0,
            dim1=dim1,
            output_ndim=output_ndim,
            input_node=node.args[0],
        )


class BackwardLBPPermute(BackwardLBPStrategy):
    """Backward LBP strategy for permute."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPPermute requires input to be BackwardRelaxation, got {type(sym_input)}")

        if len(args) == 2 and isinstance(args[1], (tuple, list)):
            dims = tuple(args[1])
        else:
            dims = tuple(args[1:])

        output_ndim = len(node.meta["tensor_meta"]["shape"])
        dims = tuple(d + output_ndim if d < 0 else d for d in dims)

        return PermuteRelaxation(perm=dims, output_ndim=output_ndim, input_node=node.args[0])


class BackwardLBPSelect(BackwardLBPStrategy):
    """Backward LBP strategy for select."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPSelect requires input to be BackwardRelaxation, got {type(sym_input)}")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        index = args[2] if len(args) > 2 else kwargs.get("index", 0)
        source_shape = node.args[0].meta["tensor_meta"]["shape"]

        input_ndim = len(source_shape)
        if dim < 0:
            dim += input_ndim

        return SelectRelaxation(
            dim=dim,
            index=index,
            source_shape=source_shape,
            input_node=node.args[0],
        )


class BackwardLBPGetItem(BackwardLBPStrategy):
    """Backward LBP strategy for operator.getitem."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, _ = tape.resolve_args(node)
        sym_input = args[0]
        index = args[1]

        if not isinstance(sym_input, BackwardRelaxation):
            raise TypeError(f"BackwardLBPGetItem requires input to be BackwardRelaxation, got {type(sym_input)}")

        source_shape = node.args[0].meta["tensor_meta"]["shape"]
        output_shape = node.meta["tensor_meta"]["shape"]

        return GetItemRelaxation(
            index=index,
            source_shape=source_shape,
            output_shape=output_shape,
            input_node=node.args[0],
        )


class BackwardLBPConcat(BackwardLBPStrategy):
    """Backward LBP strategy for torch.cat."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        input_nodes: list[fx.Node] = []
        split_sizes: list[int] = []

        for i, (t, raw_arg) in enumerate(zip(tensors, node.args[0], strict=True)):
            if not isinstance(t, BackwardRelaxation):
                raise TypeError(
                    f"BackwardLBPConcat requires all inputs to be BackwardRelaxation, but input {i} is {type(t)}"
                )
            input_nodes.append(raw_arg)
            split_sizes.append(raw_arg.meta["tensor_meta"]["shape"][dim])

        output_ndim = len(node.meta["tensor_meta"]["shape"])
        if dim < 0:
            dim += output_ndim

        return CatRelaxation(
            dim=dim,
            split_sizes=tuple(split_sizes),
            output_ndim=output_ndim,
            input_nodes=input_nodes,
        )


class BackwardLBPStack(BackwardLBPStrategy):
    """Backward LBP strategy for torch.stack."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        input_nodes: list[fx.Node] = []
        for i, (t, raw_arg) in enumerate(zip(tensors, node.args[0], strict=True)):
            if not isinstance(t, BackwardRelaxation):
                raise TypeError(
                    f"BackwardLBPStack requires all inputs to be BackwardRelaxation, but input {i} is {type(t)}"
                )
            input_nodes.append(raw_arg)

        output_ndim = len(node.meta["tensor_meta"]["shape"])
        if dim < 0:
            dim += output_ndim

        return StackRelaxation(dim=dim, output_ndim=output_ndim, input_nodes=input_nodes)
