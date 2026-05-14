"""Backward LBP strategies for pairwise (binary) operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.fx as fx
from beartype.typing import final

from ...bounds import IntervalBounds
from ...linear_operators import DenseOperator, LinearOperator
from ..linear_relaxations.alpha_resolvers import (
    resolve_div_etas,
    resolve_max_etas,
    resolve_min_etas,
    resolve_mul_etas,
)
from ..linear_relaxations.elementwise import ElementwiseParams, compute_constant_div_relaxation
from ..linear_relaxations.pairwise import (
    PairedParams,
    compute_div_relaxation,
    compute_maximum_relaxation,
    compute_minimum_relaxation,
    compute_mul_relaxation,
)
from .base import (
    BackwardContributions,
    BackwardLBPStrategy,
    BackwardRelaxation,
    IntermediateBoundsProvider,
    accumulate_a_terms,
)
from .elementwise import ElementwiseBackwardRelaxation
from .linear import ScaleRelaxation

if TYPE_CHECKING:
    from .tape import BackwardTape


@final
@dataclass
class PairedBackwardRelaxation(BackwardRelaxation):
    """Backward relaxation for pairwise (binary) operations.

    Represents a linear relaxation of the form:
        z_lower >= alpha_lower_a * a + alpha_lower_b * b + bias_lower
        z_upper <= alpha_upper_a * a + alpha_upper_b * b + bias_upper

    ``params`` carries alpha / bias tensors broadcast to the output node
    shape; ``left_shape`` / ``right_shape`` record each operand's actual
    pre-broadcast shape so :meth:`backward_through` can sum-reduce the
    per-operand ``A`` contributions back to the operand's natural shape.

    Handles the case where ``left_node == right_node`` (e.g., ``x * x``) by
    using :func:`accumulate_a_terms` to sum contributions.
    """

    params: PairedParams
    left_node: fx.Node
    right_node: fx.Node
    left_shape: tuple[int, ...]
    right_shape: tuple[int, ...]

    def predecessor_nodes(self) -> list[fx.Node]:
        """Return predecessor nodes, deduplicated for the x*x case."""
        return list(dict.fromkeys([self.left_node, self.right_node]))

    def backward_through(
        self, A_lower: LinearOperator, A_upper: LinearOperator, batch_ndim: int
    ) -> BackwardContributions:
        """Backward-propagate A-matrices through this pairwise relaxation."""
        output_shape = A_lower.output_shape

        p = self.params
        node_ndim = p.alpha_lower_a.ndim - batch_ndim
        bounded_ndim = A_lower.output_ndim + A_lower.input_ndim - p.alpha_lower_a.ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        A_l_pos = A_lower.clamp_min(0).to_dense().tensor
        A_l_neg = A_lower.clamp_max(0).to_dense().tensor
        A_u_pos = A_upper.clamp_min(0).to_dense().tensor
        A_u_neg = A_upper.clamp_max(0).to_dense().tensor

        # Left input coefficients
        new_A_lower_left = A_l_pos * bc(p.alpha_lower_a) + A_l_neg * bc(p.alpha_upper_a)
        new_A_upper_left = A_u_pos * bc(p.alpha_upper_a) + A_u_neg * bc(p.alpha_lower_a)

        # Right input coefficients
        new_A_lower_right = A_l_pos * bc(p.alpha_lower_b) + A_l_neg * bc(p.alpha_upper_b)
        new_A_upper_right = A_u_pos * bc(p.alpha_upper_b) + A_u_neg * bc(p.alpha_lower_b)

        # Bias
        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        delta_bias_lower = A_l_pos * bc(p.bias_lower) + A_l_neg * bc(p.bias_upper)
        delta_bias_upper = A_u_pos * bc(p.bias_upper) + A_u_neg * bc(p.bias_lower)
        if sum_dims:
            delta_bias_lower = delta_bias_lower.sum(dim=sum_dims)
            delta_bias_upper = delta_bias_upper.sum(dim=sum_dims)

        # Reduce per-operand contributions from the broadcast node shape
        # back to each operand's actual shape. When ``left_node == right_node``
        # (the ``x * x`` case), no reduction is needed because both shapes
        # match the (un-broadcast) node shape.
        out_ndim = len(output_shape)
        broadcast_node_shape = tuple(p.alpha_lower_a.shape[batch_ndim:])
        new_A_lower_left = _reduce_to_operand_shape(
            new_A_lower_left, broadcast_node_shape, self.left_shape, batch_ndim, out_ndim
        )
        new_A_upper_left = _reduce_to_operand_shape(
            new_A_upper_left, broadcast_node_shape, self.left_shape, batch_ndim, out_ndim
        )
        new_A_lower_right = _reduce_to_operand_shape(
            new_A_lower_right, broadcast_node_shape, self.right_shape, batch_ndim, out_ndim
        )
        new_A_upper_right = _reduce_to_operand_shape(
            new_A_upper_right, broadcast_node_shape, self.right_shape, batch_ndim, out_ndim
        )

        # Build a_terms with accumulation for same-node case (x*x)
        a_terms: dict[fx.Node, tuple[LinearOperator, LinearOperator]] = {}
        accumulate_a_terms(
            a_terms,
            self.left_node,
            DenseOperator(new_A_lower_left, output_shape=torch.Size(new_A_lower_left.shape[:out_ndim])),
            DenseOperator(new_A_upper_left, output_shape=torch.Size(new_A_upper_left.shape[:out_ndim])),
        )
        accumulate_a_terms(
            a_terms,
            self.right_node,
            DenseOperator(new_A_lower_right, output_shape=torch.Size(new_A_lower_right.shape[:out_ndim])),
            DenseOperator(new_A_upper_right, output_shape=torch.Size(new_A_upper_right.shape[:out_ndim])),
        )

        return BackwardContributions(a_terms=a_terms, bias_lower=delta_bias_lower, bias_upper=delta_bias_upper)


def _broadcast_paired_bounds(
    bounds_a: IntervalBounds,
    bounds_b: IntervalBounds,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[IntervalBounds, IntervalBounds]:
    """Broadcast two interval bounds to their common output shape.

    PyTorch's binary ops broadcast their operands to a shared output shape;
    ``compute_*_relaxation`` then produces alpha and bias tensors at that
    shape so :class:`PairedBackwardRelaxation` can apply the standard
    ``A * alpha`` step uniformly. Operand-specific contributions are later
    sum-reduced back to each operand's actual shape via
    :func:`_reduce_to_operand_shape`.
    """
    output_shape = torch.broadcast_shapes(left_shape, right_shape)
    if tuple(bounds_a.lower.shape) == tuple(output_shape) and tuple(bounds_b.lower.shape) == tuple(output_shape):
        return bounds_a, bounds_b

    def expand(b: IntervalBounds) -> IntervalBounds:
        if tuple(b.lower.shape) == tuple(output_shape):
            return b
        return IntervalBounds(
            lower=b.lower.broadcast_to(output_shape).contiguous(),
            upper=b.upper.broadcast_to(output_shape).contiguous(),
        )

    return expand(bounds_a), expand(bounds_b)


def _reduce_to_operand_shape(
    a: torch.Tensor,
    broadcast_node_shape: tuple[int, ...],
    operand_shape: tuple[int, ...],
    batch_ndim: int,
    out_ndim: int,
) -> torch.Tensor:
    """Sum a tensor's trailing node axes from broadcast shape back to operand shape.

    ``a`` has shape ``(*batch_dims, *output_dims, *broadcast_node_shape)``.
    The returned tensor has shape ``(*batch_dims, *output_dims, *operand_shape)``,
    matching what a non-broadcast operand of that shape would have produced.

    Broadcast-vs-operand alignment follows PyTorch's right-aligned convention:
    extra leading dims in the broadcast shape are summed out entirely, and
    dims where the operand has size 1 but the broadcast has size > 1 are
    summed with ``keepdim=True``.
    """
    if broadcast_node_shape == operand_shape:
        return a

    rank_extra = len(broadcast_node_shape) - len(operand_shape)
    node_axis_start = batch_ndim + out_ndim

    # Sum the rank-mismatch (extra leading node dims): these dims have no
    # corresponding operand axis, so they collapse entirely.
    if rank_extra > 0:
        sum_dims = tuple(range(node_axis_start, node_axis_start + rank_extra))
        a = a.sum(dim=sum_dims, keepdim=False)

    # Sum the size-1 broadcast dims of the operand (keep them as size-1).
    keep_sum_dims: list[int] = []
    for i, op_size in enumerate(operand_shape):
        bc_size = broadcast_node_shape[rank_extra + i]
        if op_size == 1 and bc_size > 1:
            keep_sum_dims.append(node_axis_start + i)
    if keep_sum_dims:
        a = a.sum(dim=tuple(keep_sum_dims), keepdim=True)

    return a


class BackwardLBPMul(BackwardLBPStrategy):
    """Backward LBP strategy for element-wise multiplication."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        """Build a backward relaxation for multiplication.

        Handles three cases:
        - Both operands abstract: McCormick relaxation via ``PairedBackwardRelaxation``.
        - One operand constant: ``ScaleRelaxation`` with the constant as scale.

        Parameters
        ----------
        node : fx.Node
            The multiplication node.
        tape : BackwardTape
            Tape providing argument resolution.
        bounds : IntermediateBoundsProvider
            Provider for predecessor interval bounds.

        Returns
        -------
        BackwardRelaxation
            Either a ``PairedBackwardRelaxation`` or ``ScaleRelaxation``.
        """
        args, _ = tape.resolve_args(node)
        left, right = args[0], args[1]
        left_is_abstract = isinstance(left, BackwardRelaxation)
        right_is_abstract = isinstance(right, BackwardRelaxation)

        left_node: fx.Node = node.args[0]  # ty:ignore[invalid-assignment]
        right_node: fx.Node = node.args[1]  # ty:ignore[invalid-assignment]

        if left_is_abstract and right_is_abstract:
            left_shape = tuple(tape.shape_of(left_node))
            right_shape = tuple(tape.shape_of(right_node))
            bounds_a, bounds_b = _broadcast_paired_bounds(
                bounds(left_node), bounds(right_node), left_shape, right_shape
            )
            eta_lo, eta_up = resolve_mul_etas(tape.alpha_provider, node, bounds_a)
            params = compute_mul_relaxation(
                bounds_a,
                bounds_b,
                eta_lower=eta_lo if eta_lo is not None else 0.5,
                eta_upper=eta_up if eta_up is not None else 0.5,
            )
            return PairedBackwardRelaxation(
                params=params,
                left_node=left_node,
                right_node=right_node,
                left_shape=left_shape,
                right_shape=right_shape,
            )

        if left_is_abstract:
            constant = torch.as_tensor(right, dtype=tape.dtype_of(left_node))
            scale = constant.expand(tape.shape_of(left_node))
            return ScaleRelaxation(scale=scale, input_node=left_node)

        if right_is_abstract:
            constant = torch.as_tensor(left, dtype=tape.dtype_of(right_node))
            scale = constant.expand(tape.shape_of(right_node))
            return ScaleRelaxation(scale=scale, input_node=right_node)

        raise TypeError(f"BackwardLBPMul requires at least one abstract operand, got {type(left)} and {type(right)}")


class BackwardLBPDiv(BackwardLBPStrategy):
    """Backward LBP strategy for element-wise division."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        """Build a backward relaxation for division.

        Handles three cases:
        - Both operands abstract: ``PairedBackwardRelaxation`` via ``compute_div_relaxation``.
        - Abstract / constant: ``ScaleRelaxation`` with ``1/constant`` as scale.
        - Constant / abstract: ``ElementwiseBackwardRelaxation`` via ``compute_constant_div_relaxation``.

        Parameters
        ----------
        node : fx.Node
            The division node.
        tape : BackwardTape
            Tape providing argument resolution.
        bounds : IntermediateBoundsProvider
            Provider for predecessor interval bounds.

        Returns
        -------
        BackwardRelaxation
            A ``PairedBackwardRelaxation``, ``ScaleRelaxation``, or ``ElementwiseBackwardRelaxation``.
        """
        args, _ = tape.resolve_args(node)
        left, right = args[0], args[1]
        left_is_abstract = isinstance(left, BackwardRelaxation)
        right_is_abstract = isinstance(right, BackwardRelaxation)
        left_node: fx.Node = node.args[0]  # ty:ignore[invalid-assignment]
        right_node: fx.Node = node.args[1]  # ty:ignore[invalid-assignment]

        if left_is_abstract and right_is_abstract:
            left_shape = tuple(tape.shape_of(left_node))
            right_shape = tuple(tape.shape_of(right_node))
            bounds_a, bounds_b = _broadcast_paired_bounds(
                bounds(left_node), bounds(right_node), left_shape, right_shape
            )
            eta_lo, eta_up = resolve_div_etas(tape.alpha_provider, node, bounds_a)
            params = compute_div_relaxation(
                bounds_a,
                bounds_b,
                eta_lower=eta_lo if eta_lo is not None else 0.5,
                eta_upper=eta_up if eta_up is not None else 0.5,
            )
            return PairedBackwardRelaxation(
                params=params,
                left_node=left_node,
                right_node=right_node,
                left_shape=left_shape,
                right_shape=right_shape,
            )

        if left_is_abstract:
            # abstract / constant = abstract * (1/constant)
            divisor = torch.as_tensor(right, dtype=tape.dtype_of(left_node))
            scale = (1.0 / divisor).expand(tape.shape_of(left_node))
            return ScaleRelaxation(scale=scale, input_node=left_node)

        if right_is_abstract:
            # constant / abstract
            input_bounds = bounds(right_node)
            params = compute_constant_div_relaxation(input_bounds, left)
            return ElementwiseBackwardRelaxation(params=params, input_node=right_node)

        raise TypeError(f"BackwardLBPDiv requires at least one abstract operand, got {type(left)} and {type(right)}")


class _PairwiseComparisonBackwardLBP(BackwardLBPStrategy):
    """Shared base for element-wise maximum/minimum backward LBP strategies.

    Subclasses provide the ``_compute_relaxation`` method to select between
    ``compute_maximum_relaxation`` and ``compute_minimum_relaxation``, and
    ``_resolve_etas`` to look up the alpha-CROWN overrides.
    """

    @staticmethod
    def _compute_relaxation(
        bounds_a: IntervalBounds,
        bounds_b: IntervalBounds,
        eta_lower,
        eta_upper,
    ) -> PairedParams:
        raise NotImplementedError

    @staticmethod
    def _resolve_etas(
        provider, node: fx.Node, reference: IntervalBounds
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        raise NotImplementedError

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        """Build a backward relaxation for an element-wise comparison operation.

        Handles two cases:
        - Both operands abstract: ``PairedBackwardRelaxation`` via the subclass relaxation.
        - One operand constant: expand constant to ``IntervalBounds(c, c)`` and use
          ``PairedBackwardRelaxation`` (the constant side will have zero alpha).

        Parameters
        ----------
        node : fx.Node
            The comparison node.
        tape : BackwardTape
            Tape providing argument resolution.
        bounds : IntermediateBoundsProvider
            Provider for predecessor interval bounds.

        Returns
        -------
        PairedBackwardRelaxation
            Relaxation for the comparison operation.
        """
        args, _ = tape.resolve_args(node)
        left, right = args[0], args[1]
        left_is_abstract = isinstance(left, BackwardRelaxation)
        right_is_abstract = isinstance(right, BackwardRelaxation)
        left_node: fx.Node = node.args[0]  # ty:ignore[invalid-assignment]
        right_node: fx.Node = node.args[1]  # ty:ignore[invalid-assignment]

        if left_is_abstract and right_is_abstract:
            left_shape = tuple(tape.shape_of(left_node))
            right_shape = tuple(tape.shape_of(right_node))
            bounds_a, bounds_b = _broadcast_paired_bounds(
                bounds(left_node), bounds(right_node), left_shape, right_shape
            )
            eta_lo, eta_up = self._resolve_etas(tape.alpha_provider, node, bounds_a)
            paired = self._compute_relaxation(
                bounds_a,
                bounds_b,
                eta_lower=eta_lo if eta_lo is not None else 0.5,
                eta_upper=eta_up if eta_up is not None else 0.5,
            )
            return PairedBackwardRelaxation(
                params=paired,
                left_node=left_node,
                right_node=right_node,
                left_shape=left_shape,
                right_shape=right_shape,
            )

        if left_is_abstract:
            bounds_a = bounds(left_node)
            c = torch.as_tensor(right, dtype=bounds_a.lower.dtype, device=bounds_a.lower.device)
            c = c.expand_as(bounds_a.lower)
            eta_lo, eta_up = self._resolve_etas(tape.alpha_provider, node, bounds_a)
            paired = self._compute_relaxation(
                bounds_a,
                IntervalBounds(c, c),
                eta_lower=eta_lo if eta_lo is not None else 0.5,
                eta_upper=eta_up if eta_up is not None else 0.5,
            )
            # Fold constant contribution into bias to avoid propagating through constant node
            params = ElementwiseParams(
                alpha_lower=paired.alpha_lower_a,
                alpha_upper=paired.alpha_upper_a,
                beta_lower=paired.alpha_lower_b * c + paired.bias_lower,
                beta_upper=paired.alpha_upper_b * c + paired.bias_upper,
            )
            return ElementwiseBackwardRelaxation(params=params, input_node=left_node)

        if right_is_abstract:
            bounds_b = bounds(right_node)
            c = torch.as_tensor(left, dtype=bounds_b.lower.dtype, device=bounds_b.lower.device)
            c = c.expand_as(bounds_b.lower)
            eta_lo, eta_up = self._resolve_etas(tape.alpha_provider, node, bounds_b)
            paired = self._compute_relaxation(
                IntervalBounds(c, c),
                bounds_b,
                eta_lower=eta_lo if eta_lo is not None else 0.5,
                eta_upper=eta_up if eta_up is not None else 0.5,
            )
            # Fold constant contribution into bias to avoid propagating through constant node
            params = ElementwiseParams(
                alpha_lower=paired.alpha_lower_b,
                alpha_upper=paired.alpha_upper_b,
                beta_lower=paired.alpha_lower_a * c + paired.bias_lower,
                beta_upper=paired.alpha_upper_a * c + paired.bias_upper,
            )
            return ElementwiseBackwardRelaxation(params=params, input_node=right_node)

        raise TypeError(
            f"{self.__class__.__name__} requires at least one abstract operand, got {type(left)} and {type(right)}"
        )


class BackwardLBPMaximum(_PairwiseComparisonBackwardLBP):
    """Backward LBP strategy for element-wise maximum."""

    _compute_relaxation = staticmethod(compute_maximum_relaxation)
    _resolve_etas = staticmethod(resolve_max_etas)


class BackwardLBPMinimum(_PairwiseComparisonBackwardLBP):
    """Backward LBP strategy for element-wise minimum."""

    _compute_relaxation = staticmethod(compute_minimum_relaxation)
    _resolve_etas = staticmethod(resolve_min_etas)
