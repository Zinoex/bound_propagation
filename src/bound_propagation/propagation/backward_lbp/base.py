"""Backward LBP (CROWN) base abstractions: relaxations, tape, and providers.

Backward LBP propagates a per-output linear bound back through the graph.
The running coefficient ``A`` (the "A-matrix") starts as the identity at the
output and is left-multiplied by each operation's local Jacobian relaxation
on the way to the inputs:

.. math::

    A^{(k)}_L = \\big[ A^{(k+1)}_L \\big]^+ J^{(k)}_L
              + \\big[ A^{(k+1)}_L \\big]^- J^{(k)}_U
    \\quad\\text{(symmetric for } A_U\\text{)}

This is **sign decomposition** (Zhang et al. 2018, "CROWN"): the upper
relaxation is multiplied where the running A is negative, the lower where it
is positive. It is load-bearing across every nonlinear backward step (here in
:class:`IntervalLeafRelaxation`, and in
:mod:`propagation.backward_lbp.elementwise`,
:mod:`propagation.backward_lbp.pairwise`,
:mod:`propagation.backward_lbp.linear`).

The graph is recorded onto a :class:`BackwardTape` (a Wengert list) during
the forward walk; the tape replays the recorded relaxations in reverse BFS
order to build the final :class:`LinearBounds`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
import torch.fx as fx
from plum import dispatch

from ...bounds import IntervalBounds, LinearBounds
from ...linear_operators import DenseOperator, IdentityOperator, LinearOperator
from ..strategy import BoundingStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext
    from .tape import BackwardTape


@dataclass
class BackwardContributions:
    """Result of a single backward step through a relaxation.

    Attributes
    ----------
    a_terms : dict[fx.Node, tuple[LinearOperator, LinearOperator]]
        Mapping from predecessor fx.Node to (delta_A_lower, delta_A_upper) as
        :class:`LinearOperator` instances. When the same predecessor appears
        multiple times (e.g. x * x), contributions should be pre-accumulated
        before constructing this.
    bias_lower : torch.Tensor
        Bias contribution to the lower bound; shape broadcasts against the
        A-operator's ``output_shape``.
    bias_upper : torch.Tensor
        Bias contribution to the upper bound.
    """

    a_terms: dict[fx.Node, tuple[LinearOperator, LinearOperator]]
    bias_lower: torch.Tensor
    bias_upper: torch.Tensor


def _wrap_a_term_tensors(
    a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]],
    output_ndim: int,
) -> dict[fx.Node, tuple[LinearOperator, LinearOperator]]:
    """Wrap tensor a_terms in :class:`DenseOperator`.

    Convenience for strategies that compute the new A-matrix as a raw tensor
    via einsum (e.g. ``A^{(k)} = A^{(k+1)} · J^{(k)}`` rendered as a contraction)
    and need to hand the result back as a :class:`LinearOperator`.

    ``output_ndim`` is the number of leading dims in each tensor that belong
    to the A-matrix output shape (shared across all predecessors in a single
    backward step). Trailing dims are the predecessor's input feature shape.
    Using the actual tensor shape (rather than a frozen template) accommodates
    ops that expand placeholder batch dims (e.g. McCormick matmul).
    """
    result: dict[fx.Node, tuple[LinearOperator, LinearOperator]] = {}
    for node, (lower_t, upper_t) in a_terms.items():
        output_shape = torch.Size(lower_t.shape[:output_ndim])
        result[node] = (
            DenseOperator(lower_t, output_shape=output_shape),
            DenseOperator(upper_t, output_shape=output_shape),
        )
    return result


class BackwardRelaxation(ABC):
    """Abstract single-step backward relaxation for one operation.

    Each concrete subclass represents one operation in the computation graph.
    The tape calls ``backward_through`` during BFS traversal -- no recursion.
    """

    @abstractmethod
    def predecessor_nodes(self) -> list[fx.Node]:
        """Unique fx.Node predecessors this relaxation propagates A-matrices to.

        The tape uses this to build the backward subgraph and pending counts.
        Chain-breaking relaxations (e.g. IntervalLeafRelaxation) return [].
        """

    @abstractmethod
    def backward_through(
        self,
        A_lower: LinearOperator,
        A_upper: LinearOperator,
        batch_ndim: int,
    ) -> BackwardContributions:
        """Single-step backward: transform A-matrices for this operation.

        Parameters
        ----------
        A_lower : LinearOperator
            Lower A-matrix as a :class:`LinearOperator` with
            ``output_shape = (*batch, *bounded_out)`` and
            ``input_shape = (*node_dims,)``.
        A_upper : LinearOperator
            Upper A-matrix (same shape convention).
        batch_ndim : int
            Number of leading batch dimensions.

        Returns
        -------
        BackwardContributions
            Contributions to predecessor nodes (as :class:`LinearOperator`)
            and bias deltas (as :class:`torch.Tensor`).
        """


@dataclass
class IntervalLeafRelaxation(BackwardRelaxation):
    """Chain-breaking leaf: contributes only to bias, no predecessors.

    Used for operations like amax/amin that concretize their input subtree
    and produce interval bounds, breaking the symbolic chain.

    Attributes
    ----------
    lower : torch.Tensor
        Concrete lower bound tensor, shape (*batch, *node_dims).
    upper : torch.Tensor
        Concrete upper bound tensor, shape (*batch, *node_dims).
    """

    lower: torch.Tensor
    upper: torch.Tensor

    def predecessor_nodes(self) -> list[fx.Node]:
        return []

    @dispatch
    def backward_through(
        self,
        A_lower: LinearOperator,
        A_upper: LinearOperator,
        batch_ndim: int,
    ) -> BackwardContributions:
        """Sign decomposition on A, contract over node dims into bias."""
        node_ndim = self.lower.ndim - batch_ndim
        bounded_ndim = A_lower.output_ndim + A_lower.input_ndim - self.lower.ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            """Broadcast (*batch, *node) -> (*batch, *bounded_out, *node)."""
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        A_l_pos = A_lower.clamp_min(0).to_dense().tensor
        A_l_neg = A_lower.clamp_max(0).to_dense().tensor
        A_u_pos = A_upper.clamp_min(0).to_dense().tensor
        A_u_neg = A_upper.clamp_max(0).to_dense().tensor

        # Sign decomposition: positive A uses lower for lower bound, upper for upper bound
        bias_lower = A_l_pos * bc(self.lower) + A_l_neg * bc(self.upper)
        bias_upper = A_u_pos * bc(self.upper) + A_u_neg * bc(self.lower)

        # Sum over node dimensions
        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        if sum_dims:
            bias_lower = bias_lower.sum(dim=sum_dims)
            bias_upper = bias_upper.sum(dim=sum_dims)

        return BackwardContributions(a_terms={}, bias_lower=bias_lower, bias_upper=bias_upper)

    @dispatch
    def backward_through(  # noqa: F811
        self,
        A_lower: IdentityOperator,
        A_upper: IdentityOperator,
        batch_ndim: int,  # noqa: ARG002
    ) -> BackwardContributions:
        """Identity at the leaf: bias is just ``self.lower`` / ``self.upper``.

        No sign decomposition is needed because Identity is entrywise
        non-negative, so ``A_pos = A`` and ``A_neg = 0`` and the contraction
        against the interval bounds reduces to the bounds themselves.
        """
        return BackwardContributions(a_terms={}, bias_lower=self.lower, bias_upper=self.upper)


class IntermediateBoundsProvider(Protocol):
    """Callable that returns ``IntervalBounds`` for a given fx.Node.

    Backward LBP strategies request the interval bounds of a predecessor
    node through this abstraction. Different callers plug in different
    sources of bounds:

    - ``CrownBoundsProvider``: concretizes via the tape (recursive CROWN).
    - ``PrecomputedBoundsProvider``: looks up IBP bounds computed during a
      prior forward sweep (CROWN-IBP).
    """

    def __call__(self, node: fx.Node) -> IntervalBounds: ...


class CrownBoundsProvider:
    """Bounds provider that concretizes via the tape (standard CROWN).

    When ``no_grad_concretizations`` is ``True`` the recursive backward
    concretization runs under :func:`torch.no_grad`, so alpha-CROWN
    gradients do not flow through intermediate bound computation. This is
    used by the "final-only" optimization mode, where alphas affect only
    the last backward pass. The default ``False`` preserves gradients for
    the "intermediate" mode.
    """

    def __init__(
        self,
        tape: BackwardTape,
        batch_ndim: int = 0,
        *,
        no_grad_concretizations: bool = False,
    ) -> None:
        self._tape = tape
        self._batch_ndim = batch_ndim
        self._no_grad = no_grad_concretizations

    def __call__(self, node: fx.Node) -> IntervalBounds:
        if self._no_grad:
            with torch.no_grad():
                return self._tape.concretize_at(node, batch_ndim=self._batch_ndim)
        return self._tape.concretize_at(node, batch_ndim=self._batch_ndim)


class PrecomputedBoundsProvider:
    """Bounds provider backed by a pre-computed mapping (e.g. IBP bounds).

    The mapping is keyed by fx.Node name. Construct directly from a dict or
    from an existing ``PropagationContext`` (which already stores bounds
    keyed by node name).
    """

    def __init__(self, bounds: dict[str, IntervalBounds]) -> None:
        self._bounds = bounds

    @classmethod
    def from_context(cls, ctx: PropagationContext[IntervalBounds]) -> PrecomputedBoundsProvider:
        """Create a provider that reads from *ctx* lazily via ``ctx.resolve``."""
        return _ContextBoundsProvider(ctx)

    def __call__(self, node: fx.Node) -> IntervalBounds:
        try:
            return self._bounds[node.name]
        except KeyError as e:
            raise KeyError(
                f"No precomputed IntervalBounds for node {node.name!r}; known nodes: {sorted(self._bounds)}"
            ) from e


class _ContextBoundsProvider(PrecomputedBoundsProvider):
    """Adapter that reads ``IntervalBounds`` directly from a PropagationContext."""

    def __init__(self, ctx: PropagationContext[IntervalBounds]) -> None:
        self._ctx = ctx

    def __call__(self, node: fx.Node) -> IntervalBounds:
        value = self._ctx.resolve(node)
        if not isinstance(value, IntervalBounds):
            raise TypeError(f"Expected IntervalBounds for node {node.name!r}, got {type(value).__name__}")
        return value


class ForwardLBPBoundsProvider:
    """Bounds provider backed by a forward LBP context (Forward-Backward LBP).

    The forward LBP context stores :class:`LinearBounds` for abstract nodes
    and :class:`torch.Tensor` for concrete nodes (``get_attr`` or
    non-abstract calls). This provider returns :class:`IntervalBounds` by
    concretizing linear bounds and by wrapping tensors as degenerate
    intervals. Concretizations are cached per fx.Node name so each
    intermediate node is concretized at most once per backward sweep.
    """

    def __init__(self, ctx: PropagationContext[LinearBounds]) -> None:
        self._ctx = ctx
        self._cache: dict[str, IntervalBounds] = {}

    def __call__(self, node: fx.Node) -> IntervalBounds:
        if node.name in self._cache:
            return self._cache[node.name]

        value = self._ctx.resolve(node)
        if isinstance(value, LinearBounds):
            result = value.concretize()
        elif isinstance(value, IntervalBounds):
            result = value
        elif isinstance(value, torch.Tensor):
            result = IntervalBounds(value, value)
        else:
            raise TypeError(
                f"Cannot convert value of type {type(value).__name__} for node {node.name!r} to IntervalBounds"
            )

        self._cache[node.name] = result
        return result


class BackwardLBPStrategy(BoundingStrategy):
    """Base class for backward LBP strategies.

    Strategies build ``BackwardRelaxation`` objects during the forward
    traversal. The tape later calls ``backward_through`` on each relaxation
    during the backward BFS.
    """

    @property
    def method_name(self) -> str:
        return "backward_lbp"

    @abstractmethod
    def build_relaxation(
        self,
        node: fx.Node,
        tape: BackwardTape,
        bounds: IntermediateBoundsProvider,
    ) -> BackwardRelaxation:
        """Build a single-step backward relaxation for this operation.

        Parameters
        ----------
        node : fx.Node
            The fx.Node being processed.
        tape : BackwardTape
            The backward tape (provides resolve_args, get_module, fetch_attr).
        bounds : IntermediateBoundsProvider
            Callable returning ``IntervalBounds`` for a predecessor fx.Node.
            Use this whenever the strategy needs concrete interval bounds of
            an input to construct the linear relaxation; do not call
            ``tape.concretize_at`` directly.

        Returns
        -------
        BackwardRelaxation
            A BackwardRelaxation for this operation.
        """


def accumulate_a_terms(
    a_terms: dict[fx.Node, tuple[LinearOperator, LinearOperator]],
    node: fx.Node,
    delta_A_lower: LinearOperator,
    delta_A_upper: LinearOperator,
) -> None:
    """Accumulate A-matrix contributions for a predecessor node.

    Handles the case where the same predecessor appears multiple times
    (e.g. y = x * x) by summing contributions via :meth:`LinearOperator.add`.

    Parameters
    ----------
    a_terms : dict[fx.Node, tuple[LinearOperator, LinearOperator]]
        The a_terms dict to accumulate into (modified in-place).
    node : fx.Node
        The predecessor fx.Node.
    delta_A_lower : LinearOperator
        Lower A-matrix contribution.
    delta_A_upper : LinearOperator
        Upper A-matrix contribution.
    """
    if node in a_terms:
        old_l, old_u = a_terms[node]
        a_terms[node] = (old_l.add(delta_A_lower), old_u.add(delta_A_upper))
    else:
        a_terms[node] = (delta_A_lower, delta_A_upper)
