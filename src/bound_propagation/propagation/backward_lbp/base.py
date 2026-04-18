from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ..strategy import BoundingStrategy

if TYPE_CHECKING:
    from .tape import BackwardTape


@dataclass
class BackwardContributions:
    """Result of a single backward step through a relaxation.

    Attributes
    ----------
    a_terms : dict[fx.Node, tuple[torch.Tensor, torch.Tensor]]
        Mapping from predecessor fx.Node to (delta_A_lower, delta_A_upper).
        When the same predecessor appears multiple times (e.g. x * x),
        contributions should be pre-accumulated before constructing this.
    bias_lower : torch.Tensor
        Bias contribution to the lower bound, shape (*batch, *bounded_out).
    bias_upper : torch.Tensor
        Bias contribution to the upper bound, shape (*batch, *bounded_out).
    """

    a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]]
    bias_lower: torch.Tensor
    bias_upper: torch.Tensor


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
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        """Single-step backward: transform A-matrices for this operation.

        Parameters
        ----------
        A_lower : torch.Tensor
            Lower A-matrix, shape (*batch, *bounded_out, *node_dims).
        A_upper : torch.Tensor
            Upper A-matrix, shape (*batch, *bounded_out, *node_dims).
        batch_ndim : int
            Number of leading batch dimensions.

        Returns
        -------
        BackwardContributions
            Contributions to predecessor nodes and bias deltas.
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

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        """Sign decomposition on A, contract over node dims into bias."""
        node_ndim = self.lower.ndim - batch_ndim
        bounded_ndim = A_lower.ndim - self.lower.ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            """Broadcast (*batch, *node) -> (*batch, *bounded_out, *node)."""
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        A_l_pos = A_lower.clamp(min=0)
        A_l_neg = A_lower.clamp(max=0)
        A_u_pos = A_upper.clamp(min=0)
        A_u_neg = A_upper.clamp(max=0)

        # Sign decomposition: positive A uses lower for lower bound, upper for upper bound
        bias_lower = A_l_pos * bc(self.lower) + A_l_neg * bc(self.upper)
        bias_upper = A_u_pos * bc(self.upper) + A_u_neg * bc(self.lower)

        # Sum over node dimensions
        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        if sum_dims:
            bias_lower = bias_lower.sum(dim=sum_dims)
            bias_upper = bias_upper.sum(dim=sum_dims)

        return BackwardContributions(a_terms={}, bias_lower=bias_lower, bias_upper=bias_upper)


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
    ) -> BackwardRelaxation:
        """Build a single-step backward relaxation for this operation.

        Parameters
        ----------
        node : fx.Node
            The fx.Node being processed.
        tape : BackwardTape
            The backward tape (provides resolve_args, concretize_at, etc.).

        Returns
        -------
        BackwardRelaxation
            A BackwardRelaxation for this operation.
        """


def accumulate_a_terms(
    a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]],
    node: fx.Node,
    delta_A_lower: torch.Tensor,
    delta_A_upper: torch.Tensor,
) -> None:
    """Accumulate A-matrix contributions for a predecessor node.

    Handles the case where the same predecessor appears multiple times
    (e.g. y = x * x) by summing contributions.

    Parameters
    ----------
    a_terms : dict[fx.Node, tuple[torch.Tensor, torch.Tensor]]
        The a_terms dict to accumulate into (modified in-place).
    node : fx.Node
        The predecessor fx.Node.
    delta_A_lower : torch.Tensor
        Lower A-matrix contribution.
    delta_A_upper : torch.Tensor
        Upper A-matrix contribution.
    """
    if node in a_terms:
        old_l, old_u = a_terms[node]
        a_terms[node] = (old_l + delta_A_lower, old_u + delta_A_upper)
    else:
        a_terms[node] = (delta_A_lower, delta_A_upper)
