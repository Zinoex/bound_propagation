from __future__ import annotations
from bound_propagation.bounds import IntervalBounds

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ..linear_relaxations.base import SymbolicLinearRelaxation
from ..strategy import BoundingStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class BackwardLBPStrategy(BoundingStrategy):
    """Build symbolic relaxation nodes during forward traversal for backward LBP (CROWN).

    Unlike forward LBP strategies that produce ``LinearBounds`` directly,
    backward LBP strategies produce ``SymbolicLinearRelaxation`` nodes.
    The symbolic tree is then backward-concretized at the output to obtain
    ``LinearBounds``.
    """

    @property
    def method_name(self) -> str:
        return "backward_lbp"

    @abstractmethod
    def build_symbolic(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> SymbolicLinearRelaxation:
        """Build the symbolic relaxation node for this operation.

        Args:
            node: The fx.Node being processed.
            ctx: Propagation context containing stored symbolic relaxations
                 for upstream nodes, and concrete tensors for constants.

        Returns:
            A SymbolicLinearRelaxation representing this operation.
        """


def concretize_symbolic(
    sym: SymbolicLinearRelaxation,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concretize a symbolic relaxation to interval bounds via recursive CROWN.

    Constructs an identity A-matrix over the full output shape, calls
    ``backward`` to obtain ``LinearBounds``, and concretizes to intervals.

    Args:
        sym: The symbolic relaxation to concretize.
        shape: Full output shape (may include batch dims).
        dtype: Tensor dtype.
        device: Tensor device.

    Returns:
        ``(lower, upper)`` interval bounds.
    """

    identity = torch.eye(numel, dtype=dtype, device=device).reshape(*shape, *shape)
    bounds = sym.backward(identity, identity, batch_ndim=0)
    return bounds.concretize()


class BackwardLinearRelaxation(ABC):
    """Base class for symbolic relaxations used in backward LBP.

    Each operation's backward LBP strategy builds a tree of these symbolic
    relaxations, which are then concretized at the output to obtain final
    bounds.
    """

    def __init__(self):
        super().__init__()

        self.A_lower: torch.Tensor | None = None
        self.A_upper: torch.Tensor | None = None

    @abstractmethod
    def concrete_relaxation(self) -> IntervalBounds:
        

    @abstractmethod
    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        """Backward-concretize this symbolic relaxation to obtain input bounds.

        Args:
            A_lower: The lower A-matrix from the output (shape: (*batch, *output) + node_shape).
            A_upper: The upper A-matrix from the output (shape: (*batch, *output) + node_shape).
            batch_ndim: Number of leading batch dimensions in the A-matrices.

        Returns:
            LinearBounds representing the concretized input bounds.
        """

    @abstractmethod
    def symbolic_forward(self, ctx: PropagationContext) -> BackwardLinearRelaxation:
        """Compute the symbolic relaxation for this operation given upstream relaxations.

        Args:
            ctx: Propagation context containing stored symbolic relaxations
                 for upstream nodes, and concrete tensors for constants.
        Returns:
            A new BackwardLinearRelaxation representing this operation.
        """
