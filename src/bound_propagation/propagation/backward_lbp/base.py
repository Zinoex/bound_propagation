from __future__ import annotations

from abc import abstractmethod
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
    numel = 1
    for d in shape:
        numel *= d
    identity = torch.eye(numel, dtype=dtype, device=device).reshape(*shape, *shape)
    bounds = sym.backward(identity, identity, batch_ndim=0)
    return bounds.concretize()
