"""Abstract bounding strategy interface.

Defines the contract for all bounding strategies (IBP, Forward LBP,
Backward LBP / CROWN, etc.).

Strategies operate directly on ``torch.fx.Node`` objects and use a
:class:`~.context.PropagationContext` to resolve arguments and store
results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import torch.fx as fx

from ..bounds import AbstractBounds

if TYPE_CHECKING:
    from .context import PropagationContext

T = TypeVar("T", bound=AbstractBounds)


class BoundingStrategy(ABC):
    """Marker base for all bounding strategies."""

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Short identifier (e.g. ``"ibp"``, ``"forward_lbp"``)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method={self.method_name})"


class ForwardBoundingStrategy(BoundingStrategy, ABC, Generic[T]):
    """Compute output bounds from input bounds in forward order.

    Subclasses implement ``propagate_forward`` which receives the fx
    node and a :class:`PropagationContext`.  The context provides
    ``resolve_args`` to obtain concrete values or bounds for each
    argument.
    """

    @abstractmethod
    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> T:
        """Compute output bounds for *node*.

        Args:
            node: The ``torch.fx.Node`` being propagated.
            ctx: Propagation context (bounds store, module access, …).

        Returns:
            Bounds for the node's output.
        """


class BackwardBoundingStrategy(BoundingStrategy, ABC, Generic[T]):
    """Propagate linear bounds backward (CROWN-style).

    For an operation ``z = f(x₁, x₂, …)`` with accumulated linear
    bounds on *z*, the backward rule decomposes those bounds into
    contributions to each abstract input.
    """

    @abstractmethod
    def propagate_backward(
        self,
        node: fx.Node,
        output_bounds: T,
        ctx: PropagationContext,
    ) -> dict[str, T]:
        """Propagate *output_bounds* backward through *node*.

        Args:
            node: The ``torch.fx.Node`` (``z = f(…)``).
            output_bounds: Linear bounds accumulated on *z*.
            ctx: Propagation context (concrete bounds from forward pass, …).

        Returns:
            Mapping from input node **name** to the linear bounds to
            accumulate for that input.  Only abstract inputs need
            entries; constant/parameter inputs are omitted.
        """
