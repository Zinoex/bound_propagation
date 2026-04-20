"""Base classes for method-specific propagators.

Each propagation method (IBP, Forward LBP, Backward LBP) has its own
propagator class that orchestrates bound propagation through a
:class:`torch.fx.GraphModule`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch.fx as fx

from ...bounds import AbstractBounds
from ...regions import SimpleRegion


class BoundPropagator(ABC):
    """Abstract base class for method-specific bound propagators.

    A BoundPropagator walks a :class:`torch.fx.GraphModule` graph in
    topological order (forward) or reverse topological order (backward),
    dispatching each operation node to a registered bounding strategy.

    Args:
        graph_module: The traced ``fx.GraphModule`` (with metadata
            already annotated by :class:`MetadataPass`).
        registry: A :class:`TargetRegistry` mapping fx targets to
            bounding strategies.
    """

    def __init__(self, graph_module: fx.GraphModule):
        self._graph_module = graph_module

    @property
    def graph_module(self) -> fx.GraphModule:
        """The ``fx.GraphModule`` being propagated through."""
        return self._graph_module

    @abstractmethod
    def propagate(
        self,
        input_regions: Sequence[SimpleRegion],
        batch_ndim: int = 0,
    ) -> AbstractBounds:
        """Propagate bounds through the computation graph.

        Args:
            input_regions: Input regions defining bounds on each
                placeholder input.
            batch_ndim: Number of leading batch dimensions shared across
                inputs and outputs. ``0`` (the default) treats the entire
                tensor as semantic. Backward-mode propagators use this to
                build per-batch-element linear bounds; forward-mode
                propagators are shape-transparent and ignore it.

        Returns:
            The computed bounds for the model's single output.
        """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Short identifier for this propagation method."""

    # ------------------------------------------------------------------
    # Helpers shared by forward propagators
    # ------------------------------------------------------------------

    def _placeholder_nodes(self) -> list[fx.Node]:
        """Return placeholder nodes in order."""
        return [n for n in self._graph_module.graph.nodes if n.op == "placeholder"]

    def _output_node(self) -> fx.Node:
        """Return the single output node."""
        for n in self._graph_module.graph.nodes:
            if n.op == "output":
                return n
        raise RuntimeError("Graph has no output node")
