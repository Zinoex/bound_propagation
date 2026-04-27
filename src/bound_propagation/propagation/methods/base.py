"""Base classes for method-specific propagators.

Each propagation method (IBP, Forward LBP, Backward LBP, CROWN-IBP,
Forward-Backward LBP) has its own propagator class that orchestrates bound
propagation through a :class:`torch.fx.GraphModule`. Shared concrete-evaluation
plumbing lives here on :class:`BoundPropagator`; subclasses own their
method-specific dispatch and tape/context wiring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol

import torch
import torch.fx as fx

from ...bounds import AbstractBounds
from ...regions import SimpleRegion
from ..constants import CONSTANT_PRODUCING_TARGETS, evaluate_constant_producer


class _ArgResolver(Protocol):
    """Minimal interface needed by :meth:`BoundPropagator._evaluate_concrete`.

    Both :class:`PropagationContext` and :class:`BackwardTape` satisfy this
    structurally, so concrete-evaluation logic can be shared without coupling
    the base propagator to either container type.
    """

    def resolve_args(self, node: fx.Node) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def fetch_attr(self, target: str) -> Any: ...
    def get_module(self, target: str) -> torch.nn.Module: ...


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

    @staticmethod
    def _evaluate_concrete(node: fx.Node, resolver: _ArgResolver) -> torch.Tensor:
        """Concretely evaluate a non-abstract node via *resolver*.

        Constant-producing targets (``torch.zeros``, ``torch.tensor``, etc.)
        are evaluated from the node's literal kwargs without needing any
        stored values. Other nodes resolve their args/kwargs from the
        resolver and dispatch by ``node.op``.
        """
        target = node.target
        if node.op == "call_function" and target in CONSTANT_PRODUCING_TARGETS:
            return evaluate_constant_producer(node)

        args, kwargs = resolver.resolve_args(node)
        if node.op == "call_function":
            return target(*args, **kwargs)  # ty:ignore[call-non-callable]
        if node.op == "call_method":
            return getattr(args[0], target)(*args[1:], **kwargs)  # ty:ignore[invalid-argument-type]
        if node.op == "call_module":
            module = resolver.get_module(target)  # ty:ignore[invalid-argument-type]
            return module(*args, **kwargs)
        raise ValueError(f"Cannot evaluate node op={node.op!r}")
